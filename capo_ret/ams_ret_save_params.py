# coding=utf-8
"""
Lightweight version of clip_ret_ams_v3.py — only saves updated params.
Reward model on CPU after precomputation → GPU ~2-3 GB.
"""
import os
import time
import json
import logging
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.optimizer import Optimizer

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from params import parse_args
from clip_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TTA
from clip_ret_policy import tokenize_all_text, get_all_text_embeds, get_all_image_embeds


# ------------------------------------------------------------------ #
#  GradientGuidedAdam  (same as clip_ret_ams_v3.py)
# ------------------------------------------------------------------ #
class GradientGuidedAdam(Optimizer):
    def __init__(self, params, lr=5e-3, betas=(0.9, 0.999), eps=1e-8,
                 gamma=0.1, T_horizon=10):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        gamma=gamma, T_horizon=T_horizon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr, betas, eps, gamma = (
                group['lr'], group['betas'], group['eps'], group['gamma'])
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(param.data)
                    state['v'] = torch.zeros_like(param.data)
                    state['u'] = torch.zeros_like(param.data)
                m, v, u = state['m'], state['v'], state['u']
                grad = param.grad.data
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                state['step'] += 1
                bc1 = 1 - betas[0] ** state['step']
                bc2 = 1 - betas[1] ** state['step']
                denom = (v.sqrt() / bc2).add_(eps)
                u.copy_(m / bc1 / denom)
                if 'mask' not in state or state['step'] % group['T_horizon'] == 0:
                    _, idx = torch.topk(u.abs().flatten(),
                                        int(gamma * u.numel()))
                    mask = torch.zeros_like(u, dtype=torch.bool).flatten()
                    mask[idx] = True
                    state['mask'] = mask.view_as(u)
                param.data.addcmul_(u, state['mask'].float(), value=-lr)
        return loss


def kd_distill_loss_v2(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    loss_kd = -torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    return loss_kd * T_stu * T_stu


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _move_reward_to_cpu(reward_model):
    """Move reward model to CPU and fix device references."""
    reward_model = reward_model.cpu()
    reward_model.device = torch.device('cpu')
    if hasattr(reward_model, 'text_features') and reward_model.text_features is not None:
        reward_model.text_features = reward_model.text_features.cpu()
    if hasattr(reward_model, 'image_features') and reward_model.image_features is not None:
        reward_model.image_features = reward_model.image_features.cpu()
    return reward_model


def _get_trainable_state(model, only_visual):
    """Extract the trainable params from clip_model."""
    state = {}
    for n, p in model.clip_model.named_parameters():
        if only_visual and 'visual' in n:
            state[n] = p.detach().cpu().clone()
        elif not only_visual and 'visual' not in n:
            state[n] = p.detach().cpu().clone()
    return state


def _get_image_from_samples(samples):
    for key in ("image2", "image1", "image"):
        if key in samples:
            return samples[key]
    raise KeyError(f"No image key found in samples: {list(samples.keys())}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model_lavis = task.build_model(cfg)
    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task,
                        model=model_lavis, datasets=datasets)
    dataloader = runner.dataloaders['test']
    device = runner.model.device

    only_visual = (args.retrieval_task == "image2text")
    tag = "i2t" if only_visual else "t2i"
    max_samples = args.max_samples if args.max_samples > 0 else 5
    save_dir = os.path.join(args.save_prompt_dir or "./saved_params_ams_ret", tag)
    os.makedirs(save_dir, exist_ok=True)

    # ---- main TTA model on GPU ----
    print(f"\n[model] CLIPRet_TTA arch={args.arch}  task={args.retrieval_task}")
    model = CLIPRet_TTA(device, arch=args.arch, only_visual=only_visual,
                        momentum_update=bool(args.momentum_update),
                        update_freq=args.update_freq,
                        update_w=args.update_w,
                        momentum=args.tta_momentum)
    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    optimizer = GradientGuidedAdam(model.parameters(), lr=args.lr, eps=1e-06)
    optim_state = deepcopy(optimizer.state_dict())

    # ---- reward model: precompute on GPU, then move to CPU ----
    print(f"[reward] arch={args.reward_arch}  precomputing features on GPU …")
    reward_model = get_reward_model(device, args)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            text_ids = tokenize_all_text(dataloader.dataset.text, model, 128)
            if only_visual:
                text_embeds = get_all_text_embeds(text_ids, model, 128)
                model.set_text_features(text_features=text_embeds)
                reward_model.set_many_text_features(
                    dataloader.dataset.text, text_bs=128)
            else:
                image_embeds = get_all_image_embeds(dataloader, model)
                model.set_image_features(image_features=image_embeds)
                reward_model.set_image_features_with_dataloder(dataloader)

    print("[reward] moving to CPU to free GPU …")
    reward_model = _move_reward_to_cpu(reward_model)
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated(device) / 1024**2
    print(f"[gpu] after reward offload: {mem:.0f} MB allocated")

    # ---- save initial trainable params ----
    initial_state = _get_trainable_state(model, only_visual)
    torch.save(initial_state, os.path.join(save_dir, "initial.pt"))
    n_params = sum(v.numel() for v in initial_state.values())
    print(f"[saved] initial.pt  ({n_params:,} trainable params)\n")

    # ---- per-sample TTA + save ----
    if only_visual:
        for i, samples in enumerate(dataloader):
            if i >= max_samples:
                break
            t0 = time.time()
            image = _get_image_from_samples(samples).to(device)

            # reward logits on CPU
            with torch.no_grad():
                img_cpu = image.cpu().float()
                reward_model.set_image_features(images=img_cpu)
                r_logits_i, _ = reward_model.calulate_similarity()
                r_logits_i = r_logits_i.to(device)
            del img_cpu

            # TTA
            model.train()
            for step in range(args.tta_steps):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits_i, _ = model(image)
                    loss = kd_distill_loss_v2(logits_i, r_logits_i)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # save
            updated = _get_trainable_state(model, only_visual)
            torch.save(updated, os.path.join(save_dir, f"sample_{i}.pt"))
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"[saved] sample_{i}.pt  loss={loss.item():.4f}  "
                  f"{time.time()-t0:.1f}s  peak_gpu={peak:.0f}MB")

            model.eval()
            model.momentum_update_model()
            model.reset_initial()
            optimizer.load_state_dict(optim_state)
            del r_logits_i, updated
            torch.cuda.empty_cache()
    else:
        texts = dataloader.dataset.text
        for i in range(min(max_samples, len(texts))):
            t0 = time.time()
            text = texts[i]

            # reward logits on CPU
            with torch.no_grad():
                reward_model.set_text_features(captions=text)
                _, r_logits_t = reward_model.calulate_similarity()
                r_logits_t = r_logits_t.to(device)

            # TTA
            model.train()
            for step in range(args.tta_steps):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    _, logits_t = model(images=None, text=text)
                    loss = kd_distill_loss_v2(logits_t, r_logits_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # save
            updated = _get_trainable_state(model, only_visual)
            torch.save(updated, os.path.join(save_dir, f"sample_{i}.pt"))
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"[saved] sample_{i}.pt  loss={loss.item():.4f}  "
                  f"{time.time()-t0:.1f}s  peak_gpu={peak:.0f}MB")

            model.eval()
            model.momentum_update_model()
            model.reset_initial()
            optimizer.load_state_dict(optim_state)
            del r_logits_t, updated
            torch.cuda.empty_cache()

    print(f"\nDone.  Results → {save_dir}/")


if __name__ == "__main__":
    main()
