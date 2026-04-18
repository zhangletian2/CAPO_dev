"""
Lightweight ams — only saves updated text_encoder params.
  * Cloud model on CPU  (no grad, float32)
  * Device model on GPU (grad checkpointing on text_encoder → ~2 GB)
  * Default batch_size=4, max_samples=5
"""
import argparse
import os
import time
import types
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer
from torch.utils.checkpoint import checkpoint as ckpt

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip_device import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import (
    thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask,
)


# ------------------------------------------------------------------ #
#  GradientGuidedAdam (identical to ams_iccv.py)
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


# ------------------------------------------------------------------ #
#  Gradient-checkpointed TextEncoder forward
#  Trades ~10% extra time for ~90% less activation memory.
# ------------------------------------------------------------------ #
def _text_encoder_forward_ckpt(self, prompts, tokenized_prompts):
    x = prompts + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)                          # NLD -> LND
    for block in self.transformer.resblocks:
        x = ckpt(block, x, use_reentrant=False)      # recompute on backward
    x = x.permute(1, 0, 2)                          # LND -> NLD
    x = self.ln_final(x).type(self.dtype)
    if isinstance(self.text_projection, nn.Parameter):
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] \
            @ self.text_projection
    else:
        x = self.text_projection(
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)])
    return x


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def resolve_classnames(set_id):
    if len(set_id) > 1:
        return eval("{}_classes".format(set_id.lower()))
    assert set_id in ['A', 'R', 'K', 'V', 'I']
    all_cls = imagenet_classes
    if set_id in ['A', 'R', 'V']:
        mask = eval("imagenet_{}_mask".format(set_id.lower()))
        if set_id == 'R':
            return [all_cls[i] for i, m in enumerate(mask) if m]
        return [all_cls[i] for i in mask]
    return all_cls


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    first_set = args.test_sets.split("/")[0]

    # ---- cloud model: CPU, float32, no grad ----
    print(f"[cloud] Loading {args.cloud_arch} on CPU …")
    cloud_model = get_coop(args.cloud_arch, first_set,
                           'cpu', args.n_ctx, args.ctx_init)
    cloud_model.eval()
    for p in cloud_model.parameters():
        p.requires_grad_(False)

    # ---- device model: GPU, fp16, grad on text_encoder only ----
    print(f"[device] Loading {args.arch} on GPU {args.gpu} …")
    device_model = get_coop(args.arch, first_set,
                            args.gpu, args.n_ctx, args.ctx_init, type='Device')
    device_model = device_model.cuda(args.gpu)
    device_model.eval()

    # gradient checkpointing on text_encoder (saves ~3 GB on ImageNet)
    device_model.text_encoder.forward = types.MethodType(
        _text_encoder_forward_ckpt, device_model.text_encoder)

    teacher_trainable = []
    for name, param in device_model.named_parameters():
        if 'text_encoder' in name:
            param.requires_grad_(True)
            teacher_trainable.append(param)
        else:
            param.requires_grad_(False)

    optimizer = GradientGuidedAdam(teacher_trainable, args.lr)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    cudnn.benchmark = True
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    # print GPU baseline
    print(f"[gpu] baseline alloc "
          f"{torch.cuda.memory_allocated(args.gpu)/1024**2:.0f} MB / "
          f"reserved {torch.cuda.memory_reserved(args.gpu)/1024**2:.0f} MB")

    for set_id in args.test_sets.split("/"):
        print(f"\n{'='*50}\n  Dataset: {set_id}\n{'='*50}")

        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        data_transform = AugMixAugmenter(
            base_transform, preprocess,
            n_views=args.batch_size - 1, augmix=len(set_id) > 1)
        device_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(), normalize])

        classnames = resolve_classnames(set_id)
        cloud_model.reset_classnames(classnames, args.cloud_arch)
        device_model.reset_classnames(classnames, args.arch)
        # re-apply monkey-patch (reset_classnames may reload internals)
        device_model.text_encoder.forward = types.MethodType(
            _text_encoder_forward_ckpt, device_model.text_encoder)
        torch.cuda.empty_cache()

        val_dataset = build_dataset(set_id, device_transform, data_transform,
                                    args.data, mode=args.dataset_mode)
        if args.max_samples > 0 and len(val_dataset) > args.max_samples:
            import random as _rnd
            _rnd.seed(args.seed)
            indices = list(range(len(val_dataset)))
            _rnd.shuffle(indices)
            val_dataset = torch.utils.data.Subset(val_dataset,
                                                  indices[:args.max_samples])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        save_dir = os.path.join(args.save_param_dir, set_id)
        os.makedirs(save_dir, exist_ok=True)

        # ---- save initial state ----
        with torch.no_grad():
            cloud_model.reset()
            device_model.reset()
        initial_state = {
            n: p.detach().cpu().clone()
            for n, p in device_model.named_parameters()
            if 'text_encoder' in n
        }
        torch.save(initial_state, os.path.join(save_dir, "initial.pt"))
        n_params = sum(v.numel() for v in initial_state.values())
        print(f"  [saved] initial.pt  ({n_params:,} params)")

        # ---- per-sample TTA + save ----
        for si, (oris, recs, target) in enumerate(val_loader):
            if si >= args.max_samples:
                break
            t0 = time.time()

            with torch.no_grad():
                cloud_model.reset()
                device_model.reset()
            optimizer.load_state_dict(optim_state)

            if isinstance(recs, list):
                recs_gpu = torch.cat(recs, dim=0).cuda(args.gpu,
                                                       non_blocking=True)
            else:
                recs_gpu = recs.squeeze(0).cuda(args.gpu, non_blocking=True)

            loss_val = 0.0
            for _ in range(args.tta_steps):
                # --- cloud forward on CPU ---
                recs_cpu = recs_gpu.cpu().float()
                with torch.no_grad():
                    cf = cloud_model.get_image_features(recs_cpu)
                    out_c = cloud_model(None, cf).cuda(args.gpu)
                del recs_cpu, cf

                # --- device forward + backward on GPU ---
                with torch.cuda.amp.autocast():
                    ef = device_model.get_image_features(recs_gpu)
                    out_d = device_model(None, ef)
                    tau = 0.5
                    loss = criterion_div(
                        F.log_softmax(out_d / tau, dim=1),
                        F.softmax(out_c  / tau, dim=1))

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_val = loss.item()
                del ef, out_d, out_c

            # --- save updated state ---
            updated = {
                n: p.detach().cpu().clone()
                for n, p in device_model.named_parameters()
                if 'text_encoder' in n
            }
            fname = f"sample_{si}.pt"
            torch.save(updated, os.path.join(save_dir, fname))
            mem = torch.cuda.max_memory_allocated(args.gpu) / 1024**2
            print(f"  [saved] {fname}  loss={loss_val:.4f}  "
                  f"{time.time()-t0:.1f}s  peak_gpu={mem:.0f}MB")

            del recs_gpu, updated
            torch.cuda.empty_cache()

        del val_dataset, val_loader
        torch.cuda.empty_cache()

    print(f"\nAll done.  Results → {args.save_param_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AMS save text_encoder params (low VRAM)')
    parser.add_argument('data', metavar='DIR')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', default='RN50')
    parser.add_argument('-c', '--cloud_arch', default='RN50')
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N')
    parser.add_argument('--lr', default=5e-3, type=float, dest='lr')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--tta_steps', default=1, type=int)
    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--ctx_init', default=None, type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=5)
    parser.add_argument('--save_param_dir', type=str,
                        default='./saved_params_ams')
    main()
