# coding=utf-8
import os
import json
import time
import random
import logging
import datetime
from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.common.logger import MetricLogger

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from params import parse_args
from clip_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_Dynap_T2I, CLIPRet_Dynap_I2T
from lavis.models.clip_models.tokenizer import tokenize
import matplotlib.pyplot as plt


def plot_loss_curve(loss_before_list, loss_after_list, args):
    x = list(range(len(loss_before_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(x, loss_before_list, label='Before TPT (Step 0)', color='blue', linestyle='--', marker='o')
    plt.plot(x, loss_after_list, label='After TPT (Step N)', color='red', linestyle='-', marker='x')
    plt.xlabel('Text Sample Index')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss Before and After Prompt Tuning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(args.output, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Loss curve saved to {save_path}")

    plt.close()


def get_all_image_embeds(data_loader, model):
    """extract all image embeddings"""
    image_embeds = []
    for samples in tqdm(data_loader, desc="get all image embeds"):
        image = samples["image"].to(model.device)
        image_features = model.clip_model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        image_embeds.append(image_features)
    return torch.cat(image_embeds, dim=0)

def entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * logits.exp()).sum(dim=-1)

def topk_ent(logits, k=20, tau=1.0, reduction='mean', eps=1e-12):
    probs = F.softmax(logits / tau, dim=-1)  # [B, N]
    _, topk_idx = torch.topk(logits, k, dim=-1)  # [B, k]
    topk_probs = probs.gather(1, topk_idx)  # [B, k]
    renorm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + eps)
    loss = -(renorm * torch.log(renorm + eps)).sum(dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def test_time_prompt_tune_t2i(data_loader, model, optimizer, scaler, args, init_optim_state):
    model.eval()
    device = model.device
    texts = data_loader.dataset.text
    n_imgs = len(data_loader.dataset.image)

    score_matrix_i2t = torch.full((n_imgs, len(texts)), -100.0, device=device)
    score_matrix_t2i = torch.full((len(texts), n_imgs), -100.0, device=device)

    loss_before_list, loss_after_list = [], []

    for i, text in tqdm(enumerate(texts), desc="text2image retrieval", total=len(texts)):
        # ===== Reset / prompt 轮转 =====
        with torch.no_grad():
            model.reset_initial()
            if model.prompt_learner.ctx_use.count(0) == 0:
                # 轮转并清零头部 prompt
                head = model.prompt_learner.ctx_order[0]
                model.prompt_learner.ctx_use[head] = 0
                model.prompt_learner.ctx[head].zero_()

        optimizer.load_state_dict(init_optim_state)

        # ===== TTA：在线选择 + 反传，仅更新 prompt =====
        final_p_s = torch.arange(args.num_prompts, device=device)  # 0-based

        for step in range(args.tta_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits_all = model(text)  # [1, P, Nimg]

                # —— 在线选择（不做样本增广选择）——

                if args.num_prompts > 1 and args.onlinetpt:
                    p_pred = logits_all.squeeze(0).detach()  # [P, Nimg]
                    p_ent  = entropy(p_pred)                 # [P]
                    init_p = model.prompt_learner.ctx_order[0]
                    p_max  = p_pred.max(dim=-1).values       # [P]
                    plpd   = p_max[init_p] - p_max           # [P]

                    ent_order  = torch.argsort(p_ent, descending=False)
                    pos_in_ent = (ent_order == init_p).nonzero(as_tuple=False).item()
                    plpd_order = torch.argsort(plpd, descending=True)

                    tail_ent   = ent_order[min(pos_in_ent + 1, args.num_prompts - 1):]
                    head_plpd  = plpd_order[:pos_in_ent]
                    inter = np.intersect1d(
                        tail_ent.detach().cpu().numpy(),
                        head_plpd.detach().cpu().numpy()
                    )
                    if inter.size > 0:
                        p_s = torch.cat(
                            [ent_order[torch.as_tensor(inter, device=device)],
                             ent_order[[pos_in_ent]]], dim=0
                        )
                    else:
                        p_s = ent_order[[pos_in_ent]]  # 仅参考 prompt
                    logits_sel = logits_all[:, p_s, :]  # [1, K(K=1), Nimg]
                else:
                    p_s = torch.arange(logits_all.size(1), device=device)
                    logits_sel = logits_all[:, p_s, :]  # [1, K, Nimg]

                raw_pred = logits_sel.squeeze(0)       # [K, Nimg]

                # 损失：top-k-entropy
                loss = topk_ent(raw_pred, k=args.ent_top_k)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step == 0:
                loss_before_list.append(loss.item())

            # 轮转/计数
            if args.num_prompts > 1 and args.onlinetpt:
                for p_idx in p_s.detach().cpu().view(-1).tolist():
                    model.prompt_learner.ctx_order.remove(p_idx)
                    model.prompt_learner.ctx_use[p_idx] += 1
                    model.prompt_learner.ctx_order.append(p_idx)

            final_p_s = p_s

        # ===== 最终推理：用选中 prompts 干净前向 =====
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_final = model(text, selected_idx=final_p_s)  # [1, K, Nimg]
                logits_final = logits_final.mean(dim=1).squeeze(0)  # [Nimg]
                score_matrix_t2i[i] = logits_final
                loss_after_list.append(avg_entropy(logits_final.unsqueeze(0)).item())

    plot_loss_curve(loss_before_list, loss_after_list, args)
    return score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy()


def test_time_prompt_tune_i2t(data_loader, model, optimizer, scaler, args, init_optim_state):
    model.eval()
    device = model.device
    texts  = data_loader.dataset.text
    images = data_loader.dataset.image
    n_imgs, n_caps = len(images), len(texts)

    # 放 CPU，避免占 GPU 显存
    score_matrix_i2t = torch.full((n_imgs, n_caps), -100.0)  # CPU
    score_matrix_t2i = torch.full((n_caps, n_imgs), -100.0)  # 兼容返回

    loss_before_list, loss_after_list = [], []

    for i, samples in tqdm(enumerate(data_loader), desc="image2text", total=n_imgs):
        image = samples["image"].to(device)

        with torch.no_grad():
            model.prompt_learner.reset()
            if model.prompt_learner.ctx_use.count(0) == 0:
                head = model.prompt_learner.ctx_order[0]
                model.prompt_learner.ctx_use[head] = 0
                model.prompt_learner.ctx[head].zero_()

        optimizer.load_state_dict(init_optim_state)

        with torch.no_grad():
            img_feat = model.get_image_feature(image)
            _, topk_idx = model.get_top_k_caption_idx(img_feat, args.sim_top_k)

        final_p_s = torch.arange(args.num_prompts, device=device)

        for step in range(args.tta_steps):
            optimizer.zero_grad(set_to_none=True)

            # (1) 选择阶段：no_grad 跑全 P
            logits_allP = model.forward_allP_for_selection(img_feat, topk_idx)  # (K,P)

            if args.num_prompts > 1 and args.onlinetpt:
                # 与 T2I 相同的选择策略（熵+PLPD）
                p_pred = logits_allP.transpose(0,1).detach()     # (P,K)
                p_ent  = entropy(p_pred)                         # (P,)
                init_p = model.prompt_learner.ctx_order[0]
                p_max  = p_pred.max(dim=-1).values               # (P,)
                plpd   = p_max[init_p] - p_max                   # (P,)

                ent_order  = torch.argsort(p_ent, descending=False)
                pos_in_ent = (ent_order == init_p).nonzero(as_tuple=False).item()
                plpd_order = torch.argsort(plpd, descending=True)

                tail_ent   = ent_order[min(pos_in_ent + 1, args.num_prompts - 1):]
                head_plpd  = plpd_order[:pos_in_ent]
                inter = np.intersect1d(
                    tail_ent.detach().cpu().numpy(),
                    head_plpd.detach().cpu().numpy()
                )
                if inter.size > 0:
                    p_s = torch.cat(
                        [ent_order[torch.as_tensor(inter, device=device)],
                         ent_order[[pos_in_ent]]], dim=0
                    )
                else:
                    p_s = ent_order[[pos_in_ent]]
            else:
                p_s = torch.arange(model.num_p, device=device)

            # (2) 优化阶段：只对选中的 prompts 做有梯度前向
            with torch.cuda.amp.autocast():
                logits_sel = model.forward_selectedP_for_loss(img_feat, topk_idx, p_s)  # (K,|p_s|)
                raw_pred   = logits_sel.transpose(0,1)                                   # (|p_s|, K)
                loss = topk_ent(raw_pred, k=args.ent_top_k)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 轮转/计数
            if args.num_prompts > 1 and args.onlinetpt:
                for p_idx in p_s.detach().cpu().view(-1).tolist():
                    model.prompt_learner.ctx_order.remove(p_idx)
                    model.prompt_learner.ctx_use[p_idx] += 1
                    model.prompt_learner.ctx_order.append(p_idx)

            final_p_s = p_s

        # —— 优化后全量干净推理（内部均值 prompts） —— #
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_final = model.infer_all_captions(img_feat, selected_p_idx=final_p_s, batch=512)  # (Ncap,)
                score_matrix_i2t[i] = logits_final.cpu()

    # plot_loss_curve(loss_before_list, loss_after_list, args)  # 如需
    return score_matrix_i2t.numpy(), score_matrix_t2i.numpy()


def main():
    args = parse_args()

    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    # cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    job_id = "tpt"
    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=None, datasets=datasets)
    dataloader = runner.dataloaders['test']
    device = cfg.run_cfg.device

    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    if args.retrieval_task == "text2image":
        model = CLIPRet_Dynap_T2I(
            device=device,
            arch=args.arch,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            only_visual=True,
            num_p=args.num_prompts
        ).to(device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        init_optim_state = deepcopy(optimizer.state_dict())

        image_feat_path = os.path.join(cfg.run_cfg.output_dir,
                                       f"{cfg.run_cfg.test_splits[0]}_{args.arch}_image_embeds.npy")
        if os.path.exists(image_feat_path):
            image_embeds = torch.from_numpy(np.load(image_feat_path)).to(device)
        else:
            image_embeds = get_all_image_embeds(dataloader, model)
            np.save(image_feat_path, image_embeds.cpu().numpy())
        model.set_image_features(image_embeds)

        score_matrix_i2t, score_matrix_t2i = test_time_prompt_tune_t2i(dataloader, model, optimizer, scaler, args,
                                                                       init_optim_state)

    elif args.retrieval_task == "image2text":
        captions = dataloader.dataset.text
        model = CLIPRet_Dynap_I2T(
            device=device,
            captions=captions,
            arch=args.arch,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            num_p=args.num_prompts
        ).to(device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        init_optim_state = deepcopy(optimizer.state_dict())

        score_matrix_i2t, score_matrix_t2i = test_time_prompt_tune_i2t(dataloader, model, optimizer, scaler, args,
                                                                       init_optim_state)

    else:
        raise ValueError("retrieval_task must be one of ['text2image', 'image2text']")

    eval_result = task._report_metrics(score_matrix_i2t, score_matrix_t2i,
                                       dataloader.dataset.txt2img, dataloader.dataset.img2txt)

    output_filename = os.path.join(args.output, f"results_{args.retrieval_task}.json")
    for k, v in eval_result.items():
        eval_result[k] = round(v, 3)
    logging.info(eval_result)

    with open(output_filename, 'w') as fp:
        json.dump(eval_result, fp, indent=4)


if __name__ == "__main__":
    main()
