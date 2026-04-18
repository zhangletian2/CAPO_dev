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
from custom_models import CLIPRet_TPT_T2I, CLIPRet_TPT_I2T
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


def kl_loss(logits_adapt, logits_orig, eps=1e-12):
    """
    KL(p_adapt || p_orig), both logits are [1, N]
    """
    p_adapt = F.softmax(logits_adapt, dim=-1)
    p_orig = F.softmax(logits_orig.detach(), dim=-1)
    log_p_adapt = torch.log(p_adapt + eps)
    return F.kl_div(log_p_adapt, p_orig, reduction='batchmean')


def test_time_prompt_tune_t2i(data_loader, model, optimizer, scaler, args, init_optim_state):
    model.eval()
    device = model.device
    texts = data_loader.dataset.text
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    loss_before_list = []  # 绘制loss曲线
    loss_after_list = []
    for i, text in tqdm(enumerate(texts), desc="text2image retrieval", total=len(texts)):
        model.reset_initial()
        optimizer.load_state_dict(init_optim_state)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_orig = model.get_zero_shot(text)  # [1, N]
        for step in range(args.tta_steps):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(text)

                loss_topk = topk_ent(logits, args.ent_top_k)
                loss_kl = kl_loss(logits, logits_orig)
                loss = loss_topk + args.lambda_kl * loss_kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step == 0:
                loss_before_list.append(loss.item())  # 绘制loss曲线

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(text)
                score_matrix_t2i[i] = logits[0]
                loss = topk_ent(logits, args.ent_top_k)

                loss_after_list.append(loss.item())

    plot_loss_curve(loss_before_list, loss_after_list, args)
    return score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy()


def test_time_prompt_tune_i2t(data_loader, model, optimizer, scaler, args, init_optim_state):
    model.eval()
    device = model.device
    captions = data_loader.dataset.text  # 所有文本
    images = data_loader.dataset.image  # 所有图像 (tensor)

    score_matrix_i2t = torch.full((len(images), len(captions)), -100.0).to(device)
    score_matrix_t2i = torch.full((len(captions), len(images)), -100.0).to(device)

    loss_before_list = []
    loss_after_list = []

    for i, samples in tqdm(enumerate(data_loader), desc="image2text retrieval", total=len(images)):
        image = samples["image"].to(model.device)
        model.reset_initial()
        optimizer.load_state_dict(init_optim_state)
        image_feature = model.get_image_feature(image)
        for step in range(args.tta_steps):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits_orig, idx = model.get_top_k(image_feature, args.sim_top_k)

                logits = model(image_feature, idx)

                loss_topk = topk_ent(logits, args.ent_top_k)
                loss_kl = kl_loss(logits, logits_orig[:, idx])
                loss = loss_topk + args.lambda_kl * loss_kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if step == 0:
            #     loss_before_list.append(loss.item())

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(image_feature)
                score_matrix_i2t[i] = logits[0]

                # target = logits.argmax(dim=-1).detach()  # 把 top‑1 当作伪标签
                # loss = F.cross_entropy(logits, target)
                # loss_after_list.append(loss.item())

    # plot_loss_curve(loss_before_list, loss_after_list, args)
    return score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy()


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
        model = CLIPRet_TPT_T2I(
            device=device,
            arch=args.arch,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            only_visual=True
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
        model = CLIPRet_TPT_I2T(
            device=device,
            captions=captions,
            arch=args.arch,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init
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
