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


def test_time_prompt_tune_t2i(data_loader, edge_model, edge_optimizer, edge_scaler, edge_init_optim_state,
                              cloud_model, cloud_optimizer, cloud_scaler, cloud_init_optim_state, args):
    edge_model.eval()
    cloud_model.eval()
    device = edge_model.device
    texts = data_loader.dataset.text
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    from collections import deque
    all_entropy = deque(maxlen=500)

    prompt_save_dir = None
    if getattr(args, 'save_prompt_dir', None):
        prompt_save_dir = os.path.join(args.save_prompt_dir, 't2i')
        os.makedirs(prompt_save_dir, exist_ok=True)
    max_samples = getattr(args, 'max_samples', 0)

    for i, text in tqdm(enumerate(texts), desc="text2image retrieval", total=len(texts)):
        if max_samples > 0 and i >= max_samples:
            break
        edge_model.reset_initial()
        # with torch.no_grad():
        #     edge_logits = edge_model(text)
        #     sample_entropy = (-edge_logits.softmax(1) * edge_logits.log_softmax(1)).sum(1).item()
        #     all_entropy.append(sample_entropy)
        #     threshold = np.percentile(all_entropy, (100. - args.uplink_percent))
        #     if sample_entropy <= threshold:
        #         score_matrix_t2i[i] = edge_logits[0]
        #         continue
        edge_optimizer.load_state_dict(edge_init_optim_state)
        cloud_model.reset_initial()
        cloud_optimizer.load_state_dict(cloud_init_optim_state)
        for step in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                # update prompt in cloud
                cloud_logits = cloud_model(text)
                cloud_loss_topk = topk_ent(cloud_logits, args.ent_top_k)
                cloud_loss = cloud_loss_topk

                cloud_optimizer.zero_grad()
                cloud_scaler.scale(cloud_loss).backward()
                cloud_scaler.step(cloud_optimizer)
                cloud_scaler.update()
                # update align module in edge
                with torch.no_grad():
                    cloud_logits = cloud_model(text)
                edge_logits = edge_model(text)
                edge_loss_topk = topk_ent(edge_logits, args.ent_top_k)
                tau = args.tau
                edge_loss_kl = F.kl_div(F.log_softmax(edge_logits / tau, dim=1), F.softmax(cloud_logits / tau, dim=1),
                                        reduction='batchmean')
                edge_loss = edge_loss_topk + args.beta * edge_loss_kl

                edge_optimizer.zero_grad()
                edge_scaler.scale(edge_loss).backward()
                edge_scaler.step(edge_optimizer)
                edge_scaler.update()

        if prompt_save_dir is not None:
            ctx = edge_model.prompt_learner.ctx.detach().cpu().numpy()
            np.save(os.path.join(prompt_save_dir, f"sample_{i}.npy"), ctx)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = edge_model(text)
                score_matrix_t2i[i] = logits[0]

    return score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy()


def test_time_prompt_tune_i2t(data_loader, edge_model, edge_optimizer, edge_scaler, edge_init_optim_state,
                              cloud_model, cloud_optimizer, cloud_scaler, cloud_init_optim_state, args):
    edge_model.eval()
    cloud_model.eval()
    device = edge_model.device
    captions = data_loader.dataset.text  # 所有文本
    images = data_loader.dataset.image  # 所有图像 (tensor)

    score_matrix_i2t = torch.full((len(images), len(captions)), -100.0).to(device)
    score_matrix_t2i = torch.full((len(captions), len(images)), -100.0).to(device)
    from collections import deque
    all_entropy = deque(maxlen=500)
    loss_before_list = []
    loss_after_list = []

    prompt_save_dir = None
    if getattr(args, 'save_prompt_dir', None):
        prompt_save_dir = os.path.join(args.save_prompt_dir, 'i2t')
        os.makedirs(prompt_save_dir, exist_ok=True)
    max_samples = getattr(args, 'max_samples', 0)

    for i, samples in tqdm(enumerate(data_loader), desc="image2text retrieval", total=len(images)):
        if max_samples > 0 and i >= max_samples:
            break
        image = samples["image1"].to(edge_model.device)
        image_cloud = samples["image2"].to(edge_model.device)

        edge_model.reset_initial()
        edge_image_feature = edge_model.get_image_feature(image_cloud)  # 云端学生模型图片特征

        # with torch.no_grad():
        #     edge_logits = edge_model.get_zero_shot(edge_image_feature)
        #     sample_entropy = (-edge_logits.softmax(1) * edge_logits.log_softmax(1)).sum(1).item()
        #     all_entropy.append(sample_entropy)
        #     threshold = np.percentile(all_entropy, (100. - args.uplink_percent))
        #     if sample_entropy <= threshold:
        #         score_matrix_i2t[i] = edge_logits[0]
        #         continue
        cloud_model.reset_initial()
        edge_optimizer.load_state_dict(edge_init_optim_state)
        cloud_optimizer.load_state_dict(cloud_init_optim_state)
        cloud_input = image if args.cloud_orig else image_cloud
        cloud_image_feature = cloud_model.get_image_feature(cloud_input)
        with torch.no_grad():
            cloud_logits_orig, idx = cloud_model.get_top_k(cloud_image_feature, args.sim_top_k)
        for step in range(args.tta_steps):
            # update prompt in cloud
            with torch.cuda.amp.autocast():
                if args.ent_tea:
                    cloud_logits = cloud_model(cloud_image_feature, idx)
                    cloud_loss_topk = topk_ent(cloud_logits, args.ent_top_k)
                    cloud_loss = cloud_loss_topk

                    cloud_optimizer.zero_grad()
                    cloud_scaler.scale(cloud_loss).backward()
                    cloud_scaler.step(cloud_optimizer)
                    cloud_scaler.update()
                # update align module in edge
                with torch.no_grad():
                    cloud_logits = cloud_model(cloud_image_feature, idx)
                edge_logits = edge_model(edge_image_feature, idx)

                edge_loss_topk = topk_ent(edge_logits, args.ent_top_k)
                tau = args.tau
                edge_loss_kl = F.kl_div(F.log_softmax(edge_logits / tau, dim=1), F.softmax(cloud_logits / tau, dim=1), reduction='batchmean')
                if args.ent_stu:
                    edge_loss = edge_loss_topk + args.beta * edge_loss_kl
                else:
                    edge_loss = args.beta * edge_loss_kl

                edge_optimizer.zero_grad()
                edge_scaler.scale(edge_loss).backward()
                edge_scaler.step(edge_optimizer)
                edge_scaler.update()

            # if step == 0:
            #     loss_before_list.append(loss.item())

        if prompt_save_dir is not None:
            ctx = edge_model.prompt_learner.ctx.detach().cpu().numpy()
            np.save(os.path.join(prompt_save_dir, f"sample_{i}.npy"), ctx)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                edge_image_feature = edge_model.get_image_feature(image)  # 端侧模型图片特征
                logits = edge_model(edge_image_feature)
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
    cfg.pretty_print()

    job_id = "CEO"
    device = cfg.run_cfg.device

    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    if args.retrieval_task == "text2image":
        task = tasks.setup_task(cfg)
        datasets = task.build_datasets(cfg)
        runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=None, datasets=datasets)
        data_loader = runner.dataloaders['test']

        edge_model = CLIPRet_TPT_T2I(
            device=device,
            arch=args.edge,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            only_visual=True
        ).to(device)
        cloud_model = CLIPRet_TPT_T2I(
            device=device,
            arch=args.cloud,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            only_visual=True
        ).to(device)

        edge_optimizer = torch.optim.AdamW(edge_model.parameters(), lr=args.lr)
        cloud_optimizer = torch.optim.AdamW(cloud_model.parameters(), lr=args.lr)

        edge_init_optim_state = deepcopy(edge_optimizer.state_dict())
        cloud_init_optim_state = deepcopy(cloud_optimizer.state_dict())

        edge_image_feat_path = os.path.join(cfg.run_cfg.output_dir,
                                            f"{cfg.run_cfg.test_splits[0]}_{args.edge}_image_embeds.npy")
        cloud_image_feat_path = os.path.join(cfg.run_cfg.output_dir,
                                             f"{cfg.run_cfg.test_splits[0]}_{args.cloud}_image_embeds.npy")
        if os.path.exists(edge_image_feat_path):
            image_embeds = torch.from_numpy(np.load(edge_image_feat_path)).to(device)
        else:
            image_embeds = get_all_image_embeds(data_loader, edge_model)
            np.save(edge_image_feat_path, image_embeds.cpu().numpy())
        edge_model.set_image_features(image_embeds)

        if os.path.exists(cloud_image_feat_path):
            image_embeds = torch.from_numpy(np.load(cloud_image_feat_path)).to(device)
        else:
            image_embeds = get_all_image_embeds(data_loader, cloud_model)
            np.save(cloud_image_feat_path, image_embeds.cpu().numpy())
        cloud_model.set_image_features(image_embeds)

        score_matrix_i2t, score_matrix_t2i = test_time_prompt_tune_t2i(data_loader, edge_model, edge_optimizer,
                                                                       edge_scaler, edge_init_optim_state,
                                                                       cloud_model, cloud_optimizer, cloud_scaler,
                                                                       cloud_init_optim_state, args)

    elif args.retrieval_task == "image2text":
        task = tasks.setup_task(cfg)
        datasets = task.build_datasets(cfg)

        runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=None, datasets=datasets)
        data_loader = runner.dataloaders['test']
        if args.cloud_image_dir:
            data_loader.dataset.vis_root2 = args.cloud_image_dir
            logging.info(f"Override cloud image dir: {args.cloud_image_dir}")
        captions = data_loader.dataset.text
        edge_model = CLIPRet_TPT_I2T(
            device=device,
            captions=captions,
            arch=args.edge,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init
        ).to(device)
        cloud_model = CLIPRet_TPT_I2T(
            device=device,
            captions=captions,
            arch=args.cloud,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init
        ).to(device)

        edge_optimizer = torch.optim.AdamW(edge_model.parameters(), lr=args.lr)
        cloud_optimizer = torch.optim.AdamW(cloud_model.parameters(), lr=args.lr)

        edge_init_optim_state = deepcopy(edge_optimizer.state_dict())
        cloud_init_optim_state = deepcopy(cloud_optimizer.state_dict())

        score_matrix_i2t, score_matrix_t2i = test_time_prompt_tune_i2t(data_loader, edge_model, edge_optimizer,
                                                                   edge_scaler, edge_init_optim_state,
                                                                   cloud_model, cloud_optimizer, cloud_scaler,
                                                                   cloud_init_optim_state, args)
    else:
        raise ValueError("retrieval_task must be one of ['text2image', 'image2text']")

    eval_result = task._report_metrics(score_matrix_i2t, score_matrix_t2i,
                                       data_loader.dataset.txt2img, data_loader.dataset.img2txt)

    output_filename = os.path.join(args.output, f"results_{args.retrieval_task}.json")
    for k, v in eval_result.items():
        eval_result[k] = round(v, 3)
    print(f"Test with arch={args.arch}, tta steps={args.tta_steps}, lr={args.lr}")
    logging.info(eval_result)

    with open(output_filename, 'w') as fp:
        json.dump(eval_result, fp, indent=4)


if __name__ == "__main__":
    main()
