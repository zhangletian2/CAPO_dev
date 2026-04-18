#!/usr/bin/env python
# coding=utf-8
"""
CEO Offline Retrieval: Train shared prompts on val split (unsupervised),
then evaluate the frozen edge prompt on test split.

Phase 1 (Offline):
    Cloud + edge models both optimise prompt_learner.ctx on val data.
    Loss = topk-entropy (cloud) + topk-entropy (edge) + β·KL(edge ‖ cloud).
    Only prompt_learner parameters are updated; CLIP encoders stay frozen.

Phase 2 (Online):
    Freeze edge prompt.  Rebuild edge model with test captions (I2T) or
    test image features (T2I), then run retrieval evaluation.

Usage:
    # T2I
    python clip_ret_ceo_offline.py --cfg-path lavis/projects/clip/exp_coco_ret_tta.yaml \
        --retrieval_task text2image --edge RN50 --cloud ViT-B-16 --epochs 3 --n_train 500

    # I2T
    python clip_ret_ceo_offline.py --cfg-path lavis/projects/clip/exp_coco_dual_image_ret_tta.yaml \
        --retrieval_task image2text --edge RN50 --cloud ViT-B-16 --epochs 3 --n_train 500
"""

import argparse
import json
import math
import os
import random
import time
import logging
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TPT_T2I, CLIPRet_TPT_I2T


# ── losses ────────────────────────────────────────────────────────────

def topk_ent(logits, k=20, eps=1e-12):
    probs = F.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(logits, k, dim=-1)
    topk_probs = probs.gather(1, topk_idx)
    renorm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + eps)
    loss = -(renorm * torch.log(renorm + eps)).sum(dim=-1)
    return loss.mean()


# ── data helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def get_all_image_embeds(data_loader, model):
    embeds = []
    for samples in tqdm(data_loader, desc="extracting image embeds"):
        key = "image1" if "image1" in samples else "image"
        image = samples[key].to(model.device)
        feat = model.clip_model.encode_image(image)
        embeds.append(F.normalize(feat, dim=-1))
    return torch.cat(embeds, dim=0)


def cache_image_features(data_loader, model, cache_path):
    if os.path.exists(cache_path):
        print(f"  [cache] Loading {cache_path}")
        return torch.from_numpy(np.load(cache_path)).to(model.device)
    feats = get_all_image_embeds(data_loader, model)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, feats.cpu().numpy())
    print(f"  [cache] Saved  {cache_path}")
    return feats


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Offline Training
# ═══════════════════════════════════════════════════════════════════════

def _build_scheduler(optimizer, total_steps, cosine=False):
    if not cosine:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0)


def _prompt_reg(model, lam):
    """L2 penalty pulling prompt ctx back toward its initialisation."""
    if lam <= 0:
        return 0.0
    ctx = model.prompt_learner.ctx
    init = model.prompt_learner.ctx_init_state.to(ctx.device)
    return lam * (ctx - init).pow(2).sum()


def train_t2i_offline(val_loader, edge_model, cloud_model, args):
    """Train T2I prompts with mini-batch of texts (image features pre-cached).

    Each step feeds a batch of B texts → cloud/edge logits are (B, N_img).
    Matches ceo_offline.py training strategy: DataLoader-style batching +
    optional cosine LR schedule + prompt L2 regularisation to prevent drift.
    """
    edge_model.eval()
    cloud_model.eval()

    edge_opt = torch.optim.AdamW(edge_model.parameters(), lr=args.lr)
    cloud_opt = torch.optim.AdamW(cloud_model.parameters(), lr=args.cloud_lr)
    e_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    c_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    texts = val_loader.dataset.text
    indices = list(range(len(texts)))
    if 0 < args.n_train < len(texts):
        indices = sorted(random.Random(args.seed).sample(indices, args.n_train))

    bs = args.batch_size
    steps_per_epoch = math.ceil(len(indices) / bs)
    total_steps = args.epochs * steps_per_epoch
    sched_c = _build_scheduler(cloud_opt, total_steps, args.cosine)
    sched_e = _build_scheduler(edge_opt, total_steps, args.cosine)

    print(f"  T2I training: {len(indices)} texts × {args.epochs} epochs, "
          f"bs={bs}, steps/ep={steps_per_epoch}, "
          f"prompt_reg={args.prompt_reg}")

    for epoch in range(args.epochs):
        random.shuffle(indices)
        ec, ee, ek, er, n_step = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for start in tqdm(range(0, len(indices), bs),
                          desc=f"T2I ep{epoch + 1}", leave=False):
            batch_idx = indices[start:start + bs]
            text_batch = [texts[i] for i in batch_idx]

            # ① cloud prompt — entropy on topk + reg
            with torch.cuda.amp.autocast():
                cl = cloud_model(text_batch)            # (B, N_img)
                lc = topk_ent(cl, args.ent_top_k) + _prompt_reg(
                    cloud_model, args.prompt_reg)
            cloud_opt.zero_grad()
            c_scaler.scale(lc).backward()
            c_scaler.step(cloud_opt)
            c_scaler.update()
            if sched_c is not None:
                sched_c.step()

            # ② edge prompt — entropy + KL + reg
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    cl = cloud_model(text_batch)
                el = edge_model(text_batch)             # (B, N_img)
                le = topk_ent(el, args.ent_top_k)
                lk = F.kl_div(
                    F.log_softmax(el / args.tau, dim=1),
                    F.softmax(cl / args.tau, dim=1),
                    reduction="batchmean",
                )
                reg_e = _prompt_reg(edge_model, args.prompt_reg)
                loss = le + args.beta * lk + reg_e
            edge_opt.zero_grad()
            e_scaler.scale(loss).backward()
            e_scaler.step(edge_opt)
            e_scaler.update()
            if sched_e is not None:
                sched_e.step()

            ec += lc.item(); ee += le.item(); ek += lk.item()
            er += reg_e.item() if isinstance(reg_e, torch.Tensor) else reg_e
            n_step += 1

        c_lr = cloud_opt.param_groups[0]["lr"]
        e_lr = edge_opt.param_groups[0]["lr"]
        print(
            f"  Ep {epoch + 1}/{args.epochs}  "
            f"c_ent={ec / n_step:.4f}  e_ent={ee / n_step:.4f}  "
            f"kl={ek / n_step:.4f}  reg={er / n_step:.4f}  "
            f"c_lr={c_lr:.2e} e_lr={e_lr:.2e}  ({time.time() - t0:.1f}s)"
        )


def train_i2t_offline(val_loader, edge_model, cloud_model, args):
    """Train I2T prompts with gradient accumulation over mini-batches.

    Each image has its own top-k caption set, so true batching is not feasible.
    Instead we accumulate gradients over ``batch_size`` images, matching the
    effective batch size of ceo_offline.py.
    """
    edge_model.eval()
    cloud_model.eval()

    edge_opt = torch.optim.AdamW(edge_model.parameters(), lr=args.lr)
    cloud_opt = torch.optim.AdamW(cloud_model.parameters(), lr=args.cloud_lr)
    e_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    c_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    n_img = len(val_loader.dataset.image)
    indices = list(range(n_img))
    if 0 < args.n_train < n_img:
        indices = sorted(random.Random(args.seed).sample(indices, args.n_train))

    bs = args.batch_size
    steps_per_epoch = math.ceil(len(indices) / bs)
    total_steps = args.epochs * steps_per_epoch
    sched_c = _build_scheduler(cloud_opt, total_steps, args.cosine)
    sched_e = _build_scheduler(edge_opt, total_steps, args.cosine)

    print(f"  I2T training: {len(indices)} images × {args.epochs} epochs, "
          f"accum_bs={bs}, steps/ep={steps_per_epoch}, "
          f"prompt_reg={args.prompt_reg}")

    for epoch in range(args.epochs):
        random.shuffle(indices)
        ec, ee, ek, er, n_step = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()

        cloud_opt.zero_grad()
        edge_opt.zero_grad()

        for i, idx in enumerate(tqdm(indices, desc=f"I2T ep{epoch + 1}",
                                     leave=False)):
            samples = val_loader.dataset[idx]
            if "image1" in samples:
                img_ori = samples["image1"].unsqueeze(0).to(edge_model.device)
                img_rec = samples["image2"].unsqueeze(0).to(edge_model.device)
            else:
                img_ori = samples["image"].unsqueeze(0).to(edge_model.device)
                img_rec = img_ori

            cloud_input = img_ori if args.cloud_orig else img_rec
            c_feat = cloud_model.get_image_feature(cloud_input)
            e_feat = edge_model.get_image_feature(img_rec)

            with torch.no_grad():
                _, topk = cloud_model.get_top_k(c_feat, args.sim_top_k)

            cur_bs = min(bs, len(indices) - (i // bs) * bs)

            # ① cloud prompt (gradient accumulation)
            if args.ent_tea:
                with torch.cuda.amp.autocast():
                    cl = cloud_model(c_feat, topk)
                    lc = topk_ent(cl, args.ent_top_k) / cur_bs
                c_scaler.scale(lc).backward()
                ec += lc.item() * cur_bs

            # ② edge prompt (gradient accumulation)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    cl = cloud_model(c_feat, topk)
                el = edge_model(e_feat, topk)
                le = topk_ent(el, args.ent_top_k)
                lk = F.kl_div(
                    F.log_softmax(el / args.tau, dim=1),
                    F.softmax(cl / args.tau, dim=1),
                    reduction="batchmean",
                )
                if args.ent_stu:
                    loss_e = (le + args.beta * lk) / cur_bs
                else:
                    loss_e = (args.beta * lk) / cur_bs
            e_scaler.scale(loss_e).backward()
            ee += le.item(); ek += lk.item()

            # step every bs samples or at the end of epoch
            if (i + 1) % bs == 0 or (i + 1) == len(indices):
                # add prompt reg at each optimizer step (not accumulated per sample)
                if args.prompt_reg > 0:
                    reg_c = _prompt_reg(cloud_model, args.prompt_reg)
                    reg_e = _prompt_reg(edge_model, args.prompt_reg)
                    if args.ent_tea:
                        c_scaler.scale(reg_c).backward()
                    e_scaler.scale(reg_e).backward()
                    er += (reg_c.item() if isinstance(reg_c, torch.Tensor) else 0) + \
                          (reg_e.item() if isinstance(reg_e, torch.Tensor) else 0)

                if args.ent_tea:
                    c_scaler.step(cloud_opt)
                    c_scaler.update()
                    cloud_opt.zero_grad()
                    if sched_c is not None:
                        sched_c.step()

                e_scaler.step(edge_opt)
                e_scaler.update()
                edge_opt.zero_grad()
                if sched_e is not None:
                    sched_e.step()

                n_step += 1

        c_lr = cloud_opt.param_groups[0]["lr"]
        e_lr = edge_opt.param_groups[0]["lr"]
        print(
            f"  Ep {epoch + 1}/{args.epochs}  "
            f"c_ent={ec / max(n_step, 1):.4f}  e_ent={ee / len(indices):.4f}  "
            f"kl={ek / len(indices):.4f}  reg={er / max(n_step, 1):.4f}  "
            f"c_lr={c_lr:.2e} e_lr={e_lr:.2e}  ({time.time() - t0:.1f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Online Testing (frozen prompt)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_t2i(test_loader, edge_model):
    edge_model.eval()
    device = edge_model.device
    texts = test_loader.dataset.text
    n_img = len(test_loader.dataset.image)

    # Pre-compute ALL text features in batches (prompt is frozen)
    print("  Pre-computing text features for T2I eval ...")
    text_features = []
    bsz = 64
    with torch.cuda.amp.autocast():
        for j in tqdm(range(0, len(texts), bsz), desc="T2I text encode",
                      leave=False):
            batch_texts = texts[j:j + bsz]
            feat = edge_model.encode_text_with_prompt(batch_texts)
            text_features.append(feat)
    text_features = torch.cat(text_features, 0)        # (N_txt, D)
    logit_scale = edge_model.clip_model.logit_scale.exp()
    print(f"  Text features: {text_features.shape}")

    s_i2t = torch.full((n_img, len(texts)), -100.0, device=device)
    s_t2i = (logit_scale * text_features @ edge_model.image_features.t())

    return s_i2t.cpu().numpy(), s_t2i.cpu().numpy()


@torch.no_grad()
def test_i2t(test_loader, edge_model):
    edge_model.eval()
    device = edge_model.device
    n_txt = len(test_loader.dataset.text)
    n_img = len(test_loader.dataset.image)

    # Pre-compute ALL text features once (prompt is frozen)
    print("  Pre-computing text features for all captions ...")
    prompts, tokenized = edge_model.prompt_learner(None)  # all captions
    orig_forward = edge_model.clip_model.token_embedding.forward
    text_features = []
    bsz = 128
    with torch.cuda.amp.autocast():
        for j in range(0, tokenized.size(0), bsz):
            prompt_chunk = prompts[j:j + bsz]

            def _fake(x, pc=prompt_chunk):
                return pc

            edge_model.clip_model.token_embedding.forward = _fake
            chunk_feat = edge_model.clip_model.encode_text(
                tokenized[j:j + bsz].to(device))
            text_features.append(chunk_feat)
    edge_model.clip_model.token_embedding.forward = orig_forward
    text_features = F.normalize(torch.cat(text_features, 0), dim=-1)
    logit_scale = edge_model.clip_model.logit_scale.exp()
    print(f"  Text features: {text_features.shape}")

    s_i2t = torch.full((n_img, n_txt), -100.0, device=device)
    s_t2i = torch.full((n_txt, n_img), -100.0, device=device)

    for i, samples in tqdm(enumerate(test_loader), desc="I2T eval", total=n_img):
        key = "image1" if "image1" in samples else "image"
        image = samples[key].to(device)
        with torch.cuda.amp.autocast():
            feat = edge_model.get_image_feature(image)
            s_i2t[i] = logit_scale * feat @ text_features.T

    return s_i2t.cpu().numpy(), s_t2i.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# Task runners
# ═══════════════════════════════════════════════════════════════════════

def run_t2i(val_loader, test_loader, task, device, args):
    save_dir = args.save_dir
    prompt_path = os.path.join(save_dir, "prompts_t2i.pt")

    if args.load_prompt:
        print(f"  Skipping training — loading {args.load_prompt}")
        prompt_path = args.load_prompt
    else:
        print("═══ Phase 1: T2I Offline Training ═══")
        edge = CLIPRet_TPT_T2I(
            device=device, arch=args.edge,
            n_ctx=args.n_ctx, ctx_init=args.ctx_init, only_visual=True,
        ).to(device)
        cloud = CLIPRet_TPT_T2I(
            device=device, arch=args.cloud,
            n_ctx=args.n_ctx, ctx_init=args.ctx_init, only_visual=True,
        ).to(device)

        edge.set_image_features(
            cache_image_features(
                val_loader, edge,
                os.path.join(save_dir, f"val_{args.edge}_t2i_embeds.npy"),
            )
        )
        cloud.set_image_features(
            cache_image_features(
                val_loader, cloud,
                os.path.join(save_dir, f"val_{args.cloud}_t2i_embeds.npy"),
            )
        )

        train_t2i_offline(val_loader, edge, cloud, args)

        torch.save(
            {
                "edge_ctx": edge.prompt_learner.ctx.detach().cpu(),
                "cloud_ctx": cloud.prompt_learner.ctx.detach().cpu(),
                "args": vars(args),
            },
            prompt_path,
        )
        print(f"  Prompts saved → {prompt_path}")
        del edge, cloud
        torch.cuda.empty_cache()

    # ── Phase 2 ──
    print("\n═══ Phase 2: T2I Testing (frozen prompt) ═══")
    edge_test = CLIPRet_TPT_T2I(
        device=device, arch=args.edge,
        n_ctx=args.n_ctx, ctx_init=args.ctx_init, only_visual=True,
    ).to(device)

    ckpt = torch.load(prompt_path, map_location=device)
    edge_test.prompt_learner.ctx.data.copy_(ckpt["edge_ctx"].to(device))
    for p in edge_test.parameters():
        p.requires_grad_(False)

    edge_test.set_image_features(
        cache_image_features(
            test_loader, edge_test,
            os.path.join(save_dir, f"test_{args.edge}_t2i_embeds.npy"),
        )
    )

    return test_t2i(test_loader, edge_test)


def run_i2t(val_loader, test_loader, task, device, args):
    save_dir = args.save_dir
    prompt_path = os.path.join(save_dir, "prompts_i2t.pt")

    if args.load_prompt:
        print(f"  Skipping training — loading {args.load_prompt}")
        prompt_path = args.load_prompt
    else:
        print("═══ Phase 1: I2T Offline Training ═══")

        # If cloud images don't exist for val, fall back to original images
        if hasattr(val_loader.dataset, "vis_root2") and val_loader.dataset.vis_root2:
            test_img = val_loader.dataset.annotation[0]["image"]
            if not os.path.exists(os.path.join(val_loader.dataset.vis_root2, test_img)):
                print("  NOTE: Cloud images not found for val split — "
                      "using original images for both models during training.")
                val_loader.dataset.vis_root2 = val_loader.dataset.vis_root

        val_caps = val_loader.dataset.text
        print(f"  Val set: {len(val_loader.dataset.image)} images, {len(val_caps)} captions")

        edge = CLIPRet_TPT_I2T(
            device=device, captions=val_caps, arch=args.edge,
            n_ctx=args.n_ctx, ctx_init=args.ctx_init,
        ).to(device)
        cloud = CLIPRet_TPT_I2T(
            device=device, captions=val_caps, arch=args.cloud,
            n_ctx=args.n_ctx, ctx_init=args.ctx_init,
        ).to(device)

        train_i2t_offline(val_loader, edge, cloud, args)

        torch.save(
            {
                "edge_ctx": edge.prompt_learner.ctx.detach().cpu(),
                "cloud_ctx": cloud.prompt_learner.ctx.detach().cpu(),
                "args": vars(args),
            },
            prompt_path,
        )
        print(f"  Prompts saved → {prompt_path}")
        del edge, cloud
        torch.cuda.empty_cache()

    # ── Phase 2 ──
    print("\n═══ Phase 2: I2T Testing (frozen prompt) ═══")
    test_caps = test_loader.dataset.text
    print(f"  Test set: {len(test_loader.dataset.image)} images, {len(test_caps)} captions")

    edge_test = CLIPRet_TPT_I2T(
        device=device, captions=test_caps, arch=args.edge,
        n_ctx=args.n_ctx, ctx_init=args.ctx_init,
    ).to(device)

    ckpt = torch.load(prompt_path, map_location=device)
    edge_test.prompt_learner.ctx.data.copy_(ckpt["edge_ctx"].to(device))
    for p in edge_test.parameters():
        p.requires_grad_(False)

    return test_i2t(test_loader, edge_test)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    device = cfg.run_cfg.device

    task = tasks.setup_task(cfg)
    datasets_dict = task.build_datasets(cfg)
    runner = RunnerBase(
        cfg=cfg, job_id="CEO_OFFLINE", task=task, model=None, datasets=datasets_dict,
    )

    assert "val" in runner.dataloaders, (
        "Val split not found in dataloaders. "
        "Make sure the dataset config has val annotations."
    )
    val_loader = runner.dataloaders["val"]
    test_loader = runner.dataloaders["test"]

    if args.cloud_image_dir:
        for loader in [val_loader, test_loader]:
            if hasattr(loader.dataset, "vis_root2"):
                loader.dataset.vis_root2 = args.cloud_image_dir
                logging.info(f"Override vis_root2: {args.cloud_image_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.retrieval_task == "text2image":
        s_i2t, s_t2i = run_t2i(val_loader, test_loader, task, device, args)
    elif args.retrieval_task == "image2text":
        s_i2t, s_t2i = run_i2t(val_loader, test_loader, task, device, args)
    else:
        raise ValueError("retrieval_task must be 'text2image' or 'image2text'")

    eval_result = task._report_metrics(
        s_i2t, s_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt,
    )
    for k, v in eval_result.items():
        eval_result[k] = round(v, 3)

    print(f"\n{'=' * 60}")
    print(f"Task: {args.retrieval_task}   Edge: {args.edge}   Cloud: {args.cloud}")
    print(json.dumps(eval_result, indent=2))
    logging.info(eval_result)

    result_path = os.path.join(args.save_dir, f"results_{args.retrieval_task}.json")
    with open(result_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"Results → {result_path}")


def parse_args():
    p = argparse.ArgumentParser("CEO Offline Retrieval Prompt Training")

    # LAVIS config
    p.add_argument("--cfg-path", required=True, help="path to LAVIS yaml config")
    p.add_argument("--options", nargs="+", default=None,
                   help="override settings in config (key=value)")

    # task
    p.add_argument("--retrieval_task", required=True,
                   choices=["text2image", "image2text"])

    # model
    p.add_argument("--edge", default="RN50", help="edge model architecture")
    p.add_argument("--cloud", default="ViT-B-16", help="cloud model architecture")
    p.add_argument("--n_ctx", type=int, default=4, help="number of learnable context tokens")
    p.add_argument("--ctx_init", default="a_photo_of_a",
                   help="context init words (underscores → spaces)")

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32,
                   help="mini-batch size (T2I: true batch; I2T: grad accum)")
    p.add_argument("--cosine", action="store_true", default=False,
                   help="use cosine annealing LR schedule")
    p.add_argument("--n_train", type=int, default=0,
                   help="number of val samples for training (0 = all)")
    p.add_argument("--lr", type=float, default=5e-4, help="edge learning rate")
    p.add_argument("--prompt_reg", type=float, default=0.0,
                   help="L2 regularisation weight pulling prompt toward init "
                        "(0 = off, typical: 1.0–5.0)")
    p.add_argument("--cloud_lr", type=float, default=None,
                   help="cloud learning rate (default: same as --lr)")
    p.add_argument("--tau", type=float, default=0.8, help="KL temperature")
    p.add_argument("--beta", type=float, default=0.5, help="KL loss weight")
    p.add_argument("--ent_top_k", type=int, default=40,
                   help="topk confident entries for entropy loss")
    p.add_argument("--sim_top_k", type=int, default=256,
                   help="I2T: topk similar captions for prompt learning")
    p.add_argument("--ent_tea", type=int, default=1,
                   help="train cloud prompt with entropy (I2T)")
    p.add_argument("--ent_stu", type=int, default=1,
                   help="include entropy in edge loss (I2T)")
    p.add_argument("--cloud_orig", type=int, default=0,
                   help="if 1, cloud uses original image (I2T)")

    # IO
    p.add_argument("--save_dir", default="checkpoints/ceo_offline_ret",
                   help="directory for prompts, caches, and results")
    p.add_argument("--load_prompt", default=None,
                   help="path to pre-trained prompts.pt (skip training)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cloud_image_dir", type=str, default="",
                   help="override vis_root2 for dual-image dataset")

    args = p.parse_args()
    if args.cloud_lr is None:
        args.cloud_lr = args.lr
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CEO Offline Retrieval — Hyperparameters")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    hp_path = os.path.join(args.save_dir, f"hparams_{args.retrieval_task}.json")
    with open(hp_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    return args


if __name__ == "__main__":
    main()
