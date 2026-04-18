#!/usr/bin/env python3
# coding=utf-8
"""
CEO Retrieval E2E benchmark (I2T + T2I).

Per-sample metrics:
  - upload_kb / download_kb
  - t_edge_prep_ms
  - t_cloud_prompt_ms
  - t_edge_distill_ms
  - t_final_ms
"""

import argparse
import csv
import logging
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.runners.runner_base import RunnerBase

from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TPT_T2I, CLIPRet_TPT_I2T


def topk_ent(logits, k=20, tau=1.0, eps=1e-12):
    probs = F.softmax(logits / tau, dim=-1)
    _, topk_idx = torch.topk(logits, k, dim=-1)
    topk_probs = probs.gather(1, topk_idx)
    renorm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + eps)
    loss = -(renorm * torch.log(renorm + eps)).sum(dim=-1)
    return loss.mean()


def get_all_image_embeds(data_loader, model):
    image_embeds = []
    for samples in tqdm(data_loader, desc="extract image embeds"):
        image = samples["image"].to(model.device)
        image_features = model.clip_model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        image_embeds.append(image_features)
    return torch.cat(image_embeds, dim=0)


def prompt_download_kb(model):
    ctx = model.prompt_learner.ctx
    return (ctx.numel() * ctx.element_size()) / 1024.0


def comm_ms(total_kb, bw_mbps):
    return total_kb * 1024.0 * 8.0 / (bw_mbps * 1e3)


def build_runtime(args):
    cfg_obj = type("ArgsNamespace", (), vars(args))
    cfg = Config(cfg_obj)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    return cfg


def bench_i2t(args, cfg):
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    runner = RunnerBase(cfg=cfg, job_id="CEO_BENCH", task=task, model=None, datasets=datasets)
    data_loader = runner.dataloaders["test"]
    dataset = data_loader.dataset

    if args.cloud_image_dir:
        dataset.vis_root2 = args.cloud_image_dir
        logging.info("Override cloud image dir: %s", args.cloud_image_dir)

    captions = dataset.text
    device = cfg.run_cfg.device

    edge_model = CLIPRet_TPT_I2T(
        device=device,
        captions=captions,
        arch=args.edge,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
    ).to(device)
    cloud_model = CLIPRet_TPT_I2T(
        device=device,
        captions=captions,
        arch=args.cloud,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
    ).to(device)

    edge_optimizer = torch.optim.AdamW(edge_model.parameters(), lr=args.lr)
    cloud_optimizer = torch.optim.AdamW(cloud_model.parameters(), lr=args.lr)
    edge_init_optim_state = deepcopy(edge_optimizer.state_dict())
    cloud_init_optim_state = deepcopy(cloud_optimizer.state_dict())
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    records = []
    n_total = len(dataset.image)
    n_run = min(n_total, args.max_samples) if args.max_samples > 0 else n_total

    prompt_kb = prompt_download_kb(edge_model)

    for i, samples in tqdm(enumerate(data_loader), total=n_run, desc="bench i2t"):
        if i >= n_run:
            break

        image = samples["image1"].to(edge_model.device)
        image_cloud = samples["image2"].to(edge_model.device)

        edge_model.reset_initial()
        cloud_model.reset_initial()
        edge_optimizer.load_state_dict(edge_init_optim_state)
        cloud_optimizer.load_state_dict(cloud_init_optim_state)

        torch.cuda.synchronize()
        p0 = time.perf_counter()
        edge_image_feature = edge_model.get_image_feature(image_cloud)
        torch.cuda.synchronize()
        p1 = time.perf_counter()
        t_edge_prep_ms = (p1 - p0) * 1000.0

        cloud_input = image if args.cloud_orig else image_cloud
        cloud_image_feature = cloud_model.get_image_feature(cloud_input)
        with torch.no_grad():
            _, idx = cloud_model.get_top_k(cloud_image_feature, args.sim_top_k)

        t_cloud_prompt_ms = 0.0
        t_edge_distill_ms = 0.0

        for _ in range(args.tta_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast():
                if args.ent_tea:
                    cloud_logits = cloud_model(cloud_image_feature, idx)
                    cloud_loss = topk_ent(cloud_logits, args.ent_top_k)
                    cloud_optimizer.zero_grad()
                    cloud_scaler.scale(cloud_loss).backward()
                    cloud_scaler.step(cloud_optimizer)
                    cloud_scaler.update()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_cloud_prompt_ms += (t1 - t0) * 1000.0

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    cloud_logits = cloud_model(cloud_image_feature, idx)
                edge_logits = edge_model(edge_image_feature, idx)
                edge_loss_topk = topk_ent(edge_logits, args.ent_top_k)
                edge_loss_kl = F.kl_div(
                    F.log_softmax(edge_logits / args.tau, dim=1),
                    F.softmax(cloud_logits / args.tau, dim=1),
                    reduction="batchmean",
                )
                if args.ent_stu:
                    edge_loss = edge_loss_topk + args.beta * edge_loss_kl
                else:
                    edge_loss = args.beta * edge_loss_kl
                edge_optimizer.zero_grad()
                edge_scaler.scale(edge_loss).backward()
                edge_scaler.step(edge_optimizer)
                edge_scaler.update()

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t_edge_distill_ms += (t2 - t1) * 1000.0

        torch.cuda.synchronize()
        f0 = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                edge_image_feature_final = edge_model.get_image_feature(image)
                _ = edge_model(edge_image_feature_final)
        torch.cuda.synchronize()
        f1 = time.perf_counter()
        t_final_ms = (f1 - f0) * 1000.0

        rel_path = dataset.annotation[i]["image"]
        if args.cloud_orig:
            upload_path = os.path.join(dataset.vis_root, rel_path)
        else:
            upload_path = os.path.join(dataset.vis_root2, rel_path)
        upload_kb = os.path.getsize(upload_path) / 1024.0 if os.path.exists(upload_path) else 0.0

        records.append(
            {
                "task": "i2t",
                "idx": i,
                "upload_kb": round(upload_kb, 4),
                "download_kb": round(prompt_kb, 4),
                "t_edge_prep_ms": round(t_edge_prep_ms, 4),
                "t_cloud_prompt_ms": round(t_cloud_prompt_ms, 4),
                "t_edge_distill_ms": round(t_edge_distill_ms, 4),
                "t_final_ms": round(t_final_ms, 4),
            }
        )

    return records


def bench_t2i(args, cfg):
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    runner = RunnerBase(cfg=cfg, job_id="CEO_BENCH", task=task, model=None, datasets=datasets)
    data_loader = runner.dataloaders["test"]
    texts = data_loader.dataset.text
    device = cfg.run_cfg.device

    edge_model = CLIPRet_TPT_T2I(
        device=device,
        arch=args.edge,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        only_visual=True,
    ).to(device)
    cloud_model = CLIPRet_TPT_T2I(
        device=device,
        arch=args.cloud,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        only_visual=True,
    ).to(device)

    edge_optimizer = torch.optim.AdamW(edge_model.parameters(), lr=args.lr)
    cloud_optimizer = torch.optim.AdamW(cloud_model.parameters(), lr=args.lr)
    edge_init_optim_state = deepcopy(edge_optimizer.state_dict())
    cloud_init_optim_state = deepcopy(cloud_optimizer.state_dict())
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    edge_image_feat_path = os.path.join(cfg.run_cfg.output_dir, f"{cfg.run_cfg.test_splits[0]}_{args.edge}_image_embeds.npy")
    cloud_image_feat_path = os.path.join(cfg.run_cfg.output_dir, f"{cfg.run_cfg.test_splits[0]}_{args.cloud}_image_embeds.npy")

    if os.path.exists(edge_image_feat_path):
        edge_embeds = torch.from_numpy(np.load(edge_image_feat_path)).to(device)
    else:
        edge_embeds = get_all_image_embeds(data_loader, edge_model)
        np.save(edge_image_feat_path, edge_embeds.cpu().numpy())
    edge_model.set_image_features(edge_embeds)

    if os.path.exists(cloud_image_feat_path):
        cloud_embeds = torch.from_numpy(np.load(cloud_image_feat_path)).to(device)
    else:
        cloud_embeds = get_all_image_embeds(data_loader, cloud_model)
        np.save(cloud_image_feat_path, cloud_embeds.cpu().numpy())
    cloud_model.set_image_features(cloud_embeds)

    prompt_kb = prompt_download_kb(edge_model)
    n_total = len(texts)
    n_run = min(n_total, args.max_samples) if args.max_samples > 0 else n_total

    records = []
    for i, text in tqdm(enumerate(texts[:n_run]), total=n_run, desc="bench t2i"):
        edge_model.reset_initial()
        cloud_model.reset_initial()
        edge_optimizer.load_state_dict(edge_init_optim_state)
        cloud_optimizer.load_state_dict(cloud_init_optim_state)

        t_edge_prep_ms = 0.0
        t_cloud_prompt_ms = 0.0
        t_edge_distill_ms = 0.0

        for _ in range(args.tta_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast():
                cloud_logits = cloud_model(text)
                cloud_loss = topk_ent(cloud_logits, args.ent_top_k)
                cloud_optimizer.zero_grad()
                cloud_scaler.scale(cloud_loss).backward()
                cloud_scaler.step(cloud_optimizer)
                cloud_scaler.update()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_cloud_prompt_ms += (t1 - t0) * 1000.0

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    cloud_logits = cloud_model(text)
                edge_logits = edge_model(text)
                edge_loss_topk = topk_ent(edge_logits, args.ent_top_k)
                edge_loss_kl = F.kl_div(
                    F.log_softmax(edge_logits / args.tau, dim=1),
                    F.softmax(cloud_logits / args.tau, dim=1),
                    reduction="batchmean",
                )
                edge_loss = edge_loss_topk + args.beta * edge_loss_kl
                edge_optimizer.zero_grad()
                edge_scaler.scale(edge_loss).backward()
                edge_scaler.step(edge_optimizer)
                edge_scaler.update()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t_edge_distill_ms += (t2 - t1) * 1000.0

        torch.cuda.synchronize()
        f0 = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = edge_model(text)
        torch.cuda.synchronize()
        f1 = time.perf_counter()
        t_final_ms = (f1 - f0) * 1000.0

        upload_kb = len(text.encode("utf-8")) / 1024.0
        records.append(
            {
                "task": "t2i",
                "idx": i,
                "upload_kb": round(upload_kb, 4),
                "download_kb": round(prompt_kb, 4),
                "t_edge_prep_ms": round(t_edge_prep_ms, 4),
                "t_cloud_prompt_ms": round(t_cloud_prompt_ms, 4),
                "t_edge_distill_ms": round(t_edge_distill_ms, 4),
                "t_final_ms": round(t_final_ms, 4),
            }
        )

    return records


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, bandwidths):
    if not rows:
        print("No benchmark rows.")
        return
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0].keys() if k not in ("task", "idx")}
    task = rows[0]["task"]
    m = {k: v.mean() for k, v in arr.items()}
    s = {k: v.std() for k, v in arr.items()}
    compute = m["t_edge_prep_ms"] + m["t_cloud_prompt_ms"] + m["t_edge_distill_ms"] + m["t_final_ms"]
    total_kb = m["upload_kb"] + m["download_kb"]

    print("=" * 72)
    print(f"Task: {task} | samples={len(rows)}")
    print(f"upload={m['upload_kb']:.3f}±{s['upload_kb']:.3f} KB, download={m['download_kb']:.3f}±{s['download_kb']:.3f} KB")
    print(f"edge_prep={m['t_edge_prep_ms']:.2f} ms")
    print(f"cloud_prompt={m['t_cloud_prompt_ms']:.2f} ms")
    print(f"edge_distill={m['t_edge_distill_ms']:.2f} ms")
    print(f"final={m['t_final_ms']:.2f} ms")
    print(f"total_compute={compute:.2f} ms")
    print("-" * 72)
    for bw in bandwidths:
        c = comm_ms(total_kb, bw)
        print(f"BW {bw:>4} Mbps -> comm={c:7.2f} ms, total={compute + c:7.2f} ms")
    print("=" * 72)


def parse_args():
    p = argparse.ArgumentParser("CEO retrieval E2E benchmark")
    p.add_argument("--task", choices=["i2t", "t2i"], required=True)
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--options", nargs="+", default=None)
    p.add_argument("--output-csv", required=True)

    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--edge", default="RN50")
    p.add_argument("--cloud", default="ViT-B-16")
    p.add_argument("--n_ctx", type=int, default=4)
    p.add_argument("--ctx_init", type=str, default="a photo of a")
    p.add_argument("--tta_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--sim_top_k", type=int, default=256)
    p.add_argument("--ent_top_k", type=int, default=40)
    p.add_argument("--ent_tea", type=int, default=1)
    p.add_argument("--ent_stu", type=int, default=1)
    p.add_argument("--cloud_orig", type=int, default=0)
    p.add_argument("--cloud_image_dir", type=str, default="")
    p.add_argument("--output", type=str, default="/tmp/ceo_ret_e2e")
    p.add_argument("--retrieval_task", type=str, default="image2text")
    p.add_argument("--bandwidths", type=int, nargs="+", default=[1000, 100, 50, 10, 5, 1])
    return p.parse_args()


def main():
    args = parse_args()

    if args.task == "i2t":
        args.retrieval_task = "image2text"
    else:
        args.retrieval_task = "text2image"

    cfg = build_runtime(args)

    if args.task == "i2t":
        rows = bench_i2t(args, cfg)
    else:
        rows = bench_t2i(args, cfg)

    save_csv(args.output_csv, rows)
    print(f"CSV saved: {args.output_csv}")
    print_summary(rows, args.bandwidths)


if __name__ == "__main__":
    main()
