#!/usr/bin/env python3
# coding=utf-8
"""
AMS retrieval E2E latency benchmark (I2T + T2I).
"""

import argparse
import csv
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
from clip_reward import get_reward_model
from custom_models import CLIPRet_TTA
from clip_ret_policy import tokenize_all_text, get_all_text_embeds, get_all_image_embeds
from clip_ret_ams_v3 import GradientGuidedAdam


def kd_distill_loss_v2(logits_student, logits_teacher, t_stu=1.0, t_tea=1.0):
    log_pred_student = F.log_softmax(logits_student / t_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / t_tea, dim=1)
    loss_kd = -torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    return loss_kd * t_stu * t_stu


def comm_ms(total_kb, bw_mbps):
    return total_kb * 1024.0 * 8.0 / (bw_mbps * 1e3)


def parse_args():
    p = argparse.ArgumentParser("AMS retrieval E2E benchmark")
    p.add_argument("--task", choices=["i2t", "t2i"], required=True)
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--options", nargs="+", default=None)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output", default="/tmp/ams_ret_e2e")

    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--arch", default="RN50")
    p.add_argument("--reward_arch", default="ViT-B-16")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--tta_steps", type=int, default=1)
    p.add_argument("--sample_k", type=int, default=20)
    p.add_argument("--reward_process", type=int, default=1)
    p.add_argument("--process_batch", type=int, default=0)
    p.add_argument("--reward_amplify", type=int, default=0)
    p.add_argument("--multiple_reward_models", type=int, default=0)
    p.add_argument("--momentum_update", type=int, default=0)
    p.add_argument("--update_freq", type=int, default=64)
    p.add_argument("--tta_momentum", type=float, default=0.9998)
    p.add_argument("--update_w", type=float, default=1.0)
    p.add_argument("--cloud_image_dir", type=str, default="")
    p.add_argument("--bandwidths", type=int, nargs="+", default=[1000, 100, 50, 10, 5, 1])
    return p.parse_args()


def setup_runtime(args):
    args.retrieval_task = "image2text" if args.task == "i2t" else "text2image"
    cfg_obj = type("ArgsNamespace", (), vars(args))
    cfg = Config(cfg_obj)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    return cfg


def build_data(cfg, args):
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    runner = RunnerBase(cfg=cfg, job_id="AMS_BENCH", task=task, model=None, datasets=datasets)
    data_loader = runner.dataloaders["test"]
    if args.task == "i2t" and args.cloud_image_dir:
        data_loader.dataset.vis_root2 = args.cloud_image_dir
    return data_loader


def build_models(args, device):
    only_visual = args.task == "i2t"
    model = CLIPRet_TTA(
        device,
        arch=args.arch,
        only_visual=only_visual,
        momentum_update=args.momentum_update,
        update_freq=args.update_freq,
        update_w=args.update_w,
        momentum=args.tta_momentum,
    ).to(device)
    reward_model = get_reward_model(device, args)
    optimizer = GradientGuidedAdam(model.parameters(), lr=args.lr, eps=1e-6)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    return model, reward_model, optimizer, scaler


def precompute_global(data_loader, model, reward_model, text_bs=128):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            text_ids = tokenize_all_text(data_loader.dataset.text, model, text_bs)
            if model.only_visual:
                text_embeds = get_all_text_embeds(text_ids, model, text_bs)
                model.set_text_features(text_features=text_embeds)
                reward_model.set_many_text_features(data_loader.dataset.text, text_bs=text_bs)
            else:
                image_embeds = get_all_image_embeds(data_loader, model)
                model.set_image_features(image_features=image_embeds)
                reward_model.set_image_features_with_dataloder(data_loader)


def bench_i2t(data_loader, model, reward_model, optimizer, scaler, optim_state, args):
    records = []
    prompt_bytes = 0.0
    n_total = len(data_loader.dataset.image)
    n_run = min(n_total, args.max_samples) if args.max_samples > 0 else n_total

    for i, samples in tqdm(enumerate(data_loader), total=n_run, desc="ams i2t"):
        if i >= n_run:
            break
        image = samples["image1"].to(model.device)
        image_cloud = samples["image2"].to(model.device)

        # phase-1: teacher prepare
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reward_model.set_image_features(images=image_cloud)
        r_logits_per_image, _ = reward_model.calulate_similarity()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_teacher_ms = (t1 - t0) * 1000.0

        # phase-2: student adapt
        t_adapt_ms = 0.0
        for _ in range(args.tta_steps):
            torch.cuda.synchronize()
            a0 = time.perf_counter()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits_per_image, _ = model(image_cloud)
                loss = kd_distill_loss_v2(logits_per_image, r_logits_per_image)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            a1 = time.perf_counter()
            t_adapt_ms += (a1 - a0) * 1000.0

        # phase-3: final infer
        torch.cuda.synchronize()
        f0 = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_per_image, _ = model(image)
        _ = logits_per_image[0]
        torch.cuda.synchronize()
        f1 = time.perf_counter()
        t_final_ms = (f1 - f0) * 1000.0

        model.momentum_update_model()
        model.reset_initial()
        optimizer.load_state_dict(optim_state)

        rel_path = data_loader.dataset.annotation[i]["image"]
        upload_path = os.path.join(data_loader.dataset.vis_root2, rel_path)
        upload_kb = os.path.getsize(upload_path) / 1024.0 if os.path.exists(upload_path) else 0.0

        records.append(
            {
                "task": "i2t",
                "idx": i,
                "upload_kb": round(upload_kb, 4),
                "download_kb": round(prompt_bytes, 4),
                "t_teacher_ms": round(t_teacher_ms, 4),
                "t_adapt_ms": round(t_adapt_ms, 4),
                "t_final_ms": round(t_final_ms, 4),
            }
        )
    return records


def bench_t2i(data_loader, model, reward_model, optimizer, scaler, optim_state, args):
    records = []
    prompt_bytes = 0.0
    texts = data_loader.dataset.text
    n_total = len(texts)
    n_run = min(n_total, args.max_samples) if args.max_samples > 0 else n_total

    for i, text in tqdm(enumerate(texts[:n_run]), total=n_run, desc="ams t2i"):
        # phase-1: teacher prepare
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reward_model.set_text_features(captions=text)
        _, r_logits_per_text = reward_model.calulate_similarity()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_teacher_ms = (t1 - t0) * 1000.0

        # phase-2: student adapt
        t_adapt_ms = 0.0
        for _ in range(args.tta_steps):
            torch.cuda.synchronize()
            a0 = time.perf_counter()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                _, logits_per_text = model(images=None, text=text)
                loss = kd_distill_loss_v2(logits_per_text, r_logits_per_text)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            a1 = time.perf_counter()
            t_adapt_ms += (a1 - a0) * 1000.0

        # phase-3: final infer
        torch.cuda.synchronize()
        f0 = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _, logits_per_text = model(images=None, text=text)
        _ = logits_per_text[0]
        torch.cuda.synchronize()
        f1 = time.perf_counter()
        t_final_ms = (f1 - f0) * 1000.0

        model.momentum_update_model()
        model.reset_initial()
        optimizer.load_state_dict(optim_state)

        upload_kb = len(text.encode("utf-8")) / 1024.0
        records.append(
            {
                "task": "t2i",
                "idx": i,
                "upload_kb": round(upload_kb, 4),
                "download_kb": round(prompt_bytes, 4),
                "t_teacher_ms": round(t_teacher_ms, 4),
                "t_adapt_ms": round(t_adapt_ms, 4),
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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows, bandwidths):
    if not rows:
        print("No rows.")
        return
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k not in ("task", "idx")}
    m = {k: float(v.mean()) for k, v in arr.items()}
    s = {k: float(v.std()) for k, v in arr.items()}
    task = rows[0]["task"]
    compute = m["t_teacher_ms"] + m["t_adapt_ms"] + m["t_final_ms"]
    total_kb = m["upload_kb"] + m["download_kb"]
    print("=" * 72)
    print(f"AMS task={task}, samples={len(rows)}")
    print(f"teacher_prepare={m['t_teacher_ms']:.2f}±{s['t_teacher_ms']:.2f} ms")
    print(f"student_adapt={m['t_adapt_ms']:.2f}±{s['t_adapt_ms']:.2f} ms")
    print(f"final_infer={m['t_final_ms']:.2f}±{s['t_final_ms']:.2f} ms")
    print(f"total_compute={compute:.2f} ms")
    print("-" * 72)
    for bw in bandwidths:
        c = comm_ms(total_kb, bw)
        print(f"BW {bw:>4} Mbps -> comm={c:7.2f} ms, total={compute + c:7.2f} ms")
    print("=" * 72)


def main():
    args = parse_args()
    cfg = setup_runtime(args)
    data_loader = build_data(cfg, args)
    device = cfg.run_cfg.device
    model, reward_model, optimizer, scaler = build_models(args, device)
    optim_state = deepcopy(optimizer.state_dict())
    precompute_global(data_loader, model, reward_model, text_bs=128)

    if args.task == "i2t":
        rows = bench_i2t(data_loader, model, reward_model, optimizer, scaler, optim_state, args)
    else:
        rows = bench_t2i(data_loader, model, reward_model, optimizer, scaler, optim_state, args)

    save_csv(args.output_csv, rows)
    print(f"CSV saved: {args.output_csv}")
    print_summary(rows, args.bandwidths)


if __name__ == "__main__":
    main()
