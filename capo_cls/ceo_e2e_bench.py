#!/usr/bin/env python3
"""
ceo_e2e_bench.py  ——  CAPO 端到端延迟 & 通信量基准测试

测量项 (per-sample):
  1. 上行通信量  Edge→Cloud  压缩图像码流 (.bin)
  2. 下行通信量  Cloud→Edge  优化后的 prompt ctx 参数
  3. Cloud prompt TTA 优化耗时
  4. Edge KL 蒸馏耗时
  5. Edge 最终推理耗时
  6. Sync vs Async 准确率 & 延迟对比
  7. 不同带宽 (1~1000 Mbps) 下的总延迟

示例:
  # 快速测时间 (500 samples, QP1, 用内置码流表):
  python ceo_e2e_bench.py ./dataset --test_sets A \
    -a RN50 -c ViT-B/16 -b 64 --gpu 0 \
    --ctx_init a_photo_of_a --tau 0.4 --beta 0.9 \
    --uplink_percent 100 --rec_dir new \
    --max_samples 500 --output_csv results/bench_A.csv

  # 用实测码流 (QP7, 全量, 指定 bin_root):
  python ceo_e2e_bench.py ./dataset --test_sets A \
    -a RN50 -c ViT-B/16 -b 64 --gpu 0 \
    --ctx_init a_photo_of_a --rec_dir new_qp7 \
    --bin_root "${ENC2DEC_ROOT}/intermediate_qp7" \
    --max_samples 0 --output_csv results/bench_A_qp7.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip_device import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import (
    thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask,
)

BANDWIDTHS_MBPS = [1000, 100, 50, 10, 5, 1]

# Per-image compressed bitstream sizes (KB).
# QP7: measured from enc2dec/intermediate_qp7 .bin files.
# QP1: measured where available; OOD estimated via avg ratio QP1/QP7 ≈ 0.16
#       (measured on 8 fewshot datasets: mean=0.162, range 0.12–0.20).
BITSTREAM_KB = {
    "new_qp7": {
        "I": 49.28, "A": 40.00, "R": 41.42, "K": 82.05, "V": 44.59,
        "flower102": 35.19, "dtd": 52.53, "eurosat": 0.64, "cars": 58.62,
        "pets": 38.32, "food101": 29.76, "ucf101": 9.88, "sun397": 157.67,
        "aircraft": 91.24, "caltech101": 16.97,
    },
    "new": {
        "I": 7.88, "A": 6.40, "R": 6.63, "K": 13.13, "V": 7.13,
        "flower102": 5.68, "dtd": 8.85, "eurosat": 0.13, "cars": 8.03,
        "pets": 4.76, "food101": 4.24, "ucf101": 1.80, "sun397": 25.23,
        "aircraft": 14.60, "caltech101": 3.02,
    },
}

# ───────────────────────── helpers (from ceo.py) ─────────────────────────


def select_confident_samples(logits, top):
    batch_entropy = (-logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[
        : int(batch_entropy.size()[0] * top)
    ]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


# ───────────────────────── timed TTA ─────────────────────────


def test_time_tuning_timed(
    cloud_model, edge_model, rec,
    optimizer_cloud, optimizer_edge,
    cloud_scaler, edge_scaler, args,
):
    """Run TTA with separate timing for cloud vs edge phases (ms)."""
    criterion_div = nn.KLDivLoss(reduction="batchmean")
    selected_idx = None
    t_cloud_ms, t_distill_ms = 0.0, 0.0

    for _ in range(args.tta_steps):
        # ── phase 1: cloud prompt optimisation ──
        torch.cuda.synchronize()
        _t0 = time.perf_counter()

        with torch.cuda.amp.autocast():
            cloud_feat = cloud_model.get_image_features(rec)
            output = cloud_model(None, cloud_feat)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args.selection_p
                )
            loss_cloud = avg_entropy(output)
        optimizer_cloud.zero_grad()
        cloud_scaler.scale(loss_cloud).backward()
        cloud_scaler.step(optimizer_cloud)
        cloud_scaler.update()

        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        t_cloud_ms += (_t1 - _t0) * 1000

        # ── phase 2: edge distillation (runs on cloud side in real deploy) ──
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = cloud_model(None, cloud_feat)
            del cloud_feat
            edge_feat = edge_model.get_image_features(rec)
            output_device = edge_model(None, edge_feat)
            tau = args.tau
            loss_kl = args.beta * criterion_div(
                F.log_softmax(output_device / tau, dim=1),
                F.softmax(output / tau, dim=1),
            )
            output_device, _ = select_confident_samples(
                output_device, args.selection_p
            )
            loss_ent = avg_entropy(output_device)
            loss_edge = loss_ent + loss_kl
        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_edge).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()

        torch.cuda.synchronize()
        _t2 = time.perf_counter()
        t_distill_ms += (_t2 - _t1) * 1000

    return t_cloud_ms, t_distill_ms


# ───────────────────────── bitstream scanner ─────────────────────────

SET_TO_BIN = {
    "I": "I", "A": "A", "R": "R", "K": "K", "V": "V",
    "flower102": "flower102", "dtd": "dtd", "eurosat": "eurosat",
    "cars": "cars", "pets": "pets", "food101": "food101",
    "ucf101": "ucf101", "sun397": "sun397", "aircraft": "aircraft",
    "caltech101": "caltech101",
}


def scan_avg_bin_size(bin_root, set_id, num_images):
    """Return average per-image bitstream bytes from the .bin directory."""
    ds_dir = os.path.join(bin_root, SET_TO_BIN.get(set_id, set_id))
    if not os.path.isdir(ds_dir):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(ds_dir)
        for f in files
        if f.endswith(".bin")
    )
    return total / max(num_images, 1)


# ───────────────────────── comm latency helper ─────────────────────────


def comm_latency_ms(total_bytes, bw_mbps):
    """One-way communication latency: bytes → ms at given Mbps."""
    return total_bytes * 8.0 / (bw_mbps * 1e3)


# ───────────────────────── summary printer ─────────────────────────


def print_summary(records, set_id, upload_bytes, download_bytes, gpu_name):
    total = len(records)
    if total == 0:
        print("No records collected.")
        return
    uplinked = [r for r in records if r["is_uplink"]]
    n_up = len(uplinked)

    sync_acc = sum(r["correct_sync"] for r in records) / total * 100
    async_acc = sum(r["correct_async"] for r in records) / total * 100
    total_comm = upload_bytes + download_bytes

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  CAPO E2E Benchmark  |  Dataset: {set_id}  |  GPU: {gpu_name}")
    print(f"  Samples: {total}   Uplinked: {n_up} ({n_up/total*100:.1f}%)")
    print(sep)

    # ── accuracy ──
    print(f"\n  [Accuracy]")
    print(f"    Sync  (post-TTA):  {sync_acc:.2f}%")
    print(f"    Async (pre-TTA):   {async_acc:.2f}%")
    print(f"    Δ (sync − async):  {sync_acc - async_acc:+.2f}%")

    # ── communication volume ──
    print(f"\n  [Communication per uplinked sample]")
    print(f"    Upload  (compressed image): {upload_bytes/1024:8.2f} KB")
    print(f"    Download (prompt ctx):      {download_bytes/1024:8.2f} KB")
    print(f"    Total:                      {total_comm/1024:8.2f} KB")

    # effective per-sample average considering uplink rate
    eff_comm = total_comm * (n_up / total) if total > 0 else 0
    print(f"    Effective avg/sample:       {eff_comm/1024:8.2f} KB")

    # ── compute time breakdown ──
    if uplinked:
        def _s(key):
            v = [r[key] for r in uplinked]
            return np.mean(v), np.std(v)

        me, se = _s("t_entropy_ms")
        mc, sc = _s("t_cloud_ms")
        md, sd = _s("t_distill_ms")
        mf, sf = _s("t_final_ms")
        m_cloud_total = mc + md
        m_compute = me + mc + md + mf

        print(f"\n  [Compute time per uplinked sample (ms)]")
        print(f"    Edge entropy check:      {me:7.2f} ± {se:5.2f}")
        print(f"    Cloud TTA (prompt opt):  {mc:7.2f} ± {sc:5.2f}")
        print(f"    Cloud (edge distill):    {md:7.2f} ± {sd:5.2f}")
        print(f"    ── Cloud total:          {m_cloud_total:7.2f}")
        print(f"    Edge final inference:    {mf:7.2f} ± {sf:5.2f}")
        print(f"    ── Total compute:        {m_compute:7.2f}")

        # ── bandwidth-dependent latency ──
        # Sync: t_entropy + t_upload + t_cloud_total + t_download + t_final
        # Async: t_entropy only (edge predicts immediately)
        print(f"\n  [Total latency under different bandwidths (ms)]")
        print(f"    {'Bandwidth':>12s}  {'Comm':>8s}  {'Compute':>9s}"
              f"  {'Sync Total':>11s}  {'Async':>8s}")
        print(f"    {'─'*12}  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*8}")
        for bw in BANDWIDTHS_MBPS:
            cm = comm_latency_ms(total_comm, bw)
            sync_t = me + cm + m_cloud_total + mf
            async_t = me
            print(f"    {bw:>8d} Mbps  {cm:8.2f}  {m_compute:9.2f}"
                  f"  {sync_t:11.2f}  {async_t:8.2f}")

        # ── vs AMS comparison (comm only) ──
        AMS_BYTES = 9464.58 * 1024
        print(f"\n  [Communication comparison: CAPO vs AMS]")
        print(f"    CAPO: {total_comm/1024:.2f} KB/sample   "
              f"AMS: 9464.58 KB/sample")
        print(f"    {'Bandwidth':>12s}  {'CAPO comm':>11s}  {'AMS comm':>11s}"
              f"  {'Ratio':>7s}")
        print(f"    {'─'*12}  {'─'*11}  {'─'*11}  {'─'*7}")
        for bw in BANDWIDTHS_MBPS:
            c_capo = comm_latency_ms(total_comm, bw)
            c_ams = comm_latency_ms(AMS_BYTES, bw)
            ratio = c_ams / c_capo if c_capo > 0 else float("inf")
            print(f"    {bw:>8d} Mbps  {c_capo:9.2f}ms  {c_ams:9.0f}ms"
                  f"  {ratio:6.0f}×")

    print(f"\n{sep}\n")


# ───────────────────────── main benchmark loop ─────────────────────────


def run_benchmark(args):
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu)
    print(f"GPU {args.gpu}: {gpu_name}")

    set_id = args.test_sets

    # ── build models ──
    cloud_model = get_coop(
        args.cloud_arch, set_id, args.gpu, args.n_ctx, args.ctx_init,
    )
    edge_model = get_coop(
        args.arch, set_id, args.gpu, args.n_ctx, args.ctx_init, type="Device",
    )

    cloud_trainable, edge_trainable = [], []
    for name, param in cloud_model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad_(True)
            cloud_trainable.append(param)
        else:
            param.requires_grad_(False)
    for name, param in edge_model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad_(True)
            edge_trainable.append(param)
        else:
            param.requires_grad_(False)

    cloud_model = cloud_model.cuda(args.gpu)
    edge_model = edge_model.cuda(args.gpu)

    optimizer_cloud = optim.AdamW(cloud_trainable, args.lr)
    optimizer_edge = optim.AdamW(edge_trainable, args.lr)
    cloud_optim_state = deepcopy(optimizer_cloud.state_dict())
    edge_optim_state = deepcopy(optimizer_edge.state_dict())
    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cudnn.benchmark = True

    # ── prompt size ──
    ctx = edge_model.prompt_learner.ctx
    prompt_bytes = ctx.numel() * ctx.element_size()
    print(f"Prompt: {args.n_ctx} tokens × {ctx.shape[-1]}d  "
          f"dtype={ctx.dtype}  size={prompt_bytes} B ({prompt_bytes/1024:.2f} KB)")

    # ── data transforms ──
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
    ])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    data_transform = AugMixAugmenter(
        base_transform, preprocess,
        n_views=args.batch_size - 1, augmix=len(set_id) > 1,
    )
    device_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(), normalize,
    ])

    # ── classnames ──
    if len(set_id) > 1:
        classnames = eval(f"{set_id.lower()}_classes")
    else:
        classnames_all = imagenet_classes
        if set_id in ["A", "R", "V"]:
            label_mask = eval(f"imagenet_{set_id.lower()}_mask")
            if set_id == "R":
                classnames = [classnames_all[i]
                              for i, m in enumerate(label_mask) if m]
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all

    cloud_model.reset_classnames(classnames, args.cloud_arch)
    edge_model.reset_classnames(classnames, args.arch)

    # ── dataset ──
    val_dataset = build_dataset(
        set_id, device_transform, data_transform, args.data,
        mode=args.dataset_mode, cloud_use_ori=args.cloud_use_ori,
        rec_dir=args.rec_dir,
    )
    if 0 < args.max_samples < len(val_dataset):
        import random as _rnd
        indices = list(range(len(val_dataset)))
        _rnd.seed(args.seed)
        _rnd.shuffle(indices)
        val_dataset = torch.utils.data.Subset(
            val_dataset, indices[: args.max_samples]
        )

    n_images = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── average bitstream size ──
    if args.bin_root and os.path.isdir(args.bin_root):
        avg_upload_bytes = scan_avg_bin_size(args.bin_root, set_id, n_images)
        src = f"measured from {args.bin_root}"
    elif args.rec_dir in BITSTREAM_KB and set_id in BITSTREAM_KB[args.rec_dir]:
        avg_upload_bytes = BITSTREAM_KB[args.rec_dir][set_id] * 1024
        src = f"built-in table ({args.rec_dir})"
    else:
        avg_upload_bytes = 8.0 * 1024
        src = "fallback estimate"
    print(f"Avg bitstream/image: {avg_upload_bytes/1024:.2f} KB  ({src})")

    # ── benchmark loop ──
    cloud_model.eval()
    edge_model.eval()
    with torch.no_grad():
        cloud_model.reset()
        edge_model.reset()
    all_entropy = deque(maxlen=500)
    records = []
    warmup = args.warmup

    print(f"\n{'─'*50}")
    print(f"Benchmarking {n_images} samples  (warmup={warmup})")
    print(f"  uplink={args.uplink_percent}%  selection_p={args.selection_p}"
          f"  tau={args.tau}  beta={args.beta}  tta_steps={args.tta_steps}")
    print(f"{'─'*50}")

    for i, (oris, recs, target) in enumerate(val_loader):
        # ── prepare tensors ──
        if isinstance(oris, list):
            ori = oris[0].cuda(args.gpu, non_blocking=True)
        else:
            if len(oris.size()) > 4:
                oris = oris.squeeze(0)
            oris = oris.cuda(args.gpu, non_blocking=True)
            ori = oris

        if isinstance(recs, list):
            for k in range(len(recs)):
                recs[k] = recs[k].cuda(args.gpu, non_blocking=True)
        else:
            if len(recs.size()) > 4:
                recs = recs.squeeze(0)
            recs = recs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        if isinstance(recs, list):
            recs = torch.cat(recs, dim=0)

        # ── step 1: edge entropy check  (always, needed for async) ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            ori_feat = edge_model.get_image_features(ori)
            with torch.cuda.amp.autocast():
                output_pre = edge_model(None, ori_feat)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_entropy_ms = (t1 - t0) * 1000

        sample_ent = (
            -output_pre.softmax(1) * output_pre.log_softmax(1)
        ).sum(1).item()
        all_entropy.append(sample_ent)

        if args.uplink_percent >= 100.0:
            is_uplink = True
        else:
            thr = np.percentile(all_entropy, 100.0 - args.uplink_percent)
            is_uplink = sample_ent > thr

        pred_async = output_pre.argmax(1).item()

        # ── step 2: TTA  (only for uplinked samples) ──
        t_cloud_ms = 0.0
        t_distill_ms = 0.0
        t_final_ms = 0.0

        if is_uplink:
            with torch.no_grad():
                cloud_model.reset()
                edge_model.reset()
            optimizer_cloud.load_state_dict(cloud_optim_state)
            optimizer_edge.load_state_dict(edge_optim_state)

            t_cloud_ms, t_distill_ms = test_time_tuning_timed(
                cloud_model, edge_model, recs,
                optimizer_cloud, optimizer_edge,
                cloud_scaler, edge_scaler, args,
            )

            # ── step 3: edge final inference (post-TTA) ──
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                ori_feat = edge_model.get_image_features(ori)
                with torch.cuda.amp.autocast():
                    output_post = edge_model(None, ori_feat)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            t_final_ms = (t3 - t2) * 1000

            pred_sync = output_post.argmax(1).item()
        else:
            pred_sync = pred_async

        tgt = target.item()

        if i >= warmup:
            records.append({
                "idx": i,
                "is_uplink": int(is_uplink),
                "target": tgt,
                "pred_sync": pred_sync,
                "pred_async": pred_async,
                "correct_sync": int(pred_sync == tgt),
                "correct_async": int(pred_async == tgt),
                "upload_kb": avg_upload_bytes / 1024,
                "download_kb": prompt_bytes / 1024,
                "t_entropy_ms": round(t_entropy_ms, 4),
                "t_cloud_ms": round(t_cloud_ms, 4),
                "t_distill_ms": round(t_distill_ms, 4),
                "t_final_ms": round(t_final_ms, 4),
            })

        if (i + 1) % args.print_freq == 0:
            n_done = max(len(records), 1)
            acc_s = sum(r["correct_sync"] for r in records) / n_done * 100
            acc_a = sum(r["correct_async"] for r in records) / n_done * 100
            print(f"  [{i+1:>6d}/{n_images}]  "
                  f"sync={acc_s:5.2f}%  async={acc_a:5.2f}%  "
                  f"cloud={t_cloud_ms:.1f}ms  distill={t_distill_ms:.1f}ms")

    # ── save CSV ──
    if args.output_csv and records:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fieldnames = list(records[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nCSV saved → {args.output_csv}  ({len(records)} rows)")

    # ── summary ──
    print_summary(records, set_id, avg_upload_bytes, prompt_bytes, gpu_name)


# ───────────────────────── CLI ─────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="CAPO E2E Benchmark — latency & communication profiling",
    )
    # same as ceo.py
    p.add_argument("data", metavar="DIR", help="dataset root")
    p.add_argument("--test_sets", type=str, default="A")
    p.add_argument("--dataset_mode", type=str, default="test")
    p.add_argument("-a", "--arch", default="RN50")
    p.add_argument("-c", "--cloud_arch", default="ViT-B/16")
    p.add_argument("--resolution", default=224, type=int)
    p.add_argument("-j", "--workers", default=8, type=int)
    p.add_argument("-b", "--batch-size", default=64, type=int)
    p.add_argument("--lr", default=5e-3, type=float)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--selection_p", default=0.1, type=float)
    p.add_argument("--tta_steps", default=1, type=int)
    p.add_argument("--n_ctx", default=4, type=int)
    p.add_argument("--ctx_init", default=None, type=str)
    p.add_argument("--cocoop", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--uplink_percent", type=float, default=100.0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--cloud_use_ori", action="store_true", default=False)
    p.add_argument("--rec_dir", type=str, default="new")
    p.add_argument("-p", "--print-freq", default=200, type=int)

    # benchmark-specific
    p.add_argument(
        "--bin_root", type=str, default="",
        help="root dir of intermediate .bin files (for upload size measurement)",
    )
    p.add_argument(
        "--output_csv", type=str, default="bench_results.csv",
        help="output CSV path for per-sample metrics",
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="skip first N samples for timing statistics (GPU warmup)",
    )

    run_benchmark(p.parse_args())


if __name__ == "__main__":
    main()
