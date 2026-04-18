#!/usr/bin/env python3
"""
ams_e2e_bench.py  ——  AMS (ICCV) 端到端延迟 & 通信量基准测试

测量项 (per-sample, 100% uplink):
  1. 上行通信量  Edge→Cloud  压缩图像码流 (.bin)
  2. 下行通信量  Cloud→Edge  text_encoder 全部参数
  3. Cloud inference 耗时
  4. Edge KL 蒸馏耗时 (text_encoder 训练)
  5. Edge 最终推理耗时
  6. 不同带宽 (1~1000 Mbps) 下的总延迟

与 CAPO (ceo_e2e_bench.py) 的关键区别:
  - AMS 对 **所有** 样本都上行到 cloud (无 entropy 筛选)
  - AMS 训练的是 edge model 的 **整个 text_encoder** (非 prompt)
  - AMS 使用 GradientGuidedAdam, lr=1e-6
  - 下行数据量 ≈ 9464 KB vs CAPO 的 3.99 KB

示例:
  python ams_e2e_bench.py ./dataset --test_sets A \
    -a RN50 -c ViT-B/16 -b 64 --gpu 0 \
    --rec_dir new --max_samples 500 \
    --output_csv results/ams_bench/bench_A.csv
"""

import argparse
import csv
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.optimizer import Optimizer

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
    imagenet_a_mask, imagenet_r_mask, imagenet_v_mask,
)

BANDWIDTHS_MBPS = [1000, 100, 50, 10, 5, 1]

BITSTREAM_KB = {
    "new": {
        "I": 7.88, "A": 6.40, "R": 6.63, "K": 13.13, "V": 7.13,
        "flower102": 5.68, "dtd": 8.85, "eurosat": 0.13, "cars": 8.03,
        "pets": 4.76, "food101": 4.24, "ucf101": 1.80, "sun397": 25.23,
        "aircraft": 14.60, "caltech101": 3.02,
    },
    "new_qp7": {
        "I": 49.28, "A": 40.00, "R": 41.42, "K": 82.05, "V": 44.59,
        "flower102": 35.19, "dtd": 52.53, "eurosat": 0.64, "cars": 58.62,
        "pets": 38.32, "food101": 29.76, "ucf101": 9.88, "sun397": 157.67,
        "aircraft": 91.24, "caltech101": 16.97,
    },
}

# ────────────────────────── GradientGuidedAdam (from ams_iccv.py) ──────────────────────────


class GradientGuidedAdam(Optimizer):
    def __init__(self, params, lr=5e-3, betas=(0.9, 0.999), eps=1e-8,
                 gamma=0.1, T_horizon=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma,
                        T_horizon=T_horizon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            gamma = group['gamma']
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
                    _, indices = torch.topk(u.abs().flatten(),
                                            int(gamma * u.numel()))
                    mask = torch.zeros_like(u, dtype=torch.bool).flatten()
                    mask[indices] = True
                    state['mask'] = mask.view_as(u)
                param.data.addcmul_(u, state['mask'].float(), value=-lr)
        return loss


# ────────────────────────── timed TTA (AMS style) ──────────────────────────


def ams_tta_timed(cloud_model, edge_model, rec, optimizer_teacher,
                  teacher_scaler, args):
    """AMS TTA: cloud inference + KL distill to edge text_encoder. Returns (ms, ms)."""
    criterion_div = nn.KLDivLoss(reduction='batchmean')
    t_cloud_ms, t_distill_ms = 0.0, 0.0

    for _ in range(args.tta_steps):
        # ── phase 1: cloud feature extraction ──
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            cloud_feat = cloud_model.get_image_features(rec)

        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        t_cloud_ms += (_t1 - _t0) * 1000

        # ── phase 2: KL distillation → edge text_encoder ──
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = cloud_model(None, cloud_feat)
            del cloud_feat
            edge_feat = edge_model.get_image_features(rec)
            output_device = edge_model(None, edge_feat)
            tau = 0.5
            loss = criterion_div(
                F.log_softmax(output_device / tau, dim=1),
                F.softmax(output / tau, dim=1),
            )
        optimizer_teacher.zero_grad()
        teacher_scaler.scale(loss).backward()
        teacher_scaler.step(optimizer_teacher)
        teacher_scaler.update()

        torch.cuda.synchronize()
        _t2 = time.perf_counter()
        t_distill_ms += (_t2 - _t1) * 1000

    return t_cloud_ms, t_distill_ms


# ────────────────────────── comm latency helper ──────────────────────────


def comm_latency_ms(total_bytes, bw_mbps):
    return total_bytes * 8.0 / (bw_mbps * 1e3)


# ────────────────────────── summary printer ──────────────────────────


def print_summary(records, set_id, upload_bytes, download_bytes, gpu_name):
    total = len(records)
    if total == 0:
        print("No records collected.")
        return

    acc = sum(r["correct"] for r in records) / total * 100
    total_comm = upload_bytes + download_bytes

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  AMS E2E Benchmark  |  Dataset: {set_id}  |  GPU: {gpu_name}")
    print(f"  Samples: {total}   (100% uplink — all samples)")
    print(sep)

    print(f"\n  [Accuracy]")
    print(f"    Post-TTA:  {acc:.2f}%")

    print(f"\n  [Communication per sample]")
    print(f"    Upload  (compressed image):   {upload_bytes/1024:10.2f} KB")
    print(f"    Download (text_encoder):      {download_bytes/1024:10.2f} KB")
    print(f"    Total:                        {total_comm/1024:10.2f} KB")

    def _s(key):
        v = [r[key] for r in records]
        return np.mean(v), np.std(v)

    mc, sc = _s("t_cloud_ms")
    md, sd = _s("t_distill_ms")
    mf, sf = _s("t_final_ms")
    m_compute = mc + md + mf

    print(f"\n  [Compute time per sample (ms)]")
    print(f"    Cloud inference:    {mc:8.2f} ± {sc:5.2f}")
    print(f"    KL distillation:   {md:8.2f} ± {sd:5.2f}")
    print(f"    Edge final infer:  {mf:8.2f} ± {sf:5.2f}")
    print(f"    ── Total compute:  {m_compute:8.2f}")

    print(f"\n  [Total latency under different bandwidths (ms)]")
    print(f"    {'Bandwidth':>12s}  {'Comm':>10s}  {'Compute':>9s}  {'Sync Total':>11s}")
    print(f"    {'─'*12}  {'─'*10}  {'─'*9}  {'─'*11}")
    for bw in BANDWIDTHS_MBPS:
        cm = comm_latency_ms(total_comm, bw)
        sync_t = cm + m_compute
        print(f"    {bw:>8d} Mbps  {cm:10.2f}  {m_compute:9.2f}  {sync_t:11.2f}")

    # ── trainable param count ──
    print(f"\n  [Trainable parameters (text_encoder)]")
    print(f"    Download size:  {download_bytes/1024:.2f} KB  "
          f"({download_bytes/1024/1024:.2f} MB)")

    print(f"\n{sep}\n")


# ────────────────────────── main benchmark loop ──────────────────────────


def run_benchmark(args):
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu)
    print(f"GPU {args.gpu}: {gpu_name}")

    set_id = args.test_sets

    # ── build models (same as ams_iccv.py) ──
    cloud_model = get_coop(args.cloud_arch, set_id, args.gpu,
                           args.n_ctx, args.ctx_init)
    edge_model = get_coop(args.arch, set_id, args.gpu,
                          args.n_ctx, args.ctx_init, type='Device')

    # AMS: freeze all cloud params, train edge text_encoder
    for param in cloud_model.parameters():
        param.requires_grad_(False)

    teacher_trainable = []
    for name, param in edge_model.named_parameters():
        if 'text_encoder' in name:
            param.requires_grad_(True)
            teacher_trainable.append(param)
        else:
            param.requires_grad_(False)

    cloud_model = cloud_model.cuda(args.gpu)
    edge_model = edge_model.cuda(args.gpu)

    optimizer_teacher = GradientGuidedAdam(teacher_trainable, args.lr)
    teacher_optim_state = deepcopy(optimizer_teacher.state_dict())
    teacher_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cudnn.benchmark = True

    # ── text_encoder param size (= download size) ──
    download_bytes = sum(
        p.numel() * p.element_size() for p in teacher_trainable
    )
    print(f"text_encoder trainable: {len(teacher_trainable)} tensors, "
          f"{download_bytes} B ({download_bytes/1024:.2f} KB, "
          f"{download_bytes/1024/1024:.2f} MB)")

    # ── data transforms (same as ams_iccv.py) ──
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
        if set_id in ['A', 'R', 'V']:
            label_mask = eval(f"imagenet_{set_id.lower()}_mask")
            if set_id == 'R':
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
        mode=args.dataset_mode, rec_dir=args.rec_dir,
    )
    if 0 < args.max_samples < len(val_dataset):
        import random as _rnd
        indices = list(range(len(val_dataset)))
        _rnd.seed(args.seed)
        _rnd.shuffle(indices)
        val_dataset = torch.utils.data.Subset(
            val_dataset, indices[:args.max_samples]
        )

    n_images = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── upload size (compressed bitstream, same as CAPO) ──
    rec_key = args.rec_dir
    if rec_key in BITSTREAM_KB and set_id in BITSTREAM_KB[rec_key]:
        avg_upload_bytes = BITSTREAM_KB[rec_key][set_id] * 1024
        src = f"built-in table ({rec_key})"
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
    records = []
    warmup = args.warmup

    print(f"\n{'─'*60}")
    print(f"AMS Benchmarking {n_images} samples  (warmup={warmup})")
    print(f"  lr={args.lr}  tta_steps={args.tta_steps}  bs={args.batch_size}")
    print(f"{'─'*60}")

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

        # ── reset model state before each sample (same as ams_iccv.py) ──
        with torch.no_grad():
            cloud_model.reset()
            edge_model.reset()
        optimizer_teacher.load_state_dict(teacher_optim_state)

        # ── TTA ──
        t_cloud_ms, t_distill_ms = ams_tta_timed(
            cloud_model, edge_model, recs,
            optimizer_teacher, teacher_scaler, args,
        )

        # ── edge final inference ──
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                ori_feat = edge_model.get_image_features(ori)
                output = edge_model(None, ori_feat)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_final_ms = (t3 - t2) * 1000

        pred = output.argmax(1).item()
        tgt = target.item()

        if i >= warmup:
            records.append({
                "idx": i,
                "target": tgt,
                "pred": pred,
                "correct": int(pred == tgt),
                "upload_kb": round(avg_upload_bytes / 1024, 2),
                "download_kb": round(download_bytes / 1024, 2),
                "t_cloud_ms": round(t_cloud_ms, 4),
                "t_distill_ms": round(t_distill_ms, 4),
                "t_final_ms": round(t_final_ms, 4),
            })

        if (i + 1) % args.print_freq == 0:
            n_done = max(len(records), 1)
            acc = sum(r["correct"] for r in records) / n_done * 100
            print(f"  [{i+1:>6d}/{n_images}]  acc={acc:5.2f}%  "
                  f"cloud={t_cloud_ms:.1f}ms  distill={t_distill_ms:.1f}ms  "
                  f"final={t_final_ms:.1f}ms")

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
    print_summary(records, set_id, avg_upload_bytes, download_bytes, gpu_name)


# ────────────────────────── CLI ──────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="AMS E2E Benchmark — latency & communication profiling",
    )
    p.add_argument("data", metavar="DIR", help="dataset root")
    p.add_argument("--test_sets", type=str, default="A")
    p.add_argument("--dataset_mode", type=str, default="test")
    p.add_argument("-a", "--arch", default="RN50")
    p.add_argument("-c", "--cloud_arch", default="ViT-B/16")
    p.add_argument("--resolution", default=224, type=int)
    p.add_argument("-j", "--workers", default=8, type=int)
    p.add_argument("-b", "--batch-size", default=64, type=int)
    p.add_argument("--lr", default=1e-6, type=float)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--tta_steps", default=1, type=int)
    p.add_argument("--n_ctx", default=4, type=int)
    p.add_argument("--ctx_init", default=None, type=str)
    p.add_argument("--cocoop", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tpt", action="store_true", default=True)
    p.add_argument("--rec_dir", type=str, default="new")
    p.add_argument("-p", "--print-freq", default=200, type=int)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--output_csv", type=str, default="results/ams_bench/bench.csv")
    p.add_argument("--warmup", type=int, default=10)

    run_benchmark(p.parse_args())


if __name__ == "__main__":
    main()
