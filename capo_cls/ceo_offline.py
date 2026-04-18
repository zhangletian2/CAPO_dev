#!/usr/bin/env python
"""
CEO Offline: Train shared prompts on ImageNet val (few-shot, no labels),
then evaluate the frozen edge prompt on ImageNet variants + 10 fewshot datasets.

Training (consistent with CEO TTA):
    Both cloud (ViT-B/16) and edge (RN50) receive reconstructed images
    (dataset/imagenet/images/new/) during prompt optimisation.
    Loss = batch entropy (cloud) + batch entropy (edge) + β·KL(edge ‖ cloud)
    Only prompt_learner parameters are updated; CLIP encoders stay frozen.

Evaluation:
    Freeze everything (including prompts).
    Edge model infers on original images: I / A / R / K / V + 10 fewshot test splits.

Usage:
    python ceo_offline.py ./dataset --n_shot 16 --epochs 5 --gpu 0
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
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
from data.datautils import build_dataset, DualRootImageFolder
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import (
    thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask,
)
from data.cls_to_names import *
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed


# ── losses ────────────────────────────────────────────────────────────

def batch_entropy(logits):
    """Per-sample entropy, averaged over the batch."""
    return (-logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


def select_confident_samples(logits, top):
    """Return (selected_logits, idx) keeping the *top* fraction with lowest entropy."""
    ent = (-logits.softmax(1) * logits.log_softmax(1)).sum(1)
    k = max(1, int(ent.size(0) * top))
    idx = torch.argsort(ent)[:k]
    return logits[idx], idx


# ── data helpers ──────────────────────────────────────────────────────

def fewshot_subsample(dataset, n_shot, seed=0):
    """Deterministically sample *n_shot* images per class from a
    DualRootImageFolder (whose ``samples`` are (path1, path2, cls_idx))."""
    class_indices = defaultdict(list)
    for idx, (f1, _, cls_idx) in enumerate(dataset.samples):
        class_indices[cls_idx].append((str(f1), idx))

    rng = random.Random(seed)
    selected = []
    for cls in sorted(class_indices.keys()):
        items = sorted(class_indices[cls])
        rng.shuffle(items)
        selected.extend(i for _, i in items[:n_shot])
    return torch.utils.data.Subset(dataset, sorted(selected))


def get_classnames_for_set(set_id):
    """Return the class-name list used by the model for *set_id*."""
    if set_id in fewshot_datasets:
        return eval("{}_classes".format(set_id.lower()))
    classnames_all = imagenet_classes
    if set_id in ('A', 'R', 'V'):
        label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
        if set_id == 'R':
            return [classnames_all[i] for i, m in enumerate(label_mask) if m]
        return [classnames_all[i] for i in label_mask]
    return classnames_all          # I, K


# ── training ──────────────────────────────────────────────────────────

def _build_scheduler(optimizer, total_steps, cosine=False):
    if not cosine:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0)


def train_offline(cloud_model, edge_model, train_loader,
                  optimizer_cloud, optimizer_edge,
                  cloud_scaler, edge_scaler, args):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    cloud_model.eval()
    edge_model.eval()

    total_steps = args.epochs * len(train_loader)
    sched_c = _build_scheduler(optimizer_cloud, total_steps, args.cosine)
    sched_e = _build_scheduler(optimizer_edge,  total_steps, args.cosine)

    sel_p = args.selection_p

    for epoch in range(args.epochs):
        ep_c, ep_e, ep_kl, n = 0.0, 0.0, 0.0, 0
        t0 = time.time()
        for oris, recs, _ in train_loader:
            imgs = oris if args.use_ori else recs
            imgs = imgs.cuda(args.gpu, non_blocking=True)

            # ① cloud prompt — entropy on confident subset
            with torch.cuda.amp.autocast():
                c_feat = cloud_model.get_image_features(imgs)
                c_out = cloud_model(None, c_feat)
                if sel_p < 1.0:
                    c_out_sel, _ = select_confident_samples(c_out, sel_p)
                else:
                    c_out_sel = c_out
                loss_c = batch_entropy(c_out_sel)
            optimizer_cloud.zero_grad()
            cloud_scaler.scale(loss_c).backward()
            cloud_scaler.step(optimizer_cloud)
            cloud_scaler.update()
            if sched_c is not None:
                sched_c.step()

            # ② edge prompt — KL on all samples + entropy on confident subset
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    c_out_det = cloud_model(None, c_feat)
                e_feat = edge_model.get_image_features(imgs)
                e_out = edge_model(None, e_feat)
                loss_kl = args.beta * criterion_kl(
                    F.log_softmax(e_out / args.tau, dim=1),
                    F.softmax(c_out_det / args.tau, dim=1))
                if sel_p < 1.0:
                    e_out_sel, _ = select_confident_samples(e_out, sel_p)
                else:
                    e_out_sel = e_out
                loss_e = batch_entropy(e_out_sel)
                loss_edge = loss_e + loss_kl
            optimizer_edge.zero_grad()
            edge_scaler.scale(loss_edge).backward()
            edge_scaler.step(optimizer_edge)
            edge_scaler.update()
            if sched_e is not None:
                sched_e.step()

            ep_c += loss_c.item(); ep_e += loss_e.item()
            ep_kl += loss_kl.item(); n += 1

        c_lr = optimizer_cloud.param_groups[0]['lr']
        e_lr = optimizer_edge.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1}/{args.epochs}  "
              f"cloud_ent={ep_c / n:.4f}  edge_ent={ep_e / n:.4f}  "
              f"kl={ep_kl / n:.4f}  "
              f"c_lr={c_lr:.2e} e_lr={e_lr:.2e}  ({elapsed:.1f}s)")


# ── evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(edge_model, test_loader, args):
    edge_model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    for oris, _recs, target in test_loader:
        oris = oris.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.cuda.amp.autocast():
            feat = edge_model.get_image_features(oris)
            output = edge_model(None, feat)
        a1, a5 = accuracy(output, target, topk=(1, 5))
        top1.update(a1[0], oris.size(0))
        top5.update(a5[0], oris.size(0))

    return top1.avg.item(), top5.avg.item()


# ── main ──────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    eval_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize])

    data_root = args.data

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1 — Offline Training on ImageNet val (few-shot, unsupervised)
    # ═══════════════════════════════════════════════════════════════════

    if args.ctx_init and args.ctx_init.lower() == 'none':
        args.ctx_init = None

    cloud_lr = args.cloud_lr if args.cloud_lr is not None else args.lr
    args._cloud_lr = cloud_lr

    tag = (f'offline_{args.n_shot}shot_ep{args.epochs}'
           f'_tau{args.tau}_beta{args.beta}'
           f'_lr{args.lr}')
    if cloud_lr != args.lr:
        tag += f'_clr{cloud_lr}'
    if args.selection_p < 1.0:
        tag += f'_sp{args.selection_p}'
    if args.cosine:
        tag += '_cos'
    if args.ctx_init is None:
        tag += '_randinit'
    if args.use_ori:
        tag += '_ori'
    tag += f'_seed{args.seed}'
    save_dir = args.save_dir or os.path.join('checkpoints', tag)
    os.makedirs(save_dir, exist_ok=True)
    prompt_path = os.path.join(save_dir, 'prompts.pt')

    # create models (ImageNet-1K class names for training)
    print("Creating models …")
    cloud_model = get_coop(args.cloud_arch, 'I', args.gpu,
                           args.n_ctx, args.ctx_init)
    edge_model  = get_coop(args.arch, 'I', args.gpu,
                           args.n_ctx, args.ctx_init, type='Device')

    cloud_trainable, edge_trainable = [], []
    for name, p in cloud_model.named_parameters():
        if 'prompt_learner' in name:
            p.requires_grad_(True); cloud_trainable.append(p)
        else:
            p.requires_grad_(False)
    for name, p in edge_model.named_parameters():
        if 'prompt_learner' in name:
            p.requires_grad_(True); edge_trainable.append(p)
        else:
            p.requires_grad_(False)

    cloud_model = cloud_model.cuda(args.gpu)
    edge_model  = edge_model.cuda(args.gpu)

    if args.load_prompt and os.path.isfile(args.load_prompt):
        # ── skip training, load existing prompts ──
        ckpt = torch.load(args.load_prompt, map_location=f'cuda:{args.gpu}')
        edge_model.prompt_learner.ctx.data.copy_(ckpt['edge_ctx'])
        print(f"Loaded edge prompt from {args.load_prompt}")
    else:
        # ── build training set ──
        print("═══ Phase 1: Offline Training ═══")
        root_ori = os.path.join(data_root, 'imagenet', 'images', 'val')
        root_rec = os.path.join(data_root, 'imagenet', 'images', args.rec_dir)
        full_ds = DualRootImageFolder(root_ori, root_rec,
                                      base_transform=eval_transform,
                                      transform=eval_transform)
        train_ds = fewshot_subsample(full_ds, args.n_shot, seed=args.seed)
        print(f"  Training set: {len(train_ds)} images "
              f"({args.n_shot}-shot × {len(full_ds.classes)} classes)")

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)

        optimizer_cloud = optim.AdamW(cloud_trainable, lr=cloud_lr)
        optimizer_edge  = optim.AdamW(edge_trainable,  lr=args.lr)
        cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
        edge_scaler  = torch.cuda.amp.GradScaler(init_scale=1000)

        train_offline(cloud_model, edge_model, train_loader,
                      optimizer_cloud, optimizer_edge,
                      cloud_scaler, edge_scaler, args)

        # save prompts
        torch.save({
            'cloud_ctx': cloud_model.prompt_learner.ctx.detach().cpu(),
            'edge_ctx':  edge_model.prompt_learner.ctx.detach().cpu(),
            'args': vars(args),
        }, prompt_path)
        print(f"  Prompts saved → {prompt_path}")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2 — Evaluation (edge model only, frozen prompt)
    # ═══════════════════════════════════════════════════════════════════

    print("\n═══ Phase 2: Evaluation (Edge, Frozen) ═══")
    for p in edge_model.parameters():
        p.requires_grad_(False)

    trained_ctx = edge_model.prompt_learner.ctx.data.clone()

    test_sets = ['I', 'A', 'R', 'K', 'V'] + list(fewshot_datasets)
    results = {}

    for set_id in test_sets:
        classnames = get_classnames_for_set(set_id)
        edge_model.reset_classnames(classnames, args.arch)
        # reset_classnames reloads prefix/suffix but does NOT touch ctx;
        # overwrite explicitly to be safe against any side-effect.
        edge_model.prompt_learner.ctx.data.copy_(trained_ctx)

        test_ds = build_dataset(set_id, eval_transform, eval_transform,
                                data_root, mode='test', cloud_use_ori=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        t1, t5 = evaluate(edge_model, test_loader, args)
        results[set_id] = {'top1': t1, 'top5': t5}
        print(f"  {set_id:>12s}:  Top-1 {t1:6.2f}%   Top-5 {t5:6.2f}%")
        del test_ds, test_loader

    # ── summary ──
    print(f"\n{'Dataset':>12s}  {'Top-1':>8s}  {'Top-5':>8s}")
    print("─" * 34)
    for s in test_sets:
        print(f"{s:>12s}  {results[s]['top1']:>8.2f}  {results[s]['top5']:>8.2f}")

    result_file = os.path.join(save_dir, 'results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CEO Offline: train on ImageNet val, eval everywhere')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('-a', '--arch', default='RN50',
                        help='edge model architecture')
    parser.add_argument('-c', '--cloud_arch', default='ViT-B/16',
                        help='cloud model architecture')
    parser.add_argument('--n_shot', type=int, default=16,
                        help='shots per class on ImageNet val')
    parser.add_argument('--epochs', type=int, default=5,
                        help='offline training epochs')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate for edge prompt optimiser')
    parser.add_argument('--cloud_lr', type=float, default=None,
                        help='learning rate for cloud prompt (default: same as --lr)')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='use cosine annealing lr schedule')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training mini-batch size')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='evaluation mini-batch size')
    parser.add_argument('--tau', type=float, default=0.4,
                        help='temperature for KL divergence')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='weight of KL loss')
    parser.add_argument('--selection_p', type=float, default=1.0,
                        help='fraction of confident samples for entropy loss (1.0=all, CEO default=0.1)')
    parser.add_argument('--n_ctx', type=int, default=4,
                        help='number of learnable context tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a',
                        help='context initialisation words (use "none" for random init)')
    parser.add_argument('--resolution', type=int, default=224,
                        help='CLIP input resolution')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-j', '--workers', type=int, default=8,
                        help='data-loading workers')
    parser.add_argument('--rec_dir', default='new',
                        help='reconstructed-image subdirectory under imagenet/images/')
    parser.add_argument('--use_ori', action='store_true', default=False,
                        help='use original images for training instead of reconstructed')
    parser.add_argument('--save_dir', default=None,
                        help='directory for saved prompts & results')
    parser.add_argument('--load_prompt', default=None,
                        help='path to a pre-trained prompts.pt (skip training)')
    main()
