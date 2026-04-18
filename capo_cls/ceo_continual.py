import argparse
import time
import json
import os
from copy import deepcopy
from collections import deque

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip_device import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import accuracy, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets


# ── helpers ──────────────────────────────────────────────────────────────

def select_confident_samples(logits, top):
    batch_entropy = (-logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[
        :int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def compute_ewc_loss(param, ref, omega, ewc_lambda):
    """EWC penalty: (λ/2) Σ Ω_i (p_i − p*_i)²  (computed in fp32)."""
    diff = (param.float() - ref.float()).pow(2)
    return 0.5 * ewc_lambda * (omega * diff).sum()


# ── core TTA step (identical to ceo.py) ─────────────────────────────────

def test_time_tuning(cloud_model, edge_model, rec,
                     optimizer_cloud, optimizer_edge,
                     cloud_scaler, edge_scaler, args):
    selected_idx = None
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            cloud_feat = cloud_model.get_image_features(rec)
            output = cloud_model(None, cloud_feat)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args.selection_p)
            loss_cloud = avg_entropy(output)

        optimizer_cloud.zero_grad()
        cloud_scaler.scale(loss_cloud).backward()
        cloud_scaler.step(optimizer_cloud)
        cloud_scaler.update()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = cloud_model(None, cloud_feat)
            del cloud_feat
            edge_feat = edge_model.get_image_features(rec)
            output_device = edge_model(None, edge_feat)
            tau = args.tau
            loss_kl = args.beta * criterion_div(
                F.log_softmax(output_device / tau, dim=1),
                F.softmax(output / tau, dim=1))
            output_device, selected_idx = select_confident_samples(
                output_device, args.selection_p)
            loss_ent = avg_entropy(output_device)
            loss_edge = loss_ent + loss_kl

        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_edge).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()


# ── core TTA step with EWC regularisation ────────────────────────────────

def test_time_tuning_ewc(cloud_model, edge_model, rec,
                         optimizer_cloud, optimizer_edge,
                         cloud_scaler, edge_scaler,
                         ewc_state, args):
    """TTA step with EWC: teacher loss = ent + ewc, student loss = ent + kl + ewc.
    After the gradient step, Fisher (omega) and reference params are updated
    via a fresh forward‑backward pass on the task losses only.
    """
    selected_idx = None
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    for j in range(args.tta_steps):
        # ── Cloud (teacher): loss_ent + loss_ewc ──
        with torch.cuda.amp.autocast():
            cloud_feat = cloud_model.get_image_features(rec)
            output = cloud_model(None, cloud_feat)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args.selection_p)
            loss_ent = avg_entropy(output)
            loss_ewc_c = compute_ewc_loss(
                cloud_model.prompt_learner.ctx,
                ewc_state['cloud_ref'], ewc_state['cloud_omega'],
                args.ewc_lambda)
            loss_cloud = loss_ent + loss_ewc_c

        optimizer_cloud.zero_grad()
        cloud_scaler.scale(loss_cloud).backward()
        cloud_scaler.step(optimizer_cloud)
        cloud_scaler.update()

        # ── Edge (student): loss_ent + loss_kl + loss_ewc ──
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = cloud_model(None, cloud_feat)
            del cloud_feat
            edge_feat = edge_model.get_image_features(rec)
            output_device = edge_model(None, edge_feat)
            tau = args.tau
            loss_kl = args.beta * criterion_div(
                F.log_softmax(output_device / tau, dim=1),
                F.softmax(output / tau, dim=1))
            output_device, selected_idx = select_confident_samples(
                output_device, args.selection_p)
            loss_ent = avg_entropy(output_device)
            loss_ewc_e = compute_ewc_loss(
                edge_model.prompt_learner.ctx,
                ewc_state['edge_ref'], ewc_state['edge_omega'],
                args.ewc_lambda)
            loss_edge = loss_ent + loss_kl + loss_ewc_e

        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_edge).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()

    # ── Fisher update (separate forward‑backward at post‑step params) ──
    _update_ewc_fisher(cloud_model, edge_model, rec, ewc_state, args)


def _update_ewc_fisher(cloud_model, edge_model, rec, ewc_state, args):
    """Update Fisher information Ω via EMA of squared task‑loss gradients,
    computed at the *current* (post‑step) parameters."""
    gamma = args.ewc_gamma
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    # Cloud Fisher: Ω_t = γΩ_{t-1} + (1−γ)(∇_p loss_ent)²
    cloud_model.prompt_learner.ctx.grad = None
    with torch.cuda.amp.autocast():
        cloud_feat = cloud_model.get_image_features(rec)
        output = cloud_model(None, cloud_feat)
        output, _ = select_confident_samples(output, args.selection_p)
        loss_c = avg_entropy(output)
    loss_c.backward()
    with torch.no_grad():
        g = cloud_model.prompt_learner.ctx.grad.float()
        ewc_state['cloud_omega'].mul_(gamma).add_((1 - gamma) * g.pow(2))
        if args.ewc_anchor == 'sample':
            ewc_state['cloud_ref'] = cloud_model.prompt_learner.ctx.detach().clone()
    cloud_model.prompt_learner.ctx.grad = None

    # Edge Fisher: Ω_t = γΩ_{t-1} + (1−γ)(∇_p (loss_ent + loss_kl))²
    edge_model.prompt_learner.ctx.grad = None
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            cloud_out = cloud_model(None, cloud_feat)
        del cloud_feat
        edge_feat = edge_model.get_image_features(rec)
        output_device = edge_model(None, edge_feat)
        tau = args.tau
        loss_kl = args.beta * criterion_div(
            F.log_softmax(output_device / tau, dim=1),
            F.softmax(cloud_out / tau, dim=1))
        output_device_sel, _ = select_confident_samples(
            output_device, args.selection_p)
        loss_ent = avg_entropy(output_device_sel)
        loss_e = loss_ent + loss_kl
    loss_e.backward()
    with torch.no_grad():
        g = edge_model.prompt_learner.ctx.grad.float()
        ewc_state['edge_omega'].mul_(gamma).add_((1 - gamma) * g.pow(2))
        if args.ewc_anchor == 'sample':
            ewc_state['edge_ref'] = edge_model.prompt_learner.ctx.detach().clone()
    edge_model.prompt_learner.ctx.grad = None


# ── core TTA step with EWC on teacher only (both kept) ───────────────────

def test_time_tuning_teacher_ewc(cloud_model, edge_model, rec,
                                 optimizer_cloud, optimizer_edge,
                                 cloud_scaler, edge_scaler,
                                 ewc_state, args):
    """TTA step with EWC on teacher only.
    Teacher loss = loss_ent + loss_ewc.
    Student loss = loss_ent + loss_kl (no EWC).
    """
    selected_idx = None
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    for j in range(args.tta_steps):
        # ── Cloud (teacher): loss_ent + loss_ewc ──
        with torch.cuda.amp.autocast():
            cloud_feat = cloud_model.get_image_features(rec)
            output = cloud_model(None, cloud_feat)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args.selection_p)
            loss_ent = avg_entropy(output)
            loss_ewc_c = compute_ewc_loss(
                cloud_model.prompt_learner.ctx,
                ewc_state['cloud_ref'], ewc_state['cloud_omega'],
                args.ewc_lambda)
            loss_cloud = loss_ent + loss_ewc_c

        optimizer_cloud.zero_grad()
        cloud_scaler.scale(loss_cloud).backward()
        cloud_scaler.step(optimizer_cloud)
        cloud_scaler.update()

        # ── Edge (student): loss_ent + loss_kl (no EWC) ──
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = cloud_model(None, cloud_feat)
            del cloud_feat
            edge_feat = edge_model.get_image_features(rec)
            output_device = edge_model(None, edge_feat)
            tau = args.tau
            loss_kl = args.beta * criterion_div(
                F.log_softmax(output_device / tau, dim=1),
                F.softmax(output / tau, dim=1))
            output_device, selected_idx = select_confident_samples(
                output_device, args.selection_p)
            loss_ent = avg_entropy(output_device)
            loss_edge = loss_ent + loss_kl

        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_edge).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()

    # ── Fisher update for teacher only ──
    _update_teacher_ewc_fisher(cloud_model, rec, ewc_state, args)


def _update_teacher_ewc_fisher(cloud_model, rec, ewc_state, args):
    """Update Fisher Ω for teacher (cloud) only."""
    gamma = args.ewc_gamma

    cloud_model.prompt_learner.ctx.grad = None
    with torch.cuda.amp.autocast():
        cloud_feat = cloud_model.get_image_features(rec)
        output = cloud_model(None, cloud_feat)
        output, _ = select_confident_samples(output, args.selection_p)
        loss_c = avg_entropy(output)
    loss_c.backward()
    with torch.no_grad():
        g = cloud_model.prompt_learner.ctx.grad.float()
        ewc_state['cloud_omega'].mul_(gamma).add_((1 - gamma) * g.pow(2))
        if args.ewc_anchor == 'sample':
            ewc_state['cloud_ref'] = cloud_model.prompt_learner.ctx.detach().clone()
    cloud_model.prompt_learner.ctx.grad = None


# ── core TTA step with EWC on student only (teacher is reset) ────────────

def test_time_tuning_student_ewc(cloud_model, edge_model, rec,
                                 optimizer_cloud, optimizer_edge,
                                 cloud_scaler, edge_scaler,
                                 ewc_state, args):
    """TTA step with EWC on student only.
    Teacher loss = loss_ent (no EWC, since teacher is reset every sample).
    Student loss = loss_ent + loss_kl + loss_ewc.
    """
    selected_idx = None
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    for j in range(args.tta_steps):
        # ── Cloud (teacher): loss_ent only ──
        with torch.cuda.amp.autocast():
            cloud_feat = cloud_model.get_image_features(rec)
            output = cloud_model(None, cloud_feat)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args.selection_p)
            loss_cloud = avg_entropy(output)

        optimizer_cloud.zero_grad()
        cloud_scaler.scale(loss_cloud).backward()
        cloud_scaler.step(optimizer_cloud)
        cloud_scaler.update()

        # ── Edge (student): loss_ent + loss_kl + loss_ewc ──
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = cloud_model(None, cloud_feat)
            del cloud_feat
            edge_feat = edge_model.get_image_features(rec)
            output_device = edge_model(None, edge_feat)
            tau = args.tau
            loss_kl = args.beta * criterion_div(
                F.log_softmax(output_device / tau, dim=1),
                F.softmax(output / tau, dim=1))
            output_device, selected_idx = select_confident_samples(
                output_device, args.selection_p)
            loss_ent = avg_entropy(output_device)
            loss_ewc_e = compute_ewc_loss(
                edge_model.prompt_learner.ctx,
                ewc_state['edge_ref'], ewc_state['edge_omega'],
                args.ewc_lambda)
            loss_edge = loss_ent + loss_kl + loss_ewc_e

        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_edge).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()

    # ── Fisher update for student only ──
    _update_student_ewc_fisher(cloud_model, edge_model, rec, ewc_state, args)


def _update_student_ewc_fisher(cloud_model, edge_model, rec, ewc_state, args):
    """Update Fisher Ω for student (edge) only — teacher is reset so no need."""
    gamma = args.ewc_gamma
    criterion_div = nn.KLDivLoss(reduction='batchmean')

    edge_model.prompt_learner.ctx.grad = None
    with torch.cuda.amp.autocast():
        cloud_feat = cloud_model.get_image_features(rec)
        with torch.no_grad():
            cloud_out = cloud_model(None, cloud_feat)
        del cloud_feat
        edge_feat = edge_model.get_image_features(rec)
        output_device = edge_model(None, edge_feat)
        tau = args.tau
        loss_kl = args.beta * criterion_div(
            F.log_softmax(output_device / tau, dim=1),
            F.softmax(cloud_out / tau, dim=1))
        output_device_sel, _ = select_confident_samples(
            output_device, args.selection_p)
        loss_ent = avg_entropy(output_device_sel)
        loss_e = loss_ent + loss_kl
    loss_e.backward()
    with torch.no_grad():
        g = edge_model.prompt_learner.ctx.grad.float()
        ewc_state['edge_omega'].mul_(gamma).add_((1 - gamma) * g.pow(2))
        if args.ewc_anchor == 'sample':
            ewc_state['edge_ref'] = edge_model.prompt_learner.ctx.detach().clone()
    edge_model.prompt_learner.ctx.grad = None


# ── data ─────────────────────────────────────────────────────────────────

def build_domain_loaders(args):
    """Build fixed-size subsets for ImageNet-V and ImageNet-K."""
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    domain_loaders = {}

    for set_id in ['V', 'K']:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        data_transform = AugMixAugmenter(
            base_transform, preprocess,
            n_views=args.batch_size - 1,
            augmix=len(set_id) > 1)
        device_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(), normalize])

        val_dataset = build_dataset(set_id, device_transform, data_transform,
                                    args.data, mode=args.dataset_mode)

        if hasattr(val_dataset, 'classes'):
            try:
                lex_to_numeric = [int(c) for c in val_dataset.classes]
                val_dataset.target_transform = \
                    lambda t, _m=lex_to_numeric: _m[t]
            except (ValueError, TypeError):
                pass

        total = len(val_dataset)
        n = min(args.samples_per_domain, total)
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(total)[:n].tolist()
        subset = torch.utils.data.Subset(val_dataset, indices)

        loader = torch.utils.data.DataLoader(
            subset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        domain_loaders[set_id] = loader
        print(f"  Domain {set_id}: {n}/{total} samples (seed={args.seed})")

    return domain_loaders


# ── evaluate (no adaptation) ─────────────────────────────────────────────

def evaluate_domain(edge_model, loader, gpu, cloud_model=None):
    """Pure inference — no gradient, no adaptation."""
    corrects = []
    cloud_corrects = [] if cloud_model is not None else None
    with torch.no_grad():
        for oris, recs, target in loader:
            if isinstance(oris, list):
                ori = oris[0].cuda(gpu, non_blocking=True)
            else:
                if len(oris.size()) > 4:
                    oris = oris.squeeze(0)
                ori = oris.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            feat = edge_model.get_image_features(ori)
            output = edge_model(None, feat)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            corrects.append(int(acc1[0].item() > 0))

            if cloud_model is not None:
                cfeat = cloud_model.get_image_features(ori)
                cout = cloud_model(None, cfeat)
                cacc1, _ = accuracy(cout, target, topk=(1, 5))
                cloud_corrects.append(int(cacc1[0].item() > 0))

    edge_acc = float(np.mean(corrects) * 100)
    if cloud_model is not None:
        cloud_acc = float(np.mean(cloud_corrects) * 100)
        return edge_acc, cloud_acc
    return edge_acc


# ── adapt (online TTA with uplink gating) ────────────────────────────────

def adapt_domain(cloud_model, edge_model, loader,
                 optimizer_cloud, optimizer_edge,
                 cloud_init_ctx, edge_init_ctx,
                 cloud_init_optim, edge_init_optim,
                 cloud_scaler, edge_scaler,
                 all_entropy, args, ewc_state=None):
    """Online per-sample TTA with entropy-based uplink gating.

    *all_entropy* is mutated in-place (deque persists across stages).
    Returns (accuracy, uplink_ratio%).
    """
    corrects = []
    cloud_corrects = []
    n_uplink = 0

    for i, (oris, recs, target) in enumerate(loader):
        # ── prepare tensors ──
        if isinstance(oris, list):
            ori = oris[0].cuda(args.gpu, non_blocking=True)
        else:
            if len(oris.size()) > 4:
                oris = oris.squeeze(0)
            ori = oris.cuda(args.gpu, non_blocking=True)

        if isinstance(recs, list):
            for k in range(len(recs)):
                recs[k] = recs[k].cuda(args.gpu, non_blocking=True)
        else:
            if len(recs.size()) > 4:
                recs = recs.squeeze(0)
            recs = recs.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        recs = torch.cat(recs, dim=0)

        # ── uplink gating (following ceo.py) ──
        if args.uplink_percent >= 100.:
            is_uplink = True
        else:
            with torch.no_grad():
                ori_feat = edge_model.get_image_features(ori)
                output = edge_model(None, ori_feat)
            sample_ent = float(
                (-output.softmax(1) * output.log_softmax(1)).sum(1).item())
            all_entropy.append(sample_ent)
            threshold = float(
                np.percentile(all_entropy, 100. - args.uplink_percent))
            is_uplink = sample_ent > threshold

        if is_uplink:
            n_uplink += 1

            # save pre-step ctx for EMA
            if args.method == 'ema':
                cloud_pre = cloud_model.prompt_learner.ctx.detach().clone()
                edge_pre  = edge_model.prompt_learner.ctx.detach().clone()
            elif args.method == 'keep_student_ema':
                edge_pre = edge_model.prompt_learner.ctx.detach().clone()

            # ── reset policy ──
            if args.method == 'reset':
                with torch.no_grad():
                    cloud_model.prompt_learner.ctx.data.copy_(cloud_init_ctx)
                    edge_model.prompt_learner.ctx.data.copy_(edge_init_ctx)
                optimizer_cloud.load_state_dict(deepcopy(cloud_init_optim))
                optimizer_edge.load_state_dict(deepcopy(edge_init_optim))
            elif args.method in ('keep_student',
                                 'keep_student_ema',
                                 'keep_student_ewc'):
                with torch.no_grad():
                    cloud_model.prompt_learner.ctx.data.copy_(cloud_init_ctx)
                optimizer_cloud.load_state_dict(deepcopy(cloud_init_optim))
            # keep_both / ema / keep_both_ewc / keep_both_teacher_ewc: no resets

            # ── TTA step ──
            if args.method == 'keep_both_ewc':
                test_time_tuning_ewc(
                    cloud_model, edge_model, recs,
                    optimizer_cloud, optimizer_edge,
                    cloud_scaler, edge_scaler, ewc_state, args)
            elif args.method == 'keep_both_teacher_ewc':
                test_time_tuning_teacher_ewc(
                    cloud_model, edge_model, recs,
                    optimizer_cloud, optimizer_edge,
                    cloud_scaler, edge_scaler, ewc_state, args)
            elif args.method == 'keep_student_ewc':
                test_time_tuning_student_ewc(
                    cloud_model, edge_model, recs,
                    optimizer_cloud, optimizer_edge,
                    cloud_scaler, edge_scaler, ewc_state, args)
            else:
                test_time_tuning(
                    cloud_model, edge_model, recs,
                    optimizer_cloud, optimizer_edge,
                    cloud_scaler, edge_scaler, args)

            # ── EMA blending ──
            if args.method == 'ema':
                d = args.ema_decay
                with torch.no_grad():
                    cloud_model.prompt_learner.ctx.data.copy_(
                        d * cloud_pre +
                        (1 - d) * cloud_model.prompt_learner.ctx.data)
                    edge_model.prompt_learner.ctx.data.copy_(
                        d * edge_pre +
                        (1 - d) * edge_model.prompt_learner.ctx.data)
            elif args.method == 'keep_student_ema':
                d = args.ema_decay
                with torch.no_grad():
                    edge_model.prompt_learner.ctx.data.copy_(
                        d * edge_pre +
                        (1 - d) * edge_model.prompt_learner.ctx.data)

            # ── post-adapt evaluation ──
            with torch.no_grad():
                ori_feat = edge_model.get_image_features(ori)
                with torch.cuda.amp.autocast():
                    output = edge_model(None, ori_feat)
                # teacher accuracy (post-adapt)
                cloud_feat = cloud_model.get_image_features(ori)
                with torch.cuda.amp.autocast():
                    cloud_output = cloud_model(None, cloud_feat)
                cacc1, _ = accuracy(cloud_output, target, topk=(1, 5))
                cloud_corrects.append(int(cacc1[0].item() > 0))

        else:
            # not uplinked — still track teacher accuracy
            with torch.no_grad():
                cloud_feat = cloud_model.get_image_features(ori)
                with torch.cuda.amp.autocast():
                    cloud_output = cloud_model(None, cloud_feat)
                cacc1, _ = accuracy(cloud_output, target, topk=(1, 5))
                cloud_corrects.append(int(cacc1[0].item() > 0))

        # 'output' = pre-adapt (not uplinked) | post-adapt (uplinked)
        acc1, _ = accuracy(output, target, topk=(1, 5))
        corrects.append(int(acc1[0].item() > 0))

        if (i + 1) % args.print_freq == 0:
            up_pct = n_uplink / (i + 1) * 100
            cloud_pct = np.mean(cloud_corrects) * 100
            print(f"    [{i+1}/{len(loader.dataset)}]  "
                  f"S_acc={np.mean(corrects)*100:.1f}%  "
                  f"T_acc={cloud_pct:.1f}%  "
                  f"uplink={up_pct:.0f}%  "
                  f"queue={len(all_entropy)}")

    up_ratio = n_uplink / len(corrects) * 100 if corrects else 0.
    cloud_acc = float(np.mean(cloud_corrects) * 100) if cloud_corrects else 0.
    return float(np.mean(corrects) * 100), float(up_ratio), cloud_acc


# ── tag / label helpers ──────────────────────────────────────────────────

def _hp_suffix(args):
    """Append training hyper-params that vary across grid searches."""
    parts = [f't{args.tau}', f'b{args.beta}', f'up{args.uplink_percent}']
    if args.selection_p != 0.1:
        parts.append(f'sp{args.selection_p}')
    if args.tta_steps != 1:
        parts.append(f's{args.tta_steps}')
    return '_'.join(parts)


def make_method_tag(args):
    if args.no_adapt:
        return 'no_adapt'
    if args.method == 'reset':
        return f'reset_{_hp_suffix(args)}'
    q = 'RQ' if args.reset_queue else 'AQ'
    hp = _hp_suffix(args)
    if args.method == 'ema':
        return f'ema_d{args.ema_decay}_{hp}_{q}'
    if args.method == 'keep_student_ema':
        return f'ks_ema_d{args.ema_decay}_{hp}_{q}'
    if args.method == 'keep_both_ewc':
        anchor = 'da' if args.ewc_anchor == 'domain' else 'sa'
        return f'kb_ewc_{anchor}_l{args.ewc_lambda}_g{args.ewc_gamma}_{hp}_{q}'
    if args.method == 'keep_both_teacher_ewc':
        anchor = 'da' if args.ewc_anchor == 'domain' else 'sa'
        return f'kb_tewc_{anchor}_l{args.ewc_lambda}_g{args.ewc_gamma}_{hp}_{q}'
    if args.method == 'keep_student_ewc':
        anchor = 'da' if args.ewc_anchor == 'domain' else 'sa'
        return f'ks_ewc_{anchor}_l{args.ewc_lambda}_g{args.ewc_gamma}_{hp}_{q}'
    return f'{args.method}_{hp}_{q}'


VARIANT_NAME = {
    'no_adapt':          'No Adapt',
    'reset':             'CEO Default',
    'keep_student':      {'AQ': 'V1.1', 'RQ': 'V1.2'},
    'keep_both':         {'AQ': 'V2.1', 'RQ': 'V2.2'},
    'ema':               {'AQ': 'V3.1', 'RQ': 'V3.2'},
    'keep_both_ewc':          {'AQ': 'V4.1', 'RQ': 'V4.2'},
    'keep_both_teacher_ewc':  {'AQ': 'V4.3', 'RQ': 'V4.4'},
    'keep_student_ema':       {'AQ': 'V5',   'RQ': 'V5-RQ'},
    'keep_student_ewc':       {'AQ': 'V6',   'RQ': 'V6-RQ'},
}


def make_method_label(args):
    if args.no_adapt:
        return 'No Adaptation (CLIP RN50)'
    m = args.method
    q_tag = 'RQ' if args.reset_queue else 'AQ'
    q_lbl = 'reset queue' if args.reset_queue else 'accum queue'
    desc = {
        'reset':            'reset T+S prompt & opt',
        'keep_student':     'keep S, reset T',
        'keep_both':        'keep T+S',
        'ema':              f'keep T+S EMA(d={args.ema_decay})',
        'keep_both_ewc':         f'keep T+S + EWC(λ={args.ewc_lambda},γ={args.ewc_gamma},{args.ewc_anchor})',
        'keep_both_teacher_ewc': f'keep T(EWC)+S, EWC(λ={args.ewc_lambda},γ={args.ewc_gamma},{args.ewc_anchor}) on T only',
        'keep_student_ema':      f'keep S EMA(d={args.ema_decay}), reset T',
        'keep_student_ewc':      f'keep S + EWC(λ={args.ewc_lambda},γ={args.ewc_gamma},{args.ewc_anchor}), reset T',
    }[m]
    vname = VARIANT_NAME[m]
    if isinstance(vname, dict):
        vname = vname[q_tag]
    return f'{vname}: {desc}, {q_lbl}'


# ── main ─────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    method_tag  = make_method_tag(args)
    method_label = make_method_label(args)

    print(f"\n{'='*70}")
    print(f"  Method       : {method_tag}")
    print(f"  Description  : {method_label}")
    print(f"  Stream       : V/K alternating x {args.n_rounds} rounds")
    print(f"  Samples      : {args.samples_per_domain}/domain  "
          f"(total {args.samples_per_domain * 2 * args.n_rounds})")
    print(f"  uplink       : {args.uplink_percent}%   "
          f"queue_reset={args.reset_queue}")
    print(f"  lr={args.lr}  tau={args.tau}  beta={args.beta}  "
          f"tta_steps={args.tta_steps}")
    print(f"{'='*70}")

    # ── build models ──
    classnames = imagenet_classes
    cloud_model = get_coop(args.cloud_arch, 'K', args.gpu,
                           args.n_ctx, args.ctx_init)
    edge_model  = get_coop(args.arch, 'K', args.gpu,
                           args.n_ctx, args.ctx_init, type='Device')

    for name, param in cloud_model.named_parameters():
        param.requires_grad_("prompt_learner" in name)
    for name, param in edge_model.named_parameters():
        param.requires_grad_("prompt_learner" in name)

    cloud_trainable = [p for p in cloud_model.parameters() if p.requires_grad]
    edge_trainable  = [p for p in edge_model.parameters()  if p.requires_grad]

    cloud_model = cloud_model.cuda(args.gpu)
    edge_model  = edge_model.cuda(args.gpu)
    cloud_model.reset_classnames(classnames, args.cloud_arch)
    edge_model.reset_classnames(classnames, args.arch)

    optimizer_cloud = optim.AdamW(cloud_trainable, args.lr)
    optimizer_edge  = optim.AdamW(edge_trainable,  args.lr)

    cloud_init_ctx   = cloud_model.prompt_learner.ctx.detach().clone()
    edge_init_ctx    = edge_model.prompt_learner.ctx.detach().clone()
    cloud_init_optim = deepcopy(optimizer_cloud.state_dict())
    edge_init_optim  = deepcopy(optimizer_edge.state_dict())

    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    edge_scaler  = torch.cuda.amp.GradScaler(init_scale=1000)
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.benchmark = True

    cloud_model.eval()
    edge_model.eval()

    # ── build data ──
    domain_loaders = build_domain_loaders(args)
    domains = ['V', 'K']

    # ── entropy queue (shared across all stages) ──
    all_entropy = deque(maxlen=500)

    # ── EWC state (shared across all stages) ──
    ewc_state = None
    if args.method in ('keep_both_ewc', 'keep_student_ewc',
                       'keep_both_teacher_ewc'):
        ewc_state = {
            'cloud_omega': torch.zeros_like(
                cloud_model.prompt_learner.ctx.data, dtype=torch.float32),
            'cloud_ref': cloud_model.prompt_learner.ctx.detach().clone(),
        }
        if args.method in ('keep_both_ewc', 'keep_student_ewc'):
            ewc_state['edge_omega'] = torch.zeros_like(
                edge_model.prompt_learner.ctx.data, dtype=torch.float32)
            ewc_state['edge_ref'] = edge_model.prompt_learner.ctx.detach().clone()

    # ── no-adapt baseline (evaluate initial model on both domains) ──
    baseline = {}
    print("\n--- No-Adapt Baseline (initial model, before any adaptation) ---")
    for d in domains:
        bl_res = evaluate_domain(edge_model, domain_loaders[d], args.gpu,
                                 cloud_model=cloud_model)
        s_bl, t_bl = bl_res
        baseline[d] = {'student': s_bl, 'teacher': t_bl}
        print(f"  {d}: S_base={s_bl:.2f}%  T_base={t_bl:.2f}%")

    # ── run stream ──
    results = {d: [] for d in domains}
    t_start = time.time()

    for rnd in range(1, args.n_rounds + 1):
        for d in domains:
            visit  = len(results[d]) + 1
            loader = domain_loaders[d]

            # queue reset at domain switch (skip for the very first stage)
            is_first_stage = (rnd == 1 and d == domains[0])
            if args.reset_queue and not is_first_stage:
                all_entropy.clear()

            print(f"\n--- Round {rnd}, Domain {d}, Visit #{visit}  "
                  f"(queue={len(all_entropy)}) ---")

            # 1) pre-adapt accuracy
            pre_result = evaluate_domain(edge_model, loader, args.gpu,
                                         cloud_model=cloud_model)
            pre_acc, pre_cloud_acc = pre_result
            print(f"  S_acc'_{d}({visit}) = {pre_acc:.2f}%  "
                  f"T_acc'_{d}({visit}) = {pre_cloud_acc:.2f}%  (pre-adapt)")

            # 2) online adaptation
            if args.no_adapt:
                post_acc = pre_acc
                up_ratio = 0.
                cloud_acc = pre_cloud_acc
            else:
                post_acc, up_ratio, cloud_acc = adapt_domain(
                    cloud_model, edge_model, loader,
                    optimizer_cloud, optimizer_edge,
                    cloud_init_ctx, edge_init_ctx,
                    cloud_init_optim, edge_init_optim,
                    cloud_scaler, edge_scaler,
                    all_entropy, args, ewc_state=ewc_state)

            print(f"  S_acc_{d}({visit})  = {post_acc:.2f}%  "
                  f"T_acc_{d}({visit})  = {cloud_acc:.2f}%  "
                  f"(post-adapt, uplink={up_ratio:.0f}%)")

            # update EWC anchors at domain boundary
            if ewc_state is not None and args.ewc_anchor == 'domain':
                ewc_state['cloud_ref'] = \
                    cloud_model.prompt_learner.ctx.detach().clone()
                if 'edge_ref' in ewc_state:
                    ewc_state['edge_ref'] = \
                        edge_model.prompt_learner.ctx.detach().clone()

            results[d].append({
                'visit': visit, 'round': rnd,
                'pre_acc': pre_acc, 'post_acc': post_acc,
                'pre_cloud_acc': pre_cloud_acc,
                'cloud_acc': cloud_acc,
                'uplink_ratio': up_ratio,
            })

    elapsed = time.time() - t_start

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  {method_tag} — {method_label}   ({elapsed:.0f}s)")
    print(f"{'='*70}")

    hdr = f"{'Visit':>6}"
    for d in domains:
        hdr += f"  | S'_{d:>2}   S_{d:>2}   T'_{d:>2}   T_{d:>2}  up%"
    print(hdr)
    print('-' * 100)

    # baseline row
    row = f"  {'Base':>4}"
    for d in domains:
        sb = baseline[d]['student']
        tb = baseline[d]['teacher']
        row += f"  | {sb:6.2f}    —    {tb:6.2f}    —     —"
    print(row)
    print('-' * 100)

    for t in range(args.n_rounds):
        row = f"  {t+1:>4}"
        for d in domains:
            v = results[d][t]
            row += (f"  | {v['pre_acc']:6.2f}  {v['post_acc']:6.2f}"
                    f"  {v['pre_cloud_acc']:6.2f}  {v['cloud_acc']:6.2f}"
                    f"  {v['uplink_ratio']:3.0f}")
        print(row)

    print('-' * 100)
    row = f"  {'Mean':>4}"
    mean_post = {}
    for d in domains:
        mp = np.mean([v['pre_acc']       for v in results[d]])
        ma = np.mean([v['post_acc']      for v in results[d]])
        mcp = np.mean([v['pre_cloud_acc'] for v in results[d]])
        mca = np.mean([v['cloud_acc']    for v in results[d]])
        mu = np.mean([v['uplink_ratio']  for v in results[d]])
        mean_post[d] = ma
        row += f"  | {mp:6.2f}  {ma:6.2f}  {mcp:6.2f}  {mca:6.2f}  {mu:3.0f}"
    print(row)

    # ── Gain over no-adapt baseline ──
    print(f"\n--- Gain over No-Adapt Baseline ---")
    for d in domains:
        sb = baseline[d]['student']
        ma = mean_post[d]
        gain = ma - sb
        print(f"  S_{d}: mean_post={ma:.2f} - base={sb:.2f} = {gain:+.2f}")

    # ── ΔAcc & BT ──
    print(f"\n--- ΔAcc & Backward Transfer (Student) ---")
    deltas = []
    for d in domains:
        first = results[d][0]['post_acc']
        last  = results[d][-1]['post_acc']
        delta = last - first
        deltas.append(delta)
        print(f"  ΔS_{d} = {last:.2f} - {first:.2f} = {delta:+.2f}")
    bt = float(np.mean(deltas))
    print(f"  S_BT = {bt:+.2f}")

    print(f"\n--- ΔAcc & Backward Transfer (Teacher) ---")
    deltas_cloud = []
    for d in domains:
        first = results[d][0]['cloud_acc']
        last  = results[d][-1]['cloud_acc']
        delta = last - first
        deltas_cloud.append(delta)
        print(f"  ΔT_{d} = {last:.2f} - {first:.2f} = {delta:+.2f}")
    bt_cloud = float(np.mean(deltas_cloud))
    print(f"  T_BT = {bt_cloud:+.2f}")
    print(f"{'='*70}")

    # ── save ──
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {
        'method': method_tag,
        'label': method_label,
        'elapsed_s': elapsed,
        'uplink_percent': args.uplink_percent,
        'reset_queue': args.reset_queue,
        'baseline': baseline,
        'results': results,
        'BT': bt,
    }
    for d in domains:
        metrics[f'mean_pre_acc_{d}']  = float(
            np.mean([v['pre_acc']  for v in results[d]]))
        metrics[f'mean_post_acc_{d}'] = float(
            np.mean([v['post_acc'] for v in results[d]]))
        metrics[f'delta_acc_{d}'] = float(
            results[d][-1]['post_acc'] - results[d][0]['post_acc'])

    path = os.path.join(args.output_dir, f'results_{method_tag}.json')
    with open(path, 'w') as f:
        json.dump({'metrics': metrics, 'args': vars(args)}, f, indent=2)
    print(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Continual TTA: V/K Alternating Stream')

    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--samples_per_domain', type=int, default=500)
    parser.add_argument('--n_rounds', type=int, default=10)
    parser.add_argument('--dataset_mode', type=str, default='test')

    parser.add_argument('-a', '--arch', default='RN50')
    parser.add_argument('-c', '--cloud_arch', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('-p', '--print-freq', default=100, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--selection_p', default=0.1, type=float)
    parser.add_argument('--tta_steps', default=1, type=int)
    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.9)

    parser.add_argument('--uplink_percent', type=float, default=50.,
                        help='percentage of high-entropy samples to uplink '
                             '(100 = always uplink)')
    parser.add_argument('--output_dir', type=str, default='logs/continual_vk')

    parser.add_argument('--no_adapt', action='store_true',
                        help='no-adapt baseline: CLIP RN50 inference only')
    parser.add_argument('--method', type=str, default='reset',
                        choices=['reset', 'keep_student', 'keep_both',
                                 'ema', 'keep_both_ewc',
                                 'keep_both_teacher_ewc',
                                 'keep_student_ema', 'keep_student_ewc'],
                        help='reset=CEO default; keep_student=V1; '
                             'keep_both=V2; ema=V3; keep_both_ewc=V4; '
                             'keep_both_teacher_ewc=V4.3; '
                             'keep_student_ema=V5; keep_student_ewc=V6')
    parser.add_argument('--reset_queue', action='store_true',
                        help='clear entropy queue at each domain switch '
                             '(default: accumulate)')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--ewc_lambda', type=float, default=100.,
                        help='EWC regularisation strength λ')
    parser.add_argument('--ewc_gamma', type=float, default=0.99,
                        help='EWC Fisher EMA decay γ')
    parser.add_argument('--ewc_anchor', type=str, default='sample',
                        choices=['sample', 'domain'],
                        help='EWC ref anchor: sample=update every sample; '
                             'domain=update at domain boundaries')
    parser.add_argument('--deterministic', action='store_true',
                        help='disable cudnn.benchmark for reproducibility')

    main()
