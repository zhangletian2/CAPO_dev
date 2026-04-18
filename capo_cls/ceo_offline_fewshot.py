#!/usr/bin/env python
"""
CEO Offline (Per-Dataset): Train separate prompts on each fewshot dataset's
val split (few-shot, no labels), then evaluate each frozen prompt on its
corresponding test split.

Training (per dataset):
    Both cloud (ViT-B/16) and edge (RN50) receive reconstructed images.
    Loss = batch entropy (cloud) + batch entropy (edge) + beta * KL(edge || cloud)
    Only prompt_learner parameters are updated; CLIP encoders stay frozen.

Evaluation (per dataset):
    Freeze everything (including prompts).
    Edge model infers on original images from the test split.

Usage:
    python ceo_offline_fewshot.py ./dataset --n_shot 16 --epochs 5 --gpu 0
    python ceo_offline_fewshot.py ./dataset --datasets dtd,flower102 --gpu 0
"""

import argparse
import json
import os
import time

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
from data.datautils import ID_to_DIRNAME
from data.fewshot_datasets import fewshot_datasets, build_fewshot_dataset
from data.cls_to_names import *
from utils.tools import Summary, AverageMeter, accuracy, set_random_seed


# ── losses ────────────────────────────────────────────────────────────

def batch_entropy(logits):
    """Per-sample entropy, averaged over the batch."""
    return (-logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()


def select_confident_samples(logits, top):
    """Keep the top-`top` fraction of samples with lowest entropy."""
    ent = (-logits.softmax(1) * logits.log_softmax(1)).sum(1)
    k = max(1, int(ent.size(0) * top))
    idx = torch.argsort(ent)[:k]
    return logits[idx], idx


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

    for epoch in range(args.epochs):
        ep_c, ep_e, ep_kl, n = 0.0, 0.0, 0.0, 0
        t0 = time.time()
        for _oris, recs, _ in train_loader:
            recs = recs.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast():
                c_feat = cloud_model.get_image_features(recs)
                c_out = cloud_model(None, c_feat)
                c_out_sel, _ = select_confident_samples(c_out, args.selection_p)
                loss_c = batch_entropy(c_out_sel)
            optimizer_cloud.zero_grad()
            cloud_scaler.scale(loss_c).backward()
            cloud_scaler.step(optimizer_cloud)
            cloud_scaler.update()
            if sched_c is not None:
                sched_c.step()

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    c_out_det = cloud_model(None, c_feat)
                e_feat = edge_model.get_image_features(recs)
                e_out = edge_model(None, e_feat)
                loss_kl = args.beta * criterion_kl(
                    F.log_softmax(e_out / args.tau, dim=1),
                    F.softmax(c_out_det / args.tau, dim=1))
                e_out_sel, _ = select_confident_samples(e_out, args.selection_p)
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
        print(f"    Epoch {epoch + 1}/{args.epochs}  "
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


# ── helpers ───────────────────────────────────────────────────────────

def get_classnames(set_id):
    return eval("{}_classes".format(set_id.lower()))


def _collect_prompt_params(model):
    """Return list of prompt_learner parameters (for optimizer creation)."""
    params = []
    for name, p in model.named_parameters():
        if 'prompt_learner' in name:
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    return params


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
    cloud_lr = args.cloud_lr if args.cloud_lr is not None else args.lr
    args._cloud_lr = cloud_lr

    # resolve dataset list
    if args.datasets:
        target_sets = [d.strip() for d in args.datasets.split(',')]
        for d in target_sets:
            assert d in fewshot_datasets, f"Unknown fewshot dataset: {d}"
    else:
        target_sets = list(fewshot_datasets)

    # output directory
    tag = (f'offline_fewshot_{args.n_shot}shot_ep{args.epochs}'
           f'_tau{args.tau}_beta{args.beta}'
           f'_lr{args.lr}')
    if cloud_lr != args.lr:
        tag += f'_clr{cloud_lr}'
    if args.selection_p < 1.0:
        tag += f'_sp{args.selection_p}'
    if args.cosine:
        tag += '_cos'
    if args.train_use_ori:
        tag += '_ori'
    tag += f'_seed{args.seed}'
    save_dir = args.save_dir or os.path.join('checkpoints', tag)
    os.makedirs(save_dir, exist_ok=True)

    # ── create models once (using first dataset's classnames) ──
    first_set = target_sets[0]
    print("Creating models …")
    cloud_model = get_coop(args.cloud_arch, first_set, args.gpu,
                           args.n_ctx, args.ctx_init)
    edge_model  = get_coop(args.arch, first_set, args.gpu,
                           args.n_ctx, args.ctx_init, type='Device')

    cloud_model = cloud_model.cuda(args.gpu)
    edge_model  = edge_model.cuda(args.gpu)

    # save initial ctx for resetting before each dataset
    cloud_ctx_init = cloud_model.prompt_learner.ctx.data.clone()
    edge_ctx_init  = edge_model.prompt_learner.ctx.data.clone()

    results = {}

    for set_id in target_sets:
        print(f"\n{'═' * 60}")
        print(f"  Dataset: {set_id}")
        print(f"{'═' * 60}")

        try:
            classnames = get_classnames(set_id)
            ds_save_dir = os.path.join(save_dir, set_id)
            os.makedirs(ds_save_dir, exist_ok=True)
            prompt_path = os.path.join(ds_save_dir, 'prompts.pt')

            # ── switch classnames ──
            cloud_model.reset_classnames(classnames, args.cloud_arch)
            edge_model.reset_classnames(classnames, args.arch)

            ds_root = os.path.join(data_root, ID_to_DIRNAME[set_id])

            # ── Phase 1: Training ──
            if args.load_prompt and os.path.isfile(
                    os.path.join(args.load_prompt, set_id, 'prompts.pt')):
                ckpt_path = os.path.join(args.load_prompt, set_id, 'prompts.pt')
                ckpt = torch.load(ckpt_path, map_location=f'cuda:{args.gpu}')
                edge_model.prompt_learner.ctx.data.copy_(ckpt['edge_ctx'])
                cloud_model.prompt_learner.ctx.data.copy_(ckpt['cloud_ctx'])
                print(f"  Loaded prompts from {ckpt_path}")
            else:
                # reset ctx to initial state
                cloud_model.prompt_learner.ctx.data.copy_(cloud_ctx_init)
                edge_model.prompt_learner.ctx.data.copy_(edge_ctx_init)

                cloud_trainable = _collect_prompt_params(cloud_model)
                edge_trainable  = _collect_prompt_params(edge_model)

                # build training set from val split, few-shot sampling
                train_ds = build_fewshot_dataset(
                    set_id, ds_root, eval_transform, eval_transform,
                    mode='val', n_shot=args.n_shot,
                    cloud_use_ori=args.train_use_ori, rec_dir=args.rec_dir)
                print(f"  Train: {len(train_ds)} images "
                      f"({args.n_shot}-shot from val split, {len(classnames)} classes)")

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

                torch.save({
                    'cloud_ctx': cloud_model.prompt_learner.ctx.detach().cpu(),
                    'edge_ctx':  edge_model.prompt_learner.ctx.detach().cpu(),
                    'set_id': set_id,
                    'args': vars(args),
                }, prompt_path)
                print(f"  Prompts saved -> {prompt_path}")
                del train_ds, train_loader

            # ── Phase 2: Evaluation (frozen prompt on test split) ──
            for p in edge_model.parameters():
                p.requires_grad_(False)

            trained_ctx = edge_model.prompt_learner.ctx.data.clone()
            edge_model.prompt_learner.ctx.data.copy_(trained_ctx)

            test_ds = build_fewshot_dataset(
                set_id, ds_root, eval_transform, eval_transform,
                mode='test', cloud_use_ori=True, rec_dir=args.rec_dir)
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=args.test_batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            t1, t5 = evaluate(edge_model, test_loader, args)
            results[set_id] = {'top1': t1, 'top5': t5}
            print(f"  Test:  Top-1 {t1:6.2f}%   Top-5 {t5:6.2f}%")
            del test_ds, test_loader

        except Exception as e:
            print(f"  SKIPPED {set_id}: {e}")
            results[set_id] = {'top1': -1, 'top5': -1, 'error': str(e)}

    # ── summary ──────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  Summary ({args.n_shot}-shot, {args.epochs} epochs)")
    print(f"{'═' * 50}")
    print(f"\n{'Dataset':>12s}  {'Top-1':>8s}  {'Top-5':>8s}")
    print("─" * 34)
    t1_vals = []
    for s in target_sets:
        t1 = results[s]['top1']
        if t1 < 0:
            print(f"{s:>12s}  {'SKIP':>8s}  {'SKIP':>8s}")
        else:
            t1_vals.append(t1)
            print(f"{s:>12s}  {t1:>8.2f}  {results[s]['top5']:>8.2f}")
    if t1_vals:
        avg_t1 = sum(t1_vals) / len(t1_vals)
        print("─" * 34)
        print(f"{'Average':>12s}  {avg_t1:>8.2f}  ({len(t1_vals)} datasets)")

    result_file = os.path.join(save_dir, 'results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CEO Offline Per-Dataset: train on each fewshot val split, '
                    'eval on each test split')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('-a', '--arch', default='RN50',
                        help='edge model architecture')
    parser.add_argument('-c', '--cloud_arch', default='ViT-B/16',
                        help='cloud model architecture')
    parser.add_argument('--datasets', default=None,
                        help='comma-separated fewshot dataset names '
                             '(default: all 10)')
    parser.add_argument('--n_shot', type=int, default=16,
                        help='shots per class from val split')
    parser.add_argument('--epochs', type=int, default=5,
                        help='offline training epochs per dataset')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate for edge prompt optimiser')
    parser.add_argument('--cloud_lr', type=float, default=None,
                        help='learning rate for cloud prompt '
                             '(default: same as --lr)')
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
    parser.add_argument('--selection_p', type=float, default=0.1,
                        help='confidence selection ratio (keep lowest-entropy fraction)')
    parser.add_argument('--n_ctx', type=int, default=4,
                        help='number of learnable context tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a',
                        help='context initialisation words')
    parser.add_argument('--resolution', type=int, default=224,
                        help='CLIP input resolution')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-j', '--workers', type=int, default=8,
                        help='data-loading workers')
    parser.add_argument('--rec_dir', default='new',
                        help='reconstructed-image subdirectory')
    parser.add_argument('--train_use_ori', action='store_true', default=False,
                        help='use original images (instead of reconstructed) for training')
    parser.add_argument('--save_dir', default=None,
                        help='directory for saved prompts & results')
    parser.add_argument('--load_prompt', default=None,
                        help='path to a save_dir with per-dataset prompts '
                             '(skip training)')
    main()
