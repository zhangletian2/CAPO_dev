#!/usr/bin/env python
"""CEO Continual 5-Domain: I → A → V → R → K

Phase 1  Continual adaptation — online TTA through 5 domains sequentially.
Phase 2  Frozen evaluation   — test the final adapted prompt on every domain.

Methods
-------
    no_adapt   CLIP RN50 baseline (no adaptation)
    reset      CEO default — reset T+S prompt & optimiser per sample
    tewc_rq    keep T+S, teacher-only EWC, Reset Queue at domain switch
    tewc_aq    keep T+S, teacher-only EWC, Accumulate Queue

Usage
-----
    python ceo_continual_5d.py ./dataset --method tewc_rq --gpu 0
"""

import argparse
import json
import os
import time
from copy import deepcopy
from collections import deque

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip_device import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask
from utils.tools import accuracy, set_random_seed

DOMAIN_LABEL = {
    'I': 'IN-val', 'A': 'IN-A', 'V': 'IN-V2',
    'R': 'IN-R',   'K': 'IN-Sketch',
}


def _hp_suffix(args):
    parts = [f't{args.tau}', f'b{args.beta}', f'up{args.uplink_percent}']
    if args.lr != 5e-3:
        parts.append(f'lr{args.lr}')
    if args.selection_p != 0.1:
        parts.append(f'sp{args.selection_p}')
    if args.tta_steps != 1:
        parts.append(f's{args.tta_steps}')
    return '_'.join(parts)


def make_method_tag(args):
    if args.method == 'no_adapt':
        return 'no_adapt'
    hp = _hp_suffix(args)
    if args.method == 'reset':
        return f'reset_{hp}'
    anchor = 'da' if args.ewc_anchor == 'domain' else 'sa'
    q = 'RQ' if args.method == 'tewc_rq' else 'AQ'
    return f'tewc_{anchor}_l{args.ewc_lambda}_g{args.ewc_gamma}_{hp}_{q}'


# ── helpers ──────────────────────────────────────────────────────────

def select_confident_samples(logits, top):
    ent = (-logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(ent)[:int(ent.size(0) * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    avg_logits = torch.clamp(avg_logits, min=torch.finfo(avg_logits.dtype).min)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def compute_ewc_loss(param, ref, omega, lam):
    diff = (param.float() - ref.float()).pow(2)
    return 0.5 * lam * (omega * diff).sum()


def get_classnames(set_id):
    """Return classname list aligned with DualRootImageFolder folder order.

    I / K / V → full 1000 ImageNet classes (V uses target_transform for
    lexicographic→numeric mapping).
    A → 200-class subset via imagenet_a_mask.
    R → 200-class subset via imagenet_r_mask.
    """
    if set_id in ('I', 'K', 'V'):
        return imagenet_classes
    if set_id == 'A':
        return [imagenet_classes[i] for i in imagenet_a_mask]
    if set_id == 'R':
        return [imagenet_classes[i] for i, m in enumerate(imagenet_r_mask) if m]
    raise ValueError(f"unknown domain {set_id}")


# ── TTA steps ────────────────────────────────────────────────────────

def tta_step(cloud, edge, rec, opt_c, opt_e, sc_c, sc_e, args):
    """Vanilla TTA: entropy (cloud) + entropy+KL (edge)."""
    sel = None
    kl_fn = nn.KLDivLoss(reduction='batchmean')
    for _ in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            cf = cloud.get_image_features(rec)
            co = cloud(None, cf)
            co, sel = (co[sel], sel) if sel is not None \
                else select_confident_samples(co, args.selection_p)
            loss_c = avg_entropy(co)
        opt_c.zero_grad()
        sc_c.scale(loss_c).backward()
        sc_c.step(opt_c); sc_c.update()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                co = cloud(None, cf)
            del cf
            ef = edge.get_image_features(rec)
            eo = edge(None, ef)
            lkl = args.beta * kl_fn(
                F.log_softmax(eo / args.tau, 1),
                F.softmax(co / args.tau, 1))
            eo, sel = select_confident_samples(eo, args.selection_p)
            loss_e = avg_entropy(eo) + lkl
        opt_e.zero_grad()
        sc_e.scale(loss_e).backward()
        sc_e.step(opt_e); sc_e.update()


def tta_step_tewc(cloud, edge, rec, opt_c, opt_e, sc_c, sc_e, ewc, args):
    """TTA with EWC on teacher only: ent+EWC (cloud) + ent+KL (edge)."""
    sel = None
    kl_fn = nn.KLDivLoss(reduction='batchmean')
    for _ in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            cf = cloud.get_image_features(rec)
            co = cloud(None, cf)
            co, sel = (co[sel], sel) if sel is not None \
                else select_confident_samples(co, args.selection_p)
            loss_ent = avg_entropy(co)
            loss_ewc = compute_ewc_loss(
                cloud.prompt_learner.ctx,
                ewc['ref'], ewc['omega'], args.ewc_lambda)
            loss_c = loss_ent + loss_ewc
        opt_c.zero_grad()
        sc_c.scale(loss_c).backward()
        sc_c.step(opt_c); sc_c.update()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                co = cloud(None, cf)
            del cf
            ef = edge.get_image_features(rec)
            eo = edge(None, ef)
            lkl = args.beta * kl_fn(
                F.log_softmax(eo / args.tau, 1),
                F.softmax(co / args.tau, 1))
            eo, sel = select_confident_samples(eo, args.selection_p)
            loss_e = avg_entropy(eo) + lkl
        opt_e.zero_grad()
        sc_e.scale(loss_e).backward()
        sc_e.step(opt_e); sc_e.update()

    # Fisher update (teacher only)
    gamma = args.ewc_gamma
    cloud.prompt_learner.ctx.grad = None
    with torch.cuda.amp.autocast():
        cf = cloud.get_image_features(rec)
        co = cloud(None, cf)
        co, _ = select_confident_samples(co, args.selection_p)
        loss = avg_entropy(co)
    loss.backward()
    with torch.no_grad():
        g = cloud.prompt_learner.ctx.grad.float()
        ewc['omega'].mul_(gamma).add_((1 - gamma) * g.pow(2))
        if args.ewc_anchor == 'sample':
            ewc['ref'] = cloud.prompt_learner.ctx.detach().clone()
    cloud.prompt_learner.ctx.grad = None


# ── data ─────────────────────────────────────────────────────────────

def build_domain_loaders(domains, args):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    loaders = {}
    for sid in domains:
        base_tf = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        pre = transforms.Compose([transforms.ToTensor(), normalize])
        aug_tf = AugMixAugmenter(base_tf, pre,
                                 n_views=args.batch_size - 1,
                                 augmix=len(sid) > 1)
        dev_tf = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(), normalize])

        ds = build_dataset(sid, dev_tf, aug_tf, args.data,
                           mode=args.dataset_mode)

        # V folders are numeric strings sorted lexicographically → remap
        if hasattr(ds, 'classes'):
            try:
                mapping = [int(c) for c in ds.classes]
                ds.target_transform = lambda t, _m=mapping: _m[t]
            except (ValueError, TypeError):
                pass

        total = len(ds)
        n = min(args.samples_per_domain, total)
        idx = np.random.RandomState(args.seed).permutation(total)[:n].tolist()
        subset = torch.utils.data.Subset(ds, idx)
        loaders[sid] = torch.utils.data.DataLoader(
            subset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        print(f"  {sid} ({DOMAIN_LABEL[sid]:>10s}): {n}/{total}")
    return loaders


# ── evaluate (no adaptation) ─────────────────────────────────────────

@torch.no_grad()
def evaluate_domain(edge, loader, gpu, cloud=None):
    ok, cok = [], []
    for oris, _recs, target in loader:
        if isinstance(oris, list):
            ori = oris[0]
        else:
            ori = oris.squeeze(0) if oris.dim() > 4 else oris
        ori = ori.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            out = edge(None, edge.get_image_features(ori))
        a1, _ = accuracy(out, target, topk=(1, 5))
        ok.append(int(a1[0].item() > 0))

        if cloud is not None:
            with torch.cuda.amp.autocast():
                cout = cloud(None, cloud.get_image_features(ori))
            ca1, _ = accuracy(cout, target, topk=(1, 5))
            cok.append(int(ca1[0].item() > 0))

    s_acc = float(np.mean(ok) * 100)
    if cloud is not None:
        return s_acc, float(np.mean(cok) * 100)
    return s_acc


# ── adapt (online TTA with uplink gating) ────────────────────────────

def adapt_domain(cloud, edge, loader,
                 opt_c, opt_e, init_c_ctx, init_e_ctx,
                 init_c_opt, init_e_opt, sc_c, sc_e,
                 entropy_q, args, ewc=None):
    is_reset = (args.method == 'reset')
    corrects, t_corrects, n_up = [], [], 0

    for i, (oris, recs, target) in enumerate(loader):
        # ── prepare tensors ──
        if isinstance(oris, list):
            ori = oris[0]
        else:
            ori = oris.squeeze(0) if oris.dim() > 4 else oris
        ori = ori.cuda(args.gpu, non_blocking=True)

        if isinstance(recs, list):
            recs = [r.cuda(args.gpu, non_blocking=True) for r in recs]
        else:
            recs = recs.squeeze(0) if recs.dim() > 4 else recs
            recs = [recs.cuda(args.gpu, non_blocking=True)]
        target = target.cuda(args.gpu, non_blocking=True)
        rec = torch.cat(recs, dim=0)

        # ── uplink gating ──
        if args.uplink_percent >= 100.:
            uplink = True
        else:
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = edge(None, edge.get_image_features(ori))
            ent = float(
                (-output.softmax(1) * output.log_softmax(1)).sum(1).item())
            entropy_q.append(ent)
            uplink = ent > float(
                np.percentile(entropy_q, 100. - args.uplink_percent))

        if uplink:
            n_up += 1
            if is_reset:
                with torch.no_grad():
                    cloud.prompt_learner.ctx.data.copy_(init_c_ctx)
                    edge.prompt_learner.ctx.data.copy_(init_e_ctx)
                opt_c.load_state_dict(deepcopy(init_c_opt))
                opt_e.load_state_dict(deepcopy(init_e_opt))
                tta_step(cloud, edge, rec,
                         opt_c, opt_e, sc_c, sc_e, args)
            else:
                tta_step_tewc(cloud, edge, rec,
                              opt_c, opt_e, sc_c, sc_e, ewc, args)

            with torch.no_grad(), torch.cuda.amp.autocast():
                output = edge(None, edge.get_image_features(ori))
                cout = cloud(None, cloud.get_image_features(ori))
            ca, _ = accuracy(cout, target, topk=(1, 5))
            t_corrects.append(int(ca[0].item() > 0))
        else:
            with torch.no_grad(), torch.cuda.amp.autocast():
                cout = cloud(None, cloud.get_image_features(ori))
            ca, _ = accuracy(cout, target, topk=(1, 5))
            t_corrects.append(int(ca[0].item() > 0))

        a1, _ = accuracy(output, target, topk=(1, 5))
        corrects.append(int(a1[0].item() > 0))

        if (i + 1) % args.print_freq == 0:
            print(f"    [{i+1}/{len(loader.dataset)}]  "
                  f"S={np.mean(corrects)*100:.1f}%  "
                  f"T={np.mean(t_corrects)*100:.1f}%  "
                  f"up={n_up/(i+1)*100:.0f}%")

    up_pct = n_up / len(corrects) * 100 if corrects else 0.
    return (float(np.mean(corrects) * 100),
            float(up_pct),
            float(np.mean(t_corrects) * 100) if t_corrects else 0.)


# ── main ─────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    domains = list(args.domains)
    is_tewc = args.method.startswith('tewc')
    reset_q = (args.method == 'tewc_rq')

    method_tag = make_method_tag(args)
    desc = {
        'no_adapt': 'No adaptation (CLIP RN50)',
        'reset':    'CEO default — reset T+S per sample',
        'tewc_rq':  f'keep T+S, T-EWC(λ={args.ewc_lambda},γ={args.ewc_gamma}), RQ',
        'tewc_aq':  f'keep T+S, T-EWC(λ={args.ewc_lambda},γ={args.ewc_gamma}), AQ',
    }[args.method]

    print(f"\n{'='*70}")
    print(f"  Method : {method_tag}")
    print(f"  Desc   : {desc}")
    print(f"  Stream : {' -> '.join(domains)}")
    print(f"  N      : {args.samples_per_domain}/domain")
    print(f"  uplink : {args.uplink_percent}%   "
          f"tau={args.tau}  beta={args.beta}  lr={args.lr}")
    print(f"{'='*70}")

    # ── build models (init with full ImageNet-1K classnames) ──
    cloud = get_coop(args.cloud_arch, 'I', args.gpu,
                     args.n_ctx, args.ctx_init)
    edge = get_coop(args.arch, 'I', args.gpu,
                    args.n_ctx, args.ctx_init, type='Device')

    for n, p in cloud.named_parameters():
        p.requires_grad_('prompt_learner' in n)
    for n, p in edge.named_parameters():
        p.requires_grad_('prompt_learner' in n)

    c_train = [p for p in cloud.parameters() if p.requires_grad]
    e_train = [p for p in edge.parameters()  if p.requires_grad]

    cloud = cloud.cuda(args.gpu)
    edge  = edge.cuda(args.gpu)
    cloud.reset_classnames(imagenet_classes, args.cloud_arch)
    edge.reset_classnames(imagenet_classes, args.arch)

    opt_c = optim.AdamW(c_train, args.lr)
    opt_e = optim.AdamW(e_train, args.lr)
    init_c_ctx = cloud.prompt_learner.ctx.detach().clone()
    init_e_ctx = edge.prompt_learner.ctx.detach().clone()
    init_c_opt = deepcopy(opt_c.state_dict())
    init_e_opt = deepcopy(opt_e.state_dict())

    sc_c = torch.cuda.amp.GradScaler(init_scale=1000)
    sc_e = torch.cuda.amp.GradScaler(init_scale=1000)
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.benchmark = True

    cloud.eval()
    edge.eval()

    # ── data ──
    print("\nBuilding datasets ...")
    loaders = build_domain_loaders(domains, args)

    entropy_q = deque(maxlen=500)
    ewc = None
    if is_tewc:
        ewc = {
            'omega': torch.zeros_like(
                cloud.prompt_learner.ctx.data, dtype=torch.float32),
            'ref': cloud.prompt_learner.ctx.detach().clone(),
        }

    # ── no-adapt baseline (initial model, before any adaptation) ──
    baseline = {}
    print("\n--- No-Adapt Baseline (initial model) ---")
    for d in domains:
        cn = get_classnames(d)
        cloud.reset_classnames(cn, args.cloud_arch)
        edge.reset_classnames(cn, args.arch)
        if args.eval_teacher:
            s_bl, t_bl = evaluate_domain(edge, loaders[d], args.gpu, cloud)
            baseline[d] = {'student': s_bl, 'teacher': t_bl}
            print(f"  {d}: S_base={s_bl:.2f}%  T_base={t_bl:.2f}%")
        else:
            s_bl = evaluate_domain(edge, loaders[d], args.gpu)
            baseline[d] = {'student': s_bl}
            print(f"  {d}: S_base={s_bl:.2f}%")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 1 — Continual Adaptation
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  Phase 1: Continual Adaptation")
    print(f"{'='*70}")

    p1 = {}
    t0 = time.time()

    for idx, d in enumerate(domains):
        cn = get_classnames(d)
        cloud.reset_classnames(cn, args.cloud_arch)
        edge.reset_classnames(cn, args.arch)

        if reset_q and idx > 0:
            entropy_q.clear()

        print(f"\n--- {d} ({DOMAIN_LABEL[d]})  "
              f"[{idx+1}/{len(domains)}]  queue={len(entropy_q)} ---")

        # pre-adapt eval: first domain reuses baseline, skip redundant eval
        if idx == 0:
            pre_s = baseline[d]['student']
            pre_t = baseline[d].get('teacher')
        elif args.eval_teacher:
            pre_s, pre_t = evaluate_domain(edge, loaders[d], args.gpu, cloud)
        else:
            pre_s = evaluate_domain(edge, loaders[d], args.gpu)
            pre_t = None

        if pre_t is not None:
            print(f"  Pre : S={pre_s:.2f}%  T={pre_t:.2f}%")
        else:
            print(f"  Pre : S={pre_s:.2f}%")

        # adapt
        if args.method == 'no_adapt':
            post_s = pre_s
            up = 0.
            post_t = pre_t if pre_t is not None else 0.
        else:
            post_s, up, post_t = adapt_domain(
                cloud, edge, loaders[d],
                opt_c, opt_e, init_c_ctx, init_e_ctx,
                init_c_opt, init_e_opt, sc_c, sc_e,
                entropy_q, args, ewc=ewc)

        print(f"  Post: S={post_s:.2f}%  T={post_t:.2f}%  uplink={up:.0f}%")

        # EWC domain-boundary anchor
        if ewc is not None and args.ewc_anchor == 'domain':
            ewc['ref'] = cloud.prompt_learner.ctx.detach().clone()

        p1[d] = dict(pre_s=pre_s, pre_t=pre_t,
                     post_s=post_s, post_t=post_t, uplink=up)

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════
    #  Phase 2 — Frozen Evaluation (final prompt, no adaptation)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  Phase 2: Frozen Evaluation")
    print(f"{'='*70}")

    p2 = {}
    for d in domains:
        cn = get_classnames(d)
        cloud.reset_classnames(cn, args.cloud_arch)
        edge.reset_classnames(cn, args.arch)

        if args.eval_teacher:
            s, t = evaluate_domain(edge, loaders[d], args.gpu, cloud)
            p2[d] = dict(s=s, t=t)
            print(f"  {d} ({DOMAIN_LABEL[d]:>10s}): S={s:.2f}%  T={t:.2f}%")
        else:
            s = evaluate_domain(edge, loaders[d], args.gpu)
            p2[d] = dict(s=s)
            print(f"  {d} ({DOMAIN_LABEL[d]:>10s}): S={s:.2f}%")

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    w = 70
    print(f"\n{'='*w}")
    print(f"  {method_tag} | {desc}   ({elapsed:.0f}s)")
    print(f"{'='*w}")

    # Baseline table
    print("\n  Baseline (initial model, no adaptation):")
    hdr_bl = f"  {'Dom':>5}  {'S_base':>7}"
    if args.eval_teacher:
        hdr_bl += f"  {'T_base':>7}"
    print(hdr_bl)
    print('  ' + '-' * (len(hdr_bl) - 2))
    for d in domains:
        row = f"  {d:>5}  {baseline[d]['student']:7.2f}"
        if 'teacher' in baseline[d]:
            row += f"  {baseline[d]['teacher']:7.2f}"
        print(row)
    m_bl = np.mean([baseline[d]['student'] for d in domains])
    print(f"  {'Mean':>5}  {m_bl:7.2f}")

    # Phase 1 table
    print("\n  Phase 1 (Adaptation):")
    hdr = f"  {'Dom':>5}  {'Pre-S':>7}  {'Post-S':>7}  {'up%':>5}"
    if args.eval_teacher:
        hdr += f"  {'Pre-T':>7}  {'Post-T':>7}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for d in domains:
        r = p1[d]
        row = f"  {d:>5}  {r['pre_s']:7.2f}  {r['post_s']:7.2f}  {r['uplink']:4.0f}%"
        if args.eval_teacher and r['pre_t'] is not None:
            row += f"  {r['pre_t']:7.2f}  {r['post_t']:7.2f}"
        print(row)
    m1 = np.mean([p1[d]['post_s'] for d in domains])
    print(f"  {'Mean':>5}  {'':>7}  {m1:7.2f}")

    # Phase 2 table
    print("\n  Phase 2 (Frozen):")
    hdr2 = f"  {'Dom':>5}  {'S':>7}"
    if args.eval_teacher:
        hdr2 += f"  {'T':>7}"
    print(hdr2)
    print('  ' + '-' * (len(hdr2) - 2))
    for d in domains:
        row = f"  {d:>5}  {p2[d]['s']:7.2f}"
        if 't' in p2[d]:
            row += f"  {p2[d]['t']:7.2f}"
        print(row)
    m2 = np.mean([p2[d]['s'] for d in domains])
    print(f"  {'Mean':>5}  {m2:7.2f}")

    # Gain over baseline
    print(f"\n  Gain over Baseline (Post-adapt - Baseline):")
    for d in domains:
        gain = p1[d]['post_s'] - baseline[d]['student']
        print(f"    {d}: {p1[d]['post_s']:.2f} - {baseline[d]['student']:.2f} = {gain:+.2f}")
    mean_gain = m1 - m_bl
    print(f"    Mean: {mean_gain:+.2f}")

    # Phase 2 gain over no-adapt
    p2_gain = m2 - m_bl
    print(f"\n  Phase2 Gain over No-Adapt (Frozen - Baseline):")
    for d in domains:
        g = p2[d]['s'] - baseline[d]['student']
        print(f"    {d}: {p2[d]['s']:.2f} - {baseline[d]['student']:.2f} = {g:+.2f}")
    print(f"    Mean: {p2_gain:+.2f}")

    # Backward transfer
    print(f"\n  Backward Transfer (Frozen - Post-adapt):")
    for d in domains:
        delta = p2[d]['s'] - p1[d]['post_s']
        print(f"    D{d} = {p2[d]['s']:.2f} - {p1[d]['post_s']:.2f} = {delta:+.2f}")
    bt = np.mean([p2[d]['s'] - p1[d]['post_s'] for d in domains])
    print(f"    BT = {bt:+.2f}")
    print(f"{'='*w}")

    # ── save ──
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        'metrics': {
            'method': method_tag, 'desc': desc,
            'domains': domains, 'elapsed_s': elapsed,
            'baseline': {d: baseline[d] for d in domains},
            'mean_baseline_s': float(m_bl),
            'phase1': {d: p1[d] for d in domains},
            'phase2': {d: p2[d] for d in domains},
            'mean_post_s': float(m1),
            'mean_frozen_s': float(m2),
            'mean_gain': float(mean_gain),
            'phase2_gain': float(p2_gain),
            'BT': float(bt),
        },
        'args': vars(args),
    }
    path = os.path.join(args.output_dir, f'results_{method_tag}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CEO Continual 5-Domain: I->A->V->R->K')

    parser.add_argument('data', metavar='DIR',help='dataset root')
    parser.add_argument('--domains', default='IAVRK',
                        help='domain sequence (default: IAVRK)')
    parser.add_argument('--samples_per_domain', type=int, default=500)
    parser.add_argument('--dataset_mode', default='test')

    parser.add_argument('-a', '--arch', default='RN50')
    parser.add_argument('-c', '--cloud_arch', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('-p', '--print-freq', default=500, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--selection_p', default=0.1, type=float)
    parser.add_argument('--tta_steps', default=1, type=int)
    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--ctx_init', default='a_photo_of_a')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.9)

    parser.add_argument('--uplink_percent', type=float, default=50.)
    parser.add_argument('--method', default='tewc_rq',
                        choices=['no_adapt', 'reset', 'tewc_rq', 'tewc_aq'])
    parser.add_argument('--ewc_lambda', type=float, default=10000.)
    parser.add_argument('--ewc_gamma', type=float, default=0.95)
    parser.add_argument('--ewc_anchor', default='domain',
                        choices=['sample', 'domain'])

    parser.add_argument('--eval_teacher', action='store_true',
                        help='also evaluate teacher (cloud) model')
    parser.add_argument('--output_dir', default='logs/continual_5d')
    parser.add_argument('--deterministic', action='store_true')

    main()
