"""
tpt_continual.py — Single-model continual prompt optimisation experiment.

Study how a single CLIP model (e.g. ViT-B/16) behaves when its prompt is
continuously optimised via entropy minimisation across a V/K alternating
domain stream.  This isolates the "teacher collapse" problem observed in
ceo_continual.py's keep_both setting.

Methods
-------
  reset      — reset prompt & optimizer every sample (TPT baseline)
  continual  — keep prompt across samples, entropy loss only (no reg)
  ewc        — keep prompt + EWC regularisation
  ema        — keep prompt + EMA blending after each step
"""

import argparse
import time
import json
import os
from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import accuracy, set_random_seed
from data.cls_to_names import *


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


# ── TTA step ─────────────────────────────────────────────────────────────

def test_time_tuning_step(model, images, optimizer, scaler, ewc_state, args):
    """One TTA step: entropy minimisation (+ optional EWC) on augmented views.
    Returns the final entropy loss value for monitoring."""
    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            output = model(images)
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args.selection_p)
            loss_ent = avg_entropy(output)

            loss = loss_ent
            if args.method == 'ewc' and ewc_state is not None:
                loss_ewc = compute_ewc_loss(
                    model.prompt_learner.ctx,
                    ewc_state['ref'], ewc_state['omega'],
                    args.ewc_lambda)
                loss = loss_ent + loss_ewc

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return float(loss_ent.item())


def update_ewc_fisher(model, images, ewc_state, args):
    """Update Fisher Ω via EMA of squared entropy-loss gradients,
    computed at the *current* (post-step) parameters.
    Used only by sample-anchor EWC.
    """
    gamma = args.ewc_gamma
    model.prompt_learner.ctx.grad = None
    with torch.cuda.amp.autocast():
        output = model(images)
        output, _ = select_confident_samples(output, args.selection_p)
        loss = avg_entropy(output)
    loss.backward()
    with torch.no_grad():
        g = model.prompt_learner.ctx.grad.float()
        ewc_state['omega'].mul_(gamma).add_((1 - gamma) * g.pow(2))
        ewc_state['ref'] = model.prompt_learner.ctx.detach().clone()
    model.prompt_learner.ctx.grad = None


def compute_domain_fisher(model, loader, args):
    """Compute Ω_d = (1/N) Σ (∂L/∂θ)² over a domain loader with θ frozen."""
    fisher_acc = torch.zeros_like(
        model.prompt_learner.ctx.data, dtype=torch.float32)
    n = 0
    for _, recs, _ in loader:
        if isinstance(recs, list):
            for k in range(len(recs)):
                recs[k] = recs[k].cuda(args.gpu, non_blocking=True)
        else:
            if len(recs.size()) > 4:
                recs = recs.squeeze(0)
            recs = recs.cuda(args.gpu, non_blocking=True)
        recs = torch.cat(recs, dim=0)

        model.prompt_learner.ctx.grad = None
        with torch.cuda.amp.autocast():
            output = model(recs)
            output, _ = select_confident_samples(output, args.selection_p)
            loss = avg_entropy(output)
        loss.backward()
        with torch.no_grad():
            fisher_acc.add_(model.prompt_learner.ctx.grad.float().pow(2))
            n += 1
        model.prompt_learner.ctx.grad = None

    if n > 0:
        fisher_acc.div_(n)
    return fisher_acc


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
        clean_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(), normalize])

        val_dataset = build_dataset(set_id, clean_transform, data_transform,
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

def evaluate_domain(model, loader, gpu):
    """Pure inference — no gradient, no adaptation."""
    corrects = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for oris, recs, target in loader:
            if isinstance(oris, list):
                ori = oris[0].cuda(gpu, non_blocking=True)
            else:
                if len(oris.size()) > 4:
                    oris = oris.squeeze(0)
                ori = oris.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            output = model(ori)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            corrects.append(int(acc1[0].item() > 0))
    return float(np.mean(corrects) * 100)


# ── adapt (online TTA) ──────────────────────────────────────────────────

def adapt_domain(model, loader, optimizer, scaler,
                 init_ctx, init_optim, ewc_state, args):
    """Online per-sample TTA.  Returns (accuracy%, mean_entropy)."""
    corrects = []
    ent_vals = []

    for i, (oris, recs, target) in enumerate(loader):
        # ── prepare clean image ──
        if isinstance(oris, list):
            ori = oris[0].cuda(args.gpu, non_blocking=True)
        else:
            if len(oris.size()) > 4:
                oris = oris.squeeze(0)
            ori = oris.cuda(args.gpu, non_blocking=True)

        # ── prepare augmented views ──
        if isinstance(recs, list):
            for k in range(len(recs)):
                recs[k] = recs[k].cuda(args.gpu, non_blocking=True)
        else:
            if len(recs.size()) > 4:
                recs = recs.squeeze(0)
            recs = recs.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        recs = torch.cat(recs, dim=0)

        # ── reset policy ──
        if args.method == 'reset':
            with torch.no_grad():
                model.prompt_learner.ctx.data.copy_(init_ctx)
            optimizer.load_state_dict(deepcopy(init_optim))

        # save pre-step ctx for EMA
        if args.method == 'ema':
            pre_ctx = model.prompt_learner.ctx.detach().clone()

        # ── TTA step ──
        ent = test_time_tuning_step(
            model, recs, optimizer, scaler, ewc_state, args)
        ent_vals.append(ent)

        # ── EMA blending ──
        if args.method == 'ema':
            d = args.ema_decay
            with torch.no_grad():
                model.prompt_learner.ctx.data.copy_(
                    d * pre_ctx +
                    (1 - d) * model.prompt_learner.ctx.data)

        # ── EWC Fisher update (sample-anchor only) ──
        if (args.method == 'ewc' and ewc_state is not None
                and args.ewc_anchor == 'sample'):
            update_ewc_fisher(model, recs, ewc_state, args)

        # ── evaluate on clean image ──
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(ori)
        acc1, _ = accuracy(output, target, topk=(1, 5))
        corrects.append(int(acc1[0].item() > 0))

        if (i + 1) % args.print_freq == 0:
            drift = (model.prompt_learner.ctx.data - init_ctx).norm().item()
            print(f"    [{i+1}/{len(loader.dataset)}]  "
                  f"acc={np.mean(corrects)*100:.1f}%  "
                  f"ent={np.mean(ent_vals[-args.print_freq:]):.4f}  "
                  f"drift={drift:.4f}")

    return float(np.mean(corrects) * 100), float(np.mean(ent_vals))


# ── tag / label helpers ──────────────────────────────────────────────────

def make_method_tag(args):
    if args.no_adapt:
        return 'no_adapt'
    if args.method == 'reset':
        return 'reset'
    if args.method == 'ewc':
        anchor = 'da' if args.ewc_anchor == 'domain' else 'sa'
        return f'ewc_{anchor}_l{args.ewc_lambda}_g{args.ewc_gamma}'
    if args.method == 'ema':
        return f'ema_d{args.ema_decay}'
    return args.method


_LABEL = {
    'no_adapt':   'No Adaptation (inference only)',
    'reset':      'Reset prompt every sample (TPT baseline)',
    'continual':  'Keep prompt, entropy only (no regularisation)',
    'ewc':        'Keep prompt, entropy + EWC',
    'ema':        'Keep prompt, entropy + EMA',
}


def make_method_label(args):
    if args.no_adapt:
        return _LABEL['no_adapt']
    m = args.method
    extra = ''
    if m == 'ewc':
        anchor_desc = 'domain-anchor' if args.ewc_anchor == 'domain' \
            else 'sample-anchor'
        extra = f' (λ={args.ewc_lambda}, γ={args.ewc_gamma}, {anchor_desc})'
    elif m == 'ema':
        extra = f' (d={args.ema_decay})'
    return _LABEL[m] + extra


# ── main ─────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    method_tag = make_method_tag(args)
    method_label = make_method_label(args)

    print(f"\n{'='*70}")
    print(f"  Single-Model Continual Prompt Optimisation")
    print(f"  Method       : {method_tag}")
    print(f"  Description  : {method_label}")
    print(f"  Arch         : {args.arch}")
    print(f"  Stream       : V/K alternating x {args.n_rounds} rounds")
    print(f"  Samples      : {args.samples_per_domain}/domain  "
          f"(total {args.samples_per_domain * 2 * args.n_rounds})")
    print(f"  lr={args.lr}  tta_steps={args.tta_steps}  "
          f"selection_p={args.selection_p}")
    if args.method == 'ewc':
        print(f"  ewc: λ={args.ewc_lambda}  γ={args.ewc_gamma}  "
              f"anchor={args.ewc_anchor}")
    print(f"{'='*70}")

    # ── build model ──
    classnames = imagenet_classes
    model = get_coop(args.arch, 'K', args.gpu, args.n_ctx, args.ctx_init)

    for name, param in model.named_parameters():
        param.requires_grad_("prompt_learner" in name)

    trainable = [p for p in model.parameters() if p.requires_grad]
    model = model.cuda(args.gpu)
    model.reset_classnames(classnames, args.arch)

    optimizer = optim.AdamW(trainable, args.lr)
    init_ctx = model.prompt_learner.ctx.detach().clone()
    init_optim = deepcopy(optimizer.state_dict())

    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    cudnn.benchmark = True
    model.eval()

    print(f"  prompt shape : {tuple(model.prompt_learner.ctx.shape)}  "
          f"({model.prompt_learner.ctx.numel()} params)")

    # ── build data ──
    domain_loaders = build_domain_loaders(args)
    domains = ['V', 'K']

    # ── EWC state ──
    ewc_state = None
    if args.method == 'ewc':
        ewc_state = {
            'omega': torch.zeros_like(
                model.prompt_learner.ctx.data, dtype=torch.float32),
            'ref': model.prompt_learner.ctx.detach().clone(),
        }

    # ── baseline (no-adapt) evaluation before any adaptation ──
    print(f"\n--- Baseline (no-adapt) ---")
    for d in domains:
        base_acc = evaluate_domain(model, domain_loaders[d], args.gpu)
        print(f"  acc_{d}(0) = {base_acc:.2f}%  (no-adapt)")

    # ── run stream ──
    results = {d: [] for d in domains}
    t_start = time.time()

    for rnd in range(1, args.n_rounds + 1):
        for d in domains:
            visit = len(results[d]) + 1
            loader = domain_loaders[d]

            print(f"\n--- Round {rnd}, Domain {d}, Visit #{visit} ---")

            # 1) pre-adapt accuracy
            pre_acc = evaluate_domain(model, loader, args.gpu)
            print(f"  acc'_{d}({visit}) = {pre_acc:.2f}%  (pre-adapt)")

            # 2) online adaptation
            if args.no_adapt:
                post_acc, mean_ent = pre_acc, 0.
            else:
                post_acc, mean_ent = adapt_domain(
                    model, loader, optimizer, scaler,
                    init_ctx, init_optim, ewc_state, args)

            drift = (model.prompt_learner.ctx.data - init_ctx).norm().item()
            print(f"  acc_{d}({visit})  = {post_acc:.2f}%  "
                  f"(ent={mean_ent:.4f}, drift={drift:.4f})")

            # domain-anchor Online EWC: freeze θ*_d, compute Ω_d, merge
            if ewc_state is not None and args.ewc_anchor == 'domain':
                ewc_state['ref'] = model.prompt_learner.ctx.detach().clone()
                omega_d = compute_domain_fisher(model, loader, args)
                ewc_state['omega'].mul_(args.ewc_gamma).add_(omega_d)

            results[d].append({
                'visit': visit, 'round': rnd,
                'pre_acc': pre_acc, 'post_acc': post_acc,
                'mean_ent': mean_ent, 'prompt_drift': drift,
            })

    elapsed = time.time() - t_start

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  {method_tag} — {method_label}")
    print(f"  arch={args.arch}   ({elapsed:.0f}s)")
    print(f"{'='*70}")

    hdr = f"{'Visit':>6}"
    for d in domains:
        hdr += f"  | acc'_{d:>2}   acc_{d:>2}    ent    drift"
    print(hdr)
    print('-' * 80)

    for t in range(args.n_rounds):
        row = f"  {t+1:>4}"
        for d in domains:
            v = results[d][t]
            row += (f"  | {v['pre_acc']:6.2f}  {v['post_acc']:6.2f}"
                    f"  {v['mean_ent']:.4f}"
                    f"  {v['prompt_drift']:6.3f}")
        print(row)

    print('-' * 80)
    row = f"  {'Mean':>4}"
    for d in domains:
        mp = np.mean([v['pre_acc'] for v in results[d]])
        ma = np.mean([v['post_acc'] for v in results[d]])
        me = np.mean([v['mean_ent'] for v in results[d]])
        row += f"  | {mp:6.2f}  {ma:6.2f}  {me:.4f}        "
    print(row)

    # ── ΔAcc & BT ──
    print(f"\n--- ΔAcc & Backward Transfer ---")
    deltas = []
    for d in domains:
        first = results[d][0]['post_acc']
        last = results[d][-1]['post_acc']
        delta = last - first
        deltas.append(delta)
        print(f"  Δacc_{d} = acc_{d}({args.n_rounds}) - acc_{d}(1) "
              f"= {last:.2f} - {first:.2f} = {delta:+.2f}")
    bt = float(np.mean(deltas))
    print(f"  BT = mean(Δacc_V, Δacc_K) = {bt:+.2f}")
    print(f"{'='*70}")

    # ── save ──
    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {
        'method': method_tag,
        'label': method_label,
        'arch': args.arch,
        'elapsed_s': elapsed,
        'results': results,
        'BT': bt,
    }
    for d in domains:
        metrics[f'mean_pre_acc_{d}'] = float(
            np.mean([v['pre_acc'] for v in results[d]]))
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
        description='Single-Model Continual Prompt Optimisation: '
                    'V/K Alternating Stream')

    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--samples_per_domain', type=int, default=500)
    parser.add_argument('--n_rounds', type=int, default=10)
    parser.add_argument('--dataset_mode', type=str, default='test')

    parser.add_argument('-a', '--arch', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('-p', '--print-freq', default=100, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--selection_p', default=0.1, type=float)
    parser.add_argument('--tta_steps', default=1, type=int)
    parser.add_argument('--n_ctx', default=4, type=int)
    parser.add_argument('--ctx_init', default=None, type=str)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--output_dir', type=str,
                        default='logs/single_model_continual')

    parser.add_argument('--no_adapt', action='store_true',
                        help='no-adapt baseline: inference only')
    parser.add_argument('--method', type=str, default='continual',
                        choices=['reset', 'continual', 'ewc', 'ema'],
                        help='reset=TPT baseline; continual=no reg; '
                             'ewc=EWC reg; ema=EMA blending')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--ewc_lambda', type=float, default=100.,
                        help='EWC regularisation strength λ')
    parser.add_argument('--ewc_gamma', type=float, default=0.99,
                        help='EWC Fisher EMA decay γ')
    parser.add_argument('--ewc_anchor', type=str, default='sample',
                        choices=['sample', 'domain'],
                        help='EWC ref anchor: sample=update every sample '
                             '(original); domain=update at domain boundaries')

    main()
