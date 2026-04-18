#!/usr/bin/env python3
"""
flops_bench.py — CAPO vs AMS 各阶段 FLOPs 测量

方法: 使用 fvcore.nn.FlopCountAnalysis 测量 forward FLOPs,
      backward FLOPs ≈ 2× forward (仅可训练模块).

阶段分解:
  CAPO:
    (a) Edge selection: RN50_vis_fwd + TextEnc_fwd + logits
    (b) Cloud prompt opt: ViT_vis_fwd + TextEnc_fwd + logits + backward(prompt only)
    (c) Cloud distill:  ViT_vis_fwd(no_grad) + RN50_vis_fwd + TextEnc_fwd + logits + backward(prompt only)
    (d) Edge final infer: RN50_vis_fwd + TextEnc_fwd + logits

  AMS:
    (a) Cloud infer: ViT_vis_fwd
    (b) KL distill: Cloud_TextEnc(no_grad) + RN50_vis_fwd + Edge_TextEnc_fwd + backward(text_encoder)
    (c) Edge final infer: RN50_vis_fwd + TextEnc_fwd + logits

Usage:
  python flops_bench.py --gpu 5 --n_cls 200
  python flops_bench.py --gpu 5 --n_cls 1000
"""

import argparse
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from clip import load
from clip.custom_clip_device import (
    get_coop, TextEncoder, PromptLearner, Device_PromptLearner,
)

DOWNLOAD_ROOT = '~/.cache/clip'


def count_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def flops_visual_encoder(clip_model, image, label=""):
    """Measure FLOPs of visual encoder forward."""
    vis = clip_model.image_encoder
    vis.eval()
    inp = image.type(vis.conv1.weight.dtype)
    fca = FlopCountAnalysis(vis, inp)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    f = fca.total()
    print(f"  {label} visual encoder:  {f/1e9:.3f} GFLOPs")
    return f


def flops_text_encoder(text_enc, prompts, tokenized_prompts, label="",
                       max_cls_for_trace=200):
    """Measure FLOPs of text encoder forward (transformer + projection).
    For large n_cls, trace a subset and linearly scale."""
    text_enc.eval()
    n_cls = prompts.shape[0]
    if n_cls <= max_cls_for_trace:
        fca = FlopCountAnalysis(text_enc, (prompts, tokenized_prompts))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        f = fca.total()
    else:
        sub = min(max_cls_for_trace, 100)
        fca = FlopCountAnalysis(text_enc, (prompts[:sub], tokenized_prompts[:sub]))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        f_sub = fca.total()
        f = f_sub * n_cls // sub
        print(f"  {label} text encoder:    {f/1e9:.3f} GFLOPs  (n_cls={n_cls}, "
              f"scaled from {sub})")
        return f
    print(f"  {label} text encoder:    {f/1e9:.3f} GFLOPs  (n_cls={n_cls})")
    return f


def flops_prompt_learner(pl, label=""):
    """Measure FLOPs of prompt_learner forward (mostly concat, negligible)."""
    pl.eval()
    try:
        fca = FlopCountAnalysis(pl, ())
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        f = fca.total()
    except Exception:
        f = 0
    print(f"  {label} prompt_learner: {f/1e9:.3f} GFLOPs  (negligible)")
    return f


def measure_backward_flops_empirically(model, image, label=""):
    """Use torch.profiler to measure actual backward FLOPs."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        feat = model.image_encoder(image.type(model.dtype))
        output = model(None, feat)
        loss = output.sum()
        loss.backward()

    total_flops = sum(
        e.flops for e in prof.key_averages() if e.flops > 0
    )
    print(f"  {label} full fwd+bwd (profiler): {total_flops/1e9:.3f} GFLOPs")
    return total_flops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=5)
    p.add_argument("--n_cls", type=int, default=200,
                   help="Number of classes (200 for A/R, 1000 for I/K/V)")
    p.add_argument("--n_ctx", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Augmented views (AugMix), typically 64")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    n_cls = args.n_cls
    bs = args.batch_size

    classnames = [f"class_{i}" for i in range(n_cls)]

    print(f"{'='*70}")
    print(f"  FLOPs Benchmark: CAPO vs AMS")
    print(f"  GPU: {args.gpu}  n_cls: {n_cls}  n_ctx: {args.n_ctx}  batch: {bs}")
    print(f"{'='*70}")

    # ── Load models ──
    print("\nLoading models...")
    cloud_model = get_coop("ViT-B/16", "A", device, args.n_ctx, "a_photo_of_a")
    edge_model = get_coop("RN50", "A", device, args.n_ctx, "a_photo_of_a",
                          type='Device')

    cloud_model.reset_classnames(classnames, "ViT-B/16")
    edge_model.reset_classnames(classnames, "RN50")
    cloud_model = cloud_model.to(device).eval()
    edge_model = edge_model.to(device).eval()

    # ── Dummy inputs ──
    img_single = torch.randn(1, 3, 224, 224, device=device)
    img_batch = torch.randn(bs, 3, 224, 224, device=device)

    # ── Param counts ──
    print(f"\n{'─'*70}")
    print("  Parameter Counts")
    print(f"{'─'*70}")

    cloud_vis_params = count_params(cloud_model.image_encoder)
    edge_vis_params = count_params(edge_model.image_encoder)
    cloud_text_params = count_params(cloud_model.text_encoder)
    edge_text_params = count_params(edge_model.text_encoder)
    cloud_prompt_params = count_params(cloud_model.prompt_learner)
    edge_prompt_params = count_params(edge_model.prompt_learner)

    print(f"  Cloud (ViT-B/16) visual: {cloud_vis_params/1e6:.2f}M")
    print(f"  Edge  (RN50)     visual: {edge_vis_params/1e6:.2f}M")
    print(f"  Cloud text_encoder:      {cloud_text_params/1e6:.2f}M")
    print(f"  Edge  text_encoder:      {edge_text_params/1e6:.2f}M")
    print(f"  Cloud prompt_learner:    {cloud_prompt_params} ({cloud_prompt_params/1024:.2f}K)")
    print(f"  Edge  prompt_learner:    {edge_prompt_params} ({edge_prompt_params/1024:.2f}K)")

    # ═══════════════════════════════════════════════════════════════
    #  Forward FLOPs per submodule (using fvcore)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  Forward FLOPs per Submodule (fvcore)")
    print(f"{'─'*70}")

    # visual encoders (single image)
    f_cloud_vis_1 = flops_visual_encoder(cloud_model, img_single, "Cloud(ViT-B/16) ×1")
    f_edge_vis_1 = flops_visual_encoder(edge_model, img_single, "Edge (RN50)     ×1")

    # visual encoders (batch)
    f_cloud_vis_b = flops_visual_encoder(cloud_model, img_batch, f"Cloud(ViT-B/16) ×{bs}")
    f_edge_vis_b = flops_visual_encoder(edge_model, img_batch, f"Edge (RN50)     ×{bs}")

    # text encoder
    with torch.no_grad():
        cloud_prompts = cloud_model.prompt_learner()
        edge_prompts = edge_model.prompt_learner()
    cloud_tokens = cloud_model.prompt_learner.tokenized_prompts
    edge_tokens = edge_model.prompt_learner.tokenized_prompts

    f_cloud_text = flops_text_encoder(cloud_model.text_encoder,
                                       cloud_prompts, cloud_tokens,
                                       "Cloud(ViT-B/16)")
    f_edge_text = flops_text_encoder(edge_model.text_encoder,
                                      edge_prompts, edge_tokens,
                                      "Edge (RN50)    ")

    # logits computation (matmul: [bs, dim] × [dim, n_cls]) — analytical
    cloud_dim = cloud_prompts.shape[-1]
    edge_dim = edge_prompts.shape[-1]
    f_logits_cloud_1 = 2 * 1 * cloud_dim * n_cls
    f_logits_cloud_b = 2 * bs * cloud_dim * n_cls
    f_logits_edge_1 = 2 * 1 * edge_dim * n_cls
    f_logits_edge_b = 2 * bs * edge_dim * n_cls
    print(f"  Cloud logits (×1):  {f_logits_cloud_1/1e6:.3f} MFLOPs")
    print(f"  Cloud logits (×{bs}): {f_logits_cloud_b/1e6:.3f} MFLOPs")
    print(f"  Edge  logits (×1):  {f_logits_edge_1/1e6:.3f} MFLOPs")
    print(f"  Edge  logits (×{bs}): {f_logits_edge_b/1e6:.3f} MFLOPs")

    # ═══════════════════════════════════════════════════════════════
    #  CAPO per-stage FLOPs
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  CAPO: Per-Stage FLOPs Breakdown")
    print(f"{'='*70}")

    # (a) Edge entropy check / selection: edge_vis(×1) + text_enc + logits
    capo_a = f_edge_vis_1 + f_edge_text + f_logits_edge_1
    print(f"\n  (a) Edge Selection (entropy check):")
    print(f"      = Edge_vis(×1) + Edge_TextEnc + logits")
    print(f"      = {f_edge_vis_1/1e9:.3f} + {f_edge_text/1e9:.3f} + {f_logits_edge_1/1e9:.4f}")
    print(f"      = {capo_a/1e9:.3f} GFLOPs")

    # (b) Cloud prompt optimization (1 TTA step):
    #     fwd: cloud_vis(×bs) + cloud_text + logits + entropy
    #     bwd: ≈2× (cloud_text + logits) — only prompt_learner is trainable
    #          but gradient flows through text_encoder too
    capo_b_fwd = f_cloud_vis_b + f_cloud_text + f_logits_cloud_b
    capo_b_bwd = 2 * (f_cloud_text + f_logits_cloud_b)
    capo_b = capo_b_fwd + capo_b_bwd
    print(f"\n  (b) Cloud Prompt Optimization (1 TTA step):")
    print(f"      fwd: Cloud_vis(×{bs}) + Cloud_TextEnc + logits")
    print(f"         = {f_cloud_vis_b/1e9:.3f} + {f_cloud_text/1e9:.3f} + {f_logits_cloud_b/1e9:.4f}")
    print(f"         = {capo_b_fwd/1e9:.3f} GFLOPs")
    print(f"      bwd: ≈2× (Cloud_TextEnc + logits)  [grad through text_enc to prompt]")
    print(f"         = {capo_b_bwd/1e9:.3f} GFLOPs")
    print(f"      total = {capo_b/1e9:.3f} GFLOPs")

    # (c) Cloud distillation (1 TTA step):
    #     cloud_feat is REUSED from phase (b), so NO cloud_vis forward here.
    #     fwd: cloud_text(no_grad, reuses cloud_feat) + cloud_logits(no_grad)
    #        + edge_vis(×bs) + edge_text + edge_logits
    #     bwd: ≈2× (edge_text + edge_logits + KL)
    capo_c_cloud_nograd = f_cloud_text + f_logits_cloud_b
    capo_c_edge_fwd = f_edge_vis_b + f_edge_text + f_logits_edge_b
    capo_c_bwd = 2 * (f_edge_text + f_logits_edge_b)
    capo_c = capo_c_cloud_nograd + capo_c_edge_fwd + capo_c_bwd
    print(f"\n  (c) Cloud Distillation (1 TTA step):")
    print(f"      cloud no_grad: Cloud_TextEnc + logits  (cloud_feat reused from (b))")
    print(f"         = {capo_c_cloud_nograd/1e9:.3f} GFLOPs")
    print(f"      edge fwd: Edge_vis(×{bs}) + Edge_TextEnc + logits")
    print(f"         = {capo_c_edge_fwd/1e9:.3f} GFLOPs")
    print(f"      edge bwd: ≈2× (Edge_TextEnc + logits)")
    print(f"         = {capo_c_bwd/1e9:.3f} GFLOPs")
    print(f"      total = {capo_c/1e9:.3f} GFLOPs")

    # (d) Edge final inference: edge_vis(×1) + edge_text + logits
    capo_d = f_edge_vis_1 + f_edge_text + f_logits_edge_1
    print(f"\n  (d) Edge Final Inference:")
    print(f"      = Edge_vis(×1) + Edge_TextEnc + logits")
    print(f"      = {capo_d/1e9:.3f} GFLOPs")

    capo_cloud_total = capo_b + capo_c
    capo_edge_total = capo_a + capo_d
    capo_total = capo_a + capo_b + capo_c + capo_d

    print(f"\n  ── CAPO Summary ──")
    print(f"  Edge  total (a+d):  {capo_edge_total/1e9:.3f} GFLOPs")
    print(f"  Cloud total (b+c):  {capo_cloud_total/1e9:.3f} GFLOPs")
    print(f"  TOTAL (a+b+c+d):    {capo_total/1e9:.3f} GFLOPs")

    # ═══════════════════════════════════════════════════════════════
    #  AMS per-stage FLOPs
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  AMS: Per-Stage FLOPs Breakdown")
    print(f"{'='*70}")

    # (a) Cloud feature extraction: cloud_vis(×bs)
    ams_a = f_cloud_vis_b
    print(f"\n  (a) Cloud Feature Extraction:")
    print(f"      = Cloud_vis(×{bs})")
    print(f"      = {ams_a/1e9:.3f} GFLOPs")

    # (b) KL Distillation (text_encoder training, 1 step):
    #     cloud_feat reused from (a), so only text_enc runs (no_grad).
    #     Edge visual is frozen (get_image_features uses no_grad internally),
    #     so backward only flows through edge_text_enc.
    #     This is IDENTICAL to CAPO stage (c).
    ams_b_cloud_nograd = f_cloud_text + f_logits_cloud_b
    ams_b_edge_fwd = f_edge_vis_b + f_edge_text + f_logits_edge_b
    ams_b_bwd = 2 * (f_edge_text + f_logits_edge_b)
    ams_b = ams_b_cloud_nograd + ams_b_edge_fwd + ams_b_bwd
    print(f"\n  (b) KL Distillation (text_encoder training, 1 step):")
    print(f"      cloud no_grad: Cloud_TextEnc + logits  (cloud_feat reused from (a))")
    print(f"         = {ams_b_cloud_nograd/1e9:.3f} GFLOPs")
    print(f"      edge fwd: Edge_vis(×{bs}) + Edge_TextEnc + logits")
    print(f"         = {ams_b_edge_fwd/1e9:.3f} GFLOPs")
    print(f"      edge bwd: ≈2× (Edge_TextEnc + logits)")
    print(f"         [edge_vis frozen (no_grad inside get_image_features)]")
    print(f"         = {ams_b_bwd/1e9:.3f} GFLOPs")
    print(f"      total = {ams_b/1e9:.3f} GFLOPs")
    print(f"      NOTE: identical to CAPO stage (c)  ← same computation graph")

    # (c) Edge final inference: edge_vis(×1) + edge_text + logits
    ams_c = f_edge_vis_1 + f_edge_text + f_logits_edge_1
    print(f"\n  (c) Edge Final Inference:")
    print(f"      = Edge_vis(×1) + Edge_TextEnc + logits")
    print(f"      = {ams_c/1e9:.3f} GFLOPs")

    ams_cloud_total = ams_a + ams_b
    ams_edge_total = ams_c
    ams_total = ams_a + ams_b + ams_c

    print(f"\n  ── AMS Summary ──")
    print(f"  Edge  total (c):    {ams_edge_total/1e9:.3f} GFLOPs")
    print(f"  Cloud total (a+b):  {ams_cloud_total/1e9:.3f} GFLOPs")
    print(f"  TOTAL (a+b+c):      {ams_total/1e9:.3f} GFLOPs")

    # ═══════════════════════════════════════════════════════════════
    #  Comparison table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  CAPO vs AMS Comparison")
    print(f"{'='*70}")
    print(f"  {'Stage':<35s}  {'CAPO':>10s}  {'AMS':>10s}  {'Ratio':>7s}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}  {'─'*7}")

    rows = [
        ("Edge selection/entropy",  capo_a,  0),
        ("Cloud prompt opt",        capo_b,  0),
        ("Cloud feature extract",   0,       ams_a),
        ("Cloud/KL distillation",   capo_c,  ams_b),
        ("Edge final inference",    capo_d,  ams_c),
    ]
    for name, cv, av in rows:
        cs = f"{cv/1e9:.2f}" if cv > 0 else "—"
        as_ = f"{av/1e9:.2f}" if av > 0 else "—"
        if cv > 0 and av > 0:
            ratio = f"{av/cv:.2f}×"
        else:
            ratio = "—"
        print(f"  {name:<35s}  {cs:>10s}  {as_:>10s}  {ratio:>7s}")

    print(f"  {'─'*35}  {'─'*10}  {'─'*10}  {'─'*7}")
    print(f"  {'Edge subtotal':<35s}  {capo_edge_total/1e9:10.2f}  {ams_edge_total/1e9:10.2f}"
          f"  {ams_edge_total/capo_edge_total:.2f}×")
    print(f"  {'Cloud subtotal':<35s}  {capo_cloud_total/1e9:10.2f}  {ams_cloud_total/1e9:10.2f}"
          f"  {ams_cloud_total/capo_cloud_total:.2f}×")
    print(f"  {'TOTAL':<35s}  {capo_total/1e9:10.2f}  {ams_total/1e9:10.2f}"
          f"  {ams_total/capo_total:.2f}×")

    print(f"\n  Units: GFLOPs.  n_cls={n_cls}, batch={bs}, image=224×224")
    print(f"  Backward ≈ 2× forward for trainable modules.")
    print(f"  Edge visual frozen (get_image_features uses torch.no_grad internally).")
    print(f"  cloud_feat reused between stages → cloud_vis computed only once.")
    print(f"  CAPO distill ≡ AMS distill: same computation graph, same FLOPs.")
    print(f"")
    print(f"  CAPO trains prompt ({cloud_prompt_params} params = {cloud_prompt_params*4/1024:.2f} KB)")
    print(f"  AMS  trains text_encoder ({count_params(edge_model.text_encoder):,d} params"
          f" = {count_params(edge_model.text_encoder)*4/1024/1024:.2f} MB)")
    print(f"  → Trainable param count affects memory & comm, NOT backward FLOPs.")


if __name__ == "__main__":
    main()
