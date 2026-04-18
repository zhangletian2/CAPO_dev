# coding=utf-8
"""
Joint CEO: Classification (IN-V) + Retrieval (I2T) + Segmentation (VOC).
Shares a single prompt ctx among all three edge models, optimized by combined loss.

2-GPU layout  (--cloud_gpu):
  edge_dev  ->  RN50      backbone  (RET edge + CLS edge, shared) + MaskCLIP RN50 (SEG edge)
  cloud_dev ->  ViT-B/16  backbone  (RET cloud + CLS cloud, shared) + MaskCLIP ViT-B/16 (SEG cloud)

Environment: conda activate joint3  (denseclip clone + LAVIS deps)
"""
# ── spacy mock (unused by our pipeline, required by LAVIS → transformers) ──
import sys
import types
import importlib.machinery
for _mod_name in ['spacy', 'spacy.tokens', 'spacy.tokens.doc']:
    _m = types.ModuleType(_mod_name)
    _m.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
    sys.modules[_mod_name] = _m

import os

# Locate sibling tasks in the capo repo (this file lives in capo/capo_joint/).
# scripts/_paths.sh exports CLS_ROOT / SEG_ROOT / RET_ROOT; honor them if set,
# otherwise fall back to the vanilla layout.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_ROOT = os.environ.get('CLS_ROOT', os.path.join(_REPO_ROOT, 'capo_cls'))
RET_ROOT = os.environ.get('RET_ROOT', os.path.join(_REPO_ROOT, 'capo_ret'))
SEG_ROOT = os.environ.get('SEG_ROOT', os.path.join(_REPO_ROOT, 'capo_seg'))
# Legacy name kept for backward-compat inside this file
MASKCLIP_ROOT = SEG_ROOT
# third_party/{clip,lavis,mmseg} are reached via PYTHONPATH set by _paths.sh;
# the task-root inserts below just make flat imports (`from data.x import ...`,
# `from custom_models import ...`) work even when this file is run directly.
for _p in (CLS_ROOT, RET_ROOT, SEG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc
import argparse
import json
import time
import logging
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# --- RET imports (LAVIS) ---
import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.datasets.builders import *  # noqa: F401,F403  (registry)
from lavis.models import *             # noqa: F401,F403
from lavis.processors import *         # noqa: F401,F403
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *              # noqa: F401,F403
from custom_models import CLIPRet_TPT_I2T
from lavis_evaluate import setup_seeds

# --- CLS imports (CAPO_cls) ---
from clip.custom_clip_device import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from data.cls_to_names import *        # noqa: F401,F403
from data.imagenet_variants import imagenet_v_mask

# --- SEG imports (MaskCLIP / MMSeg) ---
from mmcv import Config as MMConfig
from mmcv.runner import load_checkpoint
from mmseg.datasets import (
    build_dataset as build_seg_dataset,
    build_dataloader as build_seg_dataloader,
)
from mmseg.models import build_segmentor
from mmseg.ops import resize

import clip as openai_clip
from clip.simple_tokenizer import SimpleTokenizer
_tokenizer = SimpleTokenizer()


# ═══════════════════════════════════════════════════════════════
#  Loss utilities
# ═══════════════════════════════════════════════════════════════

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


def topk_ent(logits, k=20, tau=1.0, reduction="mean", eps=1e-12):
    probs = F.softmax(logits / tau, dim=-1)
    _, topk_idx = torch.topk(logits, k, dim=-1)
    topk_probs = probs.gather(1, topk_idx)
    renorm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + eps)
    loss = -(renorm * torch.log(renorm + eps)).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ── SEG-specific losses (from ceo_seg.py) ──

def seg_forward_with_flip(model, img):
    logits_orig = model(img)
    logits_flip = model(img.flip(-1)).flip(-1)
    log_p1 = F.log_softmax(logits_orig, dim=1)
    log_p2 = F.log_softmax(logits_flip, dim=1)
    avg_log_probs = torch.stack([log_p1, log_p2], dim=0).logsumexp(dim=0) \
                    - np.log(2.0)
    return avg_log_probs


def marginal_entropy_loss(avg_log_probs, selection_p):
    avg_probs = avg_log_probs.exp()
    pixel_entropy = -(avg_probs * avg_log_probs).sum(dim=1)
    B, C = avg_log_probs.shape[:2]
    flat_ent = pixel_entropy.view(B, -1)
    k = max(1, int(flat_ent.shape[1] * selection_p))
    _, idx = torch.topk(flat_ent, k, dim=1, largest=False)
    probs_flat = avg_probs.view(B, C, -1)
    logp_flat = avg_log_probs.view(B, C, -1)
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1)
    sel_p = probs_flat.gather(2, idx_exp)
    sel_lp = logp_flat.gather(2, idx_exp)
    return -(sel_p * sel_lp).sum(dim=1).mean()


# ═══════════════════════════════════════════════════════════════
#  RET evaluation
# ═══════════════════════════════════════════════════════════════

def compute_i2t_recall(scores, img2txt, ks=(1, 5, 10)):
    n = scores.shape[0]
    results = {}
    for k in ks:
        correct = 0
        for i in range(n):
            _, topk_indices = scores[i].topk(k)
            gt = set(img2txt[i])
            if gt.intersection(set(topk_indices.tolist())):
                correct += 1
        results[f"i2t_R@{k}"] = round(100.0 * correct / n, 2)
    return results


# ═══════════════════════════════════════════════════════════════
#  SEG model classes (adapted from ceo_seg.py)
# ═══════════════════════════════════════════════════════════════

VOC_CLASSES = [
    'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv monitor',
]


class SegTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.positional_embedding.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        if isinstance(self.text_projection, nn.Parameter):
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] \
                @ self.text_projection
        else:
            x = self.text_projection(
                x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)])
        return x


class SegPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init=None):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = openai_clip.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
            self.n_ctx = n_ctx
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [openai_clip.tokenize(p) for p in prompts]
        ).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.classnames = classnames

    def reset(self):
        self.ctx.copy_(self.ctx_init_state)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=-2)
        return prompts


class MaskClipCEOModel(nn.Module):
    """MaskCLIP visual pipeline + CLIP prompt-tuned text pipeline."""

    def __init__(self, seg_cfg_path, seg_ckpt_path, clip_arch, classnames,
                 n_ctx=4, ctx_init=None, clip_download_root=None):
        super().__init__()
        cfg = MMConfig.fromfile(seg_cfg_path)
        cfg.model.pretrained = None
        cfg_dir = os.path.dirname(os.path.abspath(seg_cfg_path))
        maskclip_root = os.path.dirname(os.path.dirname(cfg_dir))
        dh = cfg.model.decode_head
        for key in ('text_embeddings_path', 'visual_projs_path'):
            val = dh.get(key, '')
            if val and not os.path.isabs(val):
                dh[key] = os.path.join(maskclip_root, val)
        seg_model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(seg_model, seg_ckpt_path, map_location='cpu',
                        strict=False)
        seg_model.eval()

        self.backbone = seg_model.backbone
        self.decode_head = seg_model.decode_head
        self.vit = getattr(seg_model.decode_head, 'vit', False)
        self.align_corners = seg_model.align_corners

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.decode_head.parameters():
            p.requires_grad_(False)

        dl_root = clip_download_root or os.path.expanduser('~/.cache/clip')
        _loaded = openai_clip.load(clip_arch, device='cpu',
                                   download_root=dl_root)
        clip_model = _loaded[0]
        self.text_encoder = SegTextEncoder(clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.prompt_learner = SegPromptLearner(
            clip_model, classnames, n_ctx=n_ctx, ctx_init=ctx_init)
        self.logit_scale = clip_model.logit_scale.exp().detach()

        del seg_model, clip_model
        torch.cuda.empty_cache()

    def get_text_features(self):
        prompts = self.prompt_learner()
        tokenized = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_visual_feat(self, img):
        with torch.no_grad():
            backbone_out = self.backbone(img)
            head = self.decode_head
            x = head._transform_inputs(backbone_out)
            if self.vit:
                if isinstance(x, list) and len(x) == 4:
                    x, q, k, v = x
                elif isinstance(x, list) and len(x) == 2:
                    x, _cls_token = x
                    v = None
                else:
                    v = None
                feat = head.proj(v if v is not None else x)
            else:
                v = head.v_proj(x)
                feat = head.c_proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True)
        return feat

    def forward(self, img):
        feat = self.get_visual_feat(img)
        text_features = self.get_text_features()
        text_features = text_features.to(feat.dtype)
        logits = self.logit_scale * F.conv2d(
            feat, text_features[:, :, None, None])
        return logits

    def predict(self, img, img_metas, use_flip=True):
        if use_flip:
            avg_lp = seg_forward_with_flip(self, img)
        else:
            avg_lp = F.log_softmax(self.forward(img), dim=1)
        avg_lp = resize(avg_lp, size=img.shape[2:],
                        mode='bilinear', align_corners=self.align_corners)
        ori_shape = img_metas[0]['ori_shape'][:2]
        avg_lp = resize(avg_lp, size=ori_shape,
                        mode='bilinear', align_corners=self.align_corners)
        seg_pred = avg_lp.argmax(dim=1).cpu().numpy()
        return list(seg_pred)

    def reset(self):
        self.prompt_learner.reset()


# ═══════════════════════════════════════════════════════════════
#  Backbone / text-encoder sharing
# ═══════════════════════════════════════════════════════════════

def _patch_text_encoder_with_mask(text_encoder, causal_mask):
    """Monkey-patch a TextEncoder to pass causal attn_mask to LAVIS transformer."""
    import types as _types
    text_encoder._causal_mask = causal_mask

    def _forward_with_mask(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self._causal_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        if isinstance(self.text_projection, nn.Parameter):
            x = x[torch.arange(x.shape[0]),
                  tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            x = self.text_projection(
                x[torch.arange(x.shape[0]),
                  tokenized_prompts.argmax(dim=-1)])
        return x

    text_encoder.forward = _types.MethodType(_forward_with_mask, text_encoder)


def share_backbone(cls_model, ret_model):
    """Share CLS CLIP backbone with RET (visual + text)."""
    clip = ret_model.clip_model
    cls_model.image_encoder = clip.visual
    cls_model.text_encoder.transformer = clip.transformer
    cls_model.text_encoder.positional_embedding = clip.positional_embedding
    cls_model.text_encoder.ln_final = clip.ln_final
    cls_model.text_encoder.text_projection = clip.text_projection
    cls_model.logit_scale = clip.logit_scale.data
    _patch_text_encoder_with_mask(cls_model.text_encoder, clip.attn_mask)


def share_seg_text_encoder(seg_model, ret_model):
    """Share SEG text encoder with RET CLIP (text only, visual stays MaskCLIP)."""
    clip = ret_model.clip_model
    seg_model.text_encoder.transformer = clip.transformer
    seg_model.text_encoder.positional_embedding = clip.positional_embedding
    seg_model.text_encoder.ln_final = clip.ln_final
    seg_model.text_encoder.text_projection = clip.text_projection
    seg_model.logit_scale = clip.logit_scale.exp().detach()
    _patch_text_encoder_with_mask(seg_model.text_encoder, clip.attn_mask)


# ═══════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Joint CEO: CLS (IN-V) + RET (I2T) + SEG (VOC)")

    # common
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--cloud_gpu", default=None, type=int)
    p.add_argument("--seg_gpu", default=None, type=int,
                   help="GPU for SEG cloud model (default: same as cloud_gpu)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tta_steps", default=1, type=int)
    p.add_argument("--lr", default=0.0025, type=float)
    p.add_argument("--n_ctx", default=4, type=int)
    p.add_argument("--ctx_init", default="a_photo_of_a", type=str)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--print_freq", type=int, default=100)

    # cls
    p.add_argument("--cls_data", type=str,
                   default=os.path.join(CLS_ROOT, "dataset"))
    p.add_argument("--cls_arch", type=str, default="RN50")
    p.add_argument("--cls_cloud_arch", type=str, default="ViT-B/16")
    p.add_argument("--cls_selection_p", type=float, default=0.1)
    p.add_argument("--cls_tau", type=float, default=0.4)
    p.add_argument("--cls_beta", type=float, default=0.3)
    p.add_argument("--cls_batch_size", type=int, default=64)
    p.add_argument("--cls_rec_dir", type=str, default="new")

    # ret
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--options", nargs="+")
    p.add_argument("--ret_edge", type=str, default="RN50")
    p.add_argument("--ret_cloud", type=str, default="ViT-B-16")
    p.add_argument("--ret_tau", type=float, default=0.8)
    p.add_argument("--ret_beta", type=float, default=0.5)
    p.add_argument("--ret_sim_top_k", type=int, default=256)
    p.add_argument("--ret_ent_top_k", type=int, default=40)
    p.add_argument("--ret_ent_tea", type=int, default=1)
    p.add_argument("--ret_ent_stu", type=int, default=1)
    p.add_argument("--ret_cloud_orig", type=int, default=0)
    p.add_argument("--ret_cloud_image_dir", type=str, default="")

    # seg
    p.add_argument("--seg_edge_config", type=str, required=True)
    p.add_argument("--seg_edge_ckpt", type=str, required=True)
    p.add_argument("--seg_cloud_config", type=str, required=True)
    p.add_argument("--seg_cloud_ckpt", type=str, required=True)
    p.add_argument("--seg_data_root", type=str,
                   default=os.path.join(MASKCLIP_ROOT,
                                        "data/VOCdevkit/VOC2012"))
    p.add_argument("--seg_tau", type=float, default=0.4)
    p.add_argument("--seg_beta", type=float, default=0.9)
    p.add_argument("--seg_selection_p", type=float, default=0.2)
    p.add_argument("--seg_selective_kl", type=int, default=1)
    p.add_argument("--seg_cloud_conf_p", type=float, default=0.1)
    p.add_argument("--seg_clip_download_root", type=str,
                   default=os.environ.get("CLIP_CACHE",
                                          os.path.join(CLS_ROOT, "clip_cache")))

    # joint weights
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_ret", type=float, default=1.0)
    p.add_argument("--lambda_seg", type=float, default=1.0)

    args = p.parse_args()
    if args.cloud_gpu is None:
        args.cloud_gpu = args.gpu
    if args.seg_gpu is None:
        args.seg_gpu = args.cloud_gpu
    os.makedirs(args.output, exist_ok=True)
    return args


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    with open(os.path.join(args.output, "hparams_joint3.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    edge_dev = f"cuda:{args.gpu}"
    cloud_dev = f"cuda:{args.cloud_gpu}"
    seg_dev = f"cuda:{args.seg_gpu}"
    multi_gpu = len({edge_dev, cloud_dev, seg_dev})
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    print(f"  Devices: edge={edge_dev}  cloud={cloud_dev}  seg={seg_dev}"
          f"  ({multi_gpu}-GPU)", flush=True)

    do_cls = args.lambda_cls > 0
    do_ret = args.lambda_ret > 0
    do_seg = args.lambda_seg > 0

    # ============================================================
    #  RET data & models  (LAVIS)
    # ============================================================
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets_ret = task.build_datasets(cfg)
    runner = RunnerBase(
        cfg=cfg, job_id="JointCEO3", task=task, model=None,
        datasets=datasets_ret)
    ret_loader = runner.dataloaders["test"]
    if args.ret_cloud_image_dir:
        ret_loader.dataset.vis_root2 = args.ret_cloud_image_dir
    captions = ret_loader.dataset.text

    ctx_init_str = args.ctx_init.replace("_", " ")

    ret_edge = CLIPRet_TPT_I2T(
        edge_dev, captions, args.ret_edge, args.n_ctx, ctx_init_str)
    ret_edge = ret_edge.to(edge_dev)
    gc.collect(); torch.cuda.empty_cache()

    if do_ret:
        ret_cloud = CLIPRet_TPT_I2T(
            cloud_dev, captions, args.ret_cloud, args.n_ctx, ctx_init_str)
        if edge_dev == cloud_dev:
            ret_cloud.prompt_learner.token_prefix = \
                ret_edge.prompt_learner.token_prefix
            ret_cloud.prompt_learner.token_suffix = \
                ret_edge.prompt_learner.token_suffix
            ret_cloud.prompt_learner.tokenized_prompts = \
                ret_edge.prompt_learner.tokenized_prompts
        gc.collect()
        ret_cloud = ret_cloud.to(cloud_dev)
        gc.collect(); torch.cuda.empty_cache()

    # ============================================================
    #  CLS data & models  (ImageNet-V)
    # ============================================================
    classnames_all = imagenet_classes
    classnames = [classnames_all[i] for i in imagenet_v_mask]

    if do_cls:
        cls_cloud = get_coop(
            args.cls_cloud_arch, "V", args.cloud_gpu, args.n_ctx,
            args.ctx_init)
        cls_cloud.reset_classnames(classnames, args.cls_cloud_arch)
        gc.collect(); torch.cuda.empty_cache()

    cls_edge = get_coop(
        args.cls_arch, "V", args.gpu, args.n_ctx, args.ctx_init, type="Device")
    cls_edge.reset_classnames(classnames, args.cls_arch)
    gc.collect(); torch.cuda.empty_cache()

    # ============================================================
    #  SEG models  (MaskCLIP)
    # ============================================================
    seg_classnames = VOC_CLASSES
    if do_seg:
        print(f"Building SEG cloud model ...", flush=True)
        seg_cloud = MaskClipCEOModel(
            args.seg_cloud_config, args.seg_cloud_ckpt, "ViT-B/16",
            seg_classnames, args.n_ctx, args.ctx_init,
            clip_download_root=args.seg_clip_download_root,
        ).to(seg_dev)
        gc.collect(); torch.cuda.empty_cache()

    print(f"Building SEG edge model ...", flush=True)
    seg_edge = MaskClipCEOModel(
        args.seg_edge_config, args.seg_edge_ckpt, "RN50",
        seg_classnames, args.n_ctx, args.ctx_init,
        clip_download_root=args.seg_clip_download_root,
    ).to(edge_dev)
    gc.collect(); torch.cuda.empty_cache()

    # ============================================================
    #  Share CLIP backbones
    # ============================================================
    if do_cls and do_ret:
        share_backbone(cls_cloud, ret_cloud)
    share_backbone(cls_edge, ret_edge)
    share_seg_text_encoder(seg_edge, ret_edge)
    if do_seg and do_ret and seg_dev == cloud_dev:
        share_seg_text_encoder(seg_cloud, ret_cloud)
    gc.collect()
    for dev_id in {args.gpu, args.cloud_gpu, args.seg_gpu}:
        with torch.cuda.device(dev_id):
            torch.cuda.empty_cache()

    # ============================================================
    #  Share ctx among all three edge prompt learners
    # ============================================================
    shared_ctx = ret_edge.prompt_learner.ctx
    shared_ctx_init = shared_ctx.data.detach().clone()

    cls_ctx_dim = cls_edge.prompt_learner.ctx.shape[-1]
    ret_ctx_dim = shared_ctx.shape[-1]
    seg_ctx_dim = seg_edge.prompt_learner.ctx.shape[-1]
    assert cls_ctx_dim == ret_ctx_dim == seg_ctx_dim, (
        f"ctx_dim mismatch: CLS={cls_ctx_dim}, RET={ret_ctx_dim}, "
        f"SEG={seg_ctx_dim}")
    cls_edge.prompt_learner.ctx = shared_ctx
    seg_edge.prompt_learner.ctx = shared_ctx

    # ============================================================
    #  CLS data
    # ============================================================
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    resolution = 224
    base_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution)])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    data_transform = AugMixAugmenter(
        base_transform, preprocess,
        n_views=args.cls_batch_size - 1, augmix=False)
    device_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(), normalize])
    cls_dataset = build_dataset(
        "V", device_transform, data_transform, args.cls_data,
        mode="test", rec_dir=args.cls_rec_dir)
    cls_loader = torch.utils.data.DataLoader(
        cls_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True)

    # ============================================================
    #  SEG data  (MMSeg)
    # ============================================================
    seg_cfg = MMConfig.fromfile(args.seg_edge_config)
    seg_cfg.data.test.data_root = args.seg_data_root
    seg_dataset = build_seg_dataset(seg_cfg.data.test)
    seg_loader = build_seg_dataloader(
        seg_dataset, samples_per_gpu=1, workers_per_gpu=4,
        dist=False, shuffle=False)

    # ============================================================
    #  Optimizers & scalers
    # ============================================================
    if do_cls:
        cls_cloud_trainable = []
        for n, param in cls_cloud.named_parameters():
            if "prompt_learner" in n:
                param.requires_grad_(True)
                cls_cloud_trainable.append(param)
            else:
                param.requires_grad_(False)
        cls_cloud_opt = torch.optim.AdamW(cls_cloud_trainable, lr=args.lr)
        cls_cloud_init_optim = deepcopy(cls_cloud_opt.state_dict())
        cls_cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    for n, param in cls_edge.named_parameters():
        param.requires_grad_("prompt_learner" in n)

    if do_ret:
        ret_cloud_opt = torch.optim.AdamW(
            ret_cloud.parameters(), lr=args.lr)
        ret_cloud_init_optim = deepcopy(ret_cloud_opt.state_dict())
        ret_cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    if do_seg:
        seg_cloud_trainable = [seg_cloud.prompt_learner.ctx]
        for p in seg_cloud.text_encoder.parameters():
            p.requires_grad_(False)
        seg_cloud_opt = torch.optim.AdamW(
            seg_cloud_trainable, lr=args.lr)
        seg_cloud_init_optim = deepcopy(seg_cloud_opt.state_dict())
        seg_cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    edge_opt = torch.optim.AdamW([shared_ctx], lr=args.lr)
    edge_init_optim = deepcopy(edge_opt.state_dict())
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # ============================================================
    #  Evaluation accumulators
    # ============================================================
    cls_edge.eval(); ret_edge.eval(); seg_edge.eval()
    if do_cls: cls_cloud.eval()
    if do_ret: ret_cloud.eval()
    if do_seg: seg_cloud.eval()

    criterion_div = nn.KLDivLoss(reduction="batchmean")

    top1_correct = 0; top5_correct = 0; cls_total = 0

    num_images_ret = len(ret_loader.dataset.image)
    num_captions = len(captions)
    score_matrix_i2t = torch.full(
        (num_images_ret, num_captions), -100.0).to(edge_dev)

    seg_results = []

    # iterators
    cls_iter = iter(cls_loader)
    ret_iter = enumerate(ret_loader)
    seg_iter = enumerate(seg_loader)

    n_cls = len(cls_loader)
    n_ret = len(ret_loader)
    n_seg = len(seg_loader)
    n_samples = min(n_cls, n_ret, n_seg)
    if args.max_samples > 0:
        n_samples = min(n_samples, args.max_samples)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Joint3 CEO — {n_samples} samples "
          f"(CLS={n_cls}, RET={n_ret}, SEG={n_seg})", flush=True)
    print(f"  CLS: tau={args.cls_tau}, beta={args.cls_beta}", flush=True)
    print(f"  RET: tau={args.ret_tau}, beta={args.ret_beta}", flush=True)
    print(f"  SEG: tau={args.seg_tau}, beta={args.seg_beta}, "
          f"sel_p={args.seg_selection_p}", flush=True)
    print(f"  Joint: l_cls={args.lambda_cls}, l_ret={args.lambda_ret}, "
          f"l_seg={args.lambda_seg}, lr={args.lr}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    t_start = time.time()

    for step_i in tqdm(range(n_samples), desc="Joint3 CEO"):
        # ── fetch CLS sample ──
        try:
            cls_oris, cls_recs, cls_target = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_loader)
            cls_oris, cls_recs, cls_target = next(cls_iter)

        # ── fetch RET sample ──
        ret_i, ret_samples = next(ret_iter)

        # ── fetch SEG sample ──
        seg_i, seg_data = next(seg_iter)

        # ── prepare CLS tensors ──
        if isinstance(cls_oris, list):
            cls_ori = cls_oris[0].to(edge_dev, non_blocking=True)
        else:
            cls_ori = cls_oris.to(edge_dev, non_blocking=True)
        if isinstance(cls_recs, list):
            cls_recs = [r.to(edge_dev, non_blocking=True) for r in cls_recs]
            cls_recs_cat = torch.cat(cls_recs, dim=0)
        else:
            cls_recs_cat = cls_recs.to(edge_dev, non_blocking=True)
        cls_target = cls_target.to(edge_dev, non_blocking=True)

        # ── prepare RET tensors ──
        ret_image = ret_samples["image1"].to(edge_dev)
        ret_image_cloud = ret_samples["image2"].to(edge_dev)

        # ── prepare SEG tensors ──
        seg_imgs = seg_data['img']
        seg_metas = seg_data['img_metas']
        if isinstance(seg_imgs, list):
            seg_imgs = seg_imgs[0]
        if hasattr(seg_imgs, 'data'):
            seg_imgs = seg_imgs.data
        if isinstance(seg_imgs, list):
            seg_imgs = seg_imgs[0]
        seg_img = seg_imgs.to(edge_dev)

        if isinstance(seg_metas, list):
            seg_metas = seg_metas[0]
        if hasattr(seg_metas, 'data'):
            seg_metas = seg_metas.data
        while isinstance(seg_metas, list) and seg_metas \
                and isinstance(seg_metas[0], list):
            seg_metas = seg_metas[0]
        seg_img_metas = seg_metas

        # ── reset all prompts & optimizers ──
        with torch.no_grad():
            if do_cls: cls_cloud.reset()
            shared_ctx.data.copy_(shared_ctx_init)
            if do_ret: ret_cloud.reset_initial()
            if do_seg: seg_cloud.reset()

        if do_cls: cls_cloud_opt.load_state_dict(cls_cloud_init_optim)
        if do_ret: ret_cloud_opt.load_state_dict(ret_cloud_init_optim)
        if do_seg: seg_cloud_opt.load_state_dict(seg_cloud_init_optim)
        edge_opt.load_state_dict(edge_init_optim)

        # ── precompute frozen image features ──
        with torch.no_grad():
            if args.lambda_cls > 0:
                cls_cloud_feat = cls_cloud.get_image_features(
                    cls_recs_cat.to(cloud_dev))
                cls_edge_feat = cls_edge.get_image_features(cls_recs_cat)

            if args.lambda_ret > 0:
                ret_cloud_input = (ret_image if args.ret_cloud_orig
                                   else ret_image_cloud).to(cloud_dev)
                ret_cloud_img_feat = ret_cloud.get_image_feature(
                    ret_cloud_input)
                ret_edge_img_feat = ret_edge.get_image_feature(
                    ret_image_cloud)

                _, ret_topk_idx = ret_cloud.get_top_k(
                    ret_cloud_img_feat, args.ret_sim_top_k)
                ret_topk_idx_edge = (ret_topk_idx.to(edge_dev)
                                     if edge_dev != cloud_dev
                                     else ret_topk_idx)

        # ── TTA steps ──
        selected_idx = None
        for _ in range(args.tta_steps):

            # ──── (1) CLS cloud update ────
            if args.lambda_cls > 0:
                with torch.cuda.amp.autocast():
                    cls_cloud_out = cls_cloud(None, cls_cloud_feat)
                    if selected_idx is not None:
                        cls_cloud_sel = cls_cloud_out[selected_idx]
                    else:
                        cls_cloud_sel, selected_idx = \
                            select_confident_samples(
                                cls_cloud_out, args.cls_selection_p)
                    cls_cloud_loss = avg_entropy(cls_cloud_sel)
                cls_cloud_opt.zero_grad()
                cls_cloud_scaler.scale(cls_cloud_loss).backward()
                cls_cloud_scaler.step(cls_cloud_opt)
                cls_cloud_scaler.update()

                if step_i < 5 or (step_i + 1) % args.print_freq == 0:
                    print(f"  [loss {step_i}] CLS cloud: "
                          f"ent={cls_cloud_loss.item():.4f}", flush=True)

            # ──── (2) RET cloud update ────
            if args.lambda_ret > 0 and args.ret_ent_tea:
                with torch.cuda.amp.autocast():
                    ret_cloud_logits = ret_cloud(
                        ret_cloud_img_feat, ret_topk_idx)
                    ret_cloud_loss = topk_ent(
                        ret_cloud_logits, args.ret_ent_top_k)
                ret_cloud_opt.zero_grad()
                ret_cloud_scaler.scale(ret_cloud_loss).backward()
                ret_cloud_scaler.step(ret_cloud_opt)
                ret_cloud_scaler.update()

                if step_i < 5 or (step_i + 1) % args.print_freq == 0:
                    print(f"  [loss {step_i}] RET cloud: "
                          f"ent={ret_cloud_loss.item():.4f}", flush=True)

            # ──── (3) SEG cloud update ────
            if args.lambda_seg > 0:
                with torch.cuda.amp.autocast():
                    seg_cloud_avg_lp = seg_forward_with_flip(
                        seg_cloud, seg_img.to(seg_dev))
                    seg_cloud_loss = marginal_entropy_loss(
                        seg_cloud_avg_lp, args.seg_selection_p)
                seg_cloud_opt.zero_grad()
                seg_cloud_scaler.scale(seg_cloud_loss).backward()
                seg_cloud_scaler.step(seg_cloud_opt)
                seg_cloud_scaler.update()

                if step_i < 5 or (step_i + 1) % args.print_freq == 0:
                    print(f"  [loss {step_i}] SEG cloud: "
                          f"ent={seg_cloud_loss.item():.4f}", flush=True)

            # ──── Joint edge update (split backward) ────
            edge_opt.zero_grad()
            _any_edge = False

            # (a) CLS edge
            if args.lambda_cls > 0:
                _any_edge = True
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        cls_cloud_logits_d = cls_cloud(
                            None, cls_cloud_feat).to(edge_dev)
                    cls_edge_out = cls_edge(None, cls_edge_feat)
                    cls_edge_kl = args.cls_beta * criterion_div(
                        F.log_softmax(
                            cls_edge_out / args.cls_tau, dim=1),
                        F.softmax(
                            cls_cloud_logits_d / args.cls_tau, dim=1))
                    cls_edge_sel, selected_idx = \
                        select_confident_samples(
                            cls_edge_out, args.cls_selection_p)
                    cls_edge_ent = avg_entropy(cls_edge_sel)
                    cls_edge_loss = args.lambda_cls * (
                        cls_edge_ent + cls_edge_kl)
                edge_scaler.scale(cls_edge_loss).backward()

                if step_i < 5 or (step_i + 1) % args.print_freq == 0:
                    print(f"  [loss {step_i}] CLS edge: "
                          f"ent={cls_edge_ent.item():.4f}  "
                          f"kl={cls_edge_kl.item():.4f}  "
                          f"total={cls_edge_loss.item():.4f}",
                          flush=True)
                del cls_edge_out, cls_edge_kl, cls_edge_sel
                del cls_edge_ent, cls_edge_loss, cls_cloud_logits_d

            # (b) RET edge
            if args.lambda_ret > 0:
                _any_edge = True
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        ret_cloud_logits_d = ret_cloud(
                            ret_cloud_img_feat,
                            ret_topk_idx).to(edge_dev)
                    ret_edge_logits = ret_edge(
                        ret_edge_img_feat, ret_topk_idx_edge)
                    ret_edge_ent = topk_ent(
                        ret_edge_logits, args.ret_ent_top_k)
                    ret_edge_kl = F.kl_div(
                        F.log_softmax(
                            ret_edge_logits / args.ret_tau, dim=1),
                        F.softmax(
                            ret_cloud_logits_d / args.ret_tau, dim=1),
                        reduction="batchmean")
                    if args.ret_ent_stu:
                        ret_edge_loss = args.lambda_ret * (
                            ret_edge_ent
                            + args.ret_beta * ret_edge_kl)
                    else:
                        ret_edge_loss = args.lambda_ret * (
                            args.ret_beta * ret_edge_kl)
                edge_scaler.scale(ret_edge_loss).backward()

                if step_i < 5 or (step_i + 1) % args.print_freq == 0:
                    print(f"  [loss {step_i}] RET edge: "
                          f"ent={ret_edge_ent.item():.4f}  "
                          f"kl={ret_edge_kl.item():.4f}  "
                          f"total={ret_edge_loss.item():.4f}",
                          flush=True)
                del ret_edge_logits, ret_edge_ent, ret_edge_kl
                del ret_edge_loss, ret_cloud_logits_d

            # (c) SEG edge
            if args.lambda_seg > 0:
                _any_edge = True
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        seg_cloud_avg_lp = seg_forward_with_flip(
                            seg_cloud, seg_img.to(seg_dev))

                    seg_edge_avg_lp = seg_forward_with_flip(
                        seg_edge, seg_img)

                    edge_size = seg_edge_avg_lp.shape[2:]
                    cloud_lp = resize(
                        seg_cloud_avg_lp.to(edge_dev),
                        size=edge_size,
                        mode='bilinear', align_corners=False)
                    cloud_lp = cloud_lp - cloud_lp.logsumexp(
                        dim=1, keepdim=True)

                    edge_lp_t = F.log_softmax(
                        seg_edge_avg_lp / args.seg_tau, dim=1)
                    cloud_p_t = F.softmax(
                        cloud_lp / args.seg_tau, dim=1)
                    kl_per_pixel = F.kl_div(
                        edge_lp_t, cloud_p_t,
                        reduction='none').sum(dim=1)

                    if args.seg_selective_kl:
                        cloud_ent = -(
                            cloud_lp.exp() * cloud_lp).sum(dim=1)
                        flat_ent = cloud_ent.view(
                            cloud_ent.shape[0], -1)
                        k_sel = max(1, int(
                            flat_ent.shape[1] * args.seg_cloud_conf_p))
                        _, sel_idx = torch.topk(
                            flat_ent, k_sel, dim=1, largest=False)
                        kl_flat = kl_per_pixel.view(
                            kl_per_pixel.shape[0], -1)
                        sel_kl = kl_flat.gather(1, sel_idx)
                        seg_kl_loss = args.seg_beta * (
                            args.seg_tau ** 2) * sel_kl.mean()
                    else:
                        seg_kl_loss = args.seg_beta * (
                            args.seg_tau ** 2) * kl_per_pixel.mean()

                    seg_ent_loss = marginal_entropy_loss(
                        seg_edge_avg_lp, args.seg_selection_p)
                    seg_edge_loss = args.lambda_seg * (
                        seg_ent_loss + seg_kl_loss)

                edge_scaler.scale(seg_edge_loss).backward()

                if step_i < 5 or (step_i + 1) % args.print_freq == 0:
                    print(f"  [loss {step_i}] SEG edge: "
                          f"ent={seg_ent_loss.item():.4f}  "
                          f"kl={seg_kl_loss.item():.4f}  "
                          f"total={seg_edge_loss.item():.4f}",
                          flush=True)
                del seg_cloud_avg_lp, seg_edge_avg_lp
                del seg_ent_loss, seg_kl_loss, seg_edge_loss

            if _any_edge:
                edge_scaler.step(edge_opt)
                edge_scaler.update()

        # ── CLS test ──
        with torch.no_grad():
            cls_test_feat = cls_edge.get_image_features(cls_ori)
            with torch.cuda.amp.autocast():
                cls_output = cls_edge(None, cls_test_feat)
        _, pred = cls_output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(cls_target.view(1, -1).expand_as(pred))
        top1_correct += correct[:1].reshape(-1).float().sum(0).item()
        top5_correct += correct[:5].reshape(-1).float().sum(0).item()
        cls_total += cls_target.size(0)

        # ── RET test ──
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                ret_test_feat = ret_edge.get_image_feature(ret_image)
                logits = ret_edge(ret_test_feat)
                score_matrix_i2t[ret_i] = logits[0]

        # ── SEG test ──
        with torch.no_grad():
            seg_pred = seg_edge.predict(seg_img, seg_img_metas)
        seg_pre_eval = seg_dataset.pre_eval(seg_pred, indices=[seg_i])
        seg_results.extend(seg_pre_eval)

        if (step_i + 1) % args.print_freq == 0:
            elapsed = time.time() - t_start
            print(
                f"  [{step_i + 1}/{n_samples}]  "
                f"CLS Top1={100.0 * top1_correct / cls_total:.2f}%  "
                f"({elapsed:.0f}s)", flush=True)

    # ============================================================
    #  Final results
    # ============================================================
    cls_top1 = round(100.0 * top1_correct / cls_total, 2)
    cls_top5 = round(100.0 * top5_correct / cls_total, 2)

    scores_cpu = score_matrix_i2t[:n_samples].detach().cpu()
    img2txt = ret_loader.dataset.img2txt
    ret_results = compute_i2t_recall(scores_cpu, img2txt)

    seg_eval = seg_dataset.evaluate(seg_results, metric='mIoU')

    results = {
        "cls_top1": cls_top1,
        "cls_top5": cls_top5,
        "cls_samples": cls_total,
        **ret_results,
        "ret_samples": n_samples,
        "seg_mIoU": round(seg_eval.get('mIoU', 0) * 100, 2),
        "seg_aAcc": round(seg_eval.get('aAcc', 0) * 100, 2),
        "seg_samples": n_samples,
        "lambda_cls": args.lambda_cls,
        "lambda_ret": args.lambda_ret,
        "lambda_seg": args.lambda_seg,
    }

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"  CLS (ImageNet-V): Top1={cls_top1}%  Top5={cls_top5}%  "
          f"({cls_total} samples)", flush=True)
    print(f"  RET (I2T):        {ret_results}  "
          f"({n_samples} samples)", flush=True)
    print(f"  SEG (VOC):        mIoU={results['seg_mIoU']}%  "
          f"aAcc={results['seg_aAcc']}%  "
          f"({n_samples} samples)", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)

    out_file = os.path.join(args.output, "results_joint3.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
