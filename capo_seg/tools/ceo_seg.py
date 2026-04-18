"""
CEO-style edge-cloud test-time prompt tuning for MaskCLIP segmentation.

Cloud model (ViT-B/16) optimizes prompt via pixel-level entropy minimization.
Edge model (RN50) distills from cloud via KL divergence + entropy minimization.
Final prediction is made by the edge model.

Usage:
    cd MaskCLIP
    python tools/ceo_seg.py \
        --cloud-config configs/maskclip/maskclip_vit16_512x512_ade20k.py \
        --cloud-ckpt pretrain/ViT16_clip_backbone.pth \
        --edge-config configs/maskclip/maskclip_r50_512x512_ade20k.py \
        --edge-ckpt pretrain/RN50_clip_backbone.pth \
        --cloud-arch ViT-B/16 --edge-arch RN50 --dataset ade
"""

import argparse
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.ops import resize

import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer

_tokenizer = SimpleTokenizer()

# ============================================================
# Segmentation class names (must match prompt_engineering.py)
# ============================================================

ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
    'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
    'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
    'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
    'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
    'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
    'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
    'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
    'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
    'clock', 'flag',
]

VOC_CLASSES = [
    'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv monitor',
]

DATASET_CLASSES = {
    'ade': ADE20K_CLASSES,
    'voc': VOC_CLASSES,
}


# ============================================================
# Text Pipeline (adapted from CAPO_cls/clip/custom_clip_device.py)
# ============================================================

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.positional_embedding.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        if isinstance(self.text_projection, nn.Parameter):
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        else:
            x = self.text_projection(
                x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)])
        return x


class PromptLearner(nn.Module):
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
            prompt = tokenize(ctx_init).to(self.device)
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

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])   # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

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


# ============================================================
# MaskClipCEOModel: MaskCLIP visual pipeline + CLIP prompt-tuned text
# ============================================================

class MaskClipCEOModel(nn.Module):
    def __init__(self, seg_cfg_path, seg_ckpt_path, clip_arch, classnames,
                 n_ctx=4, ctx_init=None, clip_download_root=None):
        super().__init__()

        # --- Visual pipeline from MaskCLIP (frozen) ---
        cfg = Config.fromfile(seg_cfg_path)
        cfg.model.pretrained = None
        seg_model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(seg_model, seg_ckpt_path, map_location='cpu', strict=False)
        seg_model.eval()

        self.backbone = seg_model.backbone
        self.decode_head = seg_model.decode_head
        self.vit = getattr(seg_model.decode_head, 'vit', False)
        self.align_corners = seg_model.align_corners

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.decode_head.parameters():
            p.requires_grad_(False)

        # --- Text pipeline from CLIP (prompt learner is trainable) ---
        dl_root = clip_download_root or os.path.expanduser('~/.cache/clip')
        clip_model, _ = clip.load(clip_arch, device='cpu', download_root=dl_root)

        self.text_encoder = TextEncoder(clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx=n_ctx, ctx_init=ctx_init)

        self.logit_scale = clip_model.logit_scale.exp().detach()

        del seg_model, clip_model
        torch.cuda.empty_cache()

    def get_text_features(self):
        """Dynamic text features from prompt learner (differentiable)."""
        prompts = self.prompt_learner()
        tokenized = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features  # [C, text_ch]

    def get_visual_feat(self, img):
        """Spatial visual features from MaskCLIP backbone + head projection."""
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
        return feat  # [B, text_ch, H', W']

    def forward(self, img):
        feat = self.get_visual_feat(img)
        text_features = self.get_text_features()
        text_features = text_features.to(feat.dtype)
        logits = self.logit_scale * F.conv2d(feat, text_features[:, :, None, None])
        return logits  # [B, C, H', W']

    def predict(self, img, img_metas, use_flip=True):
        """Forward + resize to original image size for evaluation."""
        if use_flip:
            avg_lp = forward_with_flip(self, img)          # log-avg-probs
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


# ============================================================
# Flip-augmented forward & marginal entropy (aligned with CEO)
# ============================================================

def forward_with_flip(model, img):
    """Forward original + horizontal flip, average in probability space.

    Returns log-average-probabilities so that downstream entropy / KL
    computations stay numerically stable (matches CEO's logsumexp trick).
    """
    logits_orig = model(img)                        # [B, C, H, W]
    logits_flip = model(img.flip(-1)).flip(-1)      # predict on flip, flip back

    log_p1 = F.log_softmax(logits_orig, dim=1)
    log_p2 = F.log_softmax(logits_flip, dim=1)
    # log( (p1 + p2) / 2 )  =  logsumexp(log_p1, log_p2) - log(2)
    avg_log_probs = torch.stack([log_p1, log_p2], dim=0).logsumexp(dim=0) \
                    - np.log(2.0)
    return avg_log_probs                            # [B, C, H, W]


def marginal_entropy_loss(avg_log_probs, selection_p):
    """Entropy of averaged predictions, computed over the most confident pixels.

    This mirrors CEO's avg_entropy + select_confident_samples, adapted to
    dense prediction: each pixel is treated like a "sample".
    """
    avg_probs = avg_log_probs.exp()
    pixel_entropy = -(avg_probs * avg_log_probs).sum(dim=1)   # [B, H, W]

    B, C = avg_log_probs.shape[:2]
    flat_ent = pixel_entropy.view(B, -1)                       # [B, HW]
    k = max(1, int(flat_ent.shape[1] * selection_p))
    _, idx = torch.topk(flat_ent, k, dim=1, largest=False)    # lowest entropy

    probs_flat = avg_probs.view(B, C, -1)
    logp_flat  = avg_log_probs.view(B, C, -1)
    idx_exp = idx.unsqueeze(1).expand(-1, C, -1)
    sel_p  = probs_flat.gather(2, idx_exp)
    sel_lp = logp_flat.gather(2, idx_exp)
    return -(sel_p * sel_lp).sum(dim=1).mean()


# ============================================================
# Test-Time Tuning (per image)
# ============================================================

def test_time_tuning(cloud_model, edge_model, img,
                     optimizer_cloud, optimizer_edge,
                     cloud_scaler, edge_scaler, args):
    # Pre-compute static cloud prediction if no_cloud_tta
    if args.no_cloud_tta and args.beta > 0:
        with torch.no_grad(), torch.cuda.amp.autocast():
            static_cloud_lp = forward_with_flip(cloud_model, img)

    for _step in range(args.tta_steps):
        # --- Step A: Update cloud prompt via marginal entropy ---
        if not args.no_cloud_tta:
            with torch.cuda.amp.autocast():
                cloud_avg_lp = forward_with_flip(cloud_model, img)
                loss_cloud = marginal_entropy_loss(cloud_avg_lp, args.selection_p)

            optimizer_cloud.zero_grad()
            cloud_scaler.scale(loss_cloud).backward()
            cloud_scaler.step(optimizer_cloud)
            cloud_scaler.update()

        # --- Step B: Update edge prompt via entropy + KL from cloud ---
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if args.no_cloud_tta:
                    cloud_avg_lp = static_cloud_lp
                else:
                    cloud_avg_lp = forward_with_flip(cloud_model, img)

            edge_avg_lp = forward_with_flip(edge_model, img)

            if args.kl_at_cloud_res:
                # Upsample edge to cloud resolution for KL
                cloud_size = cloud_avg_lp.shape[2:]
                edge_lp_for_kl = resize(
                    edge_avg_lp, size=cloud_size,
                    mode='bilinear', align_corners=False)
                edge_lp_for_kl = edge_lp_for_kl - edge_lp_for_kl.logsumexp(
                    dim=1, keepdim=True)
                cloud_lp_for_kl = cloud_avg_lp
            else:
                # Downsample cloud to edge resolution for KL
                edge_size = edge_avg_lp.shape[2:]
                cloud_lp_for_kl = resize(
                    cloud_avg_lp, size=edge_size,
                    mode='bilinear', align_corners=False)
                cloud_lp_for_kl = cloud_lp_for_kl - cloud_lp_for_kl.logsumexp(
                    dim=1, keepdim=True)
                edge_lp_for_kl = edge_avg_lp

            # Temperature-scaled KL: sum over C, per-pixel
            edge_lp_t = F.log_softmax(edge_lp_for_kl / args.tau, dim=1)
            cloud_p_t = F.softmax(cloud_lp_for_kl / args.tau, dim=1)
            kl_per_pixel = F.kl_div(
                edge_lp_t, cloud_p_t, reduction='none').sum(dim=1)  # [B, H, W]

            if args.selective_kl:
                # Only KL on cloud's most-confident pixels
                cloud_ent = -(cloud_lp_for_kl.exp() * cloud_lp_for_kl).sum(dim=1)
                flat_ent = cloud_ent.view(cloud_ent.shape[0], -1)
                k = max(1, int(flat_ent.shape[1] * args.cloud_conf_p))
                _, idx = torch.topk(flat_ent, k, dim=1, largest=False)
                kl_flat = kl_per_pixel.view(kl_per_pixel.shape[0], -1)
                sel_kl = kl_flat.gather(1, idx)
                loss_kl = args.beta * (args.tau ** 2) * sel_kl.mean()
            else:
                loss_kl = args.beta * (args.tau ** 2) * kl_per_pixel.mean()

            loss_ent = marginal_entropy_loss(edge_avg_lp, args.selection_p)
            loss_edge = loss_ent + loss_kl

        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_edge).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()


# ============================================================
# Main Evaluation Loop
# ============================================================

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    set_random_seed(args.seed)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    cudnn.benchmark = True

    classnames = DATASET_CLASSES[args.dataset]

    # --- Build models ---
    print(f'Building cloud model ({args.cloud_arch}) ...')
    cloud_model = MaskClipCEOModel(
        args.cloud_config, args.cloud_ckpt, args.cloud_arch,
        classnames, args.n_ctx, args.ctx_init,
        clip_download_root=args.clip_download_root,
    ).to(device)

    print(f'Building edge model ({args.edge_arch}) ...')
    edge_model = MaskClipCEOModel(
        args.edge_config, args.edge_ckpt, args.edge_arch,
        classnames, args.n_ctx, args.ctx_init,
        clip_download_root=args.clip_download_root,
    ).to(device)

    # --- Optimizers (only prompt_learner.ctx is trainable) ---
    cloud_trainable = [p for p in cloud_model.parameters() if p.requires_grad]
    edge_trainable = [p for p in edge_model.parameters() if p.requires_grad]
    print(f'Cloud trainable params: {sum(p.numel() for p in cloud_trainable)}')
    print(f'Edge  trainable params: {sum(p.numel() for p in edge_trainable)}')

    optimizer_cloud = optim.AdamW(cloud_trainable, args.lr)
    optimizer_edge = optim.AdamW(edge_trainable, args.lr)
    cloud_optim_state = deepcopy(optimizer_cloud.state_dict())
    edge_optim_state = deepcopy(optimizer_edge.state_dict())

    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # --- Dataset & Loader ---
    cfg = Config.fromfile(args.edge_config)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
        dist=False, shuffle=False)

    print(f'Dataset: {len(dataset)} images, {len(classnames)} classes')
    print(f'TTA: steps={args.tta_steps}, lr={args.lr}, '
          f'selection_p={args.selection_p}, tau={args.tau}, beta={args.beta}')

    cloud_model.eval()
    edge_model.eval()

    results = []
    total_time = 0.0

    for i, data in enumerate(loader):
        # Unwrap nested list / DataContainer from MMSeg loader
        imgs = data['img']
        metas = data['img_metas']
        if isinstance(imgs, list):
            imgs = imgs[0]
        if hasattr(imgs, 'data'):
            imgs = imgs.data
        if isinstance(imgs, list):
            imgs = imgs[0]
        img = imgs.to(device)

        if isinstance(metas, list):
            metas = metas[0]
        if hasattr(metas, 'data'):
            metas = metas.data
        # Flatten nested lists until we reach list-of-dicts
        while isinstance(metas, list) and metas and isinstance(metas[0], list):
            metas = metas[0]
        img_metas = metas

        start = time.time()

        # Reset prompts & optimizers per image
        with torch.no_grad():
            cloud_model.reset()
            edge_model.reset()
        optimizer_cloud.load_state_dict(cloud_optim_state)
        optimizer_edge.load_state_dict(edge_optim_state)

        # TTA
        test_time_tuning(cloud_model, edge_model, img,
                         optimizer_cloud, optimizer_edge,
                         cloud_scaler, edge_scaler, args)

        # Final prediction from edge model
        with torch.no_grad():
            seg_pred = edge_model.predict(img, img_metas)

        elapsed = time.time() - start
        total_time += elapsed

        pre_eval_result = dataset.pre_eval(seg_pred, indices=[i])
        results.extend(pre_eval_result)

        if (i + 1) % args.print_freq == 0:
            avg_t = total_time / (i + 1)
            eta = avg_t * (len(loader) - i - 1) / 60.0
            print(f'[{i + 1}/{len(loader)}]  {elapsed:.2f}s/img  '
                  f'avg {avg_t:.2f}s/img  ETA {eta:.1f}min')

    # --- Compute metrics ---
    eval_results = dataset.evaluate(results, metric='mIoU')
    print('\n========== Results ==========')
    for k, v in eval_results.items():
        print(f'  {k}: {v:.4f}')

    return eval_results


def parse_args():
    p = argparse.ArgumentParser(
        description='CEO-style test-time prompt tuning for MaskCLIP segmentation')

    p.add_argument('--cloud-config', required=True, help='MMSeg config for cloud model')
    p.add_argument('--cloud-ckpt', required=True, help='Backbone checkpoint for cloud model')
    p.add_argument('--edge-config', required=True, help='MMSeg config for edge model')
    p.add_argument('--edge-ckpt', required=True, help='Backbone checkpoint for edge model')
    p.add_argument('--cloud-arch', default='ViT-B/16', help='CLIP arch name for cloud text encoder')
    p.add_argument('--edge-arch', default='RN50', help='CLIP arch name for edge text encoder')
    p.add_argument('--dataset', default='ade', choices=['ade', 'voc'],
                   help='Dataset class-name set')

    p.add_argument('--n_ctx', default=4, type=int, help='Number of learnable prompt tokens')
    p.add_argument('--ctx_init', default=None, type=str, help='Prompt initialization words')
    p.add_argument('--tta_steps', default=1, type=int, help='TTA optimisation steps per image')
    p.add_argument('--lr', default=5e-3, type=float, help='Prompt learning rate')
    p.add_argument('--selection_p', default=0.1, type=float,
                   help='Fraction of confident pixels for entropy loss')
    p.add_argument('--tau', default=0.4, type=float, help='Temperature for KL divergence')
    p.add_argument('--beta', default=0.9, type=float, help='Weight of KL loss')

    p.add_argument('--no_cloud_tta', action='store_true',
                   help='Skip cloud entropy TTA; use initial zero-shot as teacher')
    p.add_argument('--kl_at_cloud_res', action='store_true',
                   help='Compute KL at cloud resolution (upsample edge) instead of downsampling cloud')
    p.add_argument('--selective_kl', action='store_true',
                   help='Apply KL only on pixels where cloud is confident')
    p.add_argument('--cloud_conf_p', default=0.3, type=float,
                   help='Fraction of most-confident cloud pixels for selective KL')

    p.add_argument('--gpu', default=0, type=int, help='GPU id')
    p.add_argument('--workers', default=8, type=int, help='Data loader workers')
    p.add_argument('--print-freq', default=100, type=int, help='Print frequency')
    p.add_argument('--seed', default=0, type=int, help='Random seed')
    p.add_argument('--clip-download-root', default=None, type=str,
                   help='CLIP weight cache directory (default ~/.cache/clip)')

    return p.parse_args()


if __name__ == '__main__':
    main()
