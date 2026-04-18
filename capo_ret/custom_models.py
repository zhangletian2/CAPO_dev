# coding=utf-8
import os
import math
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lavis.models.clip_models.tokenizer import tokenize
from lavis.models.clip_models.model import load_openai_model


# CLIP weights cache location. Override via env var RET_CLIP_CACHE (set in .env
# and exported by scripts/_paths.sh), or fall back to <repo>/capo_ret/pretrained/clip
# which is a symlink populated by scripts/setup.sh.
DOWNLOAD_ROOT = os.environ.get(
    'RET_CLIP_CACHE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained', 'clip')
)
DOWNLOAD_ROOT = os.path.expanduser(DOWNLOAD_ROOT)
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)


class MultiplePromptLearnerI2T(nn.Module):
    def __init__(self, clip_model, captions, n_ctx=4, ctx_init="a photo of", num_p=1):
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.num_p = num_p

        self.dtype  = clip_model.ln_final.weight.dtype
        self.device = clip_model.ln_final.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 1) 初始化 context（与单 prompt 同）
        with torch.no_grad():
            toks = tokenize(ctx_init).to(self.device)
            emb  = clip_model.token_embedding(toks).to(self.dtype)
        ctx_vec = emb[0, 1:1+n_ctx, :]                                # (n_ctx,D)
        if num_p > 1:
            ctx_vec = ctx_vec.unsqueeze(0).repeat(num_p,1,1)           # (P,n_ctx,D)
        else:
            ctx_vec = ctx_vec.unsqueeze(0)                              # (1,n_ctx,D)
        self.ctx = nn.Parameter(ctx_vec)                                # (P,n_ctx,D)
        self.ctx_init_state = self.ctx.detach().clone()

        # 2) 预构造所有 caption 的 prefix/suffix/tokenized（一次性，和单 prompt 同）
        prompts_all = [ctx_init + " " + cap.strip().rstrip('.') + "." for cap in captions]
        tokenized = torch.cat([tokenize(p) for p in prompts_all]).to(self.device)  # (N,77)
        with torch.no_grad():
            emb_all = clip_model.token_embedding(tokenized).to(self.dtype)         # (N,77,D)

        self.register_buffer("token_prefix", emb_all[:, :1, :])        # (N,1,D)
        self.register_buffer("token_suffix", emb_all[:, 1+n_ctx:, :])  # (N,77-1-n_ctx,D)
        self.register_buffer("tokenized_all", tokenized)               # (N,77)

        # prompt 调度
        self.ctx_order = list(range(num_p))
        self.ctx_use   = [0]*num_p

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)

    def build_batch(self, caption_idx, selected_p_idx=None):
        """
        caption_idx: LongTensor[K]
        selected_p_idx: Optional[LongTensor[M]]；若为 None 表示用全部 P
        return:
          prompt_emb: (K*M, 77, D)
          tokenized : (K*M, 77)
        """
        if selected_p_idx is None:
            selected_p_idx = torch.arange(self.ctx.size(0), device=self.device)

        K = caption_idx.numel()
        M = selected_p_idx.numel()
        D = self.token_prefix.size(-1)
        SufL = self.token_suffix.size(1)

        # 取 prefix/suffix/tokenized（只做索引，不做新的 tokenize/embedding）
        prefix = self.token_prefix.index_select(0, caption_idx)             # (K,1,D)
        suffix = self.token_suffix.index_select(0, caption_idx)             # (K,SufL,D)
        toks   = self.tokenized_all.index_select(0, caption_idx)            # (K,77)

        # 扩展到每个 prompt
        prefix = prefix.unsqueeze(1).expand(K, M, 1, D).reshape(K*M, 1, D)
        suffix = suffix.unsqueeze(1).expand(K, M, SufL, D).reshape(K*M, SufL, D)
        toks   = toks.unsqueeze(1).expand(K, M, -1).reshape(K*M, -1)        # (K*M,77)

        # 选择 ctx 并平铺
        ctx_sel = self.ctx.index_select(0, selected_p_idx)                  # (M,n_ctx,D)
        ctx_sel = ctx_sel.unsqueeze(0).expand(K, M, -1, -1)                 # (K,M,n_ctx,D)
        ctx_sel = ctx_sel.reshape(K*M, self.n_ctx, D)                       # (K*M,n_ctx,D)

        prompt_emb = torch.cat([prefix, ctx_sel, suffix], dim=1)            # (K*M,77,D)
        return prompt_emb, toks



class CLIPRet_Dynap_I2T(nn.Module):
    def __init__(self, device, captions, arch="ViT-B-16", n_ctx=4, ctx_init="a photo of", num_p=1):
        super().__init__()
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False).float()
        self.device = device

        self.prompt_learner = MultiplePromptLearnerI2T(
            self.clip_model, captions, n_ctx=n_ctx, ctx_init=ctx_init, num_p=num_p
        )
        self.num_p = num_p
        self.dtype = self.clip_model.ln_final.weight.dtype

        # zero-shot 文本特征用于 Top-K caption 选择
        self._build_zero_shot_text_features(captions)
        self.freeze_parameters()

    def freeze_parameters(self):
        for _, p in self.clip_model.named_parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _build_zero_shot_text_features(self, captions, bsz=256):
        feats = []
        for i in range(0, len(captions), bsz):
            tok = torch.cat([tokenize(t) for t in captions[i:i+bsz]]).to(self.device)
            f = self.clip_model.encode_text(tok)
            feats.append(f)
        self.text_features_zs = F.normalize(torch.cat(feats, 0), dim=-1)  # (N,D)

    @torch.no_grad()
    def get_image_feature(self, image):
        f = self.clip_model.encode_image(image)
        return F.normalize(f, dim=-1)  # (1,D)

    @torch.no_grad()
    def get_top_k_caption_idx(self, image_feature, k):
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (image_feature @ self.text_features_zs.t())
        idx = logits[0].topk(k=k, dim=-1).indices
        return logits, idx

    def _encode_text_with_prompts(self, caption_idx, selected_p_idx, use_grad):
        prompt_emb, tokenized = self.prompt_learner.build_batch(caption_idx, selected_p_idx)
        orig = self.clip_model.token_embedding.forward
        def fake_forward(_): return prompt_emb
        self.clip_model.token_embedding.forward = fake_forward
        try:
            if use_grad:
                feat = self.clip_model.encode_text(tokenized.to(self.device))
            else:
                with torch.no_grad():
                    feat = self.clip_model.encode_text(tokenized.to(self.device))
        finally:
            self.clip_model.token_embedding.forward = orig
        feat = F.normalize(feat, dim=-1)  # (K*M, D)
        K = caption_idx.numel()
        M = selected_p_idx.numel()
        return feat.view(K, M, -1)        # (K,M,D)

    def forward_allP_for_selection(self, image_feature, caption_idx):
        # 选择阶段：no_grad 跑全 P，拿到 (K,P) logits
        P_idx = torch.arange(self.num_p, device=self.device)
        text_feat = self._encode_text_with_prompts(caption_idx, P_idx, use_grad=False)  # (K,P,D)
        logits = self.clip_model.logit_scale.exp() * (text_feat @ image_feature[0].unsqueeze(-1)).squeeze(-1)  # (K,P)
        return logits  # (K,P)

    def forward_selectedP_for_loss(self, image_feature, caption_idx, selected_p_idx):
        # 优化阶段：只跑被选 M 个 prompts，返回 (K,M)
        text_feat = self._encode_text_with_prompts(caption_idx, selected_p_idx, use_grad=True)  # (K,M,D)
        logits = self.clip_model.logit_scale.exp() * (text_feat @ image_feature[0].unsqueeze(-1)).squeeze(-1)  # (K,M)
        return logits  # (K,M)

    @torch.no_grad()
    def infer_all_captions(self, image_feature, selected_p_idx=None, batch=512):
        # 最终推理：分批 + prompts 均值
        if selected_p_idx is None:
            selected_p_idx = torch.arange(self.num_p, device=self.device)
        N = self.text_features_zs.size(0)  # 也可重新编码全量（带 ctx）
        out = []
        for s in range(0, N, batch):
            idx = torch.arange(s, min(s+batch, N), device=self.device)
            text_feat = self._encode_text_with_prompts(idx, selected_p_idx, use_grad=False)  # (K,M,D)
            logits = self.clip_model.logit_scale.exp() * (text_feat @ image_feature[0].unsqueeze(-1)).squeeze(-1)  # (K,M)
            out.append(logits.mean(dim=1))  # (K,)
        return torch.cat(out, 0)  # (N,)


class MultiplePromptLearnerT2I(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init=None, num_p=1):
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.num_p = num_p

        # 跟随 CLIP 的 dtype/device
        self.dtype = clip_model.ln_final.weight.dtype
        self.device = clip_model.ln_final.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 初始化 soft prompt（context tokens）
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            with torch.no_grad():
                tokens = tokenize(ctx_init).to(self.device)
                embedding = clip_model.token_embedding(tokens).to(self.dtype)  # (1, 77, D)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]  # (n_ctx, D)
            # 用这个字符串作为每个 prompt 的前缀文本（不按字符索引）
            self.prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(ctx_vectors, std=0.02)
            # 把 n_ctx 个占位 token 用空格连接成字符串
            self.prompt_prefix = " ".join(["X"] * n_ctx)

        # 多 prompt 场景：把 context 复制 num_p 份 -> (num_p, n_ctx, D)
        if num_p > 1:
            ctx_vectors = ctx_vectors.unsqueeze(0).repeat(num_p, 1, 1)
        else:
            ctx_vectors = ctx_vectors.unsqueeze(0)  # (1, n_ctx, D)

        self.ctx = nn.Parameter(ctx_vectors)  # (num_p, n_ctx, D)
        self.ctx_init_state = self.ctx.detach().clone()

        # 你原先的这两个变量先保留
        self.ctx_order = list(range(num_p))
        self.ctx_use = [0] * num_p

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)

    def _build_prompt_list(self, text_list):
        """把每条文本扩展成 num_p 条，返回扁平的 list[str]，长度 B*num_p。"""
        flat_prompts = []
        for t in text_list:
            t = t.strip().rstrip(".")
            # 这里默认每个 prompt 文本前缀都一样（self.prompt_prefix）
            # 若你未来需要不同前缀，可改成 per-p 的前缀数组
            for _p in range(self.num_p):
                flat_prompts.append(f"{self.prompt_prefix} {t}.")
        return flat_prompts  # len = B*num_p

    def forward(self, text):
        # 统一成 list[str]
        if isinstance(text, str):
            text = [text]
        B = len(text)
        # 1) 展平：B*num_p 个字符串
        flat_prompts = self._build_prompt_list(text)              # len = B*num_p
        tokenized = tokenize(flat_prompts).to(self.device)        # (B*num_p, 77)
        # 2) token embedding
        with torch.no_grad():
            x = self.clip_model.token_embedding(tokenized).to(self.dtype)  # (B*num_p, 77, D)
        # 3) 替换前 n_ctx 的位置（位置 1..n_ctx）
        # 为每个样本选取对应的 prompt id：0..num_p-1 周期重复
        idx = torch.arange(B * self.num_p, device=self.device) % self.num_p  # (B*num_p,)
        # ctx_selected: (B*num_p, n_ctx, D)
        ctx_selected = self.ctx.index_select(0, idx)

        x[:, 1:1 + self.n_ctx, :] = ctx_selected

        return x, tokenized


class CLIPRet_Dynap_T2I(nn.Module):
    def __init__(self, device, arch="ViT-B-16", n_ctx=4, ctx_init="a photo of",
                 only_visual=True, num_p=1):
        super().__init__()
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False)
        self.clip_model.float()
        self.device = device

        self.prompt_learner = MultiplePromptLearnerT2I(
            self.clip_model, n_ctx=n_ctx, ctx_init=ctx_init, num_p=num_p
        )
        self.dtype = self.prompt_learner.dtype
        self.only_visual = only_visual

        self.image_features = None
        self.num_p = num_p

        self.freeze_parameters()

    def freeze_parameters(self):
        for _, p in self.clip_model.named_parameters():
            p.requires_grad = False

    def encode_text_with_prompt(self, text):
        """
        输入: text 为 str 或 List[str]，长度 B
        输出: text_features -> (B, num_p, D)，逐向量已 L2 归一化
        """
        # 准备 (B*num_p, 77, D) 的 prompt_emb 和 (B*num_p, 77) 的 tokenized
        prompt_emb, tokenized = self.prompt_learner(text)
        B = len(text) if isinstance(text, (list, tuple)) else 1

        # 注入 embedding
        orig_forward = self.clip_model.token_embedding.forward
        def fake_forward(_):
            return prompt_emb
        self.clip_model.token_embedding.forward = fake_forward

        try:
            feat = self.clip_model.encode_text(tokenized.to(self.device))  # (B*num_p, D)
        finally:
            self.clip_model.token_embedding.forward = orig_forward

        feat = F.normalize(feat, dim=-1)
        D = feat.shape[-1]
        feat = feat.view(B, self.num_p, D)  # 不聚合，保留每个 prompt 的特征
        return feat  # (B, num_p, D)

    def forward(self, text, selected_idx=None):
        """
        text: str or List[str]
        selected_idx: Optional[LongTensor] 形如 [k] 或 [B,k]（这里只用全局同一组选中的下标即可）
        return: (B, P_sel, Nimg)
        """
        if self.image_features is None:
            raise RuntimeError("image_features is None. Call set_image_features() first.")

        # 全部 prompts 的特征 (B, P, D)
        text_features = self.encode_text_with_prompt(text)
        B, P, D = text_features.shape

        # 选择子集
        if selected_idx is not None:
            if selected_idx.dim() == 1:
                text_features = text_features[:, selected_idx, :]   # (B, |p_s|, D)
            else:
                # 若你将来做 per-sample 的 p_s，这里可按 batch gather
                raise NotImplementedError("per-sample selected_idx not supported yet")

        # (B, P_sel, Nimg)
        logits = torch.matmul(text_features, self.image_features.t())
        logits = self.clip_model.logit_scale.exp() * logits
        return logits  # (B, P_sel, Nimg)

    def reset_initial(self):
        self.prompt_learner.reset()

    def set_image_features(self, image_features):
        """
        image_features: (N, D)
        """
        self.image_features = image_features

    def parameters(self, recurse=True):
        # 仅训练 prompt
        return self.prompt_learner.parameters()

    def get_zero_shot(self, text):
        """
        纯 CLIP zero-shot，不使用 learnable prompt。
        返回形状: (B, Nimg)
        """
        if self.image_features is None:
            raise RuntimeError("image_features is None. Call set_image_features() first.")

        if isinstance(text, str):
            text = [text]
        B = len(text)

        text_features = self.clip_model.encode_text(tokenize(text).to(self.device))
        text_features = F.normalize(text_features, dim=-1)            # (B, D)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (text_features @ self.image_features.t())  # (B, Nimg)
        return logits


class PromptLearnerI2T(nn.Module):
    def __init__(self, clip_model, ctx_init: str, captions: list, n_ctx: int = 4):
        super().__init__()
        self.clip_model = clip_model
        self.device = clip_model.visual.conv1.weight.device
        self.dtype = torch.float
        self.n_ctx = n_ctx
        self.captions = captions

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        else:
            ctx_init = " ".join(["X"] * n_ctx)
            ctx_dim = clip_model.ln_final.weight.shape[0]
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_init_state = ctx_vectors.detach().clone()

        self.ctx_init = ctx_init
        self.prompts = [ctx_init + " " + cap.strip().rstrip('.') + "." for cap in captions]
        tokenized = torch.cat([tokenize(p) for p in self.prompts]).to(self.device)

        # Batch embedding to avoid OOM on large caption sets (e.g. 25k)
        emb_bsz = 2048
        prefix_parts, suffix_parts = [], []
        with torch.no_grad():
            for i in range(0, tokenized.size(0), emb_bsz):
                emb_chunk = clip_model.token_embedding(tokenized[i:i+emb_bsz]).type(self.dtype)
                prefix_parts.append(emb_chunk[:, :1, :].cpu())
                suffix_parts.append(emb_chunk[:, 1 + n_ctx:, :].cpu())
                del emb_chunk
        prefix_cpu = torch.cat(prefix_parts, 0)
        suffix_cpu = torch.cat(suffix_parts, 0)
        self.register_buffer("token_prefix", prefix_cpu, persistent=True)
        self.register_buffer("token_suffix", suffix_cpu, persistent=True)

        self.tokenized_prompts = tokenized
        self.text_features = None  # 后续由 encode_all_captions() 生成

    def reset(self):
        self.ctx.data.copy_(self.ctx_init_state)

    def encode_all_captions(self):
        feat_list = []
        bsz = 128
        with torch.no_grad():
            for i in range(0, len(self.captions), bsz):
                token_chunk = tokenize(self.captions[i:i + bsz])
                chunk_feat = self.clip_model.encode_text(token_chunk.to(self.device))
                feat_list.append(chunk_feat)

        self.text_features = F.normalize(torch.cat(feat_list, dim=0), dim=-1)  # (N_caption, D)

    def forward(self, idx=None):
        N = len(self.captions)
        if idx is None:
            # 全量模式
            prefix = self.token_prefix.to(self.device)  # (N,1,D)
            suffix = self.token_suffix.to(self.device)  # (N,L_suffix,D)
            tokenized = self.tokenized_prompts
            batch_size = N
        else:
            # 只取 top-k
            prefix = self.token_prefix[idx].to(self.device)  # (k,1,D)
            suffix = self.token_suffix[idx].to(self.device)  # (k,L_suffix,D)
            tokenized = self.tokenized_prompts[idx]
            batch_size = idx.size(0)
        # ctx 还在 GPU 上，直接 expand
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)  # (k, n_ctx, D) 或 (N,n_ctx,D)

        # 拼接成最终 prompts embedding
        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # (k, 1+n_ctx+L_suffix, D)

        return prompts, tokenized
        # ctx = self.ctx.unsqueeze(0).expand(N, -1, -1)
        # prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        # tokenized = self.tokenized_prompts
        #
        # if idx is not None:
        #     prompts = prompts[idx]
        #     tokenized = tokenized[idx]
        #
        # return prompts, tokenized


class CLIPRet_TPT_I2T(nn.Module):
    def __init__(self, device, captions, arch="ViT-B-16", n_ctx=4, ctx_init=""):
        super().__init__()
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False)
        self.clip_model.float()
        self.device = device

        self.prompt_learner = PromptLearnerI2T(self.clip_model, ctx_init, captions, n_ctx=n_ctx)
        self.dtype = torch.float
        self.freeze_parameters()
        self.prompt_learner.encode_all_captions()

    def freeze_parameters(self):
        for n, p in self.clip_model.named_parameters():
            p.requires_grad = False

    def reset_initial(self):
        self.prompt_learner.reset()

    def get_image_feature(self, image):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
            return image_features

    def get_top_k(self, image_feature, k):
        text_features = self.prompt_learner.text_features
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_feature @ text_features.T
        topk_idx = logits[0].topk(k=k, dim=-1).indices
        return logits, topk_idx
    
    def get_zero_shot(self, image_feature):
        text_features = self.prompt_learner.text_features
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_feature @ text_features.T
        return logits

    def forward(self, image_feature, idx=None):
        prompts, tokenized = self.prompt_learner(idx)
        orig_forward = self.clip_model.token_embedding.forward

        text_features = []
        bsz = min(tokenized.size(0), 128)
        try:
            if idx is not None:

                for i in range(0, tokenized.size(0), bsz):
                    token_chunk = tokenized[i:i + bsz]
                    prompt_chunk = prompts[i:i + bsz]

                    def fake_forward(x, prompt_chunk=prompt_chunk):
                        return prompt_chunk

                    self.clip_model.token_embedding.forward = fake_forward
                    chunk_feat = self.clip_model.encode_text(token_chunk.to(self.device))
                    text_features.append(chunk_feat)
            else:
                with torch.no_grad():

                    for i in range(0, tokenized.size(0), bsz):
                        token_chunk = tokenized[i:i + bsz]
                        prompt_chunk = prompts[i:i + bsz]

                        def fake_forward(x, prompt_chunk=prompt_chunk):
                            return prompt_chunk

                        self.clip_model.token_embedding.forward = fake_forward
                        chunk_feat = self.clip_model.encode_text(token_chunk.to(self.device))
                        text_features.append(chunk_feat)
        finally:
            self.clip_model.token_embedding.forward = orig_forward

        text_features = torch.cat(text_features, dim=0)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_feature @ text_features.T
        return logits


class PromptLearnerT2I(nn.Module):
    def __init__(self, clip_model, n_ctx=4, ctx_init=""):
        super().__init__()
        dtype = torch.float
        device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, dim)
        self.dtype = dtype
        self.device = device
        self.n_ctx = n_ctx
        self.clip_model = clip_model

    def reset(self):
        self.ctx.data.copy_(self.ctx_init_state)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        B = len(text)
        # 1. 拼接 prompt 和原始 text（是 token 层面）
        prompt_texts = [self.prompt_prefix + " " + t.strip() + "." for t in text]
        tokenized = tokenize(prompt_texts).to(self.device)  # (B, 77)

        # 2. 做 token embedding（正确方式）
        with torch.no_grad():
            x = self.clip_model.token_embedding(tokenized).type(self.dtype)  # (B, 77, D)

        # 3. 替换中间 prompt 区域的 embedding（仅替换 n_ctx 段）
        x[:, 1:1 + self.n_ctx, :] = self.ctx.unsqueeze(0).expand(B, -1, -1)

        return x, tokenized


class CLIPRet_TPT_T2I(nn.Module):
    """
    CLIP Retrieval + Test-time Prompt Tuning (TPT): Only optimizes the text prompt at test time.
    """
    def __init__(self, device, arch="ViT-B-16", n_ctx=4, ctx_init="", only_visual=True):
        super().__init__()
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False)
        self.clip_model.float()
        self.device = device

        self.prompt_learner = PromptLearnerT2I(self.clip_model, n_ctx=n_ctx, ctx_init=ctx_init)
        self.dtype = self.prompt_learner.dtype
        self.only_visual = only_visual
        self.text_features = None
        self.image_features = None
        self.freeze_parameters()

    def freeze_parameters(self):
        # 只训练prompt，冻结其余参数
        for n, p in self.clip_model.named_parameters():
            p.requires_grad = False

    def encode_text_with_prompt(self, text):
        # 1. 构造 prompt embedding（含 prefix + ctx + suffix）
        prompt_emb, tokenized = self.prompt_learner(text)

        # 2. 保存原始 embedding 的 forward 方法
        orig_forward = self.clip_model.token_embedding.forward

        # 3. 替换 token_embedding.forward 方法
        def fake_forward(x):
            return prompt_emb

        self.clip_model.token_embedding.forward = fake_forward
        # 4. 用 CLIP 原生的 encode_text() 执行前向（此时 embedding 已被注入）
        try:
            feat_prompt = self.clip_model.encode_text(tokenized.to(self.device))
        finally:
            # 5. 恢复原来的 forward 方法，防止污染后续
            self.clip_model.token_embedding.forward = orig_forward

        return F.normalize(feat_prompt, dim=-1)

    def forward(self, text):
        text_features = self.encode_text_with_prompt(text)
        # image_features: (N, D)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = logit_scale * text_features @ self.image_features.t()
        return logits_per_text

    def reset_initial(self):
        self.prompt_learner.reset()

    def set_image_features(self, image_features):
        self.image_features = image_features

    def parameters(self, recurse=True):
        return self.prompt_learner.parameters()

    def get_zero_shot(self, text):
        text_features = self.clip_model.encode_text(tokenize(text).to(self.device))
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = logit_scale * text_features @ self.image_features.t()
        return logits_per_text


class CLIPRet_TTA(nn.Module):
    def __init__(self, device, arch="ViT-B-16", only_visual=True, momentum_update=False,
                    update_freq=256, update_w=1.0, momentum=0.9999):
        """
        Using CLIP to do image retrieval with Test Time Adaptation,
        this module will tune the whole or part parameters of CLIP image encoder or text encoder.
        Args:
            only_norm: if True, only update the normalization layer
        """
        super().__init__()
        model_path = os.path.join(DOWNLOAD_ROOT, arch + '.pt')
        self.clip_model = load_openai_model(model_path, device, jit=False)
        self.clip_model.float()

        self.device = device
        self.text_features = None
        self.image_features = None
        # freeze parameters
        self.only_visual = only_visual
        self.freeze_parameters()

        self.momentum_update = momentum_update
        self.update_freq = update_freq
        self.update_w = update_w
        self.momentum = momentum
        self.update_counter = 0
        # save model state of visual encoder
        with torch.no_grad():
            self.clip_state_dict = copy.deepcopy(self.clip_model.state_dict())
            self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
            if self.momentum_update:
                self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())

        print("\n CLIPRet_TTA model created: \n"
                "\t backbone: {}, momentum_update / momentum / update_freq / update_w: [{} / {} / {} / {}] \n".format(
                    arch, momentum_update, momentum, update_freq, self.update_w))

    def forward(self, images=None, text=None, tokenized_prompts=None):
        image_features = self.get_image_features(images) if images is not None else self.image_features
        text_features = self.get_text_features(text, tokenized_prompts) if text is not None or tokenized_prompts is not None \
                        else self.text_features

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text

    def get_text_features(self, text=None, tokenized_prompts=None):
        if tokenized_prompts is None:
            assert text is not None
            tokenized_prompts = tokenize(text).to(self.device)

        text_features = self.clip_model.encode_text(tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def get_image_features(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def set_image_features(self, images=None, image_features=None):
        if images is not None:
            self.image_features = self.get_image_features(images)
        else:
            self.image_features = image_features

    def set_text_features(self, text=None, tokenized_prompts=None, text_features=None):
        if text is not None or tokenized_prompts is not None:
            self.text_features = self.get_text_features(text, tokenized_prompts)
        else:
            assert text_features is not None
            self.text_features = text_features

    @torch.no_grad()
    def freeze_parameters(self):
        if self.only_visual:
            for n, p in self.clip_model.named_parameters():
                if 'visual' not in n:
                    p.requires_grad_(False)
            self.clip_model.transformer.eval()
            self.clip_model.ln_final.eval()
        else:
            self.clip_model.lock_image_tower(unlocked_groups=0, freeze_bn_stats=True)
            self.clip_model.visual.eval()

    @torch.no_grad()
    def reset_all(self):
        # reset the state dict of clip model, sometimes you may change the weights
        self.clip_model.load_state_dict(self.clip_state_dict)
        self.initial_state_dict = copy.deepcopy(self.clip_model.state_dict())
        if self.momentum_update:
            self.momentum_state_dict = copy.deepcopy(self.clip_model.state_dict())

    @torch.no_grad()
    def reset_initial(self):
        self.clip_model.load_state_dict(self.initial_state_dict)

    @torch.no_grad()
    def momentum_update_model(self):
        update_w = self.update_w
        if self.momentum_update:
            self.update_counter += 1
            # reload momentum state_dict
            state_dict = self.clip_model.state_dict()
            for k, v in state_dict.items():
                self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                for k, v in state_dict.items():
                    # self.initial_state_dict[k] = self.momentum_state_dict[k]
                    self.initial_state_dict[k] = (1 - update_w) * self.clip_state_dict[k] + update_w * self.momentum_state_dict[k]
            # update will be done by function self.reset_initial()

    def parameters(self, recurse: bool = True):
        if self.only_visual:
            return self.clip_model.visual.parameters()
        else:
            params = []
            for n, p in self.clip_model.named_parameters():
                if 'visual' not in n:
                    params.append(p)
            return params

    def train(self, mode: bool = True):
        super().train(mode)

        if self.only_visual:
            self.clip_model.transformer.eval()
            self.clip_model.ln_final.eval()
        else:
            self.clip_model.visual.eval()

        return self


class CLIPRet_Multiple(nn.Module):
    def __init__(self, device, arch=["ViT-B-16", "ViT-L-14"], **kwargs):
        """
        For CLIP ensemble, zero-shot
        """
        super().__init__()
        # load clip models
        clip_models = []
        self.resolutions = []
        self.default_resolutions = kwargs["default_resolutions"] if "default_resolutions" in kwargs.keys() else 224
        for ar in arch:
            model_path = os.path.join(DOWNLOAD_ROOT, ar + '.pt')
            clip_model = load_openai_model(model_path, device, jit=False)
            clip_model.float()
            clip_models.append(clip_model)
            self.resolutions.append(clip_model.visual.image_size)
        self.clip_models = nn.ModuleList(clip_models)
        self.n_model = len(self.clip_models)
        self.device = device
        self.text_features = None
        self.image_features = None

        print("\n CLIPRet_TTA model created: backbone: {} \n".format(arch))

    def forward(self, images=None, text=None, tokenized_prompts=None):
        sim_matrix_i2t_list, sim_matrix_t2i_list = [], []
        for i in range(self.n_model):
            image_features = self.image_features[i]
            text_features = self.text_features[i]
            sims_matrix_i2t = image_features @ text_features.t()
            sims_matrix_t2i = sims_matrix_i2t.t()
            sim_matrix_i2t_list.append(sims_matrix_i2t)
            sim_matrix_t2i_list.append(sims_matrix_t2i)

        sims_matrix_i2t = torch.mean(torch.stack(sim_matrix_i2t_list, dim=0), dim=0)
        sims_matrix_t2i = torch.mean(torch.stack(sim_matrix_t2i_list, dim=0), dim=0)

        return sims_matrix_i2t, sims_matrix_t2i

    @torch.no_grad()
    def extract_image_features(self, images):
        """extract image features without normalization"""
        image_features = []
        for i in range(self.n_model):
            if self.resolutions[i] != self.default_resolutions:
                tmp_images = nn.functional.interpolate(images, size=self.resolutions[i], mode='bicubic', align_corners=True)
            else:
                tmp_images = images
            image_feat = self.clip_models[i].encode_image(tmp_images).float()
            image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
            image_features.append(image_feat)

        # list of tensor with len=n_model and shape [batch_size, dim]
        return image_features

    @torch.no_grad()
    def extract_text_features(self, captions=None, tokenized_cap=None):
        text_features = []
        for i in range(self.n_model):
            if captions is not None:
                caption_tokens = tokenize(captions).to(self.device)
                text_feat = self.clip_models[i].encode_text(caption_tokens).float()

            if tokenized_cap is not None:
                text_feat = self.clip_models[i].encode_text(tokenized_cap).float()

            # normalized features
            text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
            text_features.append(text_feat)

        # list of tensor with len=n_model and shape [batch_size, dim]
        return text_features

    @torch.no_grad()
    def set_many_text_features(self, texts, text_bs=128):
        """tokenize a list of 'texts' and call self.set_class_features()"""
        num_text = len(texts)
        text_feats = [[] for _ in range(self.n_model)]
        i = 0
        while i < num_text:
            text = texts[i : min(num_text, i + text_bs)]
            input_ids = tokenize(text).to(self.device)
            text_features_list = self.extract_text_features(tokenized_cap=input_ids)
            for j in range(self.n_model):
                text_feats[j].append(text_features_list[j])
            i += text_bs

        text_feats_tensors = []
        for i in range(self.n_model):
            tensor = torch.cat(text_feats[i], dim=0)
            text_feats_tensors.append(tensor)

        self.text_features = text_feats_tensors

    @torch.no_grad()
    def set_image_features_with_dataloder(self, data_loader):
        image_feats = [[] for _ in range(self.n_model)]
        for samples in data_loader:
            image = samples["image"].to(self.device)
            image_features_list = self.extract_image_features(image)
            for j in range(self.n_model):
                image_feats[j].append(image_features_list[j])

        image_feats_tensors = []
        for i in range(self.n_model):
            tensor = torch.cat(image_feats[i], dim=0)
            image_feats_tensors.append(tensor)

        self.image_features = image_feats_tensors
