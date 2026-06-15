# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 VL vision stack (reverse-engineered — pending empirical validation).

The upstream vision *forward* for ``minimax_m3_vl`` is NOT published anywhere
(HF repo ships only config + processors; GitHub is docs-only; transformers 5.7
has only minimax_m2; sglang main has no minimax VL; original source deleted).
This module is reverse-engineered from the bundled image processor (byte-for-byte
Qwen2-VL) + the exact quantized tensor shapes, using mlx_vlm's qwen2_5_vl vision
tower as the structural template.

Confirmed architecture (see docs BUILD-STATUS):
  pixel_values[N, 1176] --3D-conv patch_embed--> [N, 1280]
    + 3D-RoPE (h,w; head_dim 80, rope on 40 dims)
    --> pre_layrnorm --> 32 CLIP-style encoder layers
        (LayerNorm; separate q/k/v/out_proj with bias; mlp fc1 1280->5120 gelu fc2;
         FULL attention per image, no window)
    --> multi_modal_projector (Linear 1280->6144, gelu, Linear 6144->6144)   [per patch]
    --> 2x2 spatial merge (concat -> 24576)
    --> patch_merge_mlp (Linear 24576->6144, gelu, Linear 6144->6144)
    --> image_embeds[N/4, 6144]  spliced at image_token_index=200025

Weight names (bundle): vision_tower.vision_model.{embeddings.patch_embedding,
pre_layrnorm, encoder.layers.N.{self_attn.{q,k,v,out}_proj, layer_norm1/2,
mlp.fc1/fc2}}; multi_modal_projector.linear_{1,2}; patch_merge_mlp.linear_{1,2}.
All vision linears 8-bit affine quantized.

STATUS: written, NOT yet empirically validated (no numerical oracle exists; must be
verified by feeding a real image and checking coherent description). Isolated
vision-tower shape test in test_vision_tower.py.

Created by Jinho Jang (eric@jangq.ai).
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn


# ── image preprocessing (ported 1:1 from the bundled Qwen2-VL image_processor) ──
IMAGE_MEAN = mx.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
IMAGE_STD = mx.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
MAX_RATIO = 200


def _round_by(n, f):
    return round(n / f) * f


def _ceil_by(n, f):
    return math.ceil(n / f) * f


def _floor_by(n, f):
    return math.floor(n / f) * f


def smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=451584):
    if max(h, w) / min(h, w) > MAX_RATIO:
        raise ValueError("aspect ratio too extreme")
    hb = max(factor, _round_by(h, factor))
    wb = max(factor, _round_by(w, factor))
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = _floor_by(h / beta, factor)
        wb = _floor_by(w / beta, factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = _ceil_by(h * beta, factor)
        wb = _ceil_by(w * beta, factor)
    return hb, wb


def preprocess_image(pil_img, patch_size=14, temporal_patch_size=2, merge_size=2,
                     max_pixels=451584):
    """PIL image -> (pixel_values[num_patches, 1176], grid_thw[1,3]). Mirrors the
    bundled MiniMaxM3VLImageProcessor (Qwen2-VL layout)."""
    import numpy as np
    from PIL import Image
    img = pil_img.convert("RGB")
    w0, h0 = img.size
    factor = patch_size * merge_size
    rh, rw = smart_resize(h0, w0, factor=factor, max_pixels=max_pixels)
    img = img.resize((rw, rh), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0          # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))                        # [3,H,W]
    x = mx.array(arr)
    x = (x - IMAGE_MEAN) / IMAGE_STD                          # normalize
    # add temporal dim (single frame -> repeat to temporal_patch_size)
    x = mx.expand_dims(x, 0)                                  # [1,3,H,W] (t=1,c,h,w)
    x = mx.expand_dims(x, 0)                                  # [B=1,t=1,3,H,W]
    if x.shape[1] % temporal_patch_size != 0:
        reps = temporal_patch_size - (x.shape[1] % temporal_patch_size)
        x = mx.concatenate([x, mx.repeat(x[:, -1:], reps, axis=1)], axis=1)
    B, gt, C = x.shape[0], x.shape[1], x.shape[2]
    gt = gt // temporal_patch_size
    gh, gw = rh // patch_size, rw // patch_size
    x = x.reshape(B, gt, temporal_patch_size, C,
                  gh // merge_size, merge_size, patch_size,
                  gw // merge_size, merge_size, patch_size)
    x = x.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    pixel_values = x.reshape(B * gt * gh * gw,
                             C * temporal_patch_size * patch_size * patch_size)
    grid_thw = mx.array([[gt, gh, gw]], dtype=mx.int32)
    return pixel_values, grid_thw


# ── 3D-RoPE (h,w spatial; t=1 for images) ──
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope_vision(t, freqs):
    cos = mx.expand_dims(mx.tile(mx.expand_dims(mx.cos(freqs), 1), (1, 1, 2)), 0)
    sin = mx.expand_dims(mx.tile(mx.expand_dims(mx.sin(freqs), 1), (1, 1, 2)), 0)
    return (t * cos) + (rotate_half(t) * sin)


class VisionRoPE(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim, self.theta = dim, theta

    def __call__(self, seqlen):
        inv = 1.0 / (self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        seq = mx.arange(int(seqlen), dtype=inv.dtype)
        return mx.outer(seq, inv)


def rot_pos_emb(grid_thw, merge_size, rope):
    pos_ids = []
    for t, h, w in grid_thw.tolist():
        hp = mx.repeat(mx.expand_dims(mx.arange(h), 1), w, axis=1)
        hp = hp.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
        hp = mx.transpose(hp, (0, 2, 1, 3)).flatten()
        wp = mx.repeat(mx.expand_dims(mx.arange(w), 0), h, axis=0)
        wp = wp.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
        wp = mx.transpose(wp, (0, 2, 1, 3)).flatten()
        pos_ids.append(mx.tile(mx.stack([hp, wp], axis=-1), (t, 1)))
    pos_ids = mx.concatenate(pos_ids, axis=0)
    max_grid = int(mx.max(grid_thw[:, 1:]))
    full = rope(max_grid)
    emb = full[pos_ids]
    return emb.reshape(pos_ids.shape[0], -1)


# ── CLIP-style encoder layer (LayerNorm, separate q/k/v/out, gelu mlp) ──
class M3VisionAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x, cu_seqlens, rope_emb):
        L = x.shape[0]
        q = self.q_proj(x).reshape(L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(L, self.num_heads, self.head_dim)
        q = apply_rope_vision(mx.expand_dims(q, 0), rope_emb)[0]
        k = apply_rope_vision(mx.expand_dims(k, 0), rope_emb)[0]
        q = mx.expand_dims(q.transpose(1, 0, 2), 0)  # [1,H,L,d]
        k = mx.expand_dims(k.transpose(1, 0, 2), 0)
        v = mx.expand_dims(v.transpose(1, 0, 2), 0)
        cs = cu_seqlens.tolist()
        outs = []
        for i in range(len(cs) - 1):
            a, b = cs[i], cs[i + 1]
            if b <= a:
                continue
            o = mx.fast.scaled_dot_product_attention(
                q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], scale=self.scale, mask=None)
            outs.append(o)
        out = mx.concatenate(outs, axis=2)[0].transpose(1, 0, 2).reshape(L, -1)
        return self.out_proj(out)


class M3VisionLayer(nn.Module):
    def __init__(self, dim, num_heads, intermediate, eps=1e-5):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = M3VisionAttention(dim, num_heads)
        self.mlp = _CLIPMlp(dim, intermediate)

    def __call__(self, x, cu_seqlens, rope_emb):
        x = x + self.self_attn(self.layer_norm1(x), cu_seqlens, rope_emb)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _CLIPMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Mlp2(nn.Module):
    """Two-linear gelu MLP (projector / patch_merge): linear_1 -> gelu -> linear_2."""
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mid_dim)
        self.linear_2 = nn.Linear(mid_dim, out_dim)
        self.act = nn.GELU()

    def __call__(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class MiniMaxM3VisionModel(nn.Module):
    """vision_tower.vision_model: 3D patchify + 3D-RoPE + 32 CLIP layers (full attn)."""
    def __init__(self, vc):
        super().__init__()
        dim = vc["hidden_size"]
        self.num_heads = vc["num_attention_heads"]
        self.merge_size = vc.get("img_token_compression_config", {}).get("spatial_merge_size", 2)
        self.embeddings = _Embeddings(vc)
        self.pre_layrnorm = nn.LayerNorm(dim, eps=vc.get("layer_norm_eps", 1e-5))
        self.encoder = _Encoder(vc)
        head_dim = dim // self.num_heads
        self.rope = VisionRoPE(head_dim // 2, theta=vc.get("rope_theta", 10000.0))

    def __call__(self, pixel_values, grid_thw):
        x = self.embeddings(pixel_values)                  # [N, dim]
        x = self.pre_layrnorm(x)
        rope_emb = rot_pos_emb(grid_thw, self.merge_size, self.rope)
        # full attention per image: cu_seqlens at image boundaries
        cu = [0]
        for t, h, w in grid_thw.tolist():
            cu.append(cu[-1] + t * h * w)
        cu_seqlens = mx.array(cu, dtype=mx.int32)
        x = self.encoder(x, cu_seqlens, rope_emb)
        return x


class _Embeddings(nn.Module):
    def __init__(self, vc):
        super().__init__()
        self.in_ch = vc.get("num_channels", 3)
        self.temporal = vc.get("img_token_compression_config", {}).get("temporal_patch_size", 2)
        self.patch = vc["patch_size"]
        self.hidden = vc["hidden_size"]
        # param path: embeddings.patch_embedding.weight  (matches bundle)
        self.patch_embedding = nn.Conv3d(
            self.in_ch, self.hidden,
            kernel_size=(self.temporal, self.patch, self.patch),
            stride=(self.temporal, self.patch, self.patch), bias=False)

    def __call__(self, pixel_values):
        x = pixel_values.reshape(-1, self.in_ch, self.temporal, self.patch, self.patch)
        x = x.moveaxis(1, 4)                      # [N, t, h, w, in] for MLX Conv3d
        x = self.patch_embedding(x)
        return x.reshape(-1, self.hidden)


class _Encoder(nn.Module):
    def __init__(self, vc):
        super().__init__()
        self.layers = [M3VisionLayer(vc["hidden_size"], vc["num_attention_heads"],
                                     vc["intermediate_size"], vc.get("layer_norm_eps", 1e-5))
                       for _ in range(vc["num_hidden_layers"])]

    def __call__(self, x, cu_seqlens, rope_emb):
        for layer in self.layers:
            x = layer(x, cu_seqlens, rope_emb)
        return x


class MiniMaxM3VLVisionStack(nn.Module):
    """Full vision path: vision_tower -> projector -> 2x2 merge -> patch_merge -> [N/4, 6144]."""
    def __init__(self, cfg):
        super().__init__()
        vc = cfg["vision_config"]
        self.merge = vc.get("img_token_compression_config", {}).get("spatial_merge_size", 2)
        vdim = vc["hidden_size"]                       # 1280
        tdim = cfg.get("projector_hidden_size", 6144)  # 6144
        self.vision_tower = _VisionTower(vc)
        # projector: 1280 -> 6144 -> 6144 (per patch)
        self.multi_modal_projector = _Mlp2(vdim, tdim, tdim)
        # patch_merge: (merge^2 * 6144) -> 6144 -> 6144
        self.patch_merge_mlp = _Mlp2(tdim * (self.merge ** 2), tdim, tdim)

    def __call__(self, pixel_values, grid_thw):
        feats = self.vision_tower.vision_model(pixel_values, grid_thw)  # [N,1280]
        proj = self.multi_modal_projector(feats)                       # [N,6144]
        n = proj.shape[0]
        merged = proj.reshape(n // (self.merge ** 2), (self.merge ** 2) * proj.shape[-1])
        return self.patch_merge_mlp(merged)                            # [N/4,6144]


class _VisionTower(nn.Module):
    def __init__(self, vc):
        super().__init__()
        self.vision_model = MiniMaxM3VisionModel(vc)
