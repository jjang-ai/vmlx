# SPDX-License-Identifier: Apache-2.0
"""High-throughput Apple Silicon runtime for Hy4-flash (`model_type='hy_v4_flash'`).

Implements:
- 4-stream FP32 Identity Hyper-Connections (iHC) + `hc_head`
- Disk-backed Engram N-gram fusion at layers `[1, 11, 21, 31]` with 9-layer background NVMe prefetch
- Hybrid MLA Attention:
  - 12 DSA layers (`H=32`, `theta=1e7`) with 4-token gated `IndexPool` compressed indexer ->
    exact Metal radix Top-511 historical groups + own group -> fused O(2048) sparse absorbed MLA
  - 30 SWA layers (`H=64`, `theta=1e4`, window=512) with learnable attention sinks, sigmoid output
    gate, and sandwich `post_proj_norm`
- Interleaved 3D M-RoPE (reducing identically to 1D interleaved RoPE for pure text)
- Dense Layer 0 MLP (unclamped SwiGLU) + 41 LatentMoE layers (`512` experts, `top-16`, latent `1536`,
  clamped SwiGLU `10.0`, unclamped shared expert, `combine_output_norm`)
- FP32 `lm_head` (`enable_lm_head_fp32=True`)
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.models.ernie5_5_moe.kernels import ernie_lm_head_forward, ernie_sparse_mla_prefill

from .cache import HyV4GrowingMLACache, HyV4LayerCache, HyV4RotatingMLACache
from .config import HyV4FlashConfig
from .engram import HyV4EngramHasher, HyV4EngramLayer
from .kernels import (
    HyV4DenseSwitchGLU,
    HyV4MXFP4SwitchGLU,
    hy4_ihc_post,
    hy4_ihc_pre,
    hy4_router_topk,
)


def _apply_interleaved_rope(
    x: mx.array,
    dims: int,
    base: float,
    offset: int,
    *,
    position_ids_3d: Optional[mx.array] = None,
    mrope_section: Tuple[int, int, int] = (12, 10, 10),
) -> mx.array:
    """Interleaved M-RoPE (non-NeoX: `traditional=True`).

    For 1-D text positions (`position_ids_3d is None`), uses `mx.fast.rope(..., traditional=True)`.
    For 3-D multimodal positions (`[3, B, L]`), builds exact interleaved per-axis angles.
    """
    if position_ids_3d is None:
        return mx.fast.rope(x, dims, traditional=True, base=float(base), scale=1.0, offset=int(offset))
    # Explicit 3-D interleaved M-RoPE on trailing `dims` coordinates
    half = dims // 2
    inv_freq = 1.0 / (float(base) ** (mx.arange(0, dims, 2, dtype=mx.float32) / float(dims)))
    # Interleaved channel assignment matching mrope_section=[12, 10, 10]
    # Axes cycle T, H, W until each section quota is exhausted
    s0, s1, s2 = int(mrope_section[0]), int(mrope_section[1]), int(mrope_section[2])
    axis_map: List[int] = []
    rem = [s0, s1, s2]
    while len(axis_map) < half:
        progressed = False
        for ax in (0, 1, 2):
            if rem[ax] > 0 and len(axis_map) < half:
                axis_map.append(ax)
                rem[ax] -= 1
                progressed = True
        if not progressed:
            axis_map.append(0)
    ax_idx = mx.array(axis_map, dtype=mx.int32)
    # position_ids_3d: [3, B, L] -> select per frequency channel -> [B, L, half]
    pos = position_ids_3d.astype(mx.float32)  # [3, B, L]
    pos_ch = mx.take(pos, ax_idx, axis=0).transpose(1, 2, 0)  # [B, L, half]
    angles = pos_ch * inv_freq[None, None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    # Broadcast across head dim if x is [B, H, L, D]
    if x.ndim == 4:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    x_f = x.astype(mx.float32)
    x_even = x_f[..., 0::2]
    x_odd = x_f[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_odd * cos + x_even * sin
    out = mx.stack([y_even, y_odd], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)


class HyV4IHCPre(nn.Module):
    """4-stream FP32 Identity Hyper-Connection module (`hc_attn_layer` / `hc_mlp_layer`)."""

    def __init__(self, config: HyV4FlashConfig) -> None:
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.hidden_size = int(config.hidden_size)
        self.rms_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)
        self.hc_magnitude = float(config.hc_magnitude)
        self.hc_fn = mx.zeros((2 * self.hc_mult, self.hc_mult * self.hidden_size), dtype=mx.float32)
        self.hc_base = mx.zeros((2 * self.hc_mult,), dtype=mx.float32)
        self.hc_scale = mx.ones((2,), dtype=mx.float32)

    def pre(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return hy4_ihc_pre(
            x,
            self.hc_fn,
            self.hc_base,
            self.hc_scale,
            rms_eps=self.rms_eps,
            hc_eps=self.hc_eps,
            hc_magnitude=self.hc_magnitude,
        )

    @staticmethod
    def post(x: mx.array, y: mx.array, g_post: mx.array) -> mx.array:
        return hy4_ihc_post(x, y, g_post)


class HyV4HCHead(nn.Module):
    """Final 4-stream -> 1-stream iHC collapse head (`model.hc_head`)."""

    def __init__(self, config: HyV4FlashConfig) -> None:
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.hidden_size = int(config.hidden_size)
        self.rms_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)
        self.hc_head_fn = mx.zeros((self.hc_mult, self.hc_mult * self.hidden_size), dtype=mx.float32)
        self.hc_head_base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.hc_head_scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, C, D = x.shape
        xf = x.astype(mx.float32)
        x_flat = xf.reshape(B, L, C * D)
        r = mx.rsqrt(mx.mean(x_flat * x_flat, axis=-1, keepdims=True) + self.rms_eps)
        mixes = (x_flat @ self.hc_head_fn.astype(mx.float32).T) * r
        g = mx.sigmoid(mixes * self.hc_head_scale.astype(mx.float32) + self.hc_head_base.astype(mx.float32)) + self.hc_eps
        return mx.sum(g[..., None] * xf, axis=2).astype(x.dtype)


class HyV4CompressedIndexer(nn.Module):
    """IndexPool compressed sparse indexer for Hy4-flash's 12 DSA layers (`compress_ratio=4`)."""

    def __init__(self, config: HyV4FlashConfig) -> None:
        super().__init__()
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.compress_ratio = int(config.compressed_indexer_compress_ratio)
        self.group_topk = int(config.compressed_indexer_group_topk)
        self.candidate_topk = int(config.compressed_indexer_candidate_topk)
        self.rope_theta = float(config.rope_theta_dsa)
        self.mrope_section = config.mrope_section
        self.rms_eps = float(config.rms_norm_eps)
        self.weight_scale = float((self.n_heads * self.head_dim) ** -0.5)

        self.query_projection = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.head_weight_projection = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.token_key_projection = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.token_key_norm = nn.RMSNorm(self.head_dim, eps=self.rms_eps)

        # FP32 4-token gated pooling parameters
        self.key_value_projection = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_gate_projection = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_norm = nn.RMSNorm(self.head_dim, eps=self.rms_eps)
        self.intra_group_ape = mx.zeros((self.compress_ratio, self.head_dim), dtype=mx.float32)

    def _pool_groups_fp32(self, tokens: mx.array, base_pos: int) -> mx.array:
        """Pool `[B, n_grp * 4, 128]` token keys -> `[B, n_grp, 128]` RoPEd BF16 group keys."""
        B, N, D = tokens.shape
        r = self.compress_ratio
        n_grp = N // r
        g = tokens.reshape(B, n_grp, r, D).astype(mx.float32)
        v = g @ self.key_value_projection.weight.astype(mx.float32).T
        gate = g @ self.key_gate_projection.weight.astype(mx.float32).T + self.intra_group_ape.astype(mx.float32)[None, None, :, :]
        probs = mx.softmax(gate, axis=2)
        pooled = mx.sum(probs * v, axis=2)  # [B, n_grp, D] FP32
        pooled = mx.fast.rms_norm(pooled, self.key_norm.weight.astype(mx.float32), self.rms_eps).astype(tokens.dtype)
        # Apply interleaved RoPE to trailing 64 dims at each group's FIRST token position (base_pos + 4*g)
        nope = pooled[..., : D - self.rope_dim]
        pe = pooled[..., D - self.rope_dim :]
        grp_pos = mx.arange(n_grp, dtype=mx.float32) * float(r) + float(base_pos)
        inv_freq = 1.0 / (self.rope_theta ** (mx.arange(0, self.rope_dim, 2, dtype=mx.float32) / float(self.rope_dim)))
        angles = grp_pos[:, None] * inv_freq[None, :]  # [n_grp, rope_dim/2]
        cos = mx.cos(angles)[None, :, :]
        sin = mx.sin(angles)[None, :, :]
        pe_f = pe.astype(mx.float32)
        pe_e, pe_o = pe_f[..., 0::2], pe_f[..., 1::2]
        pe_rot = mx.stack([pe_e * cos - pe_o * sin, pe_o * cos + pe_e * sin], axis=-1).reshape(pe.shape).astype(pooled.dtype)
        return mx.concatenate([nope, pe_rot], axis=-1)

    def __call__(
        self,
        x: mx.array,
        q_c: mx.array,
        cache: Optional[HyV4GrowingMLACache],
        offset: int,
        S: int,
        *,
        position_ids_3d: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Update IndexPool cache and return `[1, L, <=2048]` uint32 sparse token indices (or None if S <= 2048)."""
        B, L, _ = x.shape
        r = self.compress_ratio

        # 1. Compute token keys [B, L, 128] & update 4-token compressed group cache
        k_tok = self.token_key_norm(self.token_key_projection(x))
        if cache is not None:
            cache._last_in_tok_keys = k_tok if L <= 32 else None
            if cache.tail_keys is not None and cache.tail_len > 0:
                full_tok = mx.concatenate([cache.tail_keys, k_tok], axis=1)
                base_pos = cache.tail_base_pos
            else:
                full_tok = k_tok
                base_pos = offset
            total = int(full_tok.shape[1])
            n_new = total // r
            usable = n_new * r
            rem = total - usable
            if n_new > 0:
                new_grp = self._pool_groups_fp32(full_tok[:, :usable, :], base_pos)
                all_groups = cache.append_group_keys(new_grp)
            else:
                all_groups = cache.fetch_group_keys()
            if rem > 0:
                cache.tail_keys = mx.contiguous(full_tok[:, usable:, :])
                cache.tail_len = rem
                cache.tail_base_pos = base_pos + usable
            else:
                cache.tail_keys = None
                cache.tail_len = 0
                cache.tail_base_pos = base_pos + usable
        else:
            n_new = L // r
            all_groups = self._pool_groups_fp32(k_tok[:, : n_new * r, :], offset) if n_new > 0 else None

        # Bypass sparse selection when total sequence fits inside candidate_topk (2048)
        if S <= self.candidate_topk or all_groups is None:
            if cache is not None:
                cache.last_topk_indices = None
            return None

        # 2. Compute indexer queries with trailing 64-dim interleaved RoPE
        q_idx = self.query_projection(q_c).reshape(B, L, self.n_heads, self.head_dim)
        q_nope = q_idx[..., : self.head_dim - self.rope_dim]
        q_pe = _apply_interleaved_rope(
            q_idx[..., self.head_dim - self.rope_dim :].transpose(0, 2, 1, 3),
            self.rope_dim,
            self.rope_theta,
            offset,
            position_ids_3d=position_ids_3d,
            mrope_section=self.mrope_section,
        ).transpose(0, 2, 1, 3)
        q_rot = mx.concatenate([q_nope, q_pe], axis=-1)  # [B, L, 16, 128]
        weights = self.head_weight_projection(x).astype(mx.float32) * self.weight_scale  # [B, L, 16]

        G = int(all_groups.shape[1])
        scores = None
        if B == 1 and G >= 32:
            if L == 1:
                from vmlx_engine.metal.dsa_index_scores import dsa_decode_index_scores
                scores = dsa_decode_index_scores(q_rot, all_groups, weights, scale=1.0)
            else:
                from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
                scores = dsa_pool_index_scores(q_rot, all_groups, weights, scale=1.0, bf16_operands=True)
        if scores is None:
            # Generic fallback: [B, 16, L, 128] @ [B, 1, 128, G]
            q_t = q_rot.transpose(0, 2, 1, 3).astype(mx.float32)
            k_t = all_groups[:, None, :, :].astype(mx.float32)
            sc = mx.maximum(q_t @ k_t.swapaxes(-1, -2), 0.0)
            scores = mx.sum(sc * weights.transpose(0, 2, 1)[..., None], axis=1)  # [B, L, G]

        # 3. Mask g >= query_slot // 4; select top min(group_topk - 1, G) historical groups + own group!
        q_slots = mx.arange(offset, offset + L, dtype=mx.uint32)  # [L]
        own_grp = q_slots // r  # [L]
        hist_k = self.group_topk - 1  # 511

        sel_groups = None
        if B == 1 and G > hist_k:
            from vmlx_engine.metal.sparse_topk import metal_radix_topk
            sel_groups = metal_radix_topk(scores, hist_k, valid_end=own_grp)  # [1, L, 511] uint32

        if sel_groups is None:
            g_idx = mx.arange(G, dtype=mx.uint32)[None, :]
            valid_g = g_idx < own_grp[:, None]
            masked = mx.where(valid_g[None, :, :], scores, -mx.inf)
            k_eff = min(hist_k, G)
            if G > k_eff:
                sel_groups = mx.argpartition(-masked, kth=k_eff - 1, axis=-1)[..., :k_eff].astype(mx.uint32)
            else:
                sel_groups = mx.broadcast_to(g_idx[None, :, :], (B, L, G)).astype(mx.uint32)
            # Mark out-of-range groups with sentinel
            sel_groups = mx.where(sel_groups < own_grp[None, :, None], sel_groups, mx.array(0xFFFFFFFF, dtype=mx.uint32))

        # Append query's own group unconditionally -> [B, L, <=512]
        all_sel_g = mx.concatenate([sel_groups, own_grp[None, :, None]], axis=-1)
        # Expand each group g -> [4g, 4g+1, 4g+2, 4g+3], masking sentinel groups and slots > q_slot
        valid_group_mask = all_sel_g != mx.array(0xFFFFFFFF, dtype=mx.uint32)
        base_tok = all_sel_g[..., None] * r  # [B, L, G_sel, 1]
        offs = mx.arange(r, dtype=mx.uint32)[None, None, None, :]
        tok_ids = (base_tok + offs).reshape(B, L, -1)  # [B, L, <=2048]
        tok_valid = mx.repeat(valid_group_mask, r, axis=-1) & (tok_ids <= q_slots[None, :, None])
        tok_ids = mx.where(tok_valid, tok_ids, mx.array(0xFFFFFFFF, dtype=mx.uint32))
        if cache is not None:
            cache.last_topk_indices = tok_ids
        return tok_ids


class HyV4MLAAttention(nn.Module):
    """Unified Gated MLA Attention for both Hy4-flash DSA (32H, IndexPool) and SWA (64H, W=512)."""

    def __init__(self, config: HyV4FlashConfig, layer_idx: int, *, is_dsa: bool) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.is_dsa = bool(is_dsa)
        self.n_heads = int(config.num_attention_heads if is_dsa else config.swa_num_attention_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope = int(config.qk_nope_head_dim)
        self.qk_rope = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope + self.qk_rope  # 256
        self.v_head_dim = int(config.v_head_dim)  # 256
        self.scale = float(self.qk_head_dim ** -0.5)  # 1/16
        self.sliding_window = 0 if is_dsa else int(config.sliding_window)
        self.rope_theta = float(config.rope_theta_dsa if is_dsa else config.rope_theta_swa)
        self.mrope_section = config.mrope_section

        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj: Any = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # Absorbed kv_b_proj split into embed_q [H, 512, 192] & unembed_out [H, 256, 512]
        self.embed_q = mx.zeros((self.n_heads, self.kv_lora_rank, self.qk_nope), dtype=mx.bfloat16)
        self.unembed_out = mx.zeros((self.n_heads, self.v_head_dim, self.kv_lora_rank), dtype=mx.bfloat16)

        self.linear_gate: Any = nn.Linear(config.hidden_size, self.n_heads * self.v_head_dim, bias=False)
        self.o_proj: Any = nn.Linear(self.n_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.learnable_sink_param = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.post_proj_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.compressed_indexer: Optional[HyV4CompressedIndexer] = (
            HyV4CompressedIndexer(config) if is_dsa and config.use_compressed_indexer else None
        )
        self._sinks_bf16: Optional[mx.array] = None

    def set_kv_b_weight(self, kv_b_weight: mx.array) -> None:
        """Split `[H * (192 + 256), 512]` into absorbed `embed_q [H, 512, 192]` and `unembed_out [H, 256, 512]`."""
        w = kv_b_weight.reshape(self.n_heads, self.qk_nope + self.v_head_dim, self.kv_lora_rank)
        k_part, v_part = mx.split(w, [self.qk_nope], axis=1)
        # embed_q: q_nope [B, H, L, 192] @ k_part [H, 192, 512] -> [B, H, L, 512]
        # Store k_part transposed [H, 512, 192] so q_nope @ embed_q.swapaxes(-1,-2) == q_nope @ k_part!
        self.embed_q = mx.contiguous(k_part.swapaxes(-1, -2))
        self.unembed_out = mx.contiguous(v_part)

    def _sinks(self, dtype: mx.Dtype) -> mx.array:
        if dtype == mx.bfloat16 and self._sinks_bf16 is not None:
            return self._sinks_bf16
        return self.learnable_sink_param.astype(dtype)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[HyV4LayerCache] = None,
        *,
        position_ids_3d: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        inner = cache.inner if cache is not None else None
        offset = int(inner.offset) if inner is not None else 0
        rope_offset = max(0, offset + int(getattr(cache, "rope_delta", 0) or 0))

        q_c = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_c).reshape(B, L, self.n_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope], axis=-1)

        kva = self.kv_a_proj_with_mqa(x)
        c_kv, k_pe = mx.split(kva, [self.kv_lora_rank], axis=-1)
        c_kv = self.kv_a_layernorm(c_kv)[:, None, :, :]  # [B, 1, L, 512]
        k_pe = k_pe[:, None, :, :]  # [B, 1, L, 64]

        q_pe = _apply_interleaved_rope(
            q_pe, self.qk_rope, self.rope_theta, rope_offset,
            position_ids_3d=position_ids_3d, mrope_section=self.mrope_section,
        )
        k_pe = _apply_interleaved_rope(
            k_pe, self.qk_rope, self.rope_theta, rope_offset,
            position_ids_3d=position_ids_3d, mrope_section=self.mrope_section,
        )

        if inner is not None:
            if self.is_dsa:
                assert isinstance(inner, HyV4GrowingMLACache)
                c_kv, k_pe = inner.update_and_fetch_kv(c_kv, k_pe)
            else:
                assert isinstance(inner, HyV4RotatingMLACache)
                c_kv, k_pe = inner.update_and_fetch(c_kv, k_pe)

        S = int(c_kv.shape[2])
        sinks = self._sinks(q_nope.dtype)

        # DSA sparse IndexPool path (constant O(2048) memory & FLOPs up to 1M context!)
        sparse_indices = None
        if self.is_dsa and self.compressed_indexer is not None:
            dsa_cache = inner if isinstance(inner, HyV4GrowingMLACache) else None
            sparse_indices = self.compressed_indexer(
                x, q_c, dsa_cache, offset, S, position_ids_3d=position_ids_3d
            )

        if sparse_indices is not None and B == 1:
            # Absorbed query: [1, 32, L, 192] @ [32, 192, 512] -> [1, 32, L, 512]
            q_abs = q_nope @ self.embed_q.swapaxes(-1, -2)
            out_lat = ernie_sparse_mla_prefill(
                q_abs, q_pe, c_kv, k_pe, sparse_indices, self.learnable_sink_param, self.scale
            )
            out = out_lat @ self.unembed_out.swapaxes(-1, -2)  # [1, 32, L, 256]
        else:
            # Dense / SWA absorbed MLA (S <= 512 for SWA, S <= 2048 for short DSA)
            use_absorbed = (L <= 16) or (S <= 512)
            if use_absorbed:
                q_abs = q_nope @ self.embed_q.swapaxes(-1, -2)
                pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
                if L > 1:
                    q_pos = mx.arange(S - L, S, dtype=mx.int32)[:, None]
                    k_pos = mx.arange(S, dtype=mx.int32)[None, :]
                    if self.sliding_window > 0:
                        vis = (k_pos <= q_pos) & (k_pos > (q_pos - self.sliding_window))
                    else:
                        vis = k_pos <= q_pos
                    pe_scores = mx.where(vis[None, None], pe_scores, -mx.inf)
                out_lat = mx.fast.scaled_dot_product_attention(
                    q_abs, c_kv, c_kv, scale=self.scale, mask=pe_scores, sinks=sinks
                )
                out = out_lat @ self.unembed_out.swapaxes(-1, -2)
            else:
                # Long prefill on short DSA (L > 16, S <= 2048) or long SWA chunk (L > 16, S > 512):
                # expand K/V once and run banded 256-token strips to bound peak mask RAM!
                k_full = c_kv @ self.embed_q  # [B, H, S, 192]
                v_full = c_kv @ self.unembed_out.swapaxes(-1, -2)  # [B, H, S, 256]
                strip = 256
                prefix = S - L
                outs = []
                for s0 in range(0, L, strip):
                    s1 = min(L, s0 + strip)
                    sl = s1 - s0
                    k_start = max(0, prefix + s0 - self.sliding_window + 1) if self.sliding_window > 0 else 0
                    k_end = prefix + s1
                    qn_s = q_nope[:, :, s0:s1, :]
                    qp_s = q_pe[:, :, s0:s1, :]
                    ks = k_full[:, :, k_start:k_end, :]
                    vs = v_full[:, :, k_start:k_end, :]
                    kpe_s = k_pe[:, :, k_start:k_end, :]
                    pe_s = (qp_s * self.scale) @ kpe_s.swapaxes(-1, -2)
                    q_pos = mx.arange(prefix + s0, prefix + s1, dtype=mx.int32)[:, None]
                    k_pos = mx.arange(k_start, k_end, dtype=mx.int32)[None, :]
                    if self.sliding_window > 0:
                        vis = (k_pos <= q_pos) & (k_pos > (q_pos - self.sliding_window))
                    else:
                        vis = k_pos <= q_pos
                    pe_s = mx.where(vis[None, None], pe_s, -mx.inf)
                    outs.append(
                        mx.fast.scaled_dot_product_attention(
                            qn_s, ks, vs, scale=self.scale, mask=pe_s, sinks=sinks
                        )
                    )
                out = mx.concatenate(outs, axis=2)

        # Elementwise sigmoid gate + o_proj + sandwich post_proj_norm
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.v_head_dim)
        gate = mx.sigmoid(self.linear_gate(x))
        return self.post_proj_norm(self.o_proj(out * gate))


class HyV4DenseMLP(nn.Module):
    """Layer 0 dense SwiGLU MLP (unclamped) + sandwich post_fc2_norm."""

    def __init__(self, config: HyV4FlashConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.post_fc2_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.post_fc2_norm(self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x)))


class HyV4SharedMLP(nn.Module):
    """Unclamped SwiGLU shared expert on full hidden_size (3072 -> 1536 -> 3072)."""

    def __init__(self, config: HyV4FlashConfig) -> None:
        super().__init__()
        self.gate_proj: Any = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj: Any = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj: Any = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HyV4LatentMoE(nn.Module):
    """Hy4-flash LatentMoE (512 experts, top-16, latent 1536, clamped SwiGLU 10.0 + unclamped shared)."""

    def __init__(self, config: HyV4FlashConfig, *, mxfp4: bool = False) -> None:
        super().__init__()
        self.config = config
        self.top_k = int(config.num_experts_per_tok)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.norm_topk_prob = bool(config.norm_topk_prob)

        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,), dtype=mx.float32)

        self.routed_expert_down_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.routed_expert_up_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)
        if mxfp4:
            self.switch_mlp: Any = HyV4MXFP4SwitchGLU(
                config.n_routed_experts,
                config.routed_expert_hidden_size,
                config.moe_intermediate_size,
                swiglu_limit=config.swiglu_limit,
            )
        else:
            self.switch_mlp = HyV4DenseSwitchGLU(
                config.n_routed_experts,
                config.routed_expert_hidden_size,
                config.moe_intermediate_size,
                swiglu_limit=config.swiglu_limit,
            )
        self.shared_experts = HyV4SharedMLP(config)
        self.combine_output_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        ids, w = hy4_router_topk(
            x,
            self.gate.weight,
            self.e_score_correction_bias,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
        )
        z = self.routed_expert_down_proj(x)
        routed = self.routed_expert_up_proj(self.switch_mlp(z, ids, w))
        shared = self.shared_experts(x)
        return self.combine_output_norm(routed + shared)


class HyV4DecoderLayer(nn.Module):
    """Single Hy4-flash decoder layer: optional Engram -> iHC Attention -> iHC MLP/LatentMoE."""

    def __init__(self, config: HyV4FlashConfig, layer_idx: int, *, mxfp4: bool = False) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.is_dsa = config.is_dsa_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)
        self.has_engram = layer_idx in config.engram_zero_indexed_layers

        self.engram_layer: Optional[HyV4EngramLayer] = (
            HyV4EngramLayer(config, layer_idx) if self.has_engram else None
        )
        self.hc_attn_layer = HyV4IHCPre(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HyV4MLAAttention(config, layer_idx, is_dsa=self.is_dsa)

        self.hc_mlp_layer = HyV4IHCPre(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp: Any = HyV4LatentMoE(config, mxfp4=mxfp4) if self.is_moe else HyV4DenseMLP(config)

    def __call__(
        self,
        hidden_hc: mx.array,
        cache: Optional[HyV4LayerCache] = None,
        *,
        engram_embeds: Optional[mx.array] = None,
        engram_mask: Optional[mx.array] = None,
        position_ids_3d: Optional[mx.array] = None,
    ) -> mx.array:
        if self.engram_layer is not None and engram_embeds is not None:
            hidden_hc = self.engram_layer(hidden_hc, engram_embeds, valid_mask=engram_mask)

        u_a, gp_a = self.hc_attn_layer.pre(hidden_hc)
        y_a = self.self_attn(self.input_layernorm(u_a), cache=cache, position_ids_3d=position_ids_3d)
        hidden_hc = self.hc_attn_layer.post(hidden_hc, y_a, gp_a)

        u_m, gp_m = self.hc_mlp_layer.pre(hidden_hc)
        y_m = self.mlp(self.post_attention_layernorm(u_m))
        hidden_hc = self.hc_mlp_layer.post(hidden_hc, y_m, gp_m)
        return hidden_hc


class HyV4FlashModel(nn.Module):
    """Hy4-flash backbone with 9-layer pipelined background Engram NVMe prefetch."""

    def __init__(self, config: HyV4FlashConfig, *, mxfp4: bool = False) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [HyV4DecoderLayer(config, i, mxfp4=mxfp4) for i in range(config.num_hidden_layers)]
        self.hc_head = HyV4HCHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._engram_hasher: Optional[HyV4EngramHasher] = None
        self._engram_order: List[int] = sorted(config.engram_zero_indexed_layers.keys())

    def set_engram_hasher(self, hasher: HyV4EngramHasher) -> None:
        self._engram_hasher = hasher

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache: Optional[Sequence[HyV4LayerCache]] = None,
        *,
        inputs_embeds: Optional[mx.array] = None,
        position_ids_3d: Optional[mx.array] = None,
        tail_only: bool = False,
        return_last_hidden: bool = False,
    ) -> mx.array | Tuple[mx.array, mx.array]:
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(inputs)
        B, L, D = x.shape

        # Compute Engram hash indices on CPU and launch pipelined background NVMe prefetches
        # (layer 1 starts immediately; layer 11 chains behind layer 1 while GPU runs layers 1..10!).
        engram_futures: Dict[int, Any] = {}
        engram_shapes: Dict[int, Tuple[int, ...]] = {}
        engram_mask_mx: Optional[mx.array] = None
        if self._engram_hasher is not None and inputs is not None and B == 1:
            raw_ids_np = np.array(inputs[0], dtype=np.int64)
            prev_tail = cache[0].engram_tail if (cache is not None and len(cache) > 0) else None
            layer_hashes, new_tail, valid_np = self._engram_hasher.hash_for_layers(raw_ids_np, prev_tail)
            if cache is not None and len(cache) > 0:
                cache[0].engram_tail = new_tail
            engram_mask_mx = mx.array(valid_np, dtype=mx.bool_).reshape(1, L)
            prev_fut: Optional[Future] = None
            for zidx in self._engram_order:
                tbl = self.layers[zidx].engram_layer._table if self.layers[zidx].engram_layer is not None else None
                if tbl is None:
                    continue
                h_ids = layer_hashes[zidx]  # [L, 16]
                engram_shapes[zidx] = (1, L, int(h_ids.shape[1]))
                fut = tbl.prefetch(h_ids, after=prev_fut)
                engram_futures[zidx] = (tbl, fut)
                if isinstance(fut, Future):
                    prev_fut = fut

        # Broadcast [B, L, D] -> 4-stream iHC [B, L, 4, D]
        hidden_hc = mx.broadcast_to(x[:, :, None, :], (B, L, self.config.hc_mult, D))
        for i, layer in enumerate(self.layers):
            eng_emb = None
            if i in engram_futures:
                tbl, fut = engram_futures[i]
                got = fut.result() if isinstance(fut, Future) else fut
                eng_emb = tbl.finish(engram_shapes[i], got)
            lc = cache[i] if cache is not None else None
            hidden_hc = layer(
                hidden_hc,
                cache=lc,
                engram_embeds=eng_emb,
                engram_mask=engram_mask_mx,
                position_ids_3d=position_ids_3d,
            )

        if tail_only and not return_last_hidden:
            hidden_hc = hidden_hc[:, -1:, :, :]
        h = self.norm(self.hc_head(hidden_hc))
        if return_last_hidden:
            return (h[:, -1:, :] if tail_only else h), h
        return h


class HyV4FlashForCausalLM(nn.Module):
    """Top-level Hy4-flash CausalLM compatible with vMLX's direct, batched, and DFly2 engines."""

    model_type = "hy_v4_flash"

    def __init__(self, config: HyV4FlashConfig, *, mxfp4: bool = False) -> None:
        super().__init__()
        self.config = config
        self.model = HyV4FlashModel(config, mxfp4=mxfp4)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._dfly_drafter: Optional[Any] = None
        self.vision_tower: Optional[Any] = None

    def build_vl_inputs(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
    ) -> Tuple[mx.array, mx.array, int]:
        """Encode images via `vision_tower` and scatter visual embeddings + 3D M-RoPE positions."""
        if self.vision_tower is None:
            raise RuntimeError("Hy4-flash vision_tower is not loaded on this model instance")
        from .vision import compute_hy4_mrope_positions

        vis_feats = self.vision_tower(pixel_values, image_grid_thw)  # [1, N_vis, D]
        inputs_embeds = self.model.embed_tokens(input_ids)  # [1, L, D]
        vis_feats = vis_feats.astype(inputs_embeds.dtype)

        ids_list = input_ids[0].tolist()
        img_tok = int(self.config.image_token_id)
        vis_positions = [i for i, t in enumerate(ids_list) if int(t) == img_tok]
        if len(vis_positions) != int(vis_feats.shape[1]):
            raise ValueError(
                f"Hy4-flash VL placeholder mismatch: {len(vis_positions)} placeholders vs {int(vis_feats.shape[1])} vision tokens"
            )
        if vis_positions:
            idx_mx = mx.array(vis_positions, dtype=mx.int32)
            # Materialize a contiguous copy and splice visual features in one vectorized scatter
            emb_0 = mx.array(inputs_embeds[0])
            emb_0[idx_mx] = vis_feats[0]
            inputs_embeds = emb_0[None, :, :]

        merge_sz = int((self.config.vision_config or {}).get("spatial_merge_size", 2))
        grids = image_grid_thw.tolist() if isinstance(image_grid_thw, mx.array) else list(image_grid_thw)
        pos_3d, rope_delta = compute_hy4_mrope_positions(
            ids_list, grids, image_token_id=img_tok, merge_size=merge_sz
        )
        return inputs_embeds, pos_3d, rope_delta

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[HyV4LayerCache]:
        caches: List[HyV4LayerCache] = []
        for i in range(self.config.num_hidden_layers):
            if self.config.is_dsa_layer(i):
                caches.append(
                    HyV4LayerCache(
                        HyV4GrowingMLACache(self.config.compressed_indexer_compress_ratio),
                        is_dsa=True,
                    )
                )
            else:
                caches.append(
                    HyV4LayerCache(
                        HyV4RotatingMLACache(self.config.sliding_window),
                        is_dsa=False,
                    )
                )
        return caches

    def compute_logits(self, h: mx.array) -> mx.array:
        """Exact FP32-accumulated lm_head directly from BF16 weight (0 temp FP32 weight copy!)."""
        w = self.lm_head.weight
        if self.config.enable_lm_head_fp32:
            if w.dtype == mx.bfloat16 and int(h.shape[-1]) % 128 == 0 and mx.metal.is_available() and mx.default_device() == mx.gpu:
                z4 = getattr(self, "_lm_dummy4", None)
                if z4 is None:
                    z4 = mx.zeros((4,), dtype=mx.float32)
                    self._lm_dummy4 = z4
                return ernie_lm_head_forward(h, w, z4, z4, apply_multimax=False)
            return (h.astype(mx.float32) @ w.astype(mx.float32).T)
        return self.lm_head(h)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Sequence[HyV4LayerCache]] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        prefill_last_only = bool(kwargs.pop("prefill_last_logits_only", False))
        pos_3d = kwargs.pop("position_ids_3d", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is not None and image_grid_thw is not None and inputs_embeds is None and self.vision_tower is not None:
            inputs_embeds, pos_3d, rope_delta = self.build_vl_inputs(inputs, pixel_values, image_grid_thw)
            if cache is not None:
                for lc in cache:
                    lc.rope_delta = rope_delta
        h = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids_3d=pos_3d,
            tail_only=prefill_last_only,
        )
        return self.compute_logits(h)

    def forward_with_hidden(
        self,
        inputs: mx.array,
        cache: Optional[Sequence[HyV4LayerCache]],
        *,
        tail_only: bool = False,
        inputs_embeds: Optional[mx.array] = None,
        position_ids_3d: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Run target forward and return `(logits, post_norm_hidden)` for DFly2 MTPFusion."""
        h_logits, h_all = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids_3d=position_ids_3d,
            tail_only=tail_only,
            return_last_hidden=True,
        )
        return self.compute_logits(h_logits), h_all


HyV4FlashForCausalLM.__call__._vmlx_prefill_last_logits_only = True
