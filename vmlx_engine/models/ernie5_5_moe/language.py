# SPDX-License-Identifier: Apache-2.0
# Architecture reference: Baidu vLLM-ERNIE & upstream MIT mlx-vlm ernie5_5_moe / deepseek_v4
# (Copyright 2026 Neywa Labs Inc. / Apple Inc.).
"""High-throughput Apple Silicon runtime for ERNIE 5.5 Flash (ernie5_5_moe)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .cache import ErnieKVCache, ErnieLayerCache, ErniePoolingCache, ErnieRotatingCache
from .config import ModelConfig
from .kernels import (
    bf16_linear_fp32,
    ernie_hc_expand,
    ernie_lm_head_forward,
    ernie_mhc_collapse_norm,
    ernie_multi_router_scores,
    fused_situ,
    round_e4m3_per128,
)


def _scaled_dot_product_attention(q, k, v, *, scale: float, mask=None, sinks=None):
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sinks
    )


def _align_local_mask(mask: Optional[mx.array], local_len: int) -> Optional[mx.array]:
    if mask is None:
        return None
    current_len = mask.shape[-1]
    if current_len == local_len:
        return mask
    if current_len > local_len:
        return mask[..., -local_len:]
    pad_shape = (*mask.shape[:-1], local_len - current_len)
    pad = mx.ones(pad_shape, dtype=mask.dtype) if mask.dtype == mx.bool_ else mx.zeros(pad_shape, dtype=mask.dtype)
    return mx.concatenate([pad, mask], axis=-1)


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], total_len: int) -> Optional[mx.array]:
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    B, H, L, S = mask.shape
    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, total_len - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask[None, None], (B, H, L, total_len - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, total_len - S))
    return mx.concatenate([mask, pool_mask], axis=-1)


def _create_sliding_causal_mask(B: int, L: int, S: int, window_size: int) -> Optional[mx.array]:
    """Boolean causal sliding-window mask [1, 1, L, S] matching upstream mlx_vlm semantics."""
    if L == 1:
        # Single token sees all S keys in local rotating cache (S <= window_size).
        return None
    q_idx = mx.arange(S - L, S, dtype=mx.int32)[:, None]
    k_idx = mx.arange(S, dtype=mx.int32)[None, :]
    m = (k_idx <= q_idx) & (k_idx > (q_idx - window_size))
    return m[None, None]


class MultiLinear(nn.Module):
    """Per-head / per-group batched linear projection (weight: [groups, out_dims, in_dims])."""

    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        self.weight = mx.zeros((num_heads, output_dims, input_dims), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        return x @ self.weight


class MXFP8MultiLinear(nn.Module):
    """3-D batched native MXFP8 MultiLinear ([groups, out_dims, in_dims/4] u32 + scales u8).

    Keeps ERNIE's 1.242 GB `linear_o_group_proj` (`wo_a`) in exact-value MXFP8,
    halving both resident RAM and per-token DRAM reads.
    """

    def __init__(self, num_heads: int, output_dims: int, input_dims: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.weight = mx.zeros((num_heads, output_dims, input_dims // 4), dtype=mx.uint32)
        self.scales = mx.zeros((num_heads, output_dims, input_dims // 32), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x, self.weight, scales=self.scales,
            transpose=True, group_size=32, bits=8, mode="mxfp8",
        )


class ErnieRoPE(nn.Module):
    """DeepseekV4/ERNIE YaRN + freq_scale traditional RoPE on the trailing rope_dim coordinates."""

    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ) -> None:
        super().__init__()
        self.dims = int(dims)
        self.freq_scale = int(freq_scale)

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")
        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            orig_max = scaling_config["original_max_position_embeddings"]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def _corr_dim(n_rot: float) -> float:
                return dims * math.log(orig_max / (n_rot * 2.0 * math.pi)) / (2.0 * math.log(base))

            low = max(math.floor(_corr_dim(beta_fast)), 0)
            high = min(math.ceil(_corr_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001
            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1.0 - mx.clip(ramp, 0.0, 1.0)
            inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported ERNIE RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache: Dict[Tuple[int, bool], mx.array] = {}

    def _get_freqs(self, head_dim: int, inverse: bool) -> mx.array:
        key = (head_dim, inverse)
        cached = self._freqs_cache.get(key)
        if cached is None:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            mx.eval(f)
            self._freqs_cache[key] = f
            cached = f
        return cached

    def __call__(self, x: mx.array, offset: Any = 0, inverse: bool = False) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        off = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x, head_dim, traditional=True, base=None, scale=1.0, offset=off, freqs=freqs
        )


class SiTU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.beta = float(config.activation_situ_beta)
        self.linear_beta = float(config.activation_situ_linear_beta)

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        return fused_situ(up, gate, self.beta, self.linear_beta)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation = SiTU(config)
        self._gate_up_fused: Optional[nn.Module] = None

    def __call__(self, x: mx.array) -> mx.array:
        if self._gate_up_fused is not None:
            gu = self._gate_up_fused(x)
            half = gu.shape[-1] // 2
            return self.down_proj(self.activation(gu[..., half:], gu[..., :half]))
        return self.down_proj(self.activation(self.up_proj(x), self.gate_proj(x)))


class MultiRouter(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size), dtype=mx.bfloat16)
        self.weight_1 = mx.zeros_like(self.weight)
        self.routed_scaling_factor_param = mx.full(
            (config.n_routed_experts,), config.routed_scaling_factor, dtype=mx.float32
        )
        self.top_k = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob

    def __call__(self, x: mx.array, bias: mx.array) -> Tuple[mx.array, mx.array]:
        scores = ernie_multi_router_scores(x, self.weight, self.weight_1)
        ids = mx.argpartition(-(scores + bias), kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        ids = mx.stop_gradient(ids)
        weights = mx.take_along_axis(scores, ids, axis=-1)
        if self.normalize:
            weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
        return ids, weights * self.routed_scaling_factor_param[ids]


class LatentMoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = MultiRouter(config)
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,), dtype=mx.float32)
        self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=False)
        # switch_mlp is replaced at load time by ErnieSwitchGLU (fused JANGT/JANGH + SiTU + E4M3)
        self.switch_mlp: Any = None
        self.latent_norm = nn.RMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=False)
        self.shared_experts = MLP(config, config.n_shared_experts * config.moe_intermediate_size)
        self._shared_fc1_fused: Optional[nn.Module] = None
        self._shared_fc1_splits: Tuple[int, int] = (0, 0)
        self._shared_fc2_fused: Optional[nn.Module] = None

    def __call__(self, x: mx.array, router_bias: Optional[mx.array] = None) -> mx.array:
        bias = self.e_score_correction_bias if router_bias is None else router_bias
        ids, weights = self.gate(x, bias)
        if self._shared_fc1_fused is not None:
            proj = self._shared_fc1_fused(x)
            s1, s2 = self._shared_fc1_splits
            sh_gate = proj[..., :s1]
            sh_up = proj[..., s1:s2]
            latent = proj[..., s2:]
            shared_mid = self.shared_experts.activation(sh_up, sh_gate)
        else:
            latent = self.fc1_latent_proj(x)
            shared_mid = None

        if getattr(self.switch_mlp, "accepts_route_weights", False):
            combined = self.switch_mlp(latent, ids, weights)
        else:
            outputs = self.switch_mlp(latent, ids)
            combined = (outputs.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(x.dtype)

        normed_combined = self.latent_norm(combined)
        if self._shared_fc2_fused is not None and shared_mid is not None:
            return self._shared_fc2_fused(mx.concatenate([shared_mid, normed_combined], axis=-1))
        shared_out = (
            self.shared_experts.down_proj(shared_mid)
            if shared_mid is not None
            else self.shared_experts(x)
        )
        return shared_out + self.fc2_latent_proj(normed_combined)


class HyperConnection(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.mult = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        count = (2 + self.mult) * self.mult
        self.fn = mx.zeros((count, self.mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((count,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def collapse_and_norm(self, x: mx.array, norm: nn.RMSNorm) -> Tuple[mx.array, mx.array, mx.array]:
        if self.mult == 4 and x.shape[-1] % 1024 == 0:
            return ernie_mhc_collapse_norm(
                x, self.fn, self.scale, self.base, norm.weight,
                hc_iters=self.iters, hc_eps=self.eps, norm_eps=self.norm_eps,
            )
        flat = x.astype(mx.float32).flatten(-2)
        mixes = (flat @ self.fn.T) / (mx.sqrt(mx.mean(flat * flat, axis=-1, keepdims=True)) + self.norm_eps)
        pre_s, post_s, comb_s = self.scale[0], self.scale[1], self.scale[2]
        pre = mx.sigmoid(mixes[..., : self.mult] * pre_s + self.base[: self.mult]) + self.eps
        post = 2.0 * mx.sigmoid(mixes[..., self.mult : 2 * self.mult] * post_s + self.base[self.mult : 2 * self.mult])
        comb = mixes[..., 2 * self.mult :].reshape(*mixes.shape[:-1], self.mult, self.mult) * comb_s + self.base[2 * self.mult :].reshape(self.mult, self.mult)
        comb = mx.softmax(comb, axis=-1, precise=True) + self.eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + self.eps)
        for _ in range(max(self.iters - 1, 0)):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + self.eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.eps)
        collapsed = (pre[..., None] * x.astype(mx.float32)).sum(axis=-2).astype(x.dtype)
        return norm(collapsed), post, comb


class HyperHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros((self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)


class VirtualHeads(nn.Module):
    def __init__(self, heads: int, rank: int) -> None:
        super().__init__()
        self.U = mx.zeros((heads, rank), dtype=mx.bfloat16)
        self.V = mx.zeros((heads, rank), dtype=mx.bfloat16)
        self._folded_mix: Optional[mx.array] = None

    def fold(self) -> None:
        mix = (mx.eye(self.U.shape[0]) + self.V.astype(mx.float32) @ self.U.astype(mx.float32).T).astype(self.U.dtype)
        mx.eval(mix)
        self._folded_mix = mix

    def __call__(self, x: mx.array) -> mx.array:
        mix = self._folded_mix
        if mix is None:
            mix = (mx.eye(self.U.shape[0]) + self.V.astype(mx.float32) @ self.U.astype(mx.float32).T).astype(x.dtype)
        elif mix.dtype != x.dtype:
            mix = mix.astype(x.dtype)
        return mix @ x


class HCACompressor(nn.Module):
    def __init__(self, config: ModelConfig, compress_ratio: int, head_dim: int) -> None:
        super().__init__()
        self.compress_ratio = int(compress_ratio)
        self.head_dim = int(head_dim)
        self.wkv = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, head_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = ErnieRoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )
        self._w_cat: Optional[mx.array] = None

    def prepare_fused(self) -> None:
        if self.wkv is not None and self.wgate is not None:
            w_cat = mx.concatenate([self.wkv.weight, self.wgate.weight], axis=0)
            mx.eval(w_cat)
            self._w_cat = w_cat
            self.wkv = None
            self.wgate = None

    def __call__(self, x: mx.array, pool_cache: Optional[ErniePoolingCache], offset: int) -> mx.array:
        B = x.shape[0]
        if self._w_cat is not None:
            both = bf16_linear_fp32(x, self._w_cat)
            kv = both[..., : self.head_dim]
            gate = both[..., self.head_dim :]
        else:
            kv = bf16_linear_fp32(x, self.wkv.weight)
            gate = bf16_linear_fp32(x, self.wgate.weight)

        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            kv, gate, base = kv[:, :usable], gate[:, :usable], offset
        else:
            kv, gate, base = pool_cache.accumulate_windows(kv, gate, offset)

        if kv.shape[1] == 0:
            pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            kv = kv.reshape(B, -1, self.compress_ratio, self.head_dim)
            gate = gate.reshape(B, -1, self.compress_ratio, self.head_dim)
            weights = mx.softmax(gate + self.ape.astype(mx.float32), axis=-2, precise=True)
            pooled = self.norm((kv * weights).sum(axis=-2))
            pooled = self.rope(pooled[:, None], offset=base).squeeze(1).astype(x.dtype)
        return pool_cache.update_and_fetch(pooled) if pool_cache is not None else pooled


class HCAAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim ** -0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a: Any = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=config.attention_bias)
        self.o_gate = nn.Linear(config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False)
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.rope = ErnieRoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = HCACompressor(config, self.compress_ratio, self.head_dim)
        # Fused sibling MXFP8 projections installed by prepare_runtime()
        self._qa_kv_fused: Optional[nn.Module] = None
        self._qb_ogate_fused: Optional[nn.Module] = None

    def __call__(self, x: mx.array, cache: Optional[ErnieLayerCache] = None) -> mx.array:
        B, L, _ = x.shape
        local: Optional[ErnieRotatingCache] = cache[0] if cache is not None else None
        pool: Optional[ErniePoolingCache] = cache[1] if cache is not None else None
        offset = int(local.offset) if local is not None else 0

        if self._qa_kv_fused is not None:
            qa_kv = self._qa_kv_fused(x)
            qr = self.q_norm(qa_kv[..., : self.config.q_lora_rank])
            kv_raw = qa_kv[..., self.config.q_lora_rank :]
        else:
            qr = self.q_norm(self.wq_a(x))
            kv_raw = self.wkv(x)

        if self._qb_ogate_fused is not None:
            qb_og = self._qb_ogate_fused(qr)
            q_flat = qb_og[..., : self.n_heads * self.head_dim]
            ogate = qb_og[..., self.n_heads * self.head_dim :]
        else:
            q_flat = self.wq_b(qr)
            ogate = self.o_gate(qr)

        q = q_flat.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps).transpose(0, 2, 1, 3)
        q = self.rope(q, offset)
        kv = self.rope(self.kv_norm(kv_raw)[:, None], offset)

        if local is not None:
            kv, _ = local.update_and_fetch(kv)

        S_local = kv.shape[2]
        mask = _create_sliding_causal_mask(B, L, S_local, self.config.sliding_window)
        pooled = self.compressor(x, pool, offset)
        if pooled.shape[1] > 0:
            if pool is not None:
                pool_mask = pool.make_mask(L, offset)
            else:
                pool_mask = mx.arange(pooled.shape[1])[None] < (
                    mx.arange(1, L + 1)[:, None] // self.compress_ratio
                )
            if mask is None and pool_mask is not None:
                mask = mx.ones((B, 1, L, S_local), dtype=mx.bool_)
            kv = mx.concatenate((kv, pooled[:, None]), axis=2)
            mask = _extend_mask(mask, pool_mask, kv.shape[2])

        out = _scaled_dot_product_attention(
            q, kv, kv, scale=self.scale, mask=mask, sinks=self.attn_sink.astype(q.dtype)
        )
        out = self.rope(out, offset, inverse=True).transpose(0, 2, 1, 3)
        out = self.vha(out).reshape(B, L, self.o_groups, -1).transpose(0, 2, 1, 3)
        out = self.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out * mx.sigmoid(ogate))


class Indexer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.index_n_heads
        self.dim = config.index_head_dim
        self.topk = config.index_topk
        self.window = config.sliding_window
        self.wq_b = nn.Linear(config.hybrid_mla_q_lora_rank, self.heads * self.dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.rope = nn.RoPE(config.hybrid_mla_qk_rope_head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ErnieKVCache] = None,
        *,
        q_idx_pre: Optional[mx.array] = None,
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        B, L, _ = x.shape
        offset = int(cache.offset) if cache is not None else 0
        q_raw = q_idx_pre if q_idx_pre is not None else self.wq_b(qr)
        q = q_raw.reshape(B, L, self.heads, self.dim).transpose(0, 2, 1, 3)
        k = self.k_norm(self.wk(x))[:, None]
        q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), dtype=k.dtype))
        S = k.shape[2]
        if S <= self.topk + self.window:
            return mask, None

        delta = mx.arange(S)[None, :] - mx.arange(S - L, S)[:, None]
        recent = (delta <= 0) & (delta > -self.window)
        older = delta <= -self.window
        valid = older if mask is None else (older & mask)

        # Compute ReLU(q @ k^T) weighted across heads in FP32
        # For prefill (B == 1, L >= 2, S >= 32), dispatch to vMLX's M5 MPP matrix-unit
        # DSA indexer kernel (avoids materializing the [1, 32, L, S] FP32 tensor!).
        weights = self.weights_proj(x).astype(mx.float32) * ((self.heads * self.dim) ** -0.5)
        scores = None
        if B == 1 and L >= 2 and S >= 32:
            from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
            scores = dsa_pool_index_scores(
                q.transpose(0, 2, 1, 3), k[:, 0], weights, scale=1.0, bf16_operands=True
            )
            if scores is not None:
                scores = scores[:, None, :, :]
        if scores is None:
            scores = mx.maximum(q @ k.swapaxes(-1, -2), 0).astype(mx.float32)
            scores = (scores * weights.transpose(0, 2, 1)[..., None]).sum(axis=1, keepdims=True)
        scores = mx.where(valid, scores, -mx.inf)
        ids = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]
        sparse = mx.put_along_axis(mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1)
        sparse = (sparse & older) | recent
        sparse = sparse if mask is None else (sparse & mask)
        if L == 1:
            recent_ids = mx.broadcast_to(mx.arange(S - self.window, S), (B, 1, 1, self.window))
            return sparse, mx.concatenate((ids, recent_ids), axis=-1)
        return sparse, None


class MLAAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.hybrid_mla_num_attention_heads
        self.nope = config.hybrid_mla_qk_nope_head_dim
        self.rope_dim = config.hybrid_mla_qk_rope_head_dim
        self.latent = config.hybrid_mla_kv_lora_rank
        self.value_dim = config.hybrid_mla_v_head_dim
        qr = config.hybrid_mla_q_lora_rank
        self.scale = (self.nope + self.rope_dim) ** -0.5

        self.q_a_proj = nn.Linear(config.hidden_size, qr, bias=False)
        self.q_a_layernorm = nn.RMSNorm(qr, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(qr, self.heads * (self.nope + self.rope_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.latent + self.rope_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.latent, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent, self.heads)
        self.unembed_out = MultiLinear(self.latent, self.value_dim, self.heads)
        self.wo_b = nn.Linear(self.heads * self.value_dim, config.hidden_size, bias=False)
        self.o_gate = nn.Linear(qr, self.heads * self.value_dim, bias=False)
        self.vha = VirtualHeads(self.heads, config.vha_rank)
        self.indexer = Indexer(config)
        self.attn_sink = mx.zeros((self.heads,), dtype=mx.float32)
        self.rope = nn.RoPE(self.rope_dim, traditional=True, base=config.rope_theta)
        # Fused sibling MXFP8 projections installed by prepare_runtime()
        self._qa_kva_fused: Optional[nn.Module] = None
        self._qb_og_idx_fused: Optional[nn.Module] = None

    def __call__(self, x: mx.array, cache: Optional[ErnieLayerCache] = None) -> mx.array:
        B, L, _ = x.shape
        latent_cache: Optional[ErnieKVCache] = cache[0] if cache is not None else None
        index_cache: Optional[ErnieKVCache] = cache[1] if cache is not None else None
        offset = int(latent_cache.offset) if latent_cache is not None else 0

        if self._qa_kva_fused is not None:
            qkva = self._qa_kva_fused(x)
            q_rank = self.q_a_layernorm.weight.shape[0]
            qr = self.q_a_layernorm(qkva[..., :q_rank])
            kva = qkva[..., q_rank:]
        else:
            qr = self.q_a_layernorm(self.q_a_proj(x))
            kva = self.kv_a_proj_with_mqa(x)

        if self._qb_og_idx_fused is not None:
            qoi = self._qb_og_idx_fused(qr)
            s_qb = self.heads * (self.nope + self.rope_dim)
            s_og = self.heads * self.value_dim
            q_raw = qoi[..., :s_qb]
            ogate = qoi[..., s_qb : s_qb + s_og]
            q_idx_pre = qoi[..., s_qb + s_og :]
        else:
            q_raw = self.q_b_proj(qr)
            ogate = self.o_gate(qr)
            q_idx_pre = None

        q = q_raw.reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.nope], axis=-1)
        kv, k_pe = mx.split(kva, [self.latent], axis=-1)
        kv = self.kv_a_layernorm(kv)[:, None]
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe[:, None], offset)
        if latent_cache is not None:
            kv, k_pe = latent_cache.update_and_fetch(kv, k_pe)

        S = kv.shape[2]
        mask = None if L == 1 else (mx.arange(S)[None, :] <= mx.arange(S - L, S)[:, None])[None, None]
        mask, indices = self.indexer(x, qr, mask, index_cache, q_idx_pre=q_idx_pre)
        if indices is not None:
            take = indices[:, :, 0, :, None]
            kv = mx.take_along_axis(kv, take, axis=2)
            k_pe = mx.take_along_axis(k_pe, take, axis=2)
            mask = mx.take_along_axis(mask, indices, axis=-1)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, -mx.inf)

        # Keep absorbed latent MLA for BOTH decode (L == 1) AND DSpark verify (1 < L <= 16)
        # (and whenever S is much larger than L): avoids 24x KV expansion and preserves decode order!
        use_absorbed = (L <= 16)
        if use_absorbed:
            q_nope = self.embed_q(q_nope)
            k = v = kv
        else:
            k = self.embed_q(kv, transpose=False)
            v = self.unembed_out(kv)

        out = _scaled_dot_product_attention(
            q_nope, k, v, scale=self.scale, mask=pe_scores, sinks=self.attn_sink.astype(q_nope.dtype)
        )
        if use_absorbed:
            out = self.unembed_out(out)
        out = self.vha(out.transpose(0, 2, 1, 3)).flatten(-2)
        return self.wo_b(out * mx.sigmoid(ogate))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn: Any = (
            MLAAttention(config)
            if config.compress_ratios[layer_idx] == -2
            else HCAAttention(config, layer_idx)
        )
        self.ffn: Any = (
            MLP(config, config.intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else LatentMoE(config)
        )
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, x: mx.array, cache: Optional[ErnieLayerCache] = None, router_bias: Optional[mx.array] = None) -> mx.array:
        h, post, comb = self.attn_hc.collapse_and_norm(x, self.attn_norm)
        h = self.attn(h, cache)
        x = ernie_hc_expand(h, x, post, comb)
        h, post, comb = self.ffn_hc.collapse_and_norm(x, self.ffn_norm)
        h = self.ffn(h) if router_bias is None else self.ffn(h, router_bias)
        return ernie_hc_expand(h, x, post, comb)


class ErnieModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.hc_head = HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[ErnieLayerCache]] = None,
        inputs_embeds: Optional[mx.array] = None,
        capture_taps: Sequence[int] = (),
        tail_only: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.config.hc_mult, h.shape[-1]))
        if cache is None:
            cache = [None] * len(self.layers)
        tap_set = set(capture_taps)
        taps: List[mx.array] = []
        for ordinal, (layer, layer_cache) in enumerate(zip(self.layers, cache), 1):
            h = layer(h, layer_cache)
            if ordinal in tap_set:
                taps.append(mx.mean(h.astype(mx.float32), axis=-2).astype(h.dtype))
        if tail_only and h.shape[1] > 1:
            h = h[:, -1:, :, :]
        out = self.norm(self.hc_head(h))
        if tap_set:
            return out, mx.concatenate(taps, axis=-1)
        return out


class MultiMax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multimax_ranges = mx.zeros((4,), dtype=mx.float32)
        self.multimax_ts = mx.zeros((4,), dtype=mx.float32)

    def __call__(self, logits: mx.array) -> mx.array:
        logits = logits.astype(mx.float32)
        r, t = self.multimax_ranges, self.multimax_ts
        return (
            logits
            + t[0] * mx.maximum(r[0] - logits, 0)
            + t[1] * mx.maximum(logits - r[1], 0)
            + t[2] * mx.maximum(r[2] - logits, 0) ** 2
            + t[3] * mx.maximum(logits - r[3], 0) ** 2
        )


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = ErnieModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.multimax = MultiMax()

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[ErnieLayerCache]] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        prefill_last_only = bool(kwargs.pop("prefill_last_logits_only", False))
        h = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds, tail_only=prefill_last_only)
        return ernie_lm_head_forward(
            h, self.lm_head.weight, self.multimax.multimax_ranges, self.multimax.multimax_ts,
            apply_multimax=True,
        )

    def forward_with_taps(
        self,
        inputs: mx.array,
        cache: Optional[List[ErnieLayerCache]],
        taps: Sequence[int],
        *,
        tail_only: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        h, features = self.model(inputs, cache=cache, capture_taps=taps, tail_only=tail_only)
        logits = ernie_lm_head_forward(
            h, self.lm_head.weight, self.multimax.multimax_ranges, self.multimax.multimax_ts,
            apply_multimax=True,
        )
        return logits, features

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[ErnieLayerCache]:
        return [
            (
                ErnieLayerCache(ErnieKVCache(), ErnieKVCache(), is_mla=True)
                if ratio == -2
                else ErnieLayerCache(
                    ErnieRotatingCache(max_size=self.config.sliding_window),
                    ErniePoolingCache(ratio),
                    is_mla=False,
                )
            )
            for ratio in self.config.compress_ratios
        ]

    @staticmethod
    def sanitize_dict(config: ModelConfig, weights: Dict[str, mx.array], layers_by_idx: Optional[Dict[int, DecoderLayer]] = None) -> Dict[str, mx.array]:
        """Sanitize ERNIE 5.5 raw checkpoint keys into vMLX LanguageModel layout (keeping MXFP8 intact)."""
        weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
        out: Dict[str, mx.array] = {}
        for key, value in weights.items():
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match and int(match[1]) >= config.num_hidden_layers:
                continue
            if key.endswith("_scale_inv"):
                if key.removesuffix("_scale_inv") not in weights:
                    raise ValueError(f"FP8 scale without its weight: {key}")
                continue
            scale_key = key + "_scale_inv"
            if scale_key in weights:
                value = _dequantize_fp8_128(value, weights[scale_key]).astype(mx.bfloat16)
            elif value.dtype == mx.uint8 and not key.endswith(".scales"):
                raise ValueError(f"FP8 weight without its scale: {key}")

            if key == "model.mhc_contract.hc_head_fn":
                value = value.T
            key = key.replace("model.lm_head.", "multimax.")
            key = key.replace("model.mhc_contract.hc_head_", "model.hc_head.")
            for src, dst in (("self_attention", "attn"), ("mlp", "ffn")):
                prefix = f".{src}_hyper_connection."
                if prefix in key:
                    key = key.replace(prefix, f".{dst}_hc.")
                    if key.endswith("mapping_proj.weight"):
                        key = key.replace("mapping_proj.weight", "fn")
                        value = value.T
                    key = key.replace(f".{dst}_hc.bias", f".{dst}_hc.base")
            remap = {
                ".input_layernorm.": ".attn_norm.",
                ".post_attention_layernorm.": ".ffn_norm.",
                ".block_sparse_moe.": ".ffn.",
                ".mlp.": ".ffn.",
                ".self_attn.": ".attn.",
                ".core_attention.": ".",
                ".linear_q_down_proj.": ".wq_a.",
                ".linear_q_up_proj.": ".wq_b.",
                ".linear_kv_proj.": ".wkv.",
                ".q_layernorm.": ".q_norm.",
                ".kv_layernorm.": ".kv_norm.",
                ".compressor.linear_wkv.": ".compressor.wkv.",
                ".compressor.linear_wgate.": ".compressor.wgate.",
                ".attn.gate_proj.": ".attn.o_gate.",
                ".attn.o_proj.": ".attn.wo_b.",
                ".attn.vha_postmix_U": ".attn.vha.U",
                ".attn.vha_postmix_V": ".attn.vha.V",
                ".attn.softmax_offset": ".attn.attn_sink",
            }
            for src, dst in remap.items():
                key = key.replace(src, dst)
            if key.endswith(".attn.linear_o_group_proj"):
                key = key.replace(".linear_o_group_proj", ".wo_a.weight")
                value = value.reshape(config.o_groups, config.o_lora_rank, -1)
            elif key.endswith(".attn.linear_o_group_proj.scales"):
                key = key.replace(".linear_o_group_proj.scales", ".wo_a.scales")
                value = value.reshape(config.o_groups, config.o_lora_rank, -1)
            for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                key = key.replace(f".shared_experts.{src}.", f".shared_experts.{dst}.")
            out[key] = value

        for i in range(config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            for kind in ("attn", "ffn"):
                keys = [f"{prefix}.{kind}_hc.alpha_{p}" for p in ("pre", "post", "res")]
                if keys[0] in out:
                    out[f"{prefix}.{kind}_hc.scale"] = mx.concatenate([out.pop(k).reshape(-1) for k in keys])

        for i, ratio in enumerate(config.compress_ratios):
            key = f"model.layers.{i}.attn.kv_b_proj.weight"
            if ratio == -2 and key in out:
                w = out.pop(key)
                scale_key = key.removesuffix("weight") + "scales"
                if scale_key in out:
                    scales = out.pop(scale_key)
                    w = mx.dequantize(w, scales, None, group_size=32, bits=8, mode="mxfp8")
                heads = config.hybrid_mla_num_attention_heads
                nope = config.hybrid_mla_qk_nope_head_dim
                vdim = config.hybrid_mla_v_head_dim
                latent = config.hybrid_mla_kv_lora_rank
                w = w.reshape(heads, nope + vdim, latent)
                k_part, v_part = mx.split(w, [nope], axis=1)
                p = f"model.layers.{i}.attn."
                out[p + "embed_q.weight"] = mx.contiguous(k_part.swapaxes(-1, -2))
                out[p + "unembed_out.weight"] = mx.contiguous(v_part)
        return out


LanguageModel.__call__._vmlx_prefill_last_logits_only = True


def _dequantize_fp8_128(weight_u8: mx.array, scale_inv: mx.array) -> mx.array:
    """Dequantize 128x128 block-FP8 E4M3 weight with FP32 scale_inv."""
    rows, cols = weight_u8.shape
    w_f32 = mx.from_fp8(weight_u8, dtype=mx.float32)
    s_rep = mx.repeat(mx.repeat(scale_inv.astype(mx.float32), 128, axis=0), 128, axis=1)[:rows, :cols]
    return w_f32 * s_rep
