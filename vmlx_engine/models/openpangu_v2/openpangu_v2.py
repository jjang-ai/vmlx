# SPDX-License-Identifier: Apache-2.0
"""openPangu-2.0-Flash (openpangu_v2 / OpenPanguV2ForCausalLM) MLX runtime.

92B MoE (6B active), 512K ctx. Ported 1:1 from the numerically-proven
vmlx-swift implementation (branch feat/openpangu-v2) which was validated live
against the official Ascend reference (gitcode ascend-tribe/openPangu-2.0-Infer,
omni_npu naive fallbacks). Architecture:

  - MLA (q_lora 1024, kv_lora 512, qk 128 nope + 64 rope, v 128), EXPANDED
    per-head K/V caching (48 heads x 192/128), scale = 192^-0.5, no mscale.
  - 3 stateful causal depthwise convs per layer (qa_conv / compresskv_conv /
    o_conv, kernel 3), applied on the RAW latent BEFORE the layernorm, with
    residual (y = conv(x) + x). Path-dependent state lives in the layer cache.
  - 128 learned attention sinks (param_sink_compressed_kv / param_sink_k_pe)
    prepended to every layer's K/V, position-free (no RoPE), ALWAYS visible —
    boolean masks must prepend True columns (the Swift prefill-mask-polarity
    bug: a False sink column silently blinds the model to the prompt).
  - DSA + SWA hybrid 1:2: 16 dsa_layers (full attention + lightning indexer
    top-2048, no-op for K <= 2048) and 30+3 swa_layers (windows 512 / 2048).
  - mHC 4-stream hyper-connections per layer (attn + mlp) + global merge;
    Sinkhorn math reused from jang_tools.dsv4 (fused Metal kernel).
  - Sandwich norm + block_post_layernorm ([4*hidden], applied to the
    FLATTENED 4-stream residual) on 9 layers.
  - MoE: 256 routed + 1 shared, sigmoid gate, select on biased scores, weight
    with UNBIASED scores, renormalize x routed_scaling_factor (2.5), first 2
    layers dense.
  - MTP depth 3 (layers 46-48): dropped at sanitize (mtp_mode off). Native
    MTP stays detection-only for this family (same bucket as DSV4).

Diagnostics (env): OPENPANGU_NO_CONVS, OPENPANGU_NO_SINKS, OPENPANGU_NO_INDEXER,
OPENPANGU_ROPE_TRAD, OPENPANGU_LOGIT_PROBE.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_causal_mask
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

try:  # package import (registered under mlx_lm.models.openpangu_v2)
    from vmlx_engine.models.openpangu_v2.cache import (
        CONV_COMPRESS_KV,
        CONV_O,
        CONV_QA,
        OpenPanguV2LayerCache,
    )
except ImportError:  # direct file execution fallback
    from cache import (  # type: ignore
        CONV_COMPRESS_KV,
        CONV_O,
        CONV_QA,
        OpenPanguV2LayerCache,
    )

# mHC Sinkhorn split — the DSV4 hyper-connection math, numerically validated
# against the torch reference and shipped with a fused Metal kernel.
from jang_tools.dsv4.mlx_model import hc_split_sinkhorn

_HC_EPS = 1e-6  # sigmoid/softmax/sinkhorn epsilon (NOT rms_norm_eps)
logger = logging.getLogger(__name__)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "openpangu_v2"
    hidden_size: int = 2560
    num_hidden_layers: int = 46
    num_nextn_predict_layers: int = 3
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_attention_heads: int = 48
    vocab_size: int = 151552
    rms_norm_eps: float = 1e-5
    rope_theta: float = 6400000.0
    rope_interleave: bool = False
    max_position_embeddings: int = 524288
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    param_sink_number: int = 128
    index_n_heads: int = 24
    index_head_dim: int = 128
    index_topk: int = 2048
    dsa_layers: List[int] = field(default_factory=list)
    swa_layers: List[int] = field(default_factory=list)
    sliding_window_list: List[int] = field(default_factory=list)
    sliding_window: int = 512
    block_post_layernorm_idx: List[int] = field(default_factory=list)
    mhc_num_stream: int = 4
    mhc_recur_norm: int = 20
    mhc_use_gamma: bool = True
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    router_enable_expert_bias: bool = True
    router_sliding_window: int = 3
    first_k_dense_replace: int = 2
    sandwich_norm: bool = True
    tie_word_embeddings: bool = False

    def window_for(self, layer_idx: int) -> int:
        """Sliding window for a SWA layer (0 = DSA / full attention)."""
        if layer_idx in self.dsa_layers or layer_idx not in self.swa_layers:
            return 0
        pos = self.swa_layers.index(layer_idx)
        if pos < len(self.sliding_window_list):
            return int(self.sliding_window_list[pos])
        return int(self.sliding_window)


def _rope_traditional(args: ModelArgs) -> bool:
    # rope_interleave=False -> non-traditional split-half rope (Swift-proven).
    if os.environ.get("OPENPANGU_ROPE_TRAD") == "1":
        return True
    return bool(args.rope_interleave)


class OpenPanguCausalConv(nn.Module):
    """Short causal depthwise conv over the sequence axis with residual.

    Weight ships PyTorch [C, 1, k] and is transposed to MLX [C, k, 1] in
    sanitize. Reference fused op has residual_connection=1: y = conv(x) + x.
    During decode the trailing kernel-1 inputs come from the cache state.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=0, groups=channels, bias=False
        )

    def __call__(
        self, x: mx.array, state: Optional[mx.array]
    ) -> Tuple[mx.array, mx.array]:
        B, L, _ = x.shape
        pad = self.kernel_size - 1
        fresh = state is None
        if fresh:
            left = mx.zeros((B, pad, self.channels), dtype=x.dtype)
        else:
            left = state.astype(x.dtype)
        padded = mx.concatenate([left, x], axis=1)  # (B, pad+L, C)
        y = self.conv(padded)  # valid conv -> (B, L, C)
        # Proper standard causal_conv1d semantics, confirmed against the authoritative
        # omni-npu op (npu_mome.py: npu_fused_causal_conv1d, activation_mode="None",
        # bias=None, residual_connection=1, zero initial state on a fresh sequence):
        # boundary outputs at positions 0..k-2 are PARTIAL convolutions over the
        # zero-left-pad — they are NOT zeroed. (The `conv[:k-1]=0` in jointfix's
        # PanguTorchMOMEConv is a calibration-hook approximation — jointfix is
        # explicitly "hooks rather than serving-kernel parity" — not the serving op.)
        # Optional env override to A/B the two behaviors.
        if fresh and pad > 0 and L > 0 and os.environ.get("OPENPANGU_CONV_ZERO_BOUNDARY"):
            zlen = min(pad, L)
            zero_then_one = mx.concatenate(
                [mx.zeros((zlen,), dtype=y.dtype), mx.ones((L - zlen,), dtype=y.dtype)]
            )
            y = y * zero_then_one[None, :, None]
        new_state = padded[:, padded.shape[1] - pad :, :]
        return y + x, new_state


class OpenPanguV2Indexer(nn.Module):
    """DSA lightning indexer: top-`index_topk` key selection per query.

    No-op (returns None) whenever the key length is <= index_topk, so short
    contexts run exact full attention. Mirrors the proven deepseek_v32 (GLM-5.1)
    indexer with openpangu tensor names (wq_b / wk / k_norm / weights_proj).
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        # Official NPUPanguIndexer uses RMSNorm here. LayerNorm invents a bias
        # leaf that the checkpoint does not contain and changes indexer scores.
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self._logged_sparse_activation = False
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=_rope_traditional(args),
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=None,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ) -> Optional[mx.array]:
        b, s, _ = x.shape
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_norm(self.wk(x))
        k = mx.reshape(k, (b, 1, s, self.head_dim))

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((b, 1, s, 0), dtype=k.dtype))
        if k.shape[2] <= self.index_topk:
            return None
        if not self._logged_sparse_activation:
            logger.info(
                "openPangu DSA sparse indexer active: layer=%d key_length=%d "
                "topk=%d index_heads=%d",
                self.layer_idx,
                k.shape[2],
                self.index_topk,
                self.n_heads,
            )
            self._logged_sparse_activation = True
        weights_raw = self.weights_proj(x).astype(mx.float32) * (
            self.n_heads**-0.5 * self.softmax_scale
        )
        scores = None
        if b == 1 and k.shape[2] >= 32:
            if s == 1:
                from vmlx_engine.metal.dsa_index_scores import dsa_decode_index_scores
                scores = dsa_decode_index_scores(q.swapaxes(1, 2), k[:, 0], weights_raw, scale=1.0)
            else:
                from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
                scores = dsa_pool_index_scores(
                    q.swapaxes(1, 2), k[:, 0], weights_raw, scale=1.0, bf16_operands=(q.dtype == mx.bfloat16)
                )
            if scores is not None:
                scores = scores[:, None, :, :]
        if scores is None:
            scores = q @ k.swapaxes(-1, -2)
            scores = mx.maximum(scores, 0)
            weights = weights_raw.astype(scores.dtype).swapaxes(-1, -2)[..., None]
            scores = (scores * weights).sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, mx.array(-float("inf"), scores.dtype))
        if scores.dtype == mx.float32:
            from vmlx_engine.metal.sparse_topk import metal_radix_topk
            sel = metal_radix_topk(scores, self.index_topk)
            if sel is not None:
                return sel.astype(mx.int32)
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]


class OpenPanguV2Attention(nn.Module):
    """MLA + 3 causal convs + 128 prepended position-free sinks (+ DSA indexer)."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.is_dsa = layer_idx in args.dsa_layers
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.sink_count = args.param_sink_number
        self.scale = self.q_head_dim**-0.5

        h = args.hidden_size
        self.q_a_proj = nn.Linear(h, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            h, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, h, bias=False)

        self.qa_conv = OpenPanguCausalConv(self.q_lora_rank)
        self.compresskv_conv = OpenPanguCausalConv(self.kv_lora_rank)
        self.o_conv = OpenPanguCausalConv(self.num_heads * self.v_head_dim)

        self.param_sink_compressed_kv = mx.zeros(
            (self.sink_count, self.kv_lora_rank)
        )
        self.param_sink_k_pe = mx.zeros((self.sink_count, self.qk_rope_head_dim))

        self.indexer = OpenPanguV2Indexer(args, layer_idx) if self.is_dsa else None

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=_rope_traditional(args),
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=None,
        )

    def _sink_keys_values(self, dtype) -> Tuple[mx.array, mx.array]:
        """Expand the 128 learned sinks to per-head K (192) / V (128) rows.

        sink_k_nope/value = kv_b_proj(kv_a_layernorm(param_sink_compressed_kv));
        sink k_pe = raw param, NO rope (sinks are position-free)."""
        kv = self.kv_b_proj(
            self.kv_a_layernorm(self.param_sink_compressed_kv.astype(dtype))
        )
        kv = kv.reshape(
            self.sink_count, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 0, 2)  # (H, S, nope+v)
        sink_k_nope = kv[..., : self.qk_nope_head_dim]
        sink_v = kv[..., self.qk_nope_head_dim :]
        sink_k_pe = mx.broadcast_to(
            self.param_sink_k_pe.astype(dtype).reshape(
                1, self.sink_count, self.qk_rope_head_dim
            ),
            (self.num_heads, self.sink_count, self.qk_rope_head_dim),
        )
        sink_k = mx.concatenate([sink_k_nope, sink_k_pe], axis=-1)
        return (
            sink_k.reshape(1, self.num_heads, self.sink_count, self.q_head_dim),
            sink_v.reshape(1, self.num_heads, self.sink_count, self.v_head_dim),
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[OpenPanguV2LayerCache],
    ) -> mx.array:
        B, L, _ = x.shape
        no_convs = os.environ.get("OPENPANGU_NO_CONVS") is not None
        no_sinks = os.environ.get("OPENPANGU_NO_SINKS") is not None
        no_indexer = os.environ.get("OPENPANGU_NO_INDEXER") is not None

        # Q: q_a_proj -> qa_conv -> q_a_layernorm -> q_b_proj (conv on the RAW
        # latent BEFORE the layernorm — Swift port bug #10).
        q_lat = self.q_a_proj(x)
        if not no_convs:
            q_lat, qa_state = self.qa_conv(
                q_lat, cache.conv_states[CONV_QA] if cache else None
            )
            if cache is not None:
                cache.conv_states[CONV_QA] = qa_state
        q_lat = self.q_a_layernorm(q_lat)
        q = (
            self.q_b_proj(q_lat)
            .reshape(B, L, self.num_heads, self.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        # KV: kv_a_proj -> split -> compresskv_conv -> kv_a_layernorm -> kv_b_proj.
        kv_a = self.kv_a_proj_with_mqa(x)
        compressed_kv = kv_a[..., : self.kv_lora_rank]
        k_pe = kv_a[..., self.kv_lora_rank :]
        if not no_convs:
            compressed_kv, ckv_state = self.compresskv_conv(
                compressed_kv, cache.conv_states[CONV_COMPRESS_KV] if cache else None
            )
            if cache is not None:
                cache.conv_states[CONV_COMPRESS_KV] = ckv_state
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = (
            self.kv_b_proj(compressed_kv)
            .reshape(B, L, self.num_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        values = kv[..., self.qk_nope_head_dim :]

        offset = cache.kv.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)
        k_pe = mx.broadcast_to(
            k_pe, (B, self.num_heads, L, self.qk_rope_head_dim)
        )

        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        if cache is not None:
            keys, values = cache.kv.update_and_fetch(keys, values)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        # Base boolean mask over the REAL tokens actually present in `keys`.
        # Keys are contiguous in time and END at absolute position offset+L-1;
        # rotating (SWA) caches may hold fewer than offset+L keys.
        K = keys.shape[2]
        real_mask: Optional[mx.array] = None
        if L > 1:
            window = 0 if self.is_dsa else self.args.window_for(self.layer_idx)
            q_pos = mx.arange(offset, offset + L)[:, None]
            k_pos = mx.arange(offset + L - K, offset + L)[None, :]
            real_mask = q_pos >= k_pos
            if window:
                real_mask = real_mask & (q_pos < k_pos + window)

        # DSA lightning indexer: no-op for K <= index_topk.
        if self.indexer is not None and not no_indexer and cache is not None:
            topk_indices = self.indexer(
                x, q_lat, real_mask, cache=cache.indexer_kv
            )
            if topk_indices is not None:
                if L == 1:
                    idx = topk_indices[:, :, 0, :, None]
                    idx_k = mx.broadcast_to(
                        idx, (B, self.num_heads, idx.shape[-2], self.q_head_dim)
                    )
                    idx_v = mx.broadcast_to(
                        idx, (B, self.num_heads, idx.shape[-2], self.v_head_dim)
                    )
                    keys = mx.take_along_axis(keys, idx_k, axis=2)
                    values = mx.take_along_axis(values, idx_v, axis=2)
                else:
                    shape = list(topk_indices.shape)
                    shape[-1] = K
                    sparse = mx.zeros(shape, dtype=mx.bool_)
                    sparse = mx.put_along_axis(
                        sparse, topk_indices, mx.array(True), axis=-1
                    )
                    real_mask = (
                        sparse & real_mask if real_mask is not None else sparse
                    )

        # Prepend the 128 learned sinks: ALWAYS-VISIBLE columns (True on a
        # boolean mask — the Swift prefill polarity bug).
        eff_mask: Optional[mx.array] = real_mask
        if self.sink_count > 0 and not no_sinks:
            sink_k, sink_v = self._sink_keys_values(keys.dtype)
            sink_k = mx.broadcast_to(
                sink_k, (B,) + sink_k.shape[1:]
            )
            sink_v = mx.broadcast_to(
                sink_v, (B,) + sink_v.shape[1:]
            )
            keys = mx.concatenate([sink_k, keys], axis=2)
            values = mx.concatenate([sink_v, values], axis=2)
            if eff_mask is not None:
                visible = mx.ones(
                    eff_mask.shape[:-1] + (self.sink_count,), dtype=mx.bool_
                )
                eff_mask = mx.concatenate([visible, eff_mask], axis=-1)
            # mask None (decode) -> everything incl. sinks visible: correct.

        # MLA absorb-drift guard (Ornith-397B lesson): fp32 SDPA on decode.
        if L == 1:
            out = mx.fast.scaled_dot_product_attention(
                queries.astype(mx.float32),
                keys.astype(mx.float32),
                values.astype(mx.float32),
                scale=self.scale,
                mask=eff_mask,
            ).astype(x.dtype)
        else:
            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=eff_mask
            )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.v_head_dim)

        # o_conv on the concatenated head output, before o_proj.
        if not no_convs:
            out, o_state = self.o_conv(
                out, cache.conv_states[CONV_O] if cache else None
            )
            if cache is not None:
                cache.conv_states[CONV_O] = o_state
        return self.o_proj(out)


class OpenPanguV2MHCModule(nn.Module):
    """Per-layer mHC (attn_mhc_module / mlp_mhc_module), 4-stream.

    phi is a (quantized) Linear [mix=24, 4*hidden]; branch_alpha [3] is the
    per-field (pre/post/comb) scale; branch_beta [24] is the base directly;
    norm_gamma [4*hidden] is the learned RMS weight. Collapse/expand mirror
    DSV4 hyper-connections (validated); expand uses comb TRANSPOSED
    (new[j] = post[j]*block_out + sum_i comb[i,j] * residual[i])."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_stream = args.mhc_num_stream
        self.hidden_size = args.hidden_size
        self.mix_dim = (2 + args.mhc_num_stream) * args.mhc_num_stream
        self.sinkhorn_iters = args.mhc_recur_norm
        self.eps = args.rms_norm_eps
        wide = args.mhc_num_stream * args.hidden_size
        self.phi = nn.Linear(wide, self.mix_dim, bias=False)
        self.branch_alpha = mx.zeros((3,))
        self.branch_beta = mx.zeros((self.mix_dim,))
        self.norm_gamma = mx.ones((wide,))

    def _mixes(self, h: mx.array) -> mx.array:
        B, L = h.shape[0], h.shape[1]
        flat = h.reshape(B, L, self.num_stream * self.hidden_size)
        normed = mx.fast.rms_norm(
            flat.astype(mx.float32), self.norm_gamma.astype(mx.float32), self.eps
        )
        return self.phi(normed).astype(mx.float32)  # (B, L, mix)

    def collapse(self, h: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        dtype = h.dtype
        pre, post, comb = hc_split_sinkhorn(
            self._mixes(h),
            self.branch_alpha.astype(mx.float32),
            self.branch_beta.astype(mx.float32),
            hc_mult=self.num_stream,
            iters=self.sinkhorn_iters,
            eps=_HC_EPS,
        )
        x = (pre.astype(dtype)[..., None] * h).sum(axis=-2)
        return x, post, comb

    def expand(
        self,
        block_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ) -> mx.array:
        dtype = block_out.dtype
        comb_resid = comb.transpose(0, 1, 3, 2).astype(dtype) @ residual
        return post.astype(dtype)[..., None] * block_out[..., None, :] + comb_resid


class OpenPanguV2MergeMHC(nn.Module):
    """Global merge (model.merge_mhc_module): sigmoid-gated stream reduce.

    pre = sigmoid(mixes * branch_alpha_pre + branch_beta_pre) — NO +eps, NO
    sum-to-1 normalization (DSV4 HyperHead.reduce semantics)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_stream = args.mhc_num_stream
        self.hidden_size = args.hidden_size
        self.eps = args.rms_norm_eps
        wide = args.mhc_num_stream * args.hidden_size
        self.phi = nn.Linear(wide, args.mhc_num_stream, bias=False)
        self.branch_alpha_pre = mx.zeros((1,))
        self.branch_beta_pre = mx.zeros((args.mhc_num_stream,))
        self.norm_gamma = mx.ones((wide,))

    def __call__(self, h: mx.array) -> mx.array:
        dtype = h.dtype
        B, L = h.shape[0], h.shape[1]
        flat = h.reshape(B, L, self.num_stream * self.hidden_size)
        normed = mx.fast.rms_norm(
            flat.astype(mx.float32), self.norm_gamma.astype(mx.float32), self.eps
        )
        mixes = self.phi(normed).astype(mx.float32)
        pre = mx.sigmoid(
            mixes * self.branch_alpha_pre.astype(mx.float32)
            + self.branch_beta_pre.astype(mx.float32)
        )
        return (pre.astype(dtype)[..., None] * h).sum(axis=-2)


class OpenPanguV2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class OpenPanguV2MoE(nn.Module):
    """Sigmoid router: SELECT on scores + e_score_correction_bias, WEIGHT with
    unbiased scores, renormalize x routed_scaling_factor. Router `gate` ships
    fp16 (never quantized); e_score_correction_bias lives at mlp level in the
    bundle and on this module here — key paths match, no sanitize remap."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.use_expert_bias = args.router_enable_expert_bias
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        if args.n_shared_experts > 0:
            self.shared_experts = OpenPanguV2MLP(
                args.hidden_size, args.moe_intermediate_size * args.n_shared_experts
            )
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x)
        scores = mx.sigmoid(logits.astype(mx.float32))
        if self.use_expert_bias:
            choice = scores + self.e_score_correction_bias.astype(mx.float32)
        else:
            choice = scores
        inds = mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        picked = mx.take_along_axis(scores, inds, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            denom = picked.sum(axis=-1, keepdims=True) + 1e-20
            picked = picked / denom
        picked = (picked * self.routed_scaling_factor).astype(x.dtype)
        y = self.switch_mlp(x, inds)
        y = (y * picked[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        # Keep the residual stream in the input dtype (M3 fp32-residual lesson).
        return y.astype(x.dtype)


class OpenPanguV2DecoderLayer(nn.Module):
    """Sandwich-norm decoder layer on the (B, L, 4, H) mHC residual stream."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = OpenPanguV2Attention(args, layer_idx)
        if layer_idx >= args.first_k_dense_replace:
            self.mlp: nn.Module = OpenPanguV2MoE(args)
        else:
            self.mlp = OpenPanguV2MLP(args.hidden_size, args.intermediate_size)
        eps = args.rms_norm_eps
        h = args.hidden_size
        self.input_layernorm = nn.RMSNorm(h, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=eps)
        self.pre_mlp_layernorm = nn.RMSNorm(h, eps=eps)
        self.post_mlp_layernorm = nn.RMSNorm(h, eps=eps)
        if layer_idx in args.block_post_layernorm_idx:
            # Normalizes the FLATTENED 4-stream residual: weight [4*hidden].
            self.block_post_layernorm = nn.RMSNorm(args.mhc_num_stream * h, eps=eps)
        else:
            self.block_post_layernorm = None
        self.attn_mhc_module = OpenPanguV2MHCModule(args)
        self.mlp_mhc_module = OpenPanguV2MHCModule(args)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[OpenPanguV2LayerCache],
    ) -> mx.array:
        residual_a = h
        x_a, post_a, comb_a = self.attn_mhc_module.collapse(h)
        attn_out = self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x_a), mask, cache)
        )
        h = self.attn_mhc_module.expand(attn_out, residual_a, post_a, comb_a)

        residual_f = h
        x_f, post_f, comb_f = self.mlp_mhc_module.collapse(h)
        mlp_out = self.post_mlp_layernorm(self.mlp(self.pre_mlp_layernorm(x_f)))
        h = self.mlp_mhc_module.expand(mlp_out, residual_f, post_f, comb_f)

        if self.block_post_layernorm is not None:
            B, L, S, H = h.shape
            h = self.block_post_layernorm(h.reshape(B, L, S * H)).reshape(B, L, S, H)
        return h


class OpenPanguV2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            OpenPanguV2DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.merge_mhc_module = OpenPanguV2MergeMHC(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._logged_full_layer_traversal = False

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError(
                "openPangu cache/layer count mismatch: "
                f"cache={len(cache)} layers={len(self.layers)}. Refusing to "
                "silently truncate decoder execution."
            )
        # Embed, cast to bf16 (256-expert MoE safety), tile to 4 mHC streams.
        h = self.embed_tokens(inputs)
        if h.dtype == mx.float32:
            h = h.astype(mx.bfloat16)
        h = mx.repeat(h[..., None, :], self.args.mhc_num_stream, axis=-2)

        if cache is None:
            cache = [None] * len(self.layers)
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            c = cache[layer_idx]
            h = layer(h, None, c)  # per-layer masks built inside attention
        if not self._logged_full_layer_traversal:
            runtime_swa_windows = sorted(
                {
                    int(window)
                    for i in range(len(self.layers))
                    if (window := self.args.window_for(i))
                }
            )
            logger.info(
                "openPangu forward graph traversed all %d decoder layers "
                "(DSA=%d, SWA=%d, mHC_streams=%d, attention_sinks=%d, "
                "MLA_kv_rank=%d, SWA_windows=%s, max_context=%d)",
                len(self.layers),
                len(self.args.dsa_layers),
                sum(1 for i in range(len(self.layers)) if self.args.window_for(i)),
                self.args.mhc_num_stream,
                self.args.param_sink_number,
                self.args.kv_lora_rank,
                runtime_swa_windows,
                self.args.max_position_embeddings,
            )
            self._logged_full_layer_traversal = True
        merged = self.merge_mhc_module(h)
        return self.norm(merged)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OpenPanguV2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        out = self.lm_head(self.model(inputs, cache))
        if inputs.shape[1] > 1 and os.environ.get("OPENPANGU_LOGIT_PROBE"):
            last = out[0, -1, :]
            top = mx.argsort(last)[-8:]
            mx.eval(top)
            ids = [int(top[7 - i]) for i in range(8)]
            print(f"[openpangu logit-probe] top8 next-token ids: {ids}", file=sys.stderr)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[OpenPanguV2LayerCache]:
        args = self.args
        return [
            OpenPanguV2LayerCache(
                window=args.window_for(i), is_dsa=(i in args.dsa_layers)
            )
            for i in range(args.num_hidden_layers)
        ]

    def sanitize(self, weights):
        args = self.args
        out = {}
        conv_suffixes = (".qa_conv.weight", ".compresskv_conv.weight", ".o_conv.weight")
        for key, value in weights.items():
            # Drop MTP layers (46-48): mtp_mode=off runtime. Detection stays
            # metadata-driven via jang_config.mtp (depth 3, spec_decoding_ready).
            li = _layer_index(key)
            if li is not None and li >= args.num_hidden_layers:
                continue
            if "rotary_emb.inv_freq" in key:
                continue
            # Depthwise conv weight: PyTorch [C, 1, k] -> MLX Conv1d [C, k, 1].
            # OpenPanguCausalConv wraps the MLX Conv1d as ``self.conv``, so its
            # parameter path has an additional ``.conv`` component.  The
            # checkpoint does not: it stores ``qa_conv.weight`` (and the two
            # sibling convs).  Leaving the checkpoint key unchanged while
            # loading with strict=False silently preserves randomly initialized
            # conv weights in every layer.
            if key.endswith(conv_suffixes):
                if value.ndim == 3 and value.shape[1] == 1:
                    value = value.transpose(0, 2, 1)
                key = key[: -len(".weight")] + ".conv.weight"
                out[key] = value
                continue
            out[key] = value
        return out


def _layer_index(key: str) -> Optional[int]:
    marker = "model.layers."
    pos = key.find(marker)
    if pos < 0:
        return None
    rest = key[pos + len(marker) :]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None
