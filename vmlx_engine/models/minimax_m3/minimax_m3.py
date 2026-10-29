# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 (minimax_m3_vl) text runtime for vMLX / mlx-lm — full MSA decode.

Implements the real MiniMax Sparse Attention (MSA): a Lightning Indexer selects
top-k 128-token key blocks per query (per GQA group), and the main branch attends
only the selected blocks (msa.py: block-sparse Metal kernels for decode and
prefill; the additive block-mask path of transformers PR #46600 stays as the
reference / fallback). At context <= topk*block (=2048) every block is selected,
so it reduces to full causal attention (matches the validated probe, top1 0.654).
The indexer keys live in MiniMaxM3SparseCache alongside K/V.

Other blocks reuse proven mlx-lm code (validated against the torch oracle):
  swigluoai = gpt_oss.swiglu ; router = deepseek_v3.group_expert_select ;
  GemmaRMSNorm (1+w) ; partial RoPE ; per-head Gemma qk-norm.

Load via `load_minimax_m3(path)` (custom quant predicate: quantizes only the
modules the JANG bundle actually packed; indexer + norms stay fp16).

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.deepseek_v3 import group_expert_select

EAGLE3_AUX_LAYERS = (2, 30, 57)


# SwiGLU-OAI as M3 applies it (upstream MiniMaxM3VLExperts._apply_gate): gate
# clipped from above, linear part clipped to [-limit, limit], plus one. Defined
# here instead of imported from mlx_lm.models.gpt_oss, which mlx-lm 0.32 removed
# (its switch_layers.SwiGLU is the plain SiLU gate, not this). Same op order as
# mlx-lm 0.31.3's version, so outputs are bit-identical.
@partial(mx.compile, shapeless=True)
def swiglu(x_linear, x_glu, alpha: float = 1.702, limit: float = 7.0):
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    out_glu = x_glu * mx.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def _apply(module, x):
    return module(x)


class SwiGLU(nn.Module):
    """SwitchGLU activation: called as (x = up projection, gate)."""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha, self.limit = alpha, limit

    def __call__(self, x, gate):
        return swiglu(x, gate, self.alpha, self.limit)

try:
    from .cache import MiniMaxM3SparseCache, make_minimax_m3_cache
    from .m3_affine2_switch import MiniMaxM3Affine2SwitchGLU
    from . import fused_attn, msa, source_format
except Exception:
    try:
        # Served path: minimax_m3_register loads this file as mlx_lm.models.minimax_m3_vl, where `.cache` is
        # mlx-lm's own cache module. Import the siblings from the vmlx_engine package so the served model and the
        # prefix-cache restore (vmlx_engine.models.minimax_m3.cache) share ONE MiniMaxM3SparseCache class: with a
        # second copy loaded as a top-level `cache` module, Indexer's isinstance check failed on every restored
        # cache, the forward skipped the index lane, and the next DSpark verify raised "sparse index lane
        # desynchronised from K/V" (live M3.1 tool loop, 2026-10-28).
        from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache, make_minimax_m3_cache
        from vmlx_engine.models.minimax_m3.m3_affine2_switch import MiniMaxM3Affine2SwitchGLU
        from vmlx_engine.models.minimax_m3 import fused_attn, msa, source_format
    except Exception:  # pragma: no cover - standalone import without vmlx_engine
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from cache import MiniMaxM3SparseCache, make_minimax_m3_cache  # type: ignore
        from m3_affine2_switch import MiniMaxM3Affine2SwitchGLU  # type: ignore
        import msa  # type: ignore
        import source_format  # type: ignore
        fused_attn = None  # the fused attention kernels ship with vmlx_engine


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "minimax_m3_vl"
    hidden_size: int = 6144
    num_hidden_layers: int = 60
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    rotary_dim: int = 64
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 200064
    num_local_experts: int = 100
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.0
    norm_topk_prob: bool = True
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    moe_layer_freq: Optional[list] = None
    # MSA indexer
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    sparse_attention_freq: Optional[list] = None
    tie_word_embeddings: bool = False
    # VL (vision) — populated from raw config in from_dict; None = text-only.
    vision_config: Optional[dict] = None
    image_token_index: int = 200025
    video_token_index: int = 200026
    projector_hidden_size: int = 6144
    # JANGHT bundles: config.json's top-level "jangt" block and per-module "quantization" contract (routed experts are
    # trellis JANGT / Hadamard JANGH units, vmlx_engine.jangt). None for every other bundle.
    jangt: Optional[dict] = None
    quantization: Optional[dict] = None

    @classmethod
    def from_dict(cls, params):
        tc = dict(params.get("text_config", {}))
        sca = tc.get("sparse_attention_config", {}) or {}
        flat = {
            "index_n_heads": sca.get("sparse_num_index_heads", 4),
            "index_head_dim": sca.get("sparse_index_dim", 128),
            "index_block_size": sca.get("sparse_block_size", 128),
            "index_topk_blocks": sca.get("sparse_topk_blocks", 16),
            "index_local_blocks": sca.get("sparse_local_block", 1),
            "sparse_attention_freq": sca.get("sparse_attention_freq"),
        }
        merged = {**tc, **flat, "model_type": params.get("model_type", "minimax_m3_vl")}
        if "num_local_experts" in params:
            merged["num_local_experts"] = params["num_local_experts"]
        # VL: carry vision config from the raw (top-level) params so the model can
        # build the vision stack. None when the bundle has no vision_config (text-only).
        merged["vision_config"] = params.get("vision_config")
        merged["image_token_index"] = params.get("image_token_index", 200025)
        merged["video_token_index"] = params.get("video_token_index", 200026)
        merged["projector_hidden_size"] = params.get(
            "projector_hidden_size", merged.get("hidden_size", 6144))
        merged["jangt"] = params.get("jangt") or None
        merged["quantization"] = params.get("quantization") if params.get("jangt") else None
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in merged.items() if k in allowed})

    def is_moe(self, li):
        if self.moe_layer_freq is not None:
            return bool(self.moe_layer_freq[li])
        return li >= 3

    def is_sparse(self, li):
        if self.sparse_attention_freq is not None:
            return bool(self.sparse_attention_freq[li])
        return li >= 3


class GemmaRMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Indexer(nn.Module):
    """Lightning Indexer: scores idx_q vs cached idx_k, picks top-k key blocks.

    Returns an ``msa.MSASelection`` (top-k block ids per query row and KV group;
    ``dense_bias()`` gives the equivalent causal additive key mask) or None when
    every block is visible. Block = key_pos // block_size; the query's own
    (local) block is always kept.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.nh = args.index_n_heads
        self.d = args.index_head_dim
        self.block = args.index_block_size
        self.topk = args.index_topk_blocks
        self.local = args.index_local_blocks
        self.index_q_proj = nn.Linear(args.hidden_size, self.nh * self.d, bias=False)
        self.index_k_proj = nn.Linear(args.hidden_size, self.d, bias=False)
        self.index_q_norm = GemmaRMSNorm(self.d, args.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.d, args.rms_norm_eps)
        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)

    def project(self, x, offset, linear=None):
        """Index queries [B, nh, Sq, d] and keys [B, 1, Sq, d] of ``x`` at positions offset.. (norm + RoPE).
        ``linear(module, x)`` replaces the projection call (DSpark verify: decode arithmetic per row)."""
        B, Sq, _ = x.shape
        lin = _apply if linear is None else linear
        idx_q = self.index_q_norm(lin(self.index_q_proj, x).reshape(B, Sq, self.nh, self.d)).transpose(0, 2, 1, 3)
        idx_k = self.index_k_norm(lin(self.index_k_proj, x).reshape(B, Sq, 1, self.d)).transpose(0, 2, 1, 3)
        return self.rope(idx_q, offset=offset), self.rope(idx_k, offset=offset)

    def __call__(self, x, cache, offset):
        idx_q, idx_k = self.project(x, offset)
        if cache is not None and isinstance(cache, MiniMaxM3SparseCache):
            idx_k = cache.update_index(idx_k)            # [B,1,Sk,d]
        return self.select(idx_q, idx_k, offset)

    def select(self, idx_q, idx_k, offset):
        """Block selection (``msa.MSASelection``, or None = every block visible) for queries at offset.. over keys
        idx_k[:, :, :Sk]. One query row over its own key prefix is exactly a decode step's selection.

        Block score = max over the block's causal keys of idx_q . idx_k (float32), the local block(s) +inf; each
        indexer head is one KV group with its own top-k. ``msa.block_scores`` computes them with the score-sheet
        ops (``msa.reference_block_scores``) or a kernel that reproduces those bit for bit; the top-k below is the
        same op either way."""
        Sk = idx_k.shape[2]
        n_blocks = (Sk + self.block - 1) // self.block
        keep = min(self.topk, n_blocks)
        if keep >= n_blocks:
            # all blocks visible -> only causal matters; return None (full attn)
            return None
        block_scores = msa.block_scores(idx_q, idx_k, offset, self.block, self.local)   # [B, G, Sq, n_blocks]
        # top-k blocks per query and group; the attention applies causality inside them
        top_idx = mx.argpartition(-block_scores, kth=keep - 1, axis=-1)[..., :keep]  # [B,G,Sq,keep]
        return msa.MSASelection(top_idx, offset, Sk, self.block)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, li: int):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.scale = self.hd ** -0.5
        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, d, bias=False)
        self.q_norm = GemmaRMSNorm(self.hd, args.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.hd, args.rms_norm_eps)
        self.rope = nn.RoPE(args.rotary_dim, traditional=False, base=args.rope_theta)
        self.indexer = Indexer(args) if args.is_sparse(li) else None

    def project(self, x, offset, linear=None):
        """q [B, H, L, D], k / v [B, KV, L, D] of ``x`` at positions offset.. (q/k norm + partial RoPE)."""
        B, L, _ = x.shape
        lin = _apply if linear is None else linear
        q = self.q_norm(lin(self.q_proj, x).reshape(B, L, self.n_heads, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(lin(self.k_proj, x).reshape(B, L, self.n_kv, self.hd)).transpose(0, 2, 1, 3)
        v = lin(self.v_proj, x).reshape(B, L, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        return self.rope(q, offset=offset), self.rope(k, offset=offset), v

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        off = cache.offset if cache is not None else 0
        q, k, v = self.project(x, off)
        # Append K/V to the cache FIRST (matches upstream MiniMax-M3 ordering) so
        # the Lightning Indexer sees the COMPLETED post-append sequence length. If
        # the indexer ran first (on the pre-append offset) its block_bias Sk would
        # be one key short of SDPA's K once context passes the 2048 sparse
        # threshold -> broadcast crash + idx/KV desync across decode and
        # prefix-cache restore. `off` (captured pre-append) is still the correct
        # RoPE position for the new token(s).
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        # MSA selection (additive block mask), computed from the SAME hidden x
        block_bias = self.indexer(x, cache, off) if self.indexer is not None else None
        o = self.attend(q, k, v, block_bias, mask, cache)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def attend(self, q, k, v, block_bias, mask, cache):
        """Attention output [B, H, L, D] (before o_proj) of q over k / v, restricted to the MSA selection when one
        exists (``block_bias``: an ``msa.MSASelection``, or an explicit additive key mask [B, G, L, Sk])."""
        if block_bias is None:
            return scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        if isinstance(block_bias, msa.MSASelection):
            path = msa.choose_path(block_bias, q, k, mask, self.n_kv)
            if path == "decode":
                return msa.decode_kernel_attention(q, k, v, block_bias, self.scale)
            if path == "gather":
                return msa.gathered_attention(q, k, v, block_bias, self.scale)
            if path == "kernel":
                return msa.kernel_attention(q, k, v, block_bias, self.scale)
            block_bias = block_bias.dense_bias()
        # block_bias is built as float32 (0.0 / -inf). SDPA requires the
        # mask dtype to promote to the bf16 compute/output dtype, else it
        # raises "[scaled_dot_product_attention] Mask type must promote to
        # output type bfloat16" once the indexer fires past the
        # index_topk_blocks*index_block_size (=2048) token threshold.
        # 0.0 and -inf are exactly representable in bf16, so the cast is
        # lossless. (Root-cause fix for the M3 >2048-token SDPA crash;
        # gated by context length, NOT cache mode.)
        bias = block_bias.astype(q.dtype)  # [B, G, L, Sk], already causal
        if isinstance(mask, mx.array):  # batch padding etc.
            bias = bias + mask.astype(q.dtype)
        return self._grouped_attention(q, k, v, bias)

    def _grouped_attention(self, q, k, v, bias):
        """Attention where each KV group has its own key mask.

        Query heads are contiguous per KV head, so folding the groups into the
        batch axis gives one SDPA call whose [B*G, 1, L, Sk] mask broadcasts over
        the group's query heads; a dense per-head mask would be 16x larger.
        """
        B, H, L, D = q.shape
        G, Sk = bias.shape[1], k.shape[2]
        if G != self.n_kv:  # indexer heads that do not line up with KV heads
            bias = mx.repeat(bias, H // G, axis=1)
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=bias)
        o = mx.fast.scaled_dot_product_attention(
            q.reshape(B * G, H // G, L, D),
            k.reshape(B * G, 1, Sk, D),
            v.reshape(B * G, 1, Sk, D),
            scale=self.scale,
            mask=bias.reshape(B * G, 1, L, Sk),
        )
        return o.reshape(B, H, L, D)


class SwiGLUOAIMLP(nn.Module):
    def __init__(self, dim, hidden, alpha, limit):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.alpha, self.limit = alpha, limit

    def __call__(self, x, linear=None):
        if linear is None:
            return self.down_proj(swiglu(self.up_proj(x), self.gate_proj(x), self.alpha, self.limit))
        return linear(self.down_proj, swiglu(linear(self.up_proj, x), linear(self.gate_proj, x), self.alpha, self.limit))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        self.switch_mlp = MiniMaxM3Affine2SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=SwiGLU(args.swiglu_alpha, args.swiglu_limit),
            bias=False,
        )
        self.shared_experts = SwiGLUOAIMLP(args.hidden_size, args.shared_intermediate_size,
                                           args.swiglu_alpha, args.swiglu_limit)

    def routed(self, x, inds, scores):
        """Router-weighted sum of the routed experts' outputs for ``x`` (the shared expert excluded)."""
        if getattr(self.switch_mlp, "is_jangt", False):
            # JANGHT: trellis / Hadamard routed experts, gather + SwiGLU-OAI + router-weighted sum in one call
            return self.switch_mlp.routed(x, inds, scores)
        return (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)

    def __call__(self, x, linear=None, routed=None):
        """``linear(module, x)`` / ``routed(x, inds, scores)`` replace the projection / routed calls (DSpark verify)."""
        lin = _apply if linear is None else linear
        inds, scores = group_expert_select(
            lin(self.gate, x), self.e_score_correction_bias, self.top_k, 1, 1,
            self.routed_scaling_factor, self.norm_topk_prob)
        y = (self.routed if routed is None else routed)(x, inds, scores)
        # group_expert_select returns float32 scores, which promote the MoE output
        # (and thus the whole residual stream from layer 3 on) to float32 — forcing
        # every downstream attention/MLP matmul into float32 (~2x slower than bf16).
        # Cast back to the model compute dtype so the matmuls stay bf16.
        return (y + self.shared_experts(x, linear)).astype(x.dtype)


def _fused_moe_module():
    try:
        from . import fused_moe  # type: ignore
    except Exception:
        from vmlx_engine.models.minimax_m3 import fused_moe  # served path: loaded as mlx_lm.models.minimax_m3_vl
    return fused_moe


def moe_residual(mlp, x_normed, h):
    """``h + mlp(x_normed)`` -- through the fused MoE kernels (models/minimax_m3/fused_moe.py) when
    VMLX_M3_FUSED_MOE=1 and the layer / call fit them (1..16 rows), else None. The fused block computes every row
    with one-row (decode) arithmetic, bit-identical to the unfused block on one row (tests/test_minimax_m3_fused_moe.py).
    Callers: the decoder layer for one-token steps, the DSpark verify for its rows; multi-token prefill chunks keep
    the stock multi-row kernels (their arithmetic, and so the prompt cache, stays what it was)."""
    if not isinstance(mlp, SparseMoeBlock):
        return None
    fm = _fused_moe_module()
    if not fm.enabled() or not fm.supported(mlp, x_normed):
        return None
    return fm.fused_moe_residual(mlp, x_normed, h)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int):
        super().__init__()
        self.self_attn = Attention(args, li)
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = (SparseMoeBlock(args) if args.is_moe(li)
                    else SwiGLUOAIMLP(args.hidden_size, args.dense_intermediate_size,
                                      args.swiglu_alpha, args.swiglu_limit))

    def __call__(self, x, mask=None, cache=None):
        if fused_attn is not None and fused_attn.use_fused(self, x, mask, cache):
            # VMLX_M3_FUSED_ATTN: input norm + attention + residual in five Metal launches (decode / verify rows)
            h = fused_attn.attention_block(self, x, cache)
        else:
            h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        xn = self.post_attention_layernorm(h)
        # VMLX_M3_FUSED_MOE: router + routed / shared experts + residual in four launches (one decode row)
        fused = moe_residual(self.mlp, xn, h) if xn.shape[-2] == 1 else None
        return fused if fused is not None else h + self.mlp(xn)


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        *,
        return_aux: bool = False,
        aux_layers: tuple[int, ...] = EAGLE3_AUX_LAYERS,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # full-attention layers (0-2) use the standard causal mask; sparse layers
        # build their own block mask, but still need a causal fallback at short ctx.
        mask = create_attention_mask(h, cache[0])
        aux = [] if return_aux else None
        aux_set = set(aux_layers) if return_aux else set()
        for li, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c)
            if return_aux and li in aux_set:
                aux.append(h)
        out = self.norm(h)
        return (out, aux) if return_aux else out

    def forward_pieces(self, pieces, cache, *, aux_pieces=None, aux_layers: tuple[int, ...] = EAGLE3_AUX_LAYERS):
        """Prefill consecutive prompt pieces (token arrays [1, L_i], in order, appended to ``cache``) LAYER BY LAYER:
        every piece runs exactly the ops it runs as its own ``__call__`` (same rows, same matrix shapes, same cache
        rows visible to its attention), except that the routed experts of a MoE layer run once over the rows of all
        the pieces that take the sorted prefill kernel alone (``_group_routed``): that kernel's output rows do not
        depend on the other rows or on their count (tests/test_minimax_m3_cold_prefill.py), so each expert's weights
        are decoded once per group instead of once per piece. Returns the final-norm outputs per piece and, for the
        pieces in ``aux_pieces``, their outputs after ``aux_layers`` (``__call__``'s return_aux order).

        Why: the aligned prefill cuts a prompt at every turn cut (prefill_geometry), so a many-message chat runs many
        small pieces; the routed experts cost per piece ~ every expert the piece routes to, decoded once, whatever its
        row count. Pieces must stay pieces (the K / V / index projections' rows are not invariant to the row count),
        the routed experts need not."""
        hs = [self.embed_tokens(p) for p in pieces]
        masks = [None] * len(hs)
        aux_pieces = set(aux_pieces or ())
        aux_set = set(aux_layers) if aux_pieces else set()
        aux = {i: [] for i in aux_pieces}
        for li, (layer, c) in enumerate(zip(self.layers, cache)):
            xns = []
            for i, h in enumerate(hs):
                if li == 0:
                    masks[i] = create_attention_mask(h, cache[0])
                if fused_attn is not None and fused_attn.use_fused(layer, h, masks[i], c):
                    h = fused_attn.attention_block(layer, h, c)
                else:
                    h = h + layer.self_attn(layer.input_layernorm(h), masks[i], c)
                hs[i] = h
                xns.append(layer.post_attention_layernorm(h))
            outs = _group_mlp(layer.mlp, xns, hs)
            for i in range(len(hs)):
                hs[i] = outs[i]
                if li in aux_set and i in aux_pieces:
                    aux[i].append(hs[i])
        return [self.norm(h) for h in hs], aux


# Rows per piece below which the JANGT routed experts leave the sorted prefill kernel (switch.SORT_THRESHOLD slots)
def _takes_sorted_prefill(mlp, rows: int) -> bool:
    sw = getattr(mlp, "switch_mlp", None)
    if not getattr(sw, "is_jangt", False):
        return False
    try:
        from vmlx_engine.jangt import switch as SW
    except Exception:  # pragma: no cover - standalone import
        return False
    return rows * int(mlp.top_k) >= SW.SORT_THRESHOLD and SW.nax_prefill()


def _group_mlp(mlp, xns, hs):
    """``h + mlp(xn)`` for every piece of a layer-major group (DecoderLayer.__call__'s MLP half), with the routed
    experts of the pieces that take the sorted JANGT prefill kernel computed in ONE call over their rows
    (``VMLX_M3_PREFILL_GROUP_ROUTED=0``: one call per piece, for A/B)."""
    out = [None] * len(xns)
    grouped = []
    if isinstance(mlp, SparseMoeBlock) and os.environ.get("VMLX_M3_PREFILL_GROUP_ROUTED", "1").strip().lower() not in {
            "0", "false", "off", "no"}:
        grouped = [i for i, xn in enumerate(xns) if _takes_sorted_prefill(mlp, int(xn.shape[-2]))]
    if len(grouped) < 2:
        grouped = []
    for i, (xn, h) in enumerate(zip(xns, hs)):
        if i in grouped:
            continue
        fused = moe_residual(mlp, xn, h) if xn.shape[-2] == 1 else None
        out[i] = fused if fused is not None else h + mlp(xn)
    if grouped:
        routes = [group_expert_select(mlp.gate(xns[i]), mlp.e_score_correction_bias, mlp.top_k, 1, 1,
                                      mlp.routed_scaling_factor, mlp.norm_topk_prob) for i in grouped]
        rows = [int(xns[i].shape[-2]) for i in grouped]
        y = mlp.routed(mx.concatenate([xns[i] for i in grouped], axis=-2),
                       mx.concatenate([r[0] for r in routes], axis=-2), mx.concatenate([r[1] for r in routes], axis=-2))
        at = 0
        for i, n in zip(grouped, rows):
            xn = xns[i]
            out[i] = hs[i] + (y[..., at:at + n, :] + mlp.shared_experts(xn)).astype(xn.dtype)
            at += n
    return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        if args.jangt:
            # JANGHT: every routed stack becomes a JTMixedSwitchGLU BEFORE quantization (the loader's nn.quantize then
            # hands each {"mode": "jangt", ...} entry to the installed unit). The activation is M3's own SwiGLU-OAI.
            from vmlx_engine.jangt.switch import install_jangt

            self.jangt_modules = install_jangt(self, {
                "jangt": args.jangt, "quantization": args.quantization,
                "swiglu_kind": "oai", "swiglu_alpha": args.swiglu_alpha, "swiglu_limit": args.swiglu_limit})
        # VL vision stack (additive — only when the bundle declares vision_config).
        if getattr(args, "vision_config", None):
            try:
                from .minimax_m3_vl import MiniMaxM3VLVisionStack
            except Exception:
                try:  # served path (mlx_lm.models.minimax_m3_vl): the sibling lives in the vmlx_engine package
                    from vmlx_engine.models.minimax_m3.minimax_m3_vl import MiniMaxM3VLVisionStack
                except Exception:  # pragma: no cover - standalone import
                    from minimax_m3_vl import MiniMaxM3VLVisionStack  # type: ignore
            self.vision = MiniMaxM3VLVisionStack({
                "vision_config": args.vision_config,
                "projector_hidden_size": args.projector_hidden_size,
            })

    def _embed_with_media(
        self,
        inputs,
        *,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        """Splice image and video embeddings into their placeholder positions.

        The official M3 implementation routes both modalities through the same
        vision tower/projector and distinguishes them only by placeholder token.
        """
        import numpy as np
        te = self.model.embed_tokens(inputs)                    # (B, L, H)
        B, L = inputs.shape[0], inputs.shape[1]
        flat = te.reshape(B * L, te.shape[-1])
        ids_np = np.asarray(inputs).reshape(B * L)

        def _scatter(values, grid, token_id, label):
            nonlocal flat
            if values is None:
                return
            features = self.vision(values, grid)
            positions = np.where(ids_np == token_id)[0]
            if int(positions.size) != int(features.shape[0]):
                raise ValueError(
                    f"M3 VL {label} features/tokens mismatch: "
                    f"{features.shape[0]} features for {positions.size} tokens"
                )
            flat[mx.array(positions)] = features.reshape(
                -1, features.shape[-1]
            ).astype(flat.dtype)

        _scatter(pixel_values, image_grid_thw, self.args.image_token_index, "image")
        _scatter(
            pixel_values_videos,
            video_grid_thw,
            self.args.video_token_index,
            "video",
        )
        return flat.reshape(B, L, te.shape[-1])

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        *,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mask=None,  # M3 builds its own causal/block mask; accept + ignore.
        return_aux: bool = False,
        aux_layers: tuple[int, ...] = EAGLE3_AUX_LAYERS,
        **kwargs,  # absorb other MLLM kwargs (video_*, audio_*) harmlessly
    ):
        if (
            (pixel_values is not None or pixel_values_videos is not None)
            and input_embeddings is None
            and hasattr(self, "vision")
        ):
            input_embeddings = self._embed_with_media(
                inputs,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        out = self.model(
            inputs,
            cache,
            input_embeddings,
            return_aux=return_aux,
            aux_layers=aux_layers,
        )
        aux = None
        if return_aux:
            out, aux = out
        logits = (
            self.model.embed_tokens.as_linear(out)
            if self.args.tie_word_embeddings
            else self.lm_head(out)
        )
        return (logits, aux) if return_aux else logits

    def make_cache(self):
        return make_minimax_m3_cache(self.args)

    def prefill_group(self, pieces, cache, *, logits_last: bool = False, aux_pieces=None,
                      aux_layers: tuple[int, ...] = EAGLE3_AUX_LAYERS):
        """Text prefill of consecutive pieces as one layer-major group (``MiniMaxM3Model.forward_pieces``): every piece
        gets the rows its own ``__call__`` would give it. Returns (final-norm outputs per piece, aux per piece in
        ``aux_pieces``, the last piece's logits [1, L, V] when ``logits_last``)."""
        outs, aux = self.model.forward_pieces(pieces, cache, aux_pieces=aux_pieces, aux_layers=aux_layers)
        logits = None
        if logits_last:
            logits = (self.model.embed_tokens.as_linear(outs[-1]) if self.args.tie_word_embeddings
                      else self.lm_head(outs[-1]))
        return outs, aux, logits

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        out = {}
        has_vision = getattr(self.args, "vision_config", None) is not None
        for k, v in weights.items():
            if k.startswith("mtp"):
                continue
            if (k.startswith("vision_tower.") or k.startswith("multi_modal_projector.")
                    or k.startswith("patch_merge_mlp.")):
                if not has_vision:
                    continue  # text-only build: drop vision weights
                # 3D patch_embedding conv ships PyTorch layout (Cout,Cin,D,H,W);
                # MLX Conv3d wants (Cout,D,H,W,Cin).
                if k.endswith("patch_embedding.weight") and v.ndim == 5:
                    v = v.moveaxis(1, 4)
                out["vision." + k] = v
                continue
            if k.startswith("language_model.model."):
                k = "model." + k[len("language_model.model."):]
            elif k.startswith("language_model.lm_head"):
                k = "lm_head" + k[len("language_model.lm_head"):]
            k = k.replace(".block_sparse_moe.", ".mlp.")
            # indexer projections are flat on self_attn in the checkpoint; this
            # model nests them under the `indexer` submodule.
            k = k.replace(".self_attn.index_", ".self_attn.indexer.index_")
            out[k] = v
        if source_format.is_source_checkpoint(out):
            # official checkpoint (MXFP8 linears, NVFP4 experts): keep it as stored
            out = source_format.convert(self, out)
        return out
