# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 (minimax_m3_vl) text runtime for vMLX / mlx-lm — full MSA decode.

Implements the real MiniMax Sparse Attention (MSA): a Lightning Indexer selects
top-k 128-token key blocks per query (per GQA group), and the main branch attends
only the selected blocks via an additive block mask (the exact eager/sdpa path
from transformers PR #46600). At context < topk*block (=2048) every block is
selected, so it reduces to full causal attention (matches the validated probe,
top1 0.654). The indexer keys live in MiniMaxM3SparseCache alongside K/V.

Other blocks reuse proven mlx-lm code (validated against the torch oracle):
  swigluoai = gpt_oss.swiglu ; router = deepseek_v3.group_expert_select ;
  GemmaRMSNorm (1+w) ; partial RoPE ; per-head Gemma qk-norm.

Load via `load_minimax_m3(path)` (custom quant predicate: quantizes only the
modules the JANG bundle actually packed; indexer + norms stay fp16).

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.models.gpt_oss import SwiGLU, swiglu
from mlx_lm.models.deepseek_v3 import group_expert_select

try:
    from .cache import MiniMaxM3SparseCache, make_minimax_m3_cache
except Exception:  # pragma: no cover - standalone / mlx_lm-namespace import
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from cache import MiniMaxM3SparseCache, make_minimax_m3_cache  # type: ignore


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

    Returns an additive mask [B, 1, Sq, Sk] (0 on allowed keys, -inf elsewhere)
    that composes with the causal mask. Block = key_pos // block_size; the
    query's own (local) block is always kept.
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

    def __call__(self, x, cache, offset):
        B, Sq, _ = x.shape
        idx_q = self.index_q_norm(self.index_q_proj(x).reshape(B, Sq, self.nh, self.d)).transpose(0, 2, 1, 3)
        idx_k = self.index_k_norm(self.index_k_proj(x).reshape(B, Sq, 1, self.d)).transpose(0, 2, 1, 3)
        idx_q = self.rope(idx_q, offset=offset)
        idx_k = self.rope(idx_k, offset=offset)
        if cache is not None and isinstance(cache, MiniMaxM3SparseCache):
            idx_k = cache.update_index(idx_k)            # [B,1,Sk,d]
        Sk = idx_k.shape[2]
        # scores [B, nh, Sq, Sk] in fp32
        scores = (idx_q.astype(mx.float32) @ idx_k.astype(mx.float32).transpose(0, 1, 3, 2))
        q_pos = mx.arange(offset, offset + Sq).reshape(1, 1, Sq, 1)
        k_pos = mx.arange(Sk).reshape(1, 1, 1, Sk)
        scores = mx.where(k_pos > q_pos, -mx.inf, scores)
        # pad Sk to a block multiple, max-pool per block, max over heads
        n_blocks = (Sk + self.block - 1) // self.block
        pad = n_blocks * self.block - Sk
        if pad:
            scores = mx.concatenate(
                [scores, mx.full((B, self.nh, Sq, pad), -mx.inf, dtype=scores.dtype)], axis=-1)
        scores = scores.reshape(B, self.nh, Sq, n_blocks, self.block)
        block_scores = scores.max(axis=-1).max(axis=1)            # [B, Sq, n_blocks]
        # force the local block(s) of each query to always win
        q_block = ((q_pos.reshape(1, Sq) ) // self.block)         # [1, Sq]
        if self.local > 0:
            loc = mx.arange(self.local).reshape(1, 1, self.local)
            local_idx = mx.maximum(q_block.reshape(1, Sq, 1) - loc, 0)  # [1,Sq,local]
            local_idx = mx.broadcast_to(local_idx, (B, Sq, self.local))
            block_scores = mx.put_along_axis(
                block_scores, local_idx, mx.array(mx.inf, dtype=block_scores.dtype), axis=-1)
        keep = min(self.topk, n_blocks)
        if keep >= n_blocks:
            # all blocks visible -> only causal matters; return None (full attn)
            return None
        # top-k blocks per query
        top_idx = mx.argpartition(-block_scores, kth=keep - 1, axis=-1)[..., :keep]  # [B,Sq,keep]
        # build [B,1,Sq,n_blocks] keep-bias then expand to keys
        block_keep = mx.full((B, Sq, n_blocks), -mx.inf, dtype=mx.float32)
        block_keep = mx.put_along_axis(block_keep, top_idx, mx.array(0.0, dtype=mx.float32), axis=-1)
        # expand block -> key, drop pad, add head axis
        key_bias = mx.repeat(block_keep, self.block, axis=-1)[:, :, :Sk]   # [B,Sq,Sk]
        key_bias = key_bias.reshape(B, 1, Sq, Sk)
        # also enforce causal here (selected future blocks shouldn't leak)
        key_bias = mx.where(k_pos > q_pos, -mx.inf, key_bias)
        return key_bias


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

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        off = cache.offset if cache is not None else 0
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        q, k = self.rope(q, offset=off), self.rope(k, offset=off)
        # MSA selection (additive block mask), computed from the SAME hidden x
        attn_mask = mask
        if self.indexer is not None:
            block_bias = self.indexer(x, cache, off)
            if block_bias is not None:
                attn_mask = block_bias      # already causal + block-restricted
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        o = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=attn_mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class SwiGLUOAIMLP(nn.Module):
    def __init__(self, dim, hidden, alpha, limit):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.alpha, self.limit = alpha, limit

    def __call__(self, x):
        return self.down_proj(swiglu(self.up_proj(x), self.gate_proj(x), self.alpha, self.limit))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))
        self.switch_mlp = SwitchGLU(args.hidden_size, args.intermediate_size,
                                    args.num_local_experts, activation=SwiGLU(), bias=False)
        self.shared_experts = SwiGLUOAIMLP(args.hidden_size, args.shared_intermediate_size,
                                           args.swiglu_alpha, args.swiglu_limit)

    def __call__(self, x):
        inds, scores = group_expert_select(
            self.gate(x), self.e_score_correction_bias, self.top_k, 1, 1,
            self.routed_scaling_factor, self.norm_topk_prob)
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        return y + self.shared_experts(x)


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
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = GemmaRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # full-attention layers (0-2) use the standard causal mask; sparse layers
        # build their own block mask, but still need a causal fallback at short ctx.
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        return (self.model.embed_tokens.as_linear(out) if self.args.tie_word_embeddings
                else self.lm_head(out))

    def make_cache(self):
        return make_minimax_m3_cache(self.args)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if (k.startswith("vision_tower.") or k.startswith("multi_modal_projector.")
                    or k.startswith("patch_merge_mlp.") or k.startswith("mtp")):
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
        return out
