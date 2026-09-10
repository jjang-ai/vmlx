# SPDX-License-Identifier: Apache-2.0
"""Spark-X2.5 (``spark2_5``) text runtime for vMLX.

Port of ``XHToken/Spark-X2.5-4B`` (rev 5e10fcc0, apache-2.0). Neither mlx-lm
nor the installed transformers ships this architecture, so vMLX vendors it.

Architecture (4B: 36 layers, hidden 2560, vocab 131072, TIED embeddings):
  * Fused ``q_k_v_proj`` (one Linear, 4096+1024+1024 out) — kept fused so the
    checkpoint tensor name, the quantizer's per-module bit assignment and the
    AWQ fold site (``input_layernorm -> q_k_v_proj``) all line up.
  * ``out_proj`` (NOT ``o_proj``); embeddings live at ``model.embedding.weight``
    (NOT ``embed_tokens``) — a generic converter keyed on the usual names
    silently misses both.
  * Headwise attention output gate: ``g_proj`` (hidden -> n_heads) produces ONE
    scalar per head, sigmoid in fp32, multiplied into the attention output
    BEFORE ``out_proj``. Tiny (16x2560) and it gates every attention output —
    it is a passthrough tensor for quantization, never a quantized one.
  * Mixed SWA: ``layer_types`` repeats 3x sliding : 1x full -> 27 sliding
    (window 512) + 9 full attention.
  * Rope DIFFERS BY LAYER TYPE: full_attention theta 5e6 with
    partial_rotary_factor 0.25 (only 64 of 256 head dims rotate, the rest pass
    through); sliding_attention theta 1e4 with partial 1.0. HF applies
    rotate_half over the leading ``head_dim * prf`` dims, which is exactly
    ``nn.RoPE(dims=rope_dims, traditional=False)``.
  * MLP activation is **gelu**, not silu (upstream raises on anything else).
  * Plain RMSNorm, no ``+1`` shift.

Created by Jinho Jang (eric@jangq.ai) — 2026-09-08.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache

_FULL = "full_attention"
_SLIDING = "sliding_attention"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "spark2_5"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 10240
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    vocab_size: int = 131072
    sliding_window: int = 512
    hidden_act: str = "gelu"
    headwise_attn_output_gate: bool = True
    gate_attn_act_mode: str = "sigmoid"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 1048576
    layer_types: list[str] = field(default_factory=list)
    rope_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        self.rope_parameters = self.rope_parameters or {}
        if not self.layer_types:
            self.layer_types = [_FULL] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) != num_hidden_layers "
                f"({self.num_hidden_layers})"
            )
        if set(self.layer_types) - {_FULL, _SLIDING}:
            raise ValueError(f"Unsupported Spark layer_types: {self.layer_types}")
        if _SLIDING in self.layer_types and (self.sliding_window is None or self.sliding_window <= 0):
            raise ValueError("sliding_attention requires a positive sliding_window")
        for layer_type in set(self.layer_types):
            dims = self.head_dim * self.partial_rotary_for(layer_type)
            if not 0 < dims <= self.head_dim or dims != int(dims) or int(dims) % 2:
                raise ValueError(f"Invalid rotary dimensions for {layer_type}: {dims}")
        if self.hidden_act != "gelu":
            raise ValueError(f"spark2_5 only supports hidden_act='gelu', got {self.hidden_act!r}")

    def rope_theta_for(self, layer_type: str) -> float:
        return float(self.rope_parameters.get(layer_type, {}).get("rope_theta", 10000))

    def partial_rotary_for(self, layer_type: str) -> float:
        return float(self.rope_parameters.get(layer_type, {}).get("partial_rotary_factor", 1.0))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.layer_type = layer_type
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.q_k_v_proj = nn.Linear(
            args.hidden_size, self.q_dim + 2 * self.kv_dim, bias=args.attention_bias
        )
        self.g_proj = (
            nn.Linear(args.hidden_size, self.n_heads, bias=args.attention_bias)
            if args.headwise_attn_output_gate
            else None
        )
        self.out_proj = nn.Linear(self.q_dim, args.hidden_size, bias=args.attention_bias)

        self.gate_act = args.gate_attn_act_mode
        if self.gate_act not in ("sigmoid", "silu"):
            raise ValueError(f"Unsupported gate_attn_act_mode: {self.gate_act!r}")

        # Partial rotary: rotate the leading `rope_dims`, pass the tail through.
        rope_dims = int(self.head_dim * args.partial_rotary_for(layer_type))
        self.rope = nn.RoPE(
            dims=rope_dims, traditional=False, base=args.rope_theta_for(layer_type)
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        B, L, _ = x.shape

        qkv = self.q_k_v_proj(x)
        queries = qkv[..., : self.q_dim]
        keys = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        values = qkv[..., self.q_dim + self.kv_dim :]

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        if self.g_proj is not None:
            gate = self.g_proj(x).reshape(B, L, self.n_heads, 1).transpose(0, 2, 1, 3)
            gate = gate.astype(mx.float32)
            gate = mx.sigmoid(gate) if self.gate_act == "sigmoid" else nn.silu(gate)
            output = output * gate.astype(output.dtype)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = Attention(args, layer_type)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        # Upstream carries the residual stream in fp32 and casts back to the
        # projection dtype before each matmul (modeling_spark.py:243-258). The
        # matmuls stay bf16; only the accumulation is wide. Running the residual
        # in bf16 instead doubles the drift vs the fp32 reference and flips
        # near-tied top-1 tokens, so the wide accumulate is kept.
        #
        # The cast dtype comes from a NORM weight, never from a projection's
        # ``.weight``: once a bundle is quantized, ``gate_proj.weight`` is the
        # PACKED uint32 code array, and casting activations to uint32 truncates
        # every one of them to an integer. That reads as a plausible bundle —
        # weights dequantize correctly, embedding and each sublayer test clean
        # in isolation — and only shows up as garbage output. Norm weights are
        # passthrough and stay floating point in every profile.
        w_dtype = self.input_layernorm.weight.dtype
        x = x + self.self_attn(self.input_layernorm(x).astype(w_dtype), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x).astype(w_dtype))


class Spark2_5Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Upstream names this `embedding`, not `embed_tokens`.
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, lt) for lt in args.layer_types]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        h = input_embeddings if input_embeddings is not None else self.embedding(inputs)
        out_dtype = h.dtype
        h = h.astype(mx.float32)

        if cache is None:
            cache = [None] * len(self.layers)

        if len(cache) != len(self.layers):
            raise ValueError(f"Spark cache has {len(cache)} layers; expected {len(self.layers)}")

        types = self.args.layer_types
        first_full = types.index(_FULL) if _FULL in types else None
        first_sliding = types.index(_SLIDING) if _SLIDING in types else None

        full_mask = (
            create_attention_mask(h, cache[first_full]) if first_full is not None else None
        )
        sliding_mask = (
            create_attention_mask(h, cache[first_sliding], window_size=self.args.sliding_window)
            if first_sliding is not None
            else None
        )

        for layer, c in zip(self.layers, cache):
            mask = full_mask if layer.layer_type == _FULL else sliding_mask
            h = layer(h, mask, c)

        return self.norm(h).astype(out_dtype)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Spark2_5Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embedding.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Tied bundles carry no lm_head tensor; drop one if a repacker added it.
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if lt == _FULL else RotatingKVCache(max_size=self.args.sliding_window)
            for lt in self.args.layer_types
        ]
