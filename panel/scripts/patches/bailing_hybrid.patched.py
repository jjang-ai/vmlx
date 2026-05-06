# Bailing-V2.5 Hybrid (Ling-2.6-flash). MLA + Lightning-Linear-Attention + MoE + MTP.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    layer_group_size: int
    group_norm_size: int
    # MLA-specific
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    kv_lora_rank: int = 512
    rope_interleave: bool = True
    # MTP
    num_nextn_predict_layers: int = 0
    # routing
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 4
    score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_enable_routed_scaling: bool = True
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_router_enable_shared_expert: bool = True
    # general
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_traditional: bool = False
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 0.5
    head_dim: Optional[int] = None
    attention_bias: bool = False


# ---------------------------------------------------------------------------
# Linear (Lightning) Attention — reused from bailing_moe_linear.py
# ---------------------------------------------------------------------------


def recurrent_gla(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    h: Optional[mx.array] = None,
) -> mx.array:
    B, Hq, L, K = q.shape
    outputs = []
    # fp32 promotion: k.T @ v products can exceed fp16 max (65504) on prompts
    # past ~80 tokens, producing inf → NaN logits. Compute the recurrence in
    # fp32 and cast outputs back at the end.
    in_dtype = q.dtype
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    if h is not None:
        h = h.astype(mx.float32)
    exp_g = mx.exp(g.astype(mx.float32))[:, None, None]
    q = q * scale
    for t in range(L):
        q_t = q[:, :, t : t + 1]
        k_t = k[:, :, t : t + 1]
        v_t = v[:, :, t : t + 1]
        h_up = k_t.transpose(0, 1, 3, 2) @ v_t
        if h is not None:
            h = h * exp_g + h_up
        else:
            h = h_up
        o_t = q_t @ h
        outputs.append(o_t)
    # Keep output in fp32: q @ h products can also overflow fp16 (h grows to
    # ~20k, q ~40, summed over 128 → ~100M). g_norm in caller normalizes
    # magnitudes before downstream layers, so leaving fp32 here is safe and
    # retains the precision needed for MMLU-length prompts.
    return mx.concatenate(outputs, axis=2), h


class GroupRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, groups: int = 1):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


class LinearAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_qk_norm = args.use_qk_norm
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_attention_heads
        self.head_dim = args.head_dim or args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )
        self.g_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.g_norm = GroupRMSNorm(
            self.num_attention_heads * self.head_dim,
            eps=args.rms_norm_eps,
            groups=args.group_norm_size,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        self._slope = self._get_slopes()

    def _get_slopes(self) -> mx.array:
        n = self.num_attention_heads

        def power_of_2_slopes(n):
            return [
                2 ** (-(2 ** -(math.log2(n) - 3)) * (i + 1)) for i in range(n)
            ]

        if math.log2(n).is_integer():
            slopes = power_of_2_slopes(n)
        else:
            p = 2 ** math.floor(math.log2(n))
            slopes = power_of_2_slopes(p) + power_of_2_slopes(2 * p)[::2][: n - p]

        slopes = mx.array(slopes, dtype=mx.float32)
        denom = max(1, self.num_hidden_layers - 1)
        layer_pos = max(0, self.layer_idx - 1)
        layer_factor = 1 - (layer_pos / denom) + 1e-5
        return -slopes * layer_factor

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)
        qkv_mix = qkv.reshape(
            B,
            L,
            self.num_attention_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        q, k, v = mx.split(
            qkv_mix,
            [
                self.num_attention_heads,
                self.num_attention_heads + self.num_key_value_heads,
            ],
            axis=2,
        )

        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is None:
            cache = [None]
        output, cache[0] = recurrent_gla(
            q=queries,
            k=keys,
            v=values,
            g=self._slope,
            scale=self.scale,
            h=cache[0],
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        # g_norm divides by RMS, bounding magnitudes to ~O(1); after that
        # we can safely cast back to the input dtype to keep dense() in fp16
        # (preserves throughput; raw fp32 here drops decode from ~30 to <10
        # tok/s).
        output = self.g_norm(output).astype(x.dtype) * mx.sigmoid(self.g_proj(x))
        return self.dense(output)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DSV2-style, Bailing naming: q_a/q_b/kv_a/kv_b/dense)
# ---------------------------------------------------------------------------


class MLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=args.attention_bias,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=args.use_qkv_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.use_qkv_bias,
        )

        # YaRN scaling adjustment if applicable
        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = args.rope_scaling.get("factor", 1.0)
            if mscale_all_dim and scaling_factor > 1:
                s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scale = self.scale * s * s

        self.rope = initialize_rope(
            self.qk_rope_head_dim,
            args.rope_theta,
            traditional=args.rope_interleave,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(
            B, L, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)
        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)

        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


# ---------------------------------------------------------------------------
# MLP / MoE — reused with Bailing-style key naming
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        d = args.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, d, bias=args.use_bias)
        self.down_proj = nn.Linear(d, args.hidden_size, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, d, bias=args.use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def group_expert_select(
    gates: mx.array,
    e_score_correction_bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator
    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function

        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = Gate(args)
        shared_dim = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        )
        self.shared_experts = (
            MLP(args=args, intermediate_size=shared_dim * args.num_shared_experts)
            if args.num_shared_experts > 0 and args.moe_router_enable_shared_expert
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = (out * topk_weight[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


# ---------------------------------------------------------------------------
# Decoder layer (hybrid dispatch) + MTP layer
# ---------------------------------------------------------------------------


def _is_global_layer(layer_idx: int, args: ModelArgs) -> bool:
    return (
        (layer_idx + 1) % args.layer_group_size == 0
        or layer_idx
        >= args.num_hidden_layers // args.layer_group_size * args.layer_group_size
    )


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_global = _is_global_layer(layer_idx, args)

        if self.is_global:
            self.attention = MLAAttention(args)
        else:
            self.attention = LinearAttention(args, layer_idx=layer_idx)

        if args.num_experts is not None and layer_idx >= args.first_k_dense_replace:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        if self.is_global:
            r = self.attention(self.input_layernorm(x), mask, cache)
        else:
            r = self.attention(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class MTPLayer(nn.Module):
    """DeepSeek-V3-flavored MTP head: enorm/hnorm + eh_proj fuse, then MLA + MoE,
    with a final_layernorm on the way out."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attention = MLAAttention(args)
        self.mlp = SparseMoeBlock(args)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.is_global = True

    def __call__(
        self,
        input_embeds: mx.array,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        e = self.enorm(input_embeds)
        h = self.hnorm(hidden_states)
        x = self.eh_proj(mx.concatenate([e, h], axis=-1))
        residual = x
        x = self.input_layernorm(x)
        r = self.attention(x, mask, cache)
        h2 = residual + r
        r2 = self.mlp(self.post_attention_layernorm(h2))
        h2 = h2 + r2
        return self.final_layernorm(h2)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        # MTP heads come right after — same `model.layers.{i}` index convention
        for i in range(args.num_nextn_predict_layers):
            self.layers.append(MTPLayer(args, layer_idx=args.num_hidden_layers + i))
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # global / linear cache slot indices used to compute masks once
        self._first_global = next(
            (i for i, l in enumerate(self.layers) if l.is_global), 0
        )
        self._first_linear = next(
            (i for i, l in enumerate(self.layers) if not l.is_global), 0
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self._first_global])
        gla_mask = create_ssm_mask(h, cache[self._first_linear])
        offset = (
            cache[self._first_global].offset
            if cache[self._first_global] is not None
            else 0
        )

        # Skip MTP heads in the standard forward — they are inference-only
        # post-LM-head stubs that we don't currently invoke. Run only the
        # `num_hidden_layers` base decoder layers.
        for layer, c in zip(
            self.layers[: self.args.num_hidden_layers],
            cache[: self.args.num_hidden_layers],
        ):
            mask = attn_mask if layer.is_global else gla_mask
            h = layer(h, mask, c, offset=offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Stack per-expert MLPs into the SwitchGLU layout used at runtime.
        # Ling 2.6 Flash source has 73,728 per-expert keys (256 experts x 32
        # layers x 3 projs x 3 fields) with ZERO prestacked. mx.stack(...)
        # builds a lazy op that holds every popped per-expert tensor alive
        # until materialized -- without per-stack materialization, peak RAM
        # during sanitize blows up. Force materialization after each stack
        # so source tensors are released.
        _materialize = mx.eval
        n_total = self.args.num_hidden_layers + self.args.num_nextn_predict_layers
        for l in range(n_total):
            prefix = f"model.layers.{l}"
            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        e0 = f"{prefix}.mlp.experts.0.{m}.{k}"
                        if e0 in weights:
                            stacked = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            packed = mx.stack(stacked)
                            _materialize(packed)
                            del stacked
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = packed

                # gate.weight (and optional bias) → gate.gate_proj.{weight,bias}
                if f"{prefix}.mlp.gate.weight" in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(
                        f"{prefix}.mlp.gate.weight"
                    )
                if f"{prefix}.mlp.gate.bias" in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = weights.pop(
                        f"{prefix}.mlp.gate.bias"
                    )

        # MXFP4 / older converter shape repair: bundles converted by an
        # earlier `convert_ling_mxfp4.py` revision (Ling-2.6-flash-MXFP4 +
        # Ling-2.6-flash-MXFP4-CRACK observed 2026-05-05) shipped already-
        # stacked switch_mlp tensors but flattened the (out, in_per_row)
        # axes into one axis — i.e. uint32 packed weights as
        # (n_experts, out * packed_in) instead of (n_experts, out, packed_in),
        # and matching scales/biases as (n_experts, out * n_groups).
        # mlx_lm's quantized SwitchLinear strict-checks shape during
        # `model.load_weights(...)` and raises ValueError. Total elements
        # match exactly so a reshape preserves data byte-for-byte; we
        # reshape using `out=moe_intermediate_size` (gate/up) or
        # `hidden_size` (down). This is a no-op on correctly-shaped 3D
        # bundles. Applies to both base layers and the MTP head.
        moe_inter = self.args.moe_intermediate_size
        hidden = self.args.hidden_size
        for l in range(n_total):
            prefix = f"model.layers.{l}"
            if l < self.args.first_k_dense_replace:
                continue
            for proj_name, out_dim in (
                ("gate_proj", moe_inter),
                ("up_proj", moe_inter),
                ("down_proj", hidden),
            ):
                base = f"{prefix}.mlp.switch_mlp.{proj_name}"
                for field in ("weight", "scales", "biases"):
                    key = f"{base}.{field}"
                    arr = weights.get(key)
                    if arr is None or arr.ndim != 2:
                        continue
                    n_exp, flat = arr.shape
                    if flat % out_dim != 0:
                        # Shape can't be the legacy flat layout — leave alone.
                        continue
                    in_per_row = flat // out_dim
                    weights[key] = arr.reshape(n_exp, out_dim, in_per_row)

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # gate router stays at 8-bit (sigmoid scoring is sensitive)
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            # expert_bias is fp32 in source; do not downcast.
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        # Only base decoder layers participate in normal generation cache.
        # MTP cache slots are intentionally NOT created here — LanguageModel
        # forward iterates only `num_hidden_layers` and never writes to them,
        # so creating BatchKVCache wrappers for unused MTP slots leaves them
        # with `self.keys=None` and crashes BatchGenerator.extract_cache()
        # with `'NoneType' object is not subscriptable`. Index parity inside
        # LanguageModel is preserved by zip(layers[:N], cache[:N]) regardless.
        # If/when the MTP head is wired into spec-decode, allocate a separate
        # cache list for it; do not reuse this one.
        for l in self.model.layers[: self.args.num_hidden_layers]:
            if l.is_global:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=1))
        return caches
