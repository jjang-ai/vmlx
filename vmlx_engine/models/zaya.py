"""Zyphra ZAYA MLX runtime.

ZAYA alternates CCA attention layers with top-1 MOD/MoE layers.  The converted
vMLX/JANG bundles keep the CCA/router tensors native and store routed experts as
pre-stacked ``zaya_block.experts.switch_mlp`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.switch_layers import SwitchGLU


def register_mlx_lm_zaya() -> None:
    """Expose this runtime at the import path mlx-lm expects."""
    import sys

    sys.modules.setdefault("mlx_lm.models.zaya", sys.modules[__name__])


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "zaya"
    hidden_size: int = 2048
    num_hidden_layers: int = 80
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    num_query_groups: int = 2
    cca_num_q_heads: int = 8
    kv_channels: int = 128
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 131072
    rope_theta: float = 5000000.0
    norm_epsilon: float = 1e-5
    vocab_size: int = 262272
    num_experts: int = 16
    moe_router_topk: int = 1
    zaya_use_eda: bool = True
    zaya_use_mod: bool = True
    zaya_mlp_expansion: int = 256
    tie_word_embeddings: bool = True
    residual_in_fp32: bool = True
    scale_residual_merge: bool = True
    cca_time0: int = 2
    cca_time1: int = 2
    clamp_temp: bool = False


class ResidualScaleMerge(nn.Module):
    def __init__(self, dim: int, residual: bool):
        super().__init__()
        self.hidden_states_scale = mx.ones((dim,))
        self.hidden_states_bias = mx.zeros((dim,))
        if residual:
            self.residual_scale = mx.ones((dim,))
            self.residual_bias = mx.zeros((dim,))

    def __call__(
        self,
        residual: Optional[mx.array],
        hidden_states: mx.array,
    ) -> tuple[Optional[mx.array], mx.array]:
        hs = (
            hidden_states.astype(mx.float32) + self.hidden_states_bias
        ) * self.hidden_states_scale
        if residual is not None and "residual_scale" in self:
            residual = (
                residual.astype(mx.float32) + self.residual_bias
            ) * self.residual_scale
        return residual, hs


class ZayaQKV(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        head_dim = args.kv_channels
        latent_q = args.cca_num_q_heads * head_dim
        latent_k = args.num_key_value_heads * head_dim
        packed = latent_q + latent_k
        groups = packed // head_dim

        self.linear_q = nn.Linear(dim, latent_q, bias=False)
        self.linear_k = nn.Linear(dim, latent_k, bias=False)
        value_half = latent_k // 2
        self.val_proj1 = nn.Linear(dim, value_half, bias=False)
        self.val_proj2 = nn.Linear(dim, value_half, bias=False)
        self.conv_qk = [
            nn.Conv1d(packed, packed, 2, groups=packed),
            nn.Conv1d(packed, packed, 2, groups=groups),
        ]
        self.temp = mx.zeros((args.num_key_value_heads,))


class ZayaPartialRoPE(nn.Module):
    """Zyphra partial RoPE.

    mlx-lm's proportional RoPE uses the full head dimension in the frequency
    denominator. ZAYA follows Transformers/vLLM and derives frequencies from
    the rotated dimension itself (`head_dim * partial_rotary_factor`).
    """

    def __init__(self, head_dim: int, rotary_dim: int, base: float):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = base

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        return mx.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if self.rotary_dim <= 0:
            return x
        L = x.shape[-2]
        inv_freq = 1.0 / (
            self.base
            ** (
                mx.arange(0, self.rotary_dim, 2, dtype=mx.float32)
                / float(self.rotary_dim)
            )
        )
        positions = mx.arange(L, dtype=mx.float32)
        if isinstance(offset, mx.array):
            positions = positions + offset.astype(mx.float32)
        else:
            positions = positions + float(offset)
        freqs = positions[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype).reshape(1, 1, L, self.rotary_dim)
        sin = mx.sin(emb).astype(x.dtype).reshape(1, 1, L, self.rotary_dim)
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_rot = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return mx.concatenate([x_rot, x_pass], axis=-1)


class ZayaCCAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.head_dim = args.kv_channels
        self.n_heads = args.cca_num_q_heads
        self.n_kv_heads = args.num_key_value_heads
        self.gqa_groups = self.n_heads // self.n_kv_heads
        self.latent_q_dim = self.n_heads * self.head_dim
        self.latent_k_dim = self.n_kv_heads * self.head_dim
        self.packed_dim = self.latent_q_dim + self.latent_k_dim
        self.total_padding = max(0, int(args.cca_time0) - 1) + max(
            0, int(args.cca_time1) - 1
        )
        self.sqrt_head_dim = self.head_dim**0.5
        self.scale = self.head_dim**-0.5
        self.clamp_temp = args.clamp_temp

        self.qkv = ZayaQKV(args)
        self.o_proj = nn.Linear(self.latent_q_dim, args.hidden_size, bias=False)
        rotary_dim = int(self.head_dim * args.partial_rotary_factor)
        rotary_dim = max(0, min(self.head_dim, rotary_dim))
        self.rotary_dim = rotary_dim
        self.rope = ZayaPartialRoPE(
            self.head_dim,
            rotary_dim=rotary_dim,
            base=args.rope_theta,
        )

    def _state_cache(self, cache):
        if isinstance(cache, CacheList):
            return cache[0], cache[1]
        return cache, None

    def _causal_convs(self, qk: mx.array, state_cache: Optional[ArraysCache]):
        conv_state = (
            state_cache[0] if state_cache is not None and state_cache.cache else None
        )
        if conv_state is None:
            conv_state = mx.zeros(
                (qk.shape[0], self.total_padding, qk.shape[-1]),
                dtype=qk.dtype,
            )
        qk_with_state = mx.concatenate([conv_state, qk], axis=1)
        conv0 = self.qkv.conv_qk[0](qk_with_state)
        conv1 = self.qkv.conv_qk[1](conv0)

        if state_cache is not None:
            state_cache[0] = mx.stop_gradient(qk_with_state[:, -self.total_padding :, :])
        return conv1

    def _shift_hidden(self, h: mx.array, state_cache: Optional[ArraysCache]):
        prev = state_cache[1] if state_cache is not None and state_cache.cache else None
        if prev is None:
            prev = mx.zeros((h.shape[0], 1, h.shape[-1]), dtype=h.dtype)
        shifted = mx.concatenate([prev, h], axis=1)[:, :-1, :]
        if state_cache is not None:
            state_cache[1] = mx.stop_gradient(h[:, -1:, :])
        return shifted

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        kv_cache, state_cache = self._state_cache(cache)

        q_pre = self.qkv.linear_q(x).reshape(B, L, self.n_heads, self.head_dim)
        k_pre = self.qkv.linear_k(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        qk_packed = mx.concatenate(
            [q_pre.reshape(B, L, -1), k_pre.reshape(B, L, -1)],
            axis=-1,
        )
        qk_conv = self._causal_convs(qk_packed, state_cache)

        q_conv = qk_conv[..., : self.latent_q_dim].reshape(
            B, L, self.n_heads, self.head_dim
        )
        k_conv = qk_conv[..., self.latent_q_dim :].reshape(
            B, L, self.n_kv_heads, self.head_dim
        )

        k_pre_rep = (
            mx.repeat(k_pre[:, :, :, None, :], self.gqa_groups, axis=3)
            .reshape(B, L, self.n_heads, self.head_dim)
        )
        q_mean = (q_pre + k_pre_rep) * 0.5
        k_mean = q_mean.reshape(
            B, L, self.n_kv_heads, self.gqa_groups, self.head_dim
        ).mean(axis=3)

        queries = q_conv + q_mean
        keys = k_conv + k_mean

        # Match Zyphra's runtime: q/k are normalized in fp32. For this bundle
        # temp is already a learned multiplier; only exp(clamp(temp)) when the
        # config explicitly requests clamp_temp.
        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)
        q_norm = mx.maximum(mx.linalg.norm(queries, ord=2, axis=-1, keepdims=True), 1e-6)
        k_norm = mx.maximum(mx.linalg.norm(keys, ord=2, axis=-1, keepdims=True), 1e-6)
        queries = queries * (self.sqrt_head_dim / q_norm)
        temp = self.qkv.temp.astype(mx.float32)
        if self.clamp_temp:
            temp = mx.exp(mx.clip(temp, 1e-7, 2.0))
        temp = temp.reshape(1, 1, self.n_kv_heads, 1)
        keys = keys * (self.sqrt_head_dim / k_norm) * temp

        h_prev = self._shift_hidden(x, state_cache)
        v1 = self.qkv.val_proj1(x)
        v2 = self.qkv.val_proj2(h_prev)
        values = mx.concatenate([v1, v2], axis=-1).reshape(
            B, L, self.n_kv_heads, self.head_dim
        )

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = kv_cache.offset if kv_cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if kv_cache is not None:
            keys, values = kv_cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries, keys, values, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class ZayaRouter(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        d = args.zaya_mlp_expansion
        logits = args.num_experts + (1 if args.zaya_use_mod else 0)
        self.down_proj = nn.Linear(args.hidden_size, d, bias=True)
        self.rmsnorm_eda = nn.RMSNorm(d, eps=args.norm_epsilon)
        self.router_mlp = [
            nn.Linear(d, d, bias=True),
            nn.GELU(),
            nn.Linear(d, d, bias=True),
            nn.GELU(),
            nn.Linear(d, logits, bias=False),
        ]
        self.balancing_biases = mx.zeros((logits,))
        if layer_idx > 1:
            self.router_states_scale = mx.ones((d,))
        self.num_experts = args.num_experts
        self.use_mod = args.zaya_use_mod

    def __call__(
        self,
        x: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        h = self.down_proj(x.astype(mx.float32))
        if (
            prev_router_hidden_states is not None
            and "router_states_scale" in self
        ):
            h = h + prev_router_hidden_states * self.router_states_scale
        router_hidden_states_next = mx.stop_gradient(h)
        h = self.rmsnorm_eda(h)
        for layer in self.router_mlp:
            h = layer(h)
        expert_prob = mx.softmax(h, axis=-1)
        biased = expert_prob.astype(mx.float32) + self.balancing_biases
        indices = mx.argmax(biased, axis=-1, keepdims=True)
        scores = mx.take_along_axis(expert_prob, indices, axis=-1).astype(x.dtype)
        if self.use_mod:
            active = indices < self.num_experts
            safe_indices = mx.minimum(indices, self.num_experts - 1)
            return scores, safe_indices, active, router_hidden_states_next
        return scores, indices, None, router_hidden_states_next


class ZayaMoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.router = ZayaRouter(args, layer_idx)
        self.experts = type("ZayaExpertContainer", (nn.Module,), {})()
        self.experts.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.ffn_hidden_size // 2,
            args.num_experts,
            bias=False,
        )

    def __call__(
        self,
        x: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        scores, indices, active, router_hidden_states_next = self.router(
            x,
            prev_router_hidden_states=prev_router_hidden_states,
        )
        y = self.experts.switch_mlp(x, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        if active is not None:
            mod_y = x * scores
            y = mx.where(active, y, mod_y)
        return y, router_hidden_states_next


class ZayaLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        self.res_scale = ResidualScaleMerge(args.hidden_size, residual=layer_idx > 0)
        if layer_idx % 2 == 0:
            self.self_attn = ZayaCCAAttention(args)
            self.zaya_block = None
        else:
            self.self_attn = None
            self.zaya_block = ZayaMoE(args, layer_idx)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        if self.res_scale is not None:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is not None:
            residual = residual.astype(mx.float32) + hidden_states.astype(mx.float32)
        else:
            residual = hidden_states.astype(mx.float32)
        h = self.input_norm(residual.astype(self.input_norm.weight.dtype))
        if self.self_attn is not None:
            hidden_states = self.self_attn(h, mask, cache)
        else:
            hidden_states, prev_router_hidden_states = self.zaya_block(
                h, prev_router_hidden_states
            )
        return hidden_states, residual, prev_router_hidden_states


class ZayaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ZayaLayer(args, i) for i in range(args.num_hidden_layers)]
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        self.res_scale = ResidualScaleMerge(args.hidden_size, residual=True)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        first_attn_cache = None
        for c in cache:
            if isinstance(c, CacheList):
                first_attn_cache = c[0]
                break
        mask = create_attention_mask(h, first_attn_cache)
        residual = None
        prev_router_hidden_states = None
        for layer, c in zip(self.layers, cache):
            h, residual, prev_router_hidden_states = layer(
                h,
                residual,
                mask,
                c,
                prev_router_hidden_states,
            )
        residual, h = self.res_scale(residual, h)
        if residual is not None:
            h = h.astype(mx.float32) + residual.astype(mx.float32)
        else:
            h = h.astype(mx.float32)
        return self.final_norm(h.astype(self.final_norm.weight.dtype))

    def make_cache(self):
        caches = []
        for i in range(len(self.layers)):
            if i % 2 == 0:
                caches.append(CacheList(KVCache(), ArraysCache(2)))
            else:
                # mlx-lm's BatchGenerator expects one mergeable cache object
                # per layer. ZAYA MoE layers do not have time-state, but a
                # placeholder ArraysCache keeps continuous batching compatible.
                caches.append(ArraysCache(1))
        return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ZayaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self.model.make_cache()

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for key in list(weights.keys()):
            if ".self_attn.qkv.conv_qk." in key and key.endswith(".weight"):
                weights[key] = weights[key].swapaxes(-1, -2)

        for layer in range(self.args.num_hidden_layers):
            if layer % 2 == 0:
                continue
            prefix = f"model.layers.{layer}.zaya_block.experts"
            if f"{prefix}.switch_mlp.gate_proj.weight" in weights:
                continue
            gate_parts = []
            up_parts = []
            down_parts = []
            for expert in range(self.args.num_experts):
                fc1_key = f"{prefix}.local_experts.{expert}.linear_fc1.weight"
                fc2_key = f"{prefix}.local_experts.{expert}.linear_fc2.weight"
                if fc1_key not in weights or fc2_key not in weights:
                    gate_parts = []
                    break
                fc1 = weights.pop(fc1_key)
                down_parts.append(weights.pop(fc2_key))
                gate, up = mx.split(fc1, 2, axis=0)
                gate_parts.append(gate)
                up_parts.append(up)
            if gate_parts:
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = mx.stack(gate_parts)
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = mx.stack(up_parts)
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = mx.stack(down_parts)

        return weights
