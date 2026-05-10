"""Zyphra ZAYA1-VL MLX runtime adapter.

ZAYA1-VL reuses a Qwen2.5-VL vision tower, but its language trunk is not a
Qwen model.  Each text layer contains CCA attention followed by ZAYA top-1
MoE, with optional image-token LoRA paths on both sublayers.  This module is
registered under ``mlx_vlm.models.zaya1_vl`` so the stock mlx-vlm loader can
resolve real ZAYA1-VL model classes instead of routing the bundle through a
corrupt Qwen alias.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
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
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.qwen2_5_vl import VisionConfig, VisionModel

from .zaya import (
    ResidualScaleMerge,
    ZayaNoStateCache,
    ZayaPartialRoPE,
    ZayaRouter,
)


def register_mlx_vlm_zaya1_vl() -> None:
    """Expose this adapter at the import path mlx-vlm expects."""
    import sys

    sys.modules.setdefault("mlx_vlm.models.zaya1_vl", sys.modules[__name__])


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "zaya1_vl"
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    num_query_groups: int = 2
    head_dim: int = 128
    kv_channels: int | None = None
    cca_num_q_heads: int | None = None
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
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
    attention_bias: bool = False
    add_bias_linear: bool = False
    lm_head_bias: bool = False
    gated_linear_unit: bool = True
    activation_func: str = "swiglu"
    vision_lora: bool = False
    vision_lora_rank_attn: int = 8
    vision_lora_rank_mlp: int = 32
    use_lora_att: bool = False
    lora_rank: int = 0
    quantization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kv_channels is None:
            self.kv_channels = self.head_dim
        if self.cca_num_q_heads is None:
            self.cca_num_q_heads = self.num_attention_heads
        if self.num_query_groups is None:
            self.num_query_groups = self.num_key_value_heads
        if self.cca_time0 is None:
            self.cca_time0 = 2
        if self.cca_time1 is None:
            self.cca_time1 = 2


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "zaya1_vl"
    image_token_id: int = 262147
    image_token_index: int = 262147
    video_token_id: int | None = None
    vision_start_token_id: int = 255999
    vision_end_token_id: int = 256000
    vocab_size: int = 262272
    eos_token_id: int | list[int] | None = 262143
    pad_token_id: int | None = 0
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        raw = dict(params)
        flat_text_raw = {
            k: v
            for k, v in raw.items()
            if k in {f.name for f in fields(TextConfig)}
        }
        text_raw = raw.get("text_config")
        if isinstance(text_raw, dict):
            text_raw = {**flat_text_raw, **text_raw}
        else:
            text_raw = flat_text_raw
        if "rotary_base" in raw and text_raw.get("rope_theta") is None:
            text_raw["rope_theta"] = raw["rotary_base"]
        quant = text_raw.get("quantization")
        if isinstance(quant, dict):
            quant = dict(quant)
            if str(quant.get("mode", "")).lower() == "affine+mxtq":
                quant["container_mode"] = quant["mode"]
                quant["mode"] = "affine"
            if str(raw.get("weight_format", "")).lower() == "mxfp4":
                quant.setdefault("embed_bits", 8)
            text_raw["quantization"] = quant
        text_clean = {
            k: v
            for k, v in text_raw.items()
            if k in {f.name for f in fields(TextConfig)} and v is not None
        }
        params["text_config"] = dict(text_clean)
        text_config = TextConfig(**text_clean)

        vision_raw = raw.get("vision_config") or {}
        vision_fields = set(VisionConfig.__dataclass_fields__.keys())
        vision_clean = {
            k: v for k, v in vision_raw.items() if k in vision_fields and v is not None
        }
        vision_config = VisionConfig(**vision_clean)

        image_token_id = int(raw.get("image_token_id", 262147))
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            model_type=str(raw.get("model_type", "zaya1_vl")),
            image_token_id=image_token_id,
            image_token_index=image_token_id,
            video_token_id=raw.get("video_token_id"),
            vision_start_token_id=int(raw.get("vision_start_token_id", 255999)),
            vision_end_token_id=int(raw.get("vision_end_token_id", 256000)),
            vocab_size=int(raw.get("vocab_size", text_config.vocab_size)),
            eos_token_id=raw.get("eos_token_id", 262143),
            pad_token_id=raw.get("pad_token_id", 0),
            tie_word_embeddings=bool(
                raw.get("tie_word_embeddings", text_config.tie_word_embeddings)
            ),
        )


def _apply_lora(layers: list[nn.Module], x: mx.array) -> mx.array:
    h = x
    for layer in layers:
        h = layer(h)
    return h


class ZayaVLQKV(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        dim = args.hidden_size
        head_dim = int(args.kv_channels or args.head_dim)
        latent_q = int(args.cca_num_q_heads or args.num_attention_heads) * head_dim
        latent_k = args.num_key_value_heads * head_dim
        packed = latent_q + latent_k
        groups = packed // head_dim

        self.linear_q = nn.Linear(dim, latent_q, bias=args.attention_bias)
        self.linear_k = nn.Linear(dim, latent_k, bias=args.attention_bias)
        value_half = latent_k // 2
        self.val_proj1 = nn.Linear(dim, value_half, bias=args.attention_bias)
        self.val_proj2 = nn.Linear(dim, value_half, bias=args.attention_bias)
        self.conv_qk = [
            nn.Conv1d(packed, packed, int(args.cca_time0), groups=packed),
            nn.Conv1d(packed, packed, int(args.cca_time1), groups=groups),
        ]
        self.temp = mx.zeros((args.num_key_value_heads,))

        if args.vision_lora:
            rank = int(args.vision_lora_rank_attn)
            self.lora_linear_q = [
                nn.Linear(dim, rank, bias=False),
                nn.Linear(rank, latent_q, bias=False),
            ]
            self.lora_linear_k = [
                nn.Linear(dim, rank, bias=False),
                nn.Linear(rank, latent_k, bias=False),
            ]
            self.lora_val_proj1 = [
                nn.Linear(dim, rank, bias=False),
                nn.Linear(rank, value_half, bias=False),
            ]
            self.lora_val_proj2 = [
                nn.Linear(dim, rank, bias=False),
                nn.Linear(rank, value_half, bias=False),
            ]


class ZayaVLCCAAttention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.head_dim = int(args.kv_channels or args.head_dim)
        self.n_heads = int(args.cca_num_q_heads or args.num_attention_heads)
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
        self.vision_lora = args.vision_lora

        self.qkv = ZayaVLQKV(args)
        self.o_proj = nn.Linear(
            self.latent_q_dim, args.hidden_size, bias=args.attention_bias
        )
        if args.vision_lora:
            rank = int(args.vision_lora_rank_attn)
            self.lora_linear_o = [
                nn.Linear(self.latent_q_dim, rank, bias=False),
                nn.Linear(rank, args.hidden_size, bias=False),
            ]

        rotary_dim = int(self.head_dim * args.partial_rotary_factor)
        rotary_dim = max(0, min(self.head_dim, rotary_dim))
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
        image_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        kv_cache, state_cache = self._state_cache(cache)
        image_gate = (
            image_mask.astype(x.dtype)[..., None]
            if self.vision_lora and image_mask is not None
            else None
        )

        q_raw = self.qkv.linear_q(x)
        k_raw = self.qkv.linear_k(x)
        v1 = self.qkv.val_proj1(x)
        if image_gate is not None:
            q_raw = q_raw + _apply_lora(self.qkv.lora_linear_q, x) * image_gate
            k_raw = k_raw + _apply_lora(self.qkv.lora_linear_k, x) * image_gate
            v1 = v1 + _apply_lora(self.qkv.lora_val_proj1, x) * image_gate

        q_pre = q_raw.reshape(B, L, self.n_heads, self.head_dim)
        k_pre = k_raw.reshape(B, L, self.n_kv_heads, self.head_dim)
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
        v2 = self.qkv.val_proj2(h_prev)
        if image_gate is not None:
            v2 = v2 + _apply_lora(self.qkv.lora_val_proj2, h_prev) * image_gate
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
        projected = self.o_proj(out)
        if image_gate is not None:
            projected = projected + _apply_lora(self.lora_linear_o, out) * image_gate
        return projected


class ZayaVLExpertLoRA(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        rank = int(args.vision_lora_rank_mlp)
        hidden_mid = args.ffn_hidden_size // 2
        self.lora_fc1 = [
            nn.Linear(args.hidden_size, rank, bias=False),
            nn.Linear(rank, args.ffn_hidden_size, bias=False),
        ]
        self.lora_fc2 = [
            nn.Linear(hidden_mid, rank, bias=False),
            nn.Linear(rank, args.hidden_size, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = _apply_lora(self.lora_fc1, x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return _apply_lora(self.lora_fc2, swiglu(gate, up))


class ZayaVLExpertContainer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        hidden_dims = args.ffn_hidden_size // 2
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            hidden_dims,
            args.num_experts,
            bias=False,
        )
        quant = args.quantization
        if isinstance(quant, dict):
            bits = int(quant.get("bits", 4))
            group_size = int(quant.get("group_size", 64))
            mode = str(quant.get("mode", "affine"))
            self.switch_mlp.gate_proj = QuantizedSwitchLinear(
                args.hidden_size,
                hidden_dims,
                args.num_experts,
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            self.switch_mlp.up_proj = QuantizedSwitchLinear(
                args.hidden_size,
                hidden_dims,
                args.num_experts,
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            self.switch_mlp.down_proj = QuantizedSwitchLinear(
                hidden_dims,
                args.hidden_size,
                args.num_experts,
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
        if args.vision_lora:
            self.local_experts = [ZayaVLExpertLoRA(args) for _ in range(args.num_experts)]


class ZayaVLMoE(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.args = args
        self.router = ZayaRouter(args, layer_idx + 1)
        self.experts = ZayaVLExpertContainer(args)

    def _expert_lora(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array,
        active: Optional[mx.array],
        image_mask: Optional[mx.array],
    ) -> mx.array:
        if not self.args.vision_lora or image_mask is None:
            return mx.zeros_like(x)
        selected_scores = scores[..., 0]
        selected_indices = indices[..., 0]
        active_mask = active[..., 0] if active is not None else mx.ones_like(image_mask)
        out = mx.zeros_like(x)
        for expert_id, expert in enumerate(self.experts.local_experts):
            token_mask = (selected_indices == expert_id) & image_mask & active_mask
            addon = expert(x) * selected_scores[..., None]
            out = mx.where(token_mask[..., None], addon, out)
        return out

    def __call__(
        self,
        x: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ):
        scores, indices, active, router_hidden_states_next = self.router(
            x,
            prev_router_hidden_states=prev_router_hidden_states,
        )
        y = self.experts.switch_mlp(x, indices)
        y = (y * scores[..., None]).sum(axis=-2)
        y = y + self._expert_lora(x, indices, scores, active, image_mask)
        if active is not None:
            mod_y = x * scores
            y = mx.where(active, y, mod_y)
        return y, router_hidden_states_next


class ZayaVLAttentionBlock(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        self.res_scale = ResidualScaleMerge(args.hidden_size, residual=layer_idx > 0)
        self.self_attn = ZayaVLCCAAttention(args)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array],
        cache: Optional[Any],
        image_mask: Optional[mx.array],
    ):
        residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is not None:
            residual = residual.astype(mx.float32) + hidden_states.astype(mx.float32)
        else:
            residual = hidden_states.astype(mx.float32)
        h = self.input_norm(residual.astype(self.input_norm.weight.dtype))
        return self.self_attn(h, mask, cache, image_mask), residual


class ZayaVLMLPBlock(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        self.res_scale = ResidualScaleMerge(args.hidden_size, residual=True)
        self.zaya_block = ZayaVLMoE(args, layer_idx)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        prev_router_hidden_states: Optional[mx.array],
        image_mask: Optional[mx.array],
    ):
        residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is not None:
            residual = residual.astype(mx.float32) + hidden_states.astype(mx.float32)
        else:
            residual = hidden_states.astype(mx.float32)
        h = self.input_norm(residual.astype(self.input_norm.weight.dtype))
        hidden_states, prev_router_hidden_states = self.zaya_block(
            h,
            prev_router_hidden_states=prev_router_hidden_states,
            image_mask=image_mask,
        )
        return hidden_states, residual, prev_router_hidden_states


class Zaya1VLLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = ZayaVLAttentionBlock(args, layer_idx)
        self.mlp = ZayaVLMLPBlock(args, layer_idx)

    @property
    def zaya_block(self):
        return self.mlp.zaya_block

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_router_hidden_states: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ):
        hidden_states, residual = self.attn(
            hidden_states,
            residual,
            mask,
            cache,
            image_mask,
        )
        return self.mlp(
            hidden_states,
            residual,
            prev_router_hidden_states,
            image_mask,
        )


class Zaya1VLTextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        quant = args.quantization if isinstance(args.quantization, dict) else None
        if quant is not None:
            self.embed_tokens = nn.QuantizedEmbedding(
                args.vocab_size,
                args.hidden_size,
                group_size=int(quant.get("group_size", 64)),
                bits=int(quant.get("embed_bits", quant.get("bits", 4))),
                mode=str(quant.get("mode", "affine")),
            )
        else:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Zaya1VLLayer(args, i) for i in range(args.num_hidden_layers)]
        self.res_scale = ResidualScaleMerge(args.hidden_size, residual=True)
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings=None,
        image_mask: Optional[mx.array] = None,
    ):
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
                image_mask,
            )
        residual, h = self.res_scale(residual, h)
        if residual is not None:
            h = h.astype(mx.float32) + residual.astype(mx.float32)
        else:
            h = h.astype(mx.float32)
        return self.final_norm(h.astype(self.final_norm.weight.dtype))

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(2)) for _ in self.layers]


class Zaya1VLLanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig | None = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Zaya1VLTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=args.lm_head_bias
            )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return self.model.make_cache()

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        image_mask: Optional[mx.array] = None,
        **_kwargs,
    ):
        if input_embeddings is None:
            input_embeddings = inputs_embeds
        out = self.model(inputs, cache, input_embeddings, image_mask)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(logits=logits)


class LanguageModel(Zaya1VLLanguageModel):
    pass


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = Zaya1VLLanguageModel(config.text_config, config)

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def model(self):
        return self.language_model.model

    def make_cache(self):
        return self.language_model.make_cache()

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id: int,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        image_positions = input_ids == image_token_id
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(input_ids.shape[0]):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()
            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Image features and image tokens do not match for batch "
                        f"{batch_idx}: tokens={num_positions}, "
                        f"features={batch_features.shape[0]}"
                    )
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered = batch_features[feature_indices]
                batch_outputs.append(
                    mx.where(image_mask[..., None], gathered, inputs_embeds[batch_idx])
                )
                feature_start_idx += num_positions
            else:
                batch_outputs.append(inputs_embeds[batch_idx])

        return mx.stack(batch_outputs, axis=0)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("ZAYA1-VL image requests require image_grid_thw")

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        image_features = self.vision_tower(
            pixel_values.astype(dtype),
            image_grid_thw,
            output_hidden_states=False,
        )
        merged = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            image_features,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=merged)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        embeddings = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        image_mask = (input_ids == self.config.image_token_id) if pixel_values is not None else None
        return self.language_model(
            input_ids,
            embeddings.inputs_embeds,
            mask=mask,
            cache=cache,
            image_mask=image_mask,
            **kwargs,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        fixed: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith("model."):
                key = key.replace("model.", "language_model.model.", 1)
            elif key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.lm_head.", 1)

            if ".layers." in key and ".zaya_block." in key and ".mlp.zaya_block." not in key:
                key = key.replace(".zaya_block.", ".mlp.zaya_block.", 1)

            if (
                key.endswith("patch_embed.proj.weight")
                and value.ndim == 5
                and value.shape[1] in (1, 3)
                and value.shape[-1] not in (1, 3)
            ):
                value = value.transpose(0, 2, 3, 4, 1)

            if ".self_attn.qkv.conv_qk." in key and key.endswith(".weight"):
                value = value.swapaxes(-1, -2)
            fixed[key] = value

        if self.config.text_config.tie_word_embeddings:
            fixed.pop("language_model.lm_head.weight", None)
        return fixed


__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
    "Zaya1VLLanguageModel",
    "register_mlx_vlm_zaya1_vl",
]
