# SPDX-License-Identifier: Apache-2.0
"""Source-owned Step-3.7 mlx-vlm runtime support.

This module is intentionally partial: it provides the config layer needed by a
real mlx-vlm port while keeping the release blocker open until the model,
vision encoder, processor, and merge path are implemented and live-proven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models import step3p5


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _preserve_extra(target: object, params: dict[str, Any], known: set[str]) -> None:
    for key, value in params.items():
        if key not in known:
            setattr(target, key, value)


@dataclass
class TextConfig:
    model_type: str = "step3p5"
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    sliding_window: int | None = None
    use_head_wise_attn_gate: bool = False
    layer_types: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> "TextConfig":
        data = _copy_dict(params)
        known = {
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "sliding_window",
            "use_head_wise_attn_gate",
            "layer_types",
        }
        cfg = cls(
            model_type=str(data.get("model_type") or "step3p5"),
            hidden_size=data.get("hidden_size"),
            num_hidden_layers=data.get("num_hidden_layers"),
            sliding_window=data.get("sliding_window"),
            use_head_wise_attn_gate=bool(data.get("use_head_wise_attn_gate", False)),
            layer_types=list(data.get("layer_types") or []),
        )
        _preserve_extra(cfg, data, known)
        return cfg


@dataclass
class VisionConfig:
    model_type: str = "perception_encoder"
    hidden_size: int | None = None
    width: int | None = None
    layers: int | None = None
    heads: int | None = None
    patch_size: int | None = None
    image_size: int | None = None
    use_rope2d: bool = True

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> "VisionConfig":
        data = _copy_dict(params)
        known = {
            "model_type",
            "hidden_size",
            "width",
            "layers",
            "heads",
            "patch_size",
            "image_size",
            "use_rope2d",
        }
        cfg = cls(
            model_type=str(data.get("model_type") or "perception_encoder"),
            hidden_size=data.get("hidden_size"),
            width=data.get("width"),
            layers=data.get("layers"),
            heads=data.get("heads"),
            patch_size=data.get("patch_size"),
            image_size=data.get("image_size"),
            use_rope2d=bool(data.get("use_rope2d", True)),
        )
        _preserve_extra(cfg, data, known)
        return cfg


@dataclass
class ProjectorConfig:
    hidden_size: int | None = None
    text_hidden_size: int | None = None
    bias: bool = False

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> "ProjectorConfig":
        data = _copy_dict(params)
        known = {"hidden_size", "text_hidden_size", "bias"}
        cfg = cls(
            hidden_size=data.get("hidden_size"),
            text_hidden_size=data.get("text_hidden_size"),
            bias=bool(data.get("bias", False)),
        )
        _preserve_extra(cfg, data, known)
        return cfg


@dataclass
class ModelConfig:
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    projector_config: ProjectorConfig = field(default_factory=ProjectorConfig)
    model_type: str = "step3p7"
    image_token_id: int = 151679
    eos_token_id: int | list[int] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        data = _copy_dict(params)
        text_params = _copy_dict(data.get("text_config"))
        vision_params = _copy_dict(data.get("vision_config"))
        projector_params = _copy_dict(data.get("projector_config"))
        if not projector_params and vision_params and text_params:
            vision_width = vision_params.get("width") or vision_params.get("hidden_size")
            text_hidden_size = text_params.get("hidden_size")
            if vision_width is not None and text_hidden_size is not None:
                projector_params = {
                    "hidden_size": int(vision_width) * 4,
                    "text_hidden_size": int(text_hidden_size),
                    "bias": bool(data.get("projector_bias", False)),
                }

        quantization = data.get("quantization")
        if isinstance(quantization, dict):
            for key, value in list(quantization.items()):
                if not isinstance(value, dict):
                    continue
                if key.startswith(("model.", "lm_head")) and not key.startswith(
                    "language_model."
                ):
                    quantization.setdefault(f"language_model.{key}", value)

        known = {
            "text_config",
            "vision_config",
            "projector_config",
            "model_type",
            "image_token_id",
            "eos_token_id",
            "projector_bias",
            "quantization",
        }
        cfg = cls(
            text_config=TextConfig.from_dict(text_params),
            vision_config=VisionConfig.from_dict(vision_params),
            projector_config=ProjectorConfig.from_dict(projector_params),
            model_type=str(data.get("model_type") or "step3p7"),
            image_token_id=int(data.get("image_token_id", 151679)),
            eos_token_id=data.get("eos_token_id"),
        )
        _preserve_extra(cfg, data, known)
        return cfg


@dataclass
class ModelArgs(step3p5.ModelArgs):
    """Step3.7 text backbone args loaded from nested ``text_config``."""

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelArgs":
        data = _copy_dict(params)
        text_config = _copy_dict(data.get("text_config")) or data
        text_config["model_type"] = "step3p5"
        return super().from_dict(text_config)


class LanguageModel(step3p5.Model):
    """Step3.7 language tower backed by the existing Step3p5 MLX runtime."""

    def _forward_with_input_embeddings(
        self,
        inputs_embeds: mx.array,
        cache: Any = None,
    ) -> mx.array:
        x = inputs_embeds
        if x.ndim == 2:
            x = mx.expand_dims(x, 0)
        elif x.ndim != 3:
            raise ValueError(
                "Step3.7 input embeddings must be shape [L, H] or [B, L, H]."
            )

        if cache is None:
            cache = [None] * self.model.num_layers

        full_mask = None
        swa_mask = None
        if self.model._full_idx is not None:
            full_mask = create_attention_mask(x, cache[self.model._full_idx])
        if self.model._swa_idx is not None:
            swa_mask = create_attention_mask(
                x,
                cache[self.model._swa_idx],
                window_size=self.args.sliding_window,
            )

        for layer, c in zip(self.model.layers, cache):
            mask = swa_mask if layer.is_sliding else full_mask
            x = layer(x, mask=mask, cache=c)

        x = self.model.norm(x)
        return self.lm_head(x)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array | None = None,
        cache: Any = None,
        **_kwargs: Any,
    ):
        if inputs_embeds is None:
            logits = super().__call__(input_ids, cache=cache)
        else:
            logits = self._forward_with_input_embeddings(inputs_embeds, cache=cache)
        return SimpleNamespace(
            logits=logits,
            cross_attention_states=None,
            encoder_outputs=None,
        )


def _rotate_half(x: mx.array) -> mx.array:
    x = mx.reshape(x, (*x.shape[:-1], -1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    return mx.reshape(mx.stack([-x2, x1], axis=-1), (*x.shape[:-2], -1))


def _vision_activation(name: str):
    if name in {"gelu", "gelu_pytorch_tanh"}:
        return nn.gelu
    if name in {"quick_gelu", "quickgelu"}:
        return lambda x: x * mx.sigmoid(1.702 * x)
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "relu":
        return nn.relu
    raise ValueError(f"Unsupported Step3.7 vision activation: {name}")


class EncoderRope2D(nn.Module):
    """Step3.7 2D rotary embedding for vision self-attention."""

    def __init__(
        self,
        dim: int,
        use_cls_token: bool = False,
        theta: int | float = 10000,
        theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.use_cls_token = bool(use_cls_token)
        self.theta = float(theta) * float(theta_rescale_factor) ** (
            self.dim / max(self.dim - 2, 1)
        )

    def _compute_inv_freq(self, dim: int) -> mx.array:
        return 1.0 / (
            self.theta ** (mx.arange(0, dim, 2).astype(mx.float32) / max(dim, 1))
        )

    def _compute_freqs(self, positions: mx.array, inv_freq: mx.array) -> mx.array:
        freqs = mx.expand_dims(positions.astype(mx.float32), axis=-1) * inv_freq
        return mx.repeat(freqs, 2, axis=-1)

    def _freqs_for_grid(self, grid_hw: tuple[int, int]) -> mx.array:
        grid_h, grid_w = grid_hw
        half_dim = self.dim // 2
        inv_freq = self._compute_inv_freq(half_dim)
        rows = mx.arange(grid_h)
        cols = mx.arange(grid_w)
        freqs_h = mx.broadcast_to(
            mx.expand_dims(self._compute_freqs(rows, inv_freq), axis=1),
            (grid_h, grid_w, half_dim),
        )
        freqs_w = mx.broadcast_to(
            mx.expand_dims(self._compute_freqs(cols, inv_freq), axis=0),
            (grid_h, grid_w, half_dim),
        )
        freqs = mx.reshape(mx.concatenate([freqs_w, freqs_h], axis=-1), (-1, self.dim))
        if self.use_cls_token:
            freqs = mx.concatenate([mx.zeros((1, self.dim)), freqs], axis=0)
        return mx.reshape(freqs, (1, 1, freqs.shape[0], self.dim))

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        grid_hw: tuple[int, int],
    ) -> tuple[mx.array, mx.array]:
        freqs = self._freqs_for_grid(grid_hw)
        q_dtype = q.dtype
        k_dtype = k.dtype
        rot_dim = freqs.shape[-1]
        if rot_dim > q.shape[-1]:
            raise ValueError(
                "Step3.7 vision RoPE dimension exceeds attention head dim: "
                f"rot_dim={rot_dim} head_dim={q.shape[-1]}."
            )
        q_left, q_mid, q_right = q[..., :0], q[..., :rot_dim], q[..., rot_dim:]
        k_left, k_mid, k_right = k[..., :0], k[..., :rot_dim], k[..., rot_dim:]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        q_mid = q_mid * cos + _rotate_half(q_mid) * sin
        k_mid = k_mid * cos + _rotate_half(k_mid) * sin
        q = mx.concatenate([q_left, q_mid, q_right], axis=-1).astype(q_dtype)
        k = mx.concatenate([k_left, k_mid, k_right], axis=-1).astype(k_dtype)
        return q, k


class EncoderLayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float | None):
        super().__init__()
        self.gamma = (
            mx.ones((dim,)) if init_value is None else mx.full((dim,), init_value)
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states * self.gamma


class EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = _vision_activation(hidden_act)
        self.c_proj = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.c_proj(self.act_fn(self.c_fc(hidden_states)))


class EncoderVisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: int | float = 10000,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "Step3.7 vision hidden_size must be divisible by heads: "
                f"hidden_size={hidden_size} heads={num_heads}."
            )
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.rope = (
            EncoderRope2D(
                self.head_dim,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                theta_rescale_factor=rope_theta_rescale_factor,
            )
            if use_rope2d
            else None
        )

    def __call__(self, hidden_states: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        batch, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = mx.transpose(
            mx.reshape(q, (batch, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )
        k = mx.transpose(
            mx.reshape(k, (batch, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )
        v = mx.transpose(
            mx.reshape(v, (batch, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw)
        attn_output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
        )
        attn_output = mx.reshape(
            mx.transpose(attn_output, (0, 2, 1, 3)),
            (batch, seq_len, hidden_size),
        )
        return self.out_proj(attn_output)


class EncoderVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float | None = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: int | float = 10000,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.attn = EncoderVisionAttention(
            hidden_size,
            num_heads,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
            rope_theta=rope_theta,
            rope_theta_rescale_factor=rope_theta_rescale_factor,
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = EncoderMLP(hidden_size, int(hidden_size * mlp_ratio), hidden_act)
        self.ls_1 = EncoderLayerScale(hidden_size, ls_init_value)
        self.ls_2 = EncoderLayerScale(hidden_size, ls_init_value)

    def __call__(self, hidden_states: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        hidden_states = hidden_states + self.ls_1(
            self.attn(self.ln_1(hidden_states), grid_hw)
        )
        hidden_states = hidden_states + self.ls_2(self.mlp(self.ln_2(hidden_states)))
        return hidden_states


class EncoderVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float | None = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: int | float = 10000,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.resblocks = [
            EncoderVisionBlock(
                hidden_size,
                heads,
                mlp_ratio,
                hidden_act,
                layer_norm_eps,
                ls_init_value=ls_init_value,
                use_cls_token=use_cls_token,
                use_rope2d=use_rope2d,
                rope_theta=rope_theta,
                rope_theta_rescale_factor=rope_theta_rescale_factor,
            )
            for _ in range(layers)
        ]

    def __call__(self, hidden_states: mx.array, grid_hw: tuple[int, int]) -> mx.array:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw)
        return hidden_states


class VisionModel(nn.Module):
    """Step3.7 vision tower surface.

    This source-owned path implements the local MLX patch embedding and vision
    transformer surface. Live image generation remains release-blocked until it
    is proven through the packaged Electron app on a real Step3.7 image prompt.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size or config.width
        if hidden_size is None:
            raise ValueError("Step3.7 vision config requires hidden_size or width.")
        self.hidden_size = int(hidden_size)
        self.num_channels = int(getattr(config, "num_channels", 3) or 3)
        self.patch_size = int(config.patch_size or 14)
        self.image_size = int(config.image_size or 728)
        self.use_cls_token = bool(getattr(config, "use_cls_token", False))
        self.use_abs_posemb = bool(getattr(config, "use_abs_posemb", True))
        self.use_ln_pre = bool(getattr(config, "use_ln_pre", False))
        self.use_ln_post = bool(getattr(config, "use_ln_post", True))
        self.num_hidden_layers = int(config.layers or 0)
        self.num_heads = int(config.heads or 1)
        self.mlp_ratio = float(getattr(config, "mlp_ratio", 8960 / 1536) or 1.0)
        self.hidden_act = str(getattr(config, "hidden_act", "gelu") or "gelu")
        self.ls_init_value = getattr(config, "ls_init_value", None)
        self.layer_norm_eps = float(getattr(config, "layer_norm_eps", 1e-5) or 1e-5)
        self.conv1 = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.ln_pre = (
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            if self.use_ln_pre
            else None
        )
        self.ln_post = (
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            if self.use_ln_post
            else None
        )
        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)
        self.class_embedding = (
            mx.random.normal((self.hidden_size,)) * (self.hidden_size**-0.5)
            if self.use_cls_token
            else None
        )
        self.posemb_grid_size = grid_size
        self.positional_embedding = (
            mx.random.normal(
                (int(self.use_cls_token) + grid_size * grid_size, self.hidden_size)
            )
            * (self.hidden_size**-0.5)
            if self.use_abs_posemb
            else None
        )
        self.transformer = EncoderVisionTransformer(
            self.hidden_size,
            self.num_hidden_layers,
            self.num_heads,
            self.mlp_ratio,
            self.hidden_act,
            self.layer_norm_eps,
            ls_init_value=self.ls_init_value,
            use_cls_token=self.use_cls_token,
            use_rope2d=bool(getattr(config, "use_rope2d", True)),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_theta_rescale_factor=getattr(config, "rope_theta_rescale_factor", 1.0),
        )
        self.vit_downsampler1 = nn.Conv2d(
            self.hidden_size,
            self.hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2,
            self.hidden_size * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def _pixels_to_nhwc(self, pixel_values: mx.array) -> mx.array:
        if not hasattr(pixel_values, "ndim"):
            raise ValueError(
                "Step3.7 pixel_values must be an MLX array with rank-4 NCHW "
                "or NHWC shape."
            )
        if pixel_values.ndim != 4:
            raise ValueError(
                "Step3.7 pixel_values must be rank-4 NCHW or NHWC before "
                "patch embedding."
            )
        if int(pixel_values.shape[1]) == self.num_channels:
            return mx.transpose(pixel_values, (0, 2, 3, 1))
        if int(pixel_values.shape[-1]) == self.num_channels:
            return pixel_values
        raise ValueError(
            "Step3.7 pixel_values channel dimension does not match vision "
            f"num_channels={self.num_channels}: shape={pixel_values.shape}."
        )

    def sample_abs_posemb(self, grid_h: int, grid_w: int) -> mx.array:
        if self.positional_embedding is None:
            raise ValueError("Step3.7 absolute position embedding is disabled.")
        if (grid_h, grid_w) == (self.posemb_grid_size, self.posemb_grid_size):
            return mx.expand_dims(self.positional_embedding, axis=0)

        pos_embed = self.positional_embedding
        cls_token_embed = None
        if self.use_cls_token:
            cls_token_embed = pos_embed[:1]
            pos_embed = pos_embed[1:]

        pos_embed = mx.reshape(
            pos_embed,
            (1, self.posemb_grid_size, self.posemb_grid_size, self.hidden_size),
        )
        pos_embed = nn.Upsample(
            scale_factor=(
                grid_h / self.posemb_grid_size,
                grid_w / self.posemb_grid_size,
            ),
            mode="linear",
            align_corners=False,
        )(pos_embed)
        pos_embed = mx.reshape(pos_embed, (grid_h * grid_w, self.hidden_size))

        if cls_token_embed is not None:
            pos_embed = mx.concatenate([cls_token_embed, pos_embed], axis=0)
        return mx.expand_dims(pos_embed, axis=0)

    def patch_embed(self, pixel_values: mx.array) -> tuple[mx.array, tuple[int, int]]:
        pixel_values = self._pixels_to_nhwc(pixel_values)
        hidden_state = self.conv1(pixel_values)
        batch, grid_h, grid_w, channels = hidden_state.shape
        hidden_state = mx.reshape(hidden_state, (batch, grid_h * grid_w, channels))
        if self.class_embedding is not None:
            cls_token = mx.broadcast_to(
                mx.reshape(self.class_embedding, (1, 1, self.hidden_size)),
                (batch, 1, self.hidden_size),
            )
            hidden_state = mx.concatenate([cls_token, hidden_state], axis=1)
        if self.positional_embedding is not None:
            hidden_state = hidden_state + self.sample_abs_posemb(grid_h, grid_w)
        if self.ln_pre is not None:
            hidden_state = self.ln_pre(hidden_state)
        return hidden_state, (int(grid_h), int(grid_w))

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden_state, grid_hw = self.patch_embed(pixel_values)
        hidden_state = self.transformer(hidden_state, grid_hw)
        if self.ln_post is not None:
            hidden_state = self.ln_post(hidden_state)
        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]
        return hidden_state


class Step3p7Projector(nn.Module):
    """Step3.7 visual-token projector.

    The HF reference constructs ``vit_large_projector`` as a linear map from
    the downsampled vision width (``vision_config.width * 4``) into the text
    hidden size. This module implements that MLX runtime component while the
    larger vision encoder and live image path remain open.
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        if config.hidden_size is None or config.text_hidden_size is None:
            raise ValueError(
                "Step3.7 projector requires hidden_size and text_hidden_size."
            )
        self.proj = nn.Linear(
            config.hidden_size,
            config.text_hidden_size,
            bias=config.bias,
        )

    def __call__(self, image_features: mx.array) -> mx.array:
        return self.proj(image_features)


class Model(nn.Module):
    """Source-owned Step3.7 VLM module shell with real weight routing.

    This does not claim live VLM generation support. It prevents the previous
    text-only bridge behavior where vision/projector tensors were dropped
    before a real port could use or verify them.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(ModelArgs.from_dict({
            "text_config": vars(config.text_config)
        }))
        self.vision_model = VisionModel(config.vision_config)
        self.vit_large_projector = Step3p7Projector(config.projector_config)

    def _flatten_image_embeds(self, image_embeds: mx.array) -> mx.array:
        if image_embeds.ndim == 2:
            return image_embeds
        if image_embeds.ndim >= 3:
            return mx.reshape(image_embeds, (-1, image_embeds.shape[-1]))
        raise ValueError(f"Unexpected shape for image_embeds: {image_embeds.shape}")

    def _parse_and_validate_image_input(self, **kwargs: Any) -> dict[str, Any] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        patch_pixel_values = kwargs.pop("patch_pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "patch_pixel_values": patch_pixel_values,
                "num_patches": num_patches,
            }
        return {
            "type": "image_embeds",
            "image_embeds": self._flatten_image_embeds(image_embeds),
        }

    def _process_image_features(self, image_features: mx.array) -> mx.array:
        if image_features.ndim != 3:
            raise ValueError(
                "Step3.7 image features must be [batch, tokens, channels] before "
                "vision downsampling/projector."
            )
        batch, patches, channels = image_features.shape
        hw = int(patches**0.5)
        if hw * hw != patches:
            raise ValueError(
                f"Step3.7 image feature token count must be square, got {patches}."
            )
        if int(channels) != int(self.vision_model.hidden_size):
            raise ValueError(
                "Step3.7 image feature channel count does not match vision "
                f"hidden size: channels={channels} hidden_size="
                f"{self.vision_model.hidden_size}."
            )
        image_features = mx.reshape(image_features, (batch, hw, hw, channels))
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)
        batch, height, width, channels = image_features.shape
        image_features = mx.reshape(image_features, (batch, height * width, channels))
        return self.vit_large_projector(image_features)

    def _process_image_input(self, image_input: dict[str, Any]) -> list[mx.array]:
        if image_input["type"] == "image_embeds":
            return [image_input["image_embeds"]]

        image_features = self.vision_model(image_input["pixel_values"])
        patch_pixel_values = image_input.get("patch_pixel_values")
        patch_image_features = (
            self.vision_model(patch_pixel_values)
            if patch_pixel_values is not None
            else None
        )
        num_patches = image_input.get("num_patches") or []

        image_features = self._process_image_features(image_features)
        patch_image_features = (
            self._process_image_features(patch_image_features)
            if patch_image_features is not None
            else None
        )

        merged_image_features: list[mx.array] = []
        current_patch_index = 0
        for image_index, num_patch in enumerate(num_patches):
            current_features: list[mx.array] = []
            if num_patch > 0:
                if patch_image_features is None:
                    raise ValueError("Step3.7 num_patches requires patch_pixel_values.")
                patch_slice = patch_image_features[
                    current_patch_index : current_patch_index + num_patch
                ]
                current_features.append(
                    mx.reshape(patch_slice, (-1, patch_slice.shape[-1]))
                )
            current_features.append(
                mx.reshape(image_features[image_index], (-1, image_features.shape[-1]))
            )
            current_patch_index += int(num_patch)
            merged_image_features.append(mx.concatenate(current_features, axis=0))

        return merged_image_features

    def get_multimodal_embeddings(self, **kwargs: Any) -> list[mx.array] | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)

    def merge_multimodal_embeddings(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        multimodal_embeddings: list[mx.array],
    ) -> mx.array:
        token_ids = input_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("Step3.7 multimodal merge supports batch size 1.")
            token_ids = token_ids[0]
            inputs_embeds = inputs_embeds[0]

        image_positions = [
            index
            for index, token_id in enumerate(token_ids)
            if int(token_id) == self.config.image_token_id
        ]
        total_image_tokens = sum(int(embeds.shape[0]) for embeds in multimodal_embeddings)
        if len(image_positions) != total_image_tokens:
            raise ValueError(
                "Step3.7 image placeholder count does not match multimodal "
                f"embeddings: placeholders={len(image_positions)} "
                f"embeddings={total_image_tokens}."
            )

        pieces: list[mx.array] = []
        cursor = 0
        embed_cursor = 0
        flat_multimodal = (
            mx.concatenate(multimodal_embeddings, axis=0)
            if len(multimodal_embeddings) > 1
            else multimodal_embeddings[0]
        )
        for position in image_positions:
            if position > cursor:
                pieces.append(inputs_embeds[cursor:position])
            pieces.append(flat_multimodal[embed_cursor : embed_cursor + 1])
            cursor = position + 1
            embed_cursor += 1
        if cursor < inputs_embeds.shape[0]:
            pieces.append(inputs_embeds[cursor:])

        merged = mx.concatenate(pieces, axis=0) if pieces else inputs_embeds
        if input_ids.ndim == 2:
            return mx.expand_dims(merged, axis=0)
        return merged

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Any = None,
        multimodal_embeddings: list[mx.array] | None = None,
        **kwargs: Any,
    ) -> mx.array | Any:
        return_features = bool(kwargs) or (
            pixel_values is not None and not isinstance(pixel_values, list)
        )
        if isinstance(pixel_values, list) and multimodal_embeddings is None:
            multimodal_embeddings = pixel_values
            pixel_values = None
        elif pixel_values is not None:
            multimodal_embeddings = self.get_multimodal_embeddings(
                pixel_values=pixel_values,
                patch_pixel_values=kwargs.get("patch_pixel_values"),
                num_patches=kwargs.get("num_patches"),
                image_embeds=kwargs.get("image_embeds"),
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if multimodal_embeddings is None:
            merged = inputs_embeds
        else:
            merged = self.merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
            )
        if return_features:
            from mlx_vlm.models.base import InputEmbeddingsFeatures

            return InputEmbeddingsFeatures(inputs_embeds=merged)
        return merged

    def __call__(
        self,
        input_ids: mx.array,
        **kwargs: Any,
    ):
        cache = kwargs.pop("cache", None)
        multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
        if multimodal_embeddings is None:
            output = self.language_model(input_ids, cache=cache)
            return output.logits if hasattr(output, "logits") else output

        output = self.language_model(
            input_ids,
            inputs_embeds=self.get_input_embeddings(
                input_ids,
                multimodal_embeddings,
            ),
            cache=cache,
            **kwargs,
        )
        return output.logits if hasattr(output, "logits") else output

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        language_weights: dict[str, Any] = {}
        sanitized: dict[str, Any] = {}
        raw_language_had_vanilla_moe = False

        for key, value in weights.items():
            if key.startswith("model.language_model."):
                language_key = "model." + key[len("model.language_model."):]
                raw_language_had_vanilla_moe = raw_language_had_vanilla_moe or (
                    (".moe." in language_key or ".share_expert." in language_key)
                    and ".mlp." not in language_key
                )
                language_weights[language_key] = value
            elif key.startswith("language_model.model."):
                language_key = key[len("language_model."):]
                raw_language_had_vanilla_moe = raw_language_had_vanilla_moe or (
                    (".moe." in language_key or ".share_expert." in language_key)
                    and ".mlp." not in language_key
                )
                language_weights[language_key] = value
            elif key.startswith("model.vision_model."):
                vision_key = "vision_model." + key[len("model.vision_model."):]
                sanitized[self._sanitize_vision_weight_key(vision_key)] = (
                    self._sanitize_vision_weight_value(vision_key, value)
                )
            elif key.startswith("model.vit_large_projector."):
                projector_key = (
                    "vit_large_projector."
                    + key[len("model.vit_large_projector."):]
                )
                sanitized[self._sanitize_projector_weight_key(projector_key)] = value
            elif key.startswith(("vision_model.", "vit_large_projector.")):
                if key.startswith("vit_large_projector."):
                    sanitized[self._sanitize_projector_weight_key(key)] = value
                else:
                    sanitized[self._sanitize_vision_weight_key(key)] = (
                        self._sanitize_vision_weight_value(key, value)
                    )
            elif key.startswith("lm_head."):
                sanitized["language_model." + key] = value
            else:
                language_weights[key] = value

        language_weights = self.language_model.sanitize(language_weights)
        if not raw_language_had_vanilla_moe:
            norm_suffixes = (
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
                "model.norm.weight",
            )
            language_weights = {
                key: value + 1.0
                if any(key.endswith(suffix) for suffix in norm_suffixes)
                else value
                for key, value in language_weights.items()
            }
        for key, value in language_weights.items():
            sanitized["language_model." + key] = value
        return sanitized

    def _sanitize_vision_weight_key(self, key: str) -> str:
        if ".attn.in_proj_weight" in key:
            return key.replace(".attn.in_proj_weight", ".attn.qkv.weight")
        if ".attn.in_proj_bias" in key:
            return key.replace(".attn.in_proj_bias", ".attn.qkv.bias")
        return key

    def _sanitize_projector_weight_key(self, key: str) -> str:
        if key in {"vit_large_projector.weight", "vit_large_projector.bias"}:
            return key.replace("vit_large_projector.", "vit_large_projector.proj.")
        return key

    def _sanitize_vision_weight_value(self, key: str, value: Any) -> Any:
        if key.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 4:
            expected_in = None
            if key.endswith("vision_model.conv1.weight"):
                expected_in = self.config.vision_config.num_channels
            elif key.endswith("vision_model.vit_downsampler1.weight"):
                expected_in = self.vision_model.hidden_size
            elif key.endswith("vision_model.vit_downsampler2.weight"):
                expected_in = self.vision_model.hidden_size * 2
            if expected_in is not None and int(value.shape[1]) == int(expected_in):
                return mx.transpose(value, (0, 2, 3, 1))
        return value


class ImagePatcher:
    """No-heavy Step3.7 image patch planning ported from the HF reference."""

    max_image_size: int = 3024

    def determine_window_size(self, long_side: int, short_side: int) -> int:
        if long_side <= 728:
            return short_side if long_side / short_side > 1.5 else 0
        return min(short_side, 504) if long_side / short_side > 4 else 504

    def get_image_size_for_padding(
        self,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(
        self,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int]:
        if max(img_height, img_width) > self.max_image_size:
            scale_factor = self.max_image_size / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
        return img_width, img_height

    def get_image_size_for_crop(
        self,
        img_width: int,
        img_height: int,
        window_size: int,
    ) -> tuple[int, int]:
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio

        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio

        return int(width_new), int(height_new)

    def patch_crop(self, image: Any, i: int, j: int, th: int, tw: int) -> Any:
        return image.crop((j, i, j + tw, i + th))

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        windows: list[tuple[int, int, int, int]] = []
        x_num = 1
        y_num = 1
        for size, step in zip(sizes, steps, strict=True):
            size_w, size_h = size
            step_w, step_h = step

            x_num = 1 if width <= size_w else int((width - size_w + step_w - 1) // step_w + 1)
            y_num = 1 if height <= size_h else int((height - size_h + step_h - 1) // step_h + 1)
            x_start = [step_w * i for i in range(x_num)]
            y_start = [step_h * i for i in range(y_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            for y in y_start:
                for x in x_start:
                    windows.append((int(x), int(y), int(size_w), int(size_h)))
        return windows, (x_num, y_num)

    def square_pad(self, image: Any) -> Any:
        width, height = image.size
        if width == height:
            return image
        from PIL import Image

        size = max(width, height)
        padded = Image.new(image.mode, (size, size), 0)
        padded.paste(image, (0, 0))
        return padded

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(
            img_width,
            img_height,
        )
        img_width, img_height = self.get_image_size_for_preprocess(
            img_width,
            img_height,
        )
        window_size = self.determine_window_size(
            max(img_height, img_width),
            min(img_height, img_width),
        )
        if window_size == 0:
            return 0, 0

        img_width, img_height = self.get_image_size_for_crop(
            img_width,
            img_height,
            window_size,
        )
        centers, (x_num, _y_num) = self.slide_window(
            img_width,
            img_height,
            [(window_size, window_size)],
            [(window_size, window_size)],
        )
        full_rows = (len(centers) - 1) // x_num + 1
        if centers and len(centers) % x_num == 0:
            full_rows -= 1
        return len(centers), full_rows

    def newline_mask_for_patches(
        self,
        num_patches: int,
        newline_count: int,
    ) -> list[bool] | None:
        if num_patches <= 0:
            return None
        if newline_count <= 0:
            return [False] * num_patches
        stride = max(1, num_patches // (newline_count + 1))
        mask = [False] * num_patches
        for row_end in range(stride - 1, num_patches - 1, stride):
            mask[row_end] = True
        return mask

    def __call__(self, image: Any) -> tuple[Any, list[Any], list[bool] | None]:
        img_width, img_height = image.size
        new_width, new_height = self.get_image_size_for_padding(img_width, img_height)
        if (new_width, new_height) != (img_width, img_height):
            image = self.square_pad(image)
            img_width, img_height = image.size

        new_width, new_height = self.get_image_size_for_preprocess(
            img_width,
            img_height,
        )
        if (new_width, new_height) != (img_width, img_height):
            from PIL import Image

            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        window_size = self.determine_window_size(
            max(new_height, new_width),
            min(new_height, new_width),
        )
        if window_size == 0:
            return image, [], None

        crop_width, crop_height = self.get_image_size_for_crop(
            new_width,
            new_height,
            window_size,
        )
        if (crop_width, crop_height) != (new_width, new_height):
            from PIL import Image

            image_for_crop = image.resize(
                (crop_width, crop_height),
                Image.Resampling.BILINEAR,
            )
        else:
            image_for_crop = image

        centers, (x_num, _y_num) = self.slide_window(
            crop_width,
            crop_height,
            [(window_size, window_size)],
            [(window_size, window_size)],
        )
        patches = []
        newlines = []
        for patch_id, (x, y, patch_w, patch_h) in enumerate(centers):
            patches.append(self.patch_crop(image_for_crop, y, x, patch_h, patch_w))
            if (patch_id + 1) % x_num == 0:
                newlines.append(patch_id)
        if newlines and newlines[-1] == len(patches) - 1:
            newlines.pop()

        return (
            image,
            patches,
            [index in newlines for index in range(len(patches))]
            if patches
            else None,
        )


class Step3VisionProcessor:
    """Torch-free Step3.7 image normalization surface."""

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(
        self,
        size: int,
        interpolation_mode: str = "bilinear",
        patch_size: int | None = None,
    ):
        self.size = int(size)
        self.patch_size = int(patch_size if patch_size is not None else size)
        self.interpolation_mode = interpolation_mode

    @property
    def _resample(self) -> Any:
        from PIL import Image

        if self.interpolation_mode == "bicubic":
            return Image.Resampling.BICUBIC
        return Image.Resampling.BILINEAR

    def _to_pil_image(self, image: Any) -> Any:
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported Step3.7 image input type: {type(image).__name__}")

    def __call__(self, image: Any, *, is_patch: bool = False) -> dict[str, mx.array]:
        size = self.patch_size if is_patch else self.size
        pil_image = self._to_pil_image(image)
        pil_image = pil_image.resize((size, size), self._resample)
        array = np.asarray(pil_image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        array = np.transpose(array, (2, 0, 1))
        return {"pixel_values": mx.expand_dims(mx.array(array), axis=0)}


class Step3VLProcessor:
    """Step3.7 processor surface without torch."""

    image_size: int = 728
    patch_size: int = 504
    num_image_feature_size: int = 169
    num_patch_feature_size: int = 81
    image_token: str = "<im_patch>"

    def __init__(self, tokenizer: Any = None, chat_template: str | None = None, **kwargs: Any):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.patcher = ImagePatcher()
        self.image_preprocessor = Step3VisionProcessor(
            self.image_size,
            "bilinear",
            self.patch_size,
        )
        _preserve_extra(self, kwargs, set())

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                "Step3VLProcessor requires a tokenizer with apply_chat_template()."
            )
        if self.chat_template is not None and "chat_template" not in kwargs:
            kwargs["chat_template"] = self.chat_template
        return self.tokenizer.apply_chat_template(messages, *args, **kwargs)

    @property
    def image_token_id(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Step3VLProcessor requires a tokenizer for token ids.")
        return self.tokenizer.get_vocab()[self.image_token]

    @property
    def image_feature_placeholder(self) -> str:
        return self.image_token * self.num_image_feature_size

    @property
    def patch_feature_placeholder(self) -> str:
        return self.image_token * self.num_patch_feature_size

    def _token_id(self, token: str) -> int:
        if self.tokenizer is None:
            raise ValueError("Step3VLProcessor requires a tokenizer for token ids.")
        return int(self.tokenizer.convert_tokens_to_ids(token))

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.patcher.get_num_patches(
            img_width,
            img_height,
        )
        return (
            num_patches * (self.num_patch_feature_size + 2)
            + self.num_image_feature_size
            + 2
            + num_newlines
        )

    def _get_patch_replacement(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> tuple[str, list[int]]:
        if patch_newline_mask is None:
            patch_newline_mask = [False] * num_patches
        if len(patch_newline_mask) != num_patches:
            raise ValueError("patch_newline_mask length must match num_patches.")

        text = ""
        token_ids: list[int] = []
        for index in range(num_patches):
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            token_ids.extend(
                [self._token_id("<patch_start>")]
                + [self.image_token_id] * self.num_patch_feature_size
                + [self._token_id("<patch_end>")]
            )
            if patch_newline_mask[index]:
                text += "<patch_newline>"
                token_ids.append(self._token_id("<patch_newline>"))
        return text, token_ids

    def _get_image_replacement(self) -> tuple[str, list[int]]:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        token_ids = (
            [self._token_id("<im_start>")]
            + [self.image_token_id] * self.num_image_feature_size
            + [self._token_id("<im_end>")]
        )
        return text, token_ids

    def build_image_replacement(
        self,
        *,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> tuple[str, list[int]]:
        patch_text, patch_token_ids = self._get_patch_replacement(
            num_patches,
            patch_newline_mask,
        )
        image_text, image_token_ids = self._get_image_replacement()
        return patch_text + image_text, patch_token_ids + image_token_ids

    def replace_placeholder(
        self,
        text: str,
        placeholder: str,
        repls: list[str],
    ) -> str:
        parts = text.split(placeholder)
        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."
            )

        result = [parts[0]]
        for index, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[index + 1])
        return "".join(result)

    def replace_image_placeholders(
        self,
        text: str,
        images: list[dict[str, Any]],
    ) -> str:
        replacements = [
            self.build_image_replacement(
                num_patches=int(image.get("num_patches") or 0),
                patch_newline_mask=image.get("patch_newline_mask"),
            )[0]
            for image in images
        ]
        return self.replace_placeholder(text, self.image_token, replacements)

    def _split_images(self, images: list[Any]) -> list[tuple[Any, list[Any], list[bool] | None]]:
        return [
            self.patcher(self.image_preprocessor._to_pil_image(image))
            for image in images
        ]

    def _convert_images_to_pixel_values(
        self,
        images: list[Any],
        *,
        is_patch: bool = False,
    ) -> list[mx.array]:
        return [
            self.image_preprocessor(image, is_patch=is_patch)["pixel_values"]
            for image in images
        ]

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if return_tensors not in (None, "mlx", "np"):
            raise ValueError("Step3VLProcessor supports return_tensors=None, 'mlx', or 'np'.")
        text_list = [] if text is None else text if isinstance(text, list) else [text]
        image_list: list[Any]
        if images is None:
            image_list = []
        elif isinstance(images, list):
            image_list = images[0] if images and isinstance(images[0], list) else images
        else:
            image_list = [images]

        image_inputs: dict[str, Any] = {}
        if image_list:
            split_images = self._split_images(image_list)
            pixel_values = []
            patch_pixel_values = []
            patch_newline_masks = []
            num_patches = []
            replacements = []
            for raw_image, patches, patch_newline_mask in split_images:
                pixel_values.extend(self._convert_images_to_pixel_values([raw_image]))
                if patches:
                    patch_pixel_values.extend(
                        self._convert_images_to_pixel_values(
                            patches,
                            is_patch=True,
                        )
                    )
                num_patches.append(len(patches))
                replacements.append(
                    self.build_image_replacement(
                        num_patches=len(patches),
                        patch_newline_mask=patch_newline_mask,
                    )[0]
                )
                if patch_newline_mask is not None:
                    patch_newline_masks.extend(patch_newline_mask)

            image_inputs = {
                "pixel_values": mx.concatenate(pixel_values, axis=0),
                "num_patches": num_patches,
            }
            if patch_pixel_values:
                image_inputs["patch_pixel_values"] = mx.concatenate(
                    patch_pixel_values,
                    axis=0,
                )
            if patch_newline_masks:
                image_inputs["patch_newline_mask"] = mx.array(patch_newline_masks)
            if text_list:
                placeholder_count = sum(item.count(self.image_token) for item in text_list)
                if placeholder_count == 0 and replacements:
                    text_list[0] = f"{self.image_token * len(replacements)}{text_list[0]}"
                text_list = [
                    self.replace_placeholder(item, self.image_token, replacements)
                    for item in text_list
                ]

        text_inputs: dict[str, Any]
        if self.tokenizer is not None and callable(self.tokenizer):
            text_inputs = dict(self.tokenizer(text_list))
        else:
            text_inputs = {"text": text_list}
        if return_tensors == "mlx":
            for key in ("input_ids", "attention_mask"):
                value = text_inputs.get(key)
                if value is not None and not isinstance(value, mx.array):
                    text_inputs[key] = mx.array(value, dtype=mx.int32)
        elif return_tensors == "np":
            for key in ("input_ids", "attention_mask"):
                value = text_inputs.get(key)
                if value is not None and not isinstance(value, np.ndarray):
                    text_inputs[key] = np.asarray(value, dtype=np.int64)
        if return_tensors == "np":
            image_inputs = {
                key: np.array(value) if isinstance(value, mx.array) else value
                for key, value in image_inputs.items()
            }
        return {**text_inputs, **image_inputs}


__all__ = [
    "ImagePatcher",
    "LanguageModel",
    "Model",
    "ModelArgs",
    "ModelConfig",
    "ProjectorConfig",
    "Step3VisionProcessor",
    "Step3VLProcessor",
    "Step3p7Projector",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
