# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for MLX from Transformers' Inkling2 implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .interpolate import resize_bilinear_nhwc


def rotate_pairs(x):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self[name] = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x, cos, sin):
        q, k, v = [
            proj(x).reshape(1, x.shape[0], self.n_heads, self.head_dim)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]

        def rope(a):
            af = a.astype(mx.float32)
            return (
                af * cos[None, :, None] + rotate_pairs(af) * sin[None, :, None]
            ).astype(a.dtype)

        q, k = rope(q), rope(k)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(x.shape))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, int(config.hidden_size * config.mlp_ratio)
        )
        self.fc2 = nn.Linear(
            int(config.hidden_size * config.mlp_ratio), config.hidden_size
        )

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.lambda_1 = mx.ones((config.hidden_size,))
        self.lambda_2 = mx.ones((config.hidden_size,))

    def __call__(self, x, cos, sin):
        x = x + self.lambda_1 * self.self_attn(self.layernorm_before(x), cos, sin)
        return x + self.lambda_2 * self.mlp(self.layernorm_after(x))


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Linear(
            config.num_channels * config.native_patch_size**2,
            config.hidden_size,
            bias=False,
        )
        self.position_embedding = mx.zeros(
            (config.posemb_grid_size**2, config.hidden_size)
        )
        self.class_embedding = mx.zeros((config.hidden_size,))
        self.cls_position_embedding = mx.zeros((config.hidden_size,))


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [
            VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.proj_in = nn.Linear(
            config.adapter_merge_size**2 * config.hidden_size,
            config.projection_hidden_dim,
            bias=config.projector_bias,
        )
        self.proj_out = nn.Linear(
            config.projection_hidden_dim,
            config.text_hidden_size,
            bias=config.projector_bias,
        )

    def __call__(self, pixel_values, image_grid_thw=None, grid_thw=None, **kwargs):
        grid = image_grid_thw if image_grid_thw is not None else grid_thw
        if grid is None:
            raise ValueError("Inkling 2 images require image_grid_thw")
        grids = np.asarray(grid).tolist()
        if any(t != 1 or h < 1 or w < 1 for t, h, w in grids):
            raise ValueError(
                "Inkling 2 expects positive still-image grids [1, height, width]"
            )
        if sum(h * w for _, h, w in grids) != pixel_values.shape[0]:
            raise ValueError("Image grid does not match the number of image patches")
        cfg = self.config
        pixel_values = pixel_values.astype(self.pre_layernorm.weight.dtype)
        merge, patch = cfg.adapter_merge_size, cfg.native_patch_size
        outputs, start = [], 0
        for _, rows, cols in grids:
            count = rows * cols
            pixels = pixel_values[start : start + count]
            start += count
            patches = pixels.reshape(
                count, merge, patch, merge, patch, cfg.num_channels
            ).transpose(0, 1, 3, 2, 4, 5)
            x = self.embeddings.patch_embedding(patches.reshape(count * merge**2, -1))
            pos = self.embeddings.position_embedding.reshape(
                1, cfg.posemb_grid_size, cfg.posemb_grid_size, cfg.hidden_size
            )
            pos = resize_bilinear_nhwc(
                pos.astype(mx.float32),
                (rows * merge, cols * merge),
                align_corners=False,
            )
            pos = (
                pos.reshape(rows, merge, cols, merge, cfg.hidden_size)
                .transpose(0, 2, 1, 3, 4)
                .reshape(x.shape)
            )
            x = x + pos.astype(x.dtype)
            cls = (
                self.embeddings.class_embedding + self.embeddings.cls_position_embedding
            ).astype(x.dtype)
            x = self.pre_layernorm(mx.concatenate([cls[None], x], axis=0))
            yy, xx = np.meshgrid(
                np.arange(rows * merge), np.arange(cols * merge), indexing="ij"
            )
            positions = (
                np.stack([xx, yy], axis=-1)
                .reshape(rows, merge, cols, merge, 2)
                .transpose(0, 2, 1, 3, 4)
                .reshape(-1, 2)
                + 1
            )
            positions = mx.array(
                np.concatenate([np.zeros((1, 2)), positions]), mx.float32
            )
            spatial_dim = cfg.hidden_size // cfg.num_attention_heads // 2
            inv = cfg.rope_theta ** (-mx.arange(0, spatial_dim, 2) / spatial_dim)
            angles = (positions[..., None] * inv).reshape(x.shape[0], -1)
            angles = mx.repeat(angles, 2, axis=-1)
            cos, sin = mx.cos(angles), mx.sin(angles)
            for layer in self.layers:
                x = layer(x, cos, sin)
            x = x[1:].reshape(count, merge**2 * cfg.hidden_size)
            outputs.append(self.proj_out(nn.gelu(self.proj_in(x))))
        return mx.concatenate(outputs, axis=0)
