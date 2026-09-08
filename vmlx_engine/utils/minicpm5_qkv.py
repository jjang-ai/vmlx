# SPDX-License-Identifier: Apache-2.0
"""Qualified MiniCPM5 affine QKV fusion; no bundle or cache tensors are rewritten."""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.llama import Attention, Model

logger = logging.getLogger(__name__)
LAYOUT = "minicpm5_affine_qkv_v1"


def _compatible(attention):
    if type(attention) is not Attention:
        return False
    projs = [attention.q_proj, attention.k_proj, attention.v_proj]
    if any(type(p) is not nn.QuantizedLinear for p in projs):
        return False
    first = projs[0]
    # Deliberately qualify only the measured affine8/BF16 bundle layout.
    # JANGTQ codebooks and other bit/group layouts stay on their own runtime.
    if first.mode != "affine" or first.bits != 8 or first.group_size != 64:
        return False
    for p in projs:
        if (p.mode, p.bits, p.group_size) != (first.mode, first.bits, first.group_size):
            return False
        if "bias" in p or p.weight.dtype != mx.uint32 or p.scales.dtype != mx.bfloat16:
            return False
        if p.get("biases") is None or p.biases.dtype != mx.bfloat16:
            return False
        if p.weight.shape[1] != first.weight.shape[1]:
            return False
    return [p.weight.shape[0] for p in projs] == [
        attention.n_heads * attention.head_dim,
        attention.n_kv_heads * attention.head_dim,
        attention.n_kv_heads * attention.head_dim,
    ]


class FusedMiniCPM5Attention(nn.Module):
    """One packed affine projection, with the native RoPE/KV/attention contract."""

    def __init__(self, attention):
        super().__init__()
        if not _compatible(attention):
            raise ValueError("Unqualified MiniCPM5 QKV layout")
        projs = [attention.q_proj, attention.k_proj, attention.v_proj]
        self.n_heads = attention.n_heads
        self.n_kv_heads = attention.n_kv_heads
        self.head_dim = attention.head_dim
        self.scale = attention.scale
        self.rope = attention.rope
        self.o_proj = attention.o_proj
        self.weight = mx.concatenate([p.weight for p in projs], axis=0)
        self.scales = mx.concatenate([p.scales for p in projs], axis=0)
        self.biases = mx.concatenate([p.biases for p in projs], axis=0)
        self.freeze()

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        projected = mx.quantized_matmul(
            x, self.weight, scales=self.scales, biases=self.biases,
            transpose=True, group_size=64, bits=8, mode="affine",
        )
        q_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        q, k, v = mx.split(projected, [q_width, q_width + kv_width], axis=-1)
        q = q.reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        output = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


def prepare_minicpm5_qkv(model, model_path):
    """Enable only for the explicit native MiniCPM5 contract, never a name guess."""
    if type(model) is not Model:
        return False
    if getattr(model, "_vmlx_attention_projection_layout", None) == LAYOUT:
        return True
    path = Path(model_path) / "jang_config.json"
    if not path.is_file():
        return False
    metadata = json.loads(path.read_text())
    tools = metadata.get("tool_calling")
    if not isinstance(tools, dict) or tools.get("dialect") != "minicpm5_xml_function":
        return False
    layers = model.layers
    if not layers or not all(_compatible(layer.self_attn) for layer in layers):
        logger.info("MiniCPM5 QKV fusion skipped: unqualified projection layout; native path retained")
        return False
    # Validate every layer before changing any. Materialize each concatenation
    # on the loading worker so lazy dependencies cannot retain duplicate weights.
    for layer in layers:
        fused = FusedMiniCPM5Attention(layer.self_attn)
        mx.eval(fused.weight, fused.scales, fused.biases)
        layer.self_attn = fused
    model._vmlx_attention_projection_layout = LAYOUT
    logger.info("MiniCPM5 packed QKV fusion active: %d layers, affine8/64 BF16, cache layout=%s", len(layers), LAYOUT)
    return True
