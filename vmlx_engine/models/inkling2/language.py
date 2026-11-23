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


import math
from contextlib import contextmanager
from contextvars import ContextVar

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import LanguageModelOutput
from mlx_lm.models.cache import (
    ArraysCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from .primitives import (
    InklingShortConvolution,
    _cache_padding_mask,
    _next_conv_state,
    banded_additive_mask,
)
from .switch_layers import SwitchGLU
from .config import TextConfig, validate_sharding


_VERIFY_WITH_DECODE_GEMV = ContextVar("inkling_verify_with_decode_gemv", default=False)

_DSPARK_FEATURE_SCOPE = ContextVar("inkling_dspark_feature_scope", default=None)


@contextmanager
def capture_dspark_features(model, taps, sink):
    active = _DSPARK_FEATURE_SCOPE.set((model, taps, sink))
    try:
        yield
    finally:
        _DSPARK_FEATURE_SCOPE.reset(active)


def _decode_gemv_rows(x, weight):
    """Run independent decode-shaped products through one indexed dispatch."""
    width = x.shape[-1]
    rows = x.size // width
    outputs = weight.shape[0]
    if rows == 1:
        return x @ weight.T
    matrix = x.reshape(rows, width)
    if width >= 16 * outputs or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        # MLX dispatches a different reduction for very asymmetric GEMV.
        result = mx.concatenate(
            [matrix[row:row + 1] @ weight.T for row in range(rows)], axis=0
        )
    else:
        result = mx.gather_mm(
            matrix[:, None, :], weight.T[None],
            rhs_indices=mx.zeros((rows,), dtype=mx.uint32),
        )
    return result.reshape(*x.shape[:-1], outputs)


def _decode_linear(module, x):
    if _VERIFY_WITH_DECODE_GEMV.get() and x.shape[1] > 1:
        if type(module) is nn.Linear:
            result = _decode_gemv_rows(x, module.weight)
            return result + module.bias if "bias" in module else result
        if type(module) is nn.QuantizedLinear and x.shape[0] == 1:
            from .jangh import affine_decode_rows, is_affine_decode_projection

            if is_affine_decode_projection(module):
                return affine_decode_rows(x, module.weight, module.scales, module.biases)
    return module(x)


def _normalize_message_bank(states):
    if _VERIFY_WITH_DECODE_GEMV.get() and states.shape[1] > 1:
        return mx.concatenate([
            _normalize_message_bank(states[:, row:row + 1])
            for row in range(states.shape[1])
        ], axis=1)
    return mx.fast.rms_norm(states.astype(mx.float32), None, 1e-6)


def _message_attention(query, bank, width):
    if _VERIFY_WITH_DECODE_GEMV.get() and bank.shape[1] > 1:
        return mx.concatenate([
            _message_attention(query, bank[:, row:row + 1], width)
            for row in range(bank.shape[1])
        ], axis=1)
    scores = (query @ bank.swapaxes(-1, -2)) * width ** -0.5
    return mx.softmax(scores, axis=-1) @ bank


@mx.compile
def l2_clip(x, threshold, scale=None):
    dtype = x.dtype
    x = x.astype(mx.float32)
    if scale is not None:
        x = x * scale.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
    return (x * mx.minimum(threshold / (norm + 1e-6), 1.0)).astype(dtype)


@mx.compile
def _softcapped_swiglu(up, gate, gate_cap, up_cap):
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    return (
        mx.tanh(gate / gate_cap)
        * gate_cap
        * mx.sigmoid(gate)
        * mx.tanh(up / up_cap)
        * up_cap
    ).astype(dtype)


@mx.compile
def _route(x, weight, bias, top_k, route_scale, exact_rows=False):
    x, weight = x.astype(mx.float32), weight.astype(mx.float32)
    projection = _decode_gemv_rows(x, weight) if exact_rows else x @ weight.T
    scores = mx.sigmoid(projection)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


@mx.compile
def _combine_routes(out, weights):
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


@mx.compile
def _attention_residual(states, query, normalize_values):
    states = states.astype(mx.float32)
    keys = mx.fast.rms_norm(states, None, 1e-6)
    scores = (keys @ query.reshape(-1, states.shape[-1]).T) * states.shape[-1] ** -0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    return weights @ (keys if normalize_values else states)


_conv_history_kernel = mx.fast.metal_kernel(
    name="inkling_masked_conv_history",
    input_names=["state", "inputs", "mask"],
    output_names=["out"],
    ensure_row_contiguous=False,
    source=r"""
    uint index = thread_position_in_grid.x;
    int B = inputs_shape[0], L = inputs_shape[1], C = inputs_shape[2];
    int W = state_shape[1];
    if (index >= B * L * W * C) return;
    int c = index % C, w = (index / C) % W;
    int t = (index / (C * W)) % L, b = index / (C * W * L);
    int count = 0, start = 0;
    for (int j = 0; j <= t; ++j) {
        if (mask[b * mask_strides[0] + j * mask_strides[1]]) {
            if (count == 0) start = j;
            ++count;
        }
    }
    int logical = count - W + w;
    out[index] = logical < 0
        ? state[b * state_strides[0] + (W + logical) * state_strides[1]
                + c * state_strides[2]]
        : inputs[b * inputs_strides[0] + (start + logical) * inputs_strides[1]
                 + c * inputs_strides[2]];
    """,
)


@mx.compile
def _conv_state_history_fallback(state, inputs, mask):
    width = state.shape[1]
    count = mx.cumsum(mask.astype(mx.int32), axis=1)
    start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(width)[None, None, :] + count[:, :, None] - width
    indices = mx.where(
        logical >= 0,
        width + start[:, None, None] + mx.maximum(logical, 0),
        width + logical,
    )
    combined = mx.concatenate([state, inputs], axis=1)
    return mx.take_along_axis(combined[:, None], indices[..., None], axis=2)


def _conv_state_history(state, inputs, mask):
    """Select every masked prefix's state without repeated prefix forwards."""
    batch, length, channels = inputs.shape
    width = state.shape[1]
    if (
        width > 0
        and length <= 8
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    ):
        return _conv_history_kernel(
            inputs=[state, inputs, mask],
            grid=(batch * length * width * channels, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, length, width, channels)],
            output_dtypes=[state.dtype],
        )[0]
    return _conv_state_history_fallback(state, inputs, mask)


def _update_conv_cache(cache, index, state, inputs, mask):
    if cache is None:
        return
    speculating = getattr(cache, "is_speculating", False)
    if speculating and mask is not None:
        # Physical prefixes can retain different numbers of valid tokens.
        # Avoid rebuilding a separate reduction, concat and gather per prefix.
        states = _conv_state_history(state, inputs, mask)
        final = states[:, -1]
        cache.record_speculative_states(
            index, states[:, :-1] if inputs.shape[1] > 1 else None, final
        )
    else:
        final = _next_conv_state(state, inputs, mask)
        if speculating:
            cache.record_speculative_window(
                index, mx.concatenate([state, inputs], axis=1), state.shape[1]
            )
    cache[index] = final


class ShortConvolution(InklingShortConvolution):
    def __call__(self, x, cache=None, mask=None):
        # Keep the convolution, residual, and scaling in FP32. The checkpoint
        # stores BF16 taps, but rounding the convolution before adding the
        # residual changes both reference parity and prefill/decode consistency.
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]))
        if (
            not self.training
            and x.dtype == mx.bfloat16
            and self.kernel_size == 4
            and x.shape[0] <= 4
            and x.shape[1] <= 8
            and x.shape[-1] % 128 == 0
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        ):
            from .kernels import short_convolution

            # Temporal selection promotes selected BF16 values into FP32 state.
            # Invalid positions never enter that state. The convolution itself
            # reads masked inputs directly, without materializing a concat.
            _update_conv_cache(cache, self.conv_idx, state, x, mask)
            return short_convolution(state, self.conv.weight, x, mask, self.kernel_size)
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        context = mx.concatenate([state, inputs], axis=1)
        _update_conv_cache(cache, self.conv_idx, state, inputs, mask)
        out = mx.conv1d(
            context, self.conv.weight.astype(mx.float32), groups=x.shape[-1]
        )
        return ((out + xf) / math.sqrt(self.kernel_size + 1)).astype(x.dtype)


class DynamicShortConvolution(nn.Module):
    def __init__(self, config, channels, num_heads, conv_idx):
        super().__init__()
        self.kernel_size = config.sconv_kernel_size
        self.num_heads = num_heads
        self.conv_idx = conv_idx
        self.static_weight = mx.zeros((self.kernel_size, channels))
        self.weight_proj = nn.Linear(
            config.hidden_size, num_heads * self.kernel_size, bias=False
        )

    def __call__(self, x, dynamic_input, cache=None, mask=None):
        b, length, channels = x.shape
        k = self.kernel_size
        xf = x.astype(mx.float32)
        inputs = xf if mask is None else mx.where(mask[..., None], xf, 0)
        state = cache[self.conv_idx] if cache is not None else None
        if state is None:
            state = mx.zeros((b, k - 1, channels), mx.float32)
        context = mx.concatenate([state, inputs], axis=1)
        _update_conv_cache(cache, self.conv_idx, state, inputs, mask)
        dynamic = (
            _decode_linear(self.weight_proj, dynamic_input)
            .astype(mx.float32)
            .reshape(b, length, self.num_heads, 1, k)
        )
        weights = self.static_weight.T.reshape(self.num_heads, -1, k) + dynamic
        out = mx.zeros_like(xf).reshape(b, length, self.num_heads, -1)
        for lag in range(k):
            values = context[:, k - 1 - lag : k - 1 - lag + length].reshape(out.shape)
            out = out + values * weights[..., lag]
        return ((out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = self.sliding or config.rel_extent
        self.sharding_group = None
        self.head_start = 0
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.r_proj = nn.Linear(
            config.hidden_size, self.n_heads * config.d_rel, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.sink_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.q_scale = mx.ones((self.n_heads, self.head_dim))
        self.rel_scale = mx.ones((self.n_heads, config.d_rel))
        self.rel_proj = mx.zeros((config.d_rel, self.rel_extent))
        for idx, name in enumerate(("k_sconv", "v_sconv")):
            channels = self.n_kv * self.head_dim
            self[name] = (
                ShortConvolution(channels, config.sconv_kernel_size, idx)
                if self.is_sliding
                else DynamicShortConvolution(config, channels, self.n_kv, idx)
            )

    def shard(self, group):
        """Split complete heads and output rows without splitting dot products."""
        if self.sharding_group is not None:
            return
        projections = ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "o_proj")
        # Preserve the dense kernels' output-row tiles. Other configurations
        # retain replicated attention while the experts are still partitioned.
        if (
            group.size() != 2
            or self.n_heads % 2
            or self.n_kv % 2
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or any(
                type(self[name]) is not nn.Linear
                or "bias" in self[name]
                or self[name].weight.shape[0] % 256
                for name in projections
            )
            or (
                not self.is_sliding
                and any(
                    type(self[name].weight_proj) is not nn.Linear
                    or "bias" in self[name].weight_proj
                    for name in ("k_sconv", "v_sconv")
                )
            )
        ):
            return
        rank = group.rank()
        for name in projections:
            projection = self[name]
            width = projection.weight.shape[0] // 2
            projection.weight = mx.contiguous(
                projection.weight[rank * width : (rank + 1) * width]
            )
        for name in ("q_scale", "rel_scale"):
            value = self[name]
            width = value.shape[0] // 2
            self[name] = mx.contiguous(value[rank * width : (rank + 1) * width])
        for name in ("k_sconv", "v_sconv"):
            conv = self[name]
            if self.is_sliding:
                width = conv.conv.weight.shape[0] // 2
                conv.conv.weight = mx.contiguous(
                    conv.conv.weight[rank * width : (rank + 1) * width]
                )
            else:
                conv.num_heads //= 2
                width = conv.static_weight.shape[1] // 2
                conv.static_weight = mx.contiguous(
                    conv.static_weight[:, rank * width : (rank + 1) * width]
                )
                width = conv.weight_proj.weight.shape[0] // 2
                conv.weight_proj.weight = mx.contiguous(
                    conv.weight_proj.weight[rank * width : (rank + 1) * width]
                )
        self.n_heads //= 2
        self.n_kv //= 2
        self.head_start = rank * self.n_heads
        self.sharding_group = group

    def _gather_features(self, x):
        if self.sharding_group is None:
            return x
        shape = x.shape
        return (
            mx.distributed.all_gather(
                x.reshape(1, -1, shape[-1]), group=self.sharding_group
            )
            .transpose(1, 0, 2)
            .reshape(*shape[:-1], shape[-1] * self.sharding_group.size())
        )

    def _project(self, projection, x):
        if self.sharding_group is None:
            return _decode_linear(projection, x)
        # Both GEMV and GEMM dispatch can depend on N, including short batches
        # on older GPUs. Preserve the original N with zero rows for the other
        # rank, then select our complete rows. No remote weights are retained.
        width = projection.weight.shape[0]
        start = self.sharding_group.rank() * width
        weight = mx.pad(projection.weight, ((start, width - start), (0, 0)))
        return (x @ weight.T)[..., start : start + width]

    def _attention_with_sink(self, q, k, v, relative, sinks, padding, rel_proj):
        size = k.shape[2]
        mask = banded_additive_mask(
            relative,
            rel_proj,
            size - q.shape[2],
            size,
            self.sliding,
            self.rel_extent,
            left_padding=padding,
            shape_ref=k,
        )
        mask = mx.concatenate([mask, sinks], axis=-1)
        # A query-dependent sink adds a logit without any value payload.
        k = mx.concatenate([k, mx.zeros((*k.shape[:2], 1, k.shape[-1]), k.dtype)], axis=2)
        v = mx.concatenate([v, mx.zeros((*v.shape[:2], 1, v.shape[-1]), v.dtype)], axis=2)
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        pack = getattr(self, "_affine_decode_pack", None)
        if (
            pack is not None and self.sharding_group is None and b == 1
            and (length == 1 or _VERIFY_WITH_DECODE_GEMV.get())
            and x.dtype == mx.bfloat16
            and mx.metal.is_available() and mx.default_device() == mx.gpu
        ):
            include_kv = key_input is x and value_input is x
            query, relative, gate, *kv_projections = pack.project(x, include_kv=include_kv)
        else:
            query = self._project(self.q_proj, x)
            relative = self._project(self.r_proj, x)
            gate = self._project(self.gate_proj, x)
            kv_projections = ()
        if kv_projections:
            k, v = kv_projections
        else:
            k = self._project(self.k_proj, key_input)
            v = self._project(self.v_proj, value_input)
        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        q = l2_clip(
            query.reshape(
                b, length, self.n_heads, self.head_dim
            ),
            cfg.qk_l2_clip_threshold,
            self.q_scale,
        )
        k = l2_clip(
            k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold
        )
        v = v.reshape(b, length, self.n_kv, self.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(
                x.dtype
            )
        r = l2_clip(
            relative.reshape(b, length, self.n_heads, cfg.d_rel),
            cfg.rel_l2_clip_threshold,
            self.rel_scale,
        )
        if not self.is_sliding and cfg.log_scaling_n_floor is not None:
            offset = kv.offset if kv is not None else 0
            positions = mx.arange(length) + mx.array(offset)[..., None] + 1
            tau = 1 + cfg.log_scaling_alpha * mx.log(
                mx.maximum(positions / cfg.log_scaling_n_floor, 1)
            )
            q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
            r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
            # Relative logits index backward distance, so ring buffers must be
            # read in chronological order (ordinary RoPE attention does not care).
            if isinstance(kv, RotatingKVCache):
                k, v = kv._temporal_order(k), kv._temporal_order(v)
            elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        padding = getattr(kv, "left_padding", None)
        rel_proj = self.rel_proj.astype(x.dtype)
        # Keep the sink projection's complete output width on each rank.
        sinks = _decode_linear(self.sink_proj, x)
        if self.sharding_group is not None:
            sinks = sinks[..., self.head_start : self.head_start + self.n_heads]
        sinks = sinks.transpose(0, 2, 1)[..., None]
        if _VERIFY_WITH_DECODE_GEMV.get() and length > 1:
            # A multi-row SDPA uses different reduction geometry and rounds
            # some BF16 coordinates differently from the serving decoder.
            # Preserve the true chronological window of each query.
            outputs = []
            prefix = k.shape[2] - length
            for row in range(length):
                end = prefix + row + 1
                start = max(0, end - self.sliding) if self.sliding else 0
                outputs.append(self._attention_with_sink(
                    q[:, :, row:row + 1], k[:, :, start:end], v[:, :, start:end],
                    r[:, row:row + 1], sinks[:, :, row:row + 1],
                    padding - start if padding is not None else None, rel_proj,
                ))
            out = mx.concatenate(outputs, axis=2)
        else:
            out = self._attention_with_sink(q, k, v, r, sinks, padding, rel_proj)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        out = (
            out.astype(mx.float32)
            * mx.sigmoid(gate.astype(mx.float32))
        ).astype(x.dtype)
        # Every output dot still consumes all heads on one rank. Gather the
        # heads first, project complete output rows, then restore hidden order.
        return self._gather_features(
            self._project(self.o_proj, self._gather_features(out))
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap

    def __call__(self, up, gate):
        return _softcapped_swiglu(up, gate, self.gate_cap, self.up_cap)


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.sharding_group = None
        self.expert_start = 0
        self.expert_end = config.n_routed_experts
        self.gate_weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.activation = SoftcappedSwiGLU(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.intermediate_size,
            config.n_routed_experts,
            activation=self.activation,
        )

    def __call__(self, x):
        indices, weights = _route(
            x,
            self.gate_weight,
            self.e_score_correction_bias,
            self.top_k,
            self.route_scale,
            _VERIFY_WITH_DECODE_GEMV.get() and x.shape[1] > 1,
        )
        if getattr(self.switch_mlp, "is_jangh", False):
            if self.sharding_group is not None:
                raise ValueError("Inkling Mini JANGH expert sharding is unsupported")
            from .jangh import softcapped_experts

            out = softcapped_experts(self.switch_mlp, x, indices, self.activation)
        elif self.sharding_group is None:
            if (
                not self.training
                and x.shape[0] <= 4
                and 1 < x.shape[1] <= 8
                and getattr(self.switch_mlp.down_proj, "mode", None) == "nvfp4"
            ):
                routes = indices.astype(mx.int32)
                if _VERIFY_WITH_DECODE_GEMV.get():
                    out = self.switch_mlp.verify_nvfp4(x, routes)
                else:
                    out = self.switch_mlp.masked(x, routes)
            else:
                out = self.switch_mlp(x, indices)
        else:
            owned = (indices >= self.expert_start) & (indices < self.expert_end)
            local = mx.where(owned, indices.astype(mx.int32) - self.expert_start, -1)
            out = self.switch_mlp.masked(x, local)
            # Exactly one rank owns each route. Reduce route outputs before
            # weighting to retain the unsharded FP32 accumulation order.
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return _combine_routes(out, weights)


class AttentionResidual(nn.Module):
    def __init__(self, config, num_queries=1):
        super().__init__()
        self.query = mx.zeros(
            (config.hidden_size,)
            if num_queries == 1
            else (num_queries, config.hidden_size)
        )
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks, partial):
        return _attention_residual(
            mx.stack([*blocks, partial], axis=-2), self.query, self.normalize_values
        )


@mx.compile
def _blend_message_gates(x, message, gates):
    forget, gain = mx.split(mx.tanh(gates), 2, axis=-1)
    return (1 - forget) * x + gain * message


class MessageController(nn.Module):
    """Blend a previous pass's message into one sublayer input."""

    def __init__(self, config):
        super().__init__()
        rank, width = config.loop_config.gate_rank, config.hidden_size
        self.down_weight = mx.zeros((rank, 2 * width))
        self.forget_weight = mx.zeros((width, rank))
        self.gain_weight = mx.zeros((width, rank))
        self.eps = config.rms_norm_eps

    def __call__(self, x, message):
        verify = _VERIFY_WITH_DECODE_GEMV.get() and x.shape[1] > 1
        packed = getattr(self, "_verify_gate_weights", None)
        if verify and (packed is None or x.shape[0] != 1):
            # The low-rank reductions are sensitive to the decode dispatch.
            return mx.concatenate([
                self(x[:, row:row + 1], message[:, row:row + 1])
                for row in range(x.shape[1])
            ], axis=1)
        pair = mx.concatenate(
            [mx.fast.rms_norm(v, None, self.eps) for v in (x, message)], axis=-1
        )
        pair = pair.astype(self.down_weight.dtype)
        if verify:
            hidden = _decode_gemv_rows(pair, self.down_weight).astype(mx.float32)
            # Indexed batched GEMV uses a different reduction for 64-wide
            # gates. Use one serial-shaped product per position, reading both
            # gates together, then blend every position in one compiled call.
            gates = mx.concatenate([
                hidden[:, row:row + 1] @ packed.T
                for row in range(hidden.shape[1])
            ], axis=1)
            return _blend_message_gates(x, message, gates)
        hidden = (pair @ self.down_weight.T).astype(mx.float32)
        forget = mx.tanh(hidden @ self.forget_weight.T)
        gain = mx.tanh(hidden @ self.gain_weight.T)
        return (1 - forget) * x + gain * message


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.attention_input = (
            AttentionResidual(config, 1 if self.is_sliding else 3)
            if layer_idx
            else None
        )
        self.mlp_input = AttentionResidual(config)
        self.self_attn = Attention(config, layer_idx)
        self.mlp = SparseMoE(config)
        names = (
            ("input_layernorm",)
            if self.is_sliding
            else ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm")
        )
        for name in (*names, "post_attention_layernorm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 2
        )
        self.mlp_sconv = ShortConvolution(
            config.hidden_size, config.sconv_kernel_size, 3
        )
        loop = config.loop_config
        if loop is not None and loop.core_start <= layer_idx < loop.core_end:
            self.controllers = {
                name: MessageController(config) for name in ("attn", "k", "v", "mlp")
            }

    def __call__(self, x, blocks, cache=None, messages=None):
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = (
            self.attention_input(blocks, x)
            if self.attention_input is not None
            else x[..., None, :]
        )
        if self.is_sliding and messages is None:
            attn = self.self_attn(
                self.input_layernorm(mixed[..., 0, :]).astype(x.dtype),
                cache=cache,
                conv_mask=mask,
            )
        else:
            parts = [mixed[..., min(i, mixed.shape[-2] - 1), :] for i in range(3)]
            if messages is not None:
                parts = [
                    self.controllers[name](part, messages[..., i, :])
                    for i, (name, part) in enumerate(zip(("attn", "k", "v"), parts))
                ]
            norms = (
                (self.input_layernorm,) * 3
                if self.is_sliding
                else (
                    self.q_input_layernorm,
                    self.k_input_layernorm,
                    self.v_input_layernorm,
                )
            )
            parts = [norm(part).astype(x.dtype) for norm, part in zip(norms, parts)]
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        if messages is not None:
            hidden = self.controllers["mlp"](hidden, messages[..., 3, :])
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(
            x.dtype
        )
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.layer_indices = config.layer_indices
        self.loop = config.loop_config
        if self.loop is not None:
            self.message_queries = mx.zeros(
                (self.loop.core_end - self.loop.core_start, 4, config.hidden_size)
            )
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        skip_final_norm=False,
        capture_layer_ids=None,
        speculative_verify=False,
    ):
        captures = {}
        if capture_layer_ids is not None and (
            len(capture_layer_ids) != len(set(capture_layer_ids))
            or any(
                type(i) is not int or not 0 <= i <= len(self.layer_indices)
                for i in capture_layer_ids
            )
        ):
            raise ValueError("Invalid Inkling 2 capture_layer_ids")
        requested = set(capture_layer_ids or [])
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        blocks = []
        bank = None
        if cache is not None and len(cache) != 2 * len(self.layer_indices):
            raise ValueError(
                "Inkling 2 requires attention and convolution caches for every layer visit"
            )
        for visit, index in enumerate(self.layer_indices):
            if self.loop is not None and index == self.loop.core_start:
                if visit == index:
                    prelude, prelude_blocks = x, blocks.copy()
                else:
                    # The first block appended in the core is the prelude partial.
                    # Only completed core blocks and the last partial supply messages.
                    states = mx.stack([*blocks[len(prelude_blocks) + 1 :], x], axis=-2)
                    bank = _normalize_message_bank(states)
                    x, blocks = prelude, prelude_blocks.copy()
            messages = None
            if bank is not None and index < self.loop.core_end:
                query = self.message_queries[index - self.loop.core_start]
                messages = _message_attention(query, bank, x.shape[-1])
            # DSpark taps the partial block entering the layer, not its output.
            if visit in requested:
                captures[visit] = x
            visit_cache = (
                None if cache is None else (cache[2 * visit], cache[2 * visit + 1])
            )
            x = self.layers[index](x, blocks, visit_cache, messages)
            if speculative_verify and (visit + 1) % 4 == 0:
                mx.async_eval(x)
        h = self.final_input(blocks, x)[..., 0, :]
        normalized = self.norm(h).astype(x.dtype)
        if len(self.layer_indices) in requested:
            captures[len(self.layer_indices)] = normalized
        result = h if skip_final_norm else normalized
        if capture_layer_ids is not None:
            return result, [captures[i] for i in capture_layer_ids]
        return result


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sharding_group = None

    def chunked_prefill_policy(self, **kwargs):
        # Captures are per-token partial blocks; the shared prefill collector
        # retains them across chunks. Preserve the caller's target chunk size.
        return True

    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        cache=None,
        return_hidden=False,
        return_shared_kv=False,
        skip_logits=False,
        capture_layer_ids=None,
        speculative_verify=False,
        **kwargs,
    ):
        from .cache_state import inkling_verification_cache

        transaction = None
        verify_geometry = None
        feature_scope = _DSPARK_FEATURE_SCOPE.get()
        capture_for_dspark = (
            feature_scope is not None and self is feature_scope[0]
            and capture_layer_ids is None and not speculative_verify
        )
        if capture_for_dspark:
            capture_layer_ids = feature_scope[1]
        if speculative_verify:
            length = inputs.shape[1] if inputs is not None else inputs_embeds.shape[1]
            cache, transaction = inkling_verification_cache(cache, length)
            verify_geometry = _VERIFY_WITH_DECODE_GEMV.set(True)
        try:
            result = self.model(
                inputs,
                cache=cache,
                input_embeddings=inputs_embeds,
                skip_final_norm=True,
                capture_layer_ids=capture_layer_ids,
                speculative_verify=speculative_verify,
            )
            pre_norm, captured = (
                result if capture_layer_ids is not None else (result, None)
            )
            if capture_for_dspark:
                feature_scope[2](mx.concatenate(captured, axis=-1))
            return LanguageModelOutput(
                logits=(
                    None
                    if skip_logits
                    else self._logits_from_norm(self.model.norm(pre_norm))
                ),
                hidden_states=(
                    captured
                    if capture_layer_ids is not None
                    else ([pre_norm] if return_hidden else None)
                ),
                shared_kv_states={} if return_shared_kv else None,
                gdn_states=transaction,
            )
        except BaseException:
            if transaction is not None:
                transaction.abort()
            raise
        finally:
            if verify_geometry is not None:
                _VERIFY_WITH_DECODE_GEMV.reset(verify_geometry)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        count = self.config.n_routed_experts
        validate_sharding(vars(self.config), size)
        if not 0 <= rank < size:
            raise ValueError("Each distributed rank must own at least one expert")
        if self.sharding_group is not None:
            raise ValueError("Inkling 2 is already sharded")
        if size == 1:
            return
        start, end = count * rank // size, count * (rank + 1) // size
        scalar_scales = {"weight_scale_2", "weight_global_scale"}
        # Check every layer before changing any weights, including native
        # packed weights, group scales, and per-expert FP32 outer scales.
        for layer in self.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(layer.mlp.switch_mlp, name)
                for key, value in projection.parameters().items():
                    if not isinstance(value, mx.array):
                        continue
                    if key in scalar_scales and value.size == 1:
                        continue
                    if not value.ndim or value.shape[0] != count:
                        raise ValueError(
                            f"Invalid expert partition for {name}.{key}: "
                            f"expected {count} experts, got {value.shape}"
                        )
        for layer in self.layers:
            moe = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(moe.switch_mlp, name)
                projection.update(
                    {
                        key: mx.contiguous(value[start:end])
                        for key, value in projection.parameters().items()
                        if isinstance(value, mx.array)
                        and not (key in scalar_scales and value.size == 1)
                    }
                )
            # Materialize one layer at a time and release the full expert
            # buffers; lazy slices alone retain the unpartitioned allocation.
            mx.eval(moe.switch_mlp.parameters())
            mx.clear_cache()
            moe.expert_start, moe.expert_end = start, end
            moe.sharding_group = group
            layer.self_attn.shard(group)
            mx.eval(layer.self_attn.parameters())
            mx.clear_cache()
        self.sharding_group = group

    def _logits_from_norm(self, h):
        verify_rows = _VERIFY_WITH_DECODE_GEMV.get() and h.shape[1] > 1
        dense_decode_head = (
            not self.training
            and type(self.lm_head) is nn.Linear
            and "bias" not in self.lm_head
            and h.ndim == 3
            and mx.metal.is_available()
            and mx.default_device() == mx.gpu
        )
        if verify_rows and not (dense_decode_head and self.sharding_group is None):
            return mx.concatenate(
                [self._logits_from_norm(h[:, row:row + 1]) for row in range(h.shape[1])],
                axis=1,
            )
        # hf_quant_config.json stores BF16 unembedding weights but requires
        # fwd_output_dtype=float32. Promote the product before multiplication;
        # casting BF16 scores afterwards loses the accumulator's low bits.
        h = h.astype(self.model.embed_norm.weight.dtype).astype(mx.float32)
        if (
            dense_decode_head and self.sharding_group is None
            and h.shape[0] == 1 and (h.shape[1] == 1 or (verify_rows and h.shape[1] <= 5))
            and self.model.embed_norm.weight.dtype == mx.bfloat16
            and self.lm_head.weight.dtype == mx.bfloat16
            and self.lm_head.weight.shape == (201024, 2304)
        ):
            from .kernels import bf16_vocabulary_gemv

            logits = bf16_vocabulary_gemv(h, self.lm_head.weight)
        elif dense_decode_head and (verify_rows or h.shape[0] * h.shape[1] <= 15):
            # Retain the exact conversion instead of reading and promoting the
            # entire vocabulary matrix again on every decode step. Whole output
            # rows can be shared across the two expert ranks without splitting
            # any dot product or changing its FP32 accumulation.
            weight = self.lm_head.weight
            group = self.sharding_group
            partition = (
                group is not None
                and group.size() == 2
                and weight.shape[0] % group.size() == 0
            )
            size, rank = (group.size(), group.rank()) if partition else (1, 0)
            cached = self.get("_head_projection_cache")
            if cached is None or cached[0] is not weight or cached[1:3] != (size, rank):
                width = weight.shape[0] // size
                projected = weight[rank * width : (rank + 1) * width].astype(mx.float32)
                self._head_projection_cache = (weight, size, rank, projected)
            projected = self._head_projection_cache[-1]
            logits = _decode_gemv_rows(h, projected) if verify_rows else h @ projected.T
            if partition:
                width = projected.shape[0]
                logits = (
                    mx.distributed.all_gather(logits.reshape(1, -1, width), group=group)
                    .transpose(1, 0, 2)
                    .reshape(*h.shape[:-1], weight.shape[0])
                )
        else:
            logits = self.lm_head(h)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    def make_cache(self):
        # Expose both native lanes. The prefix cache pages attention windows and
        # checkpoints the convolution state at the same absolute token boundary.
        caches = []
        for index in self.model.layer_indices:
            attention = (
                RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                if self.model.layers[index].is_sliding
                else KVCache()
            )
            caches.extend((attention, ArraysCache(4)))
        return caches

    def speculative_logits_from_hidden(self, hidden):
        return self._logits_from_norm(self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden):
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs, cache):
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            skip_logits=True,
            speculative_verify=True,
        )
        return out.hidden_states[-1], {}, out.gdn_states

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        from .cache_state import rollback_speculative_cache

        return rollback_speculative_cache(caches, gdn_states, accepted, block_size)
