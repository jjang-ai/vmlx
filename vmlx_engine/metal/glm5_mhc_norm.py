"""Guarded GLM productive-AR mHC + following weighted RMS fusion.

Projection and scalar Sinkhorn are incumbent arithmetic, not ds4 arithmetic.
Reduction adapted from MLX 0.32.2 rms_norm.metal (Apple MIT below).
Opt-in, with exact shape/dtype/runtime admission. No cache mutation.
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from functools import lru_cache
import importlib.metadata
import math
import os

import mlx.core as mx

from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope_active
from vmlx_engine.metal.glm5_mhc_decode import _projection_kernel, _scalar, glm5_mhc_v3_requested

_SOURCE_TEMPLATE = r"""
    uint tid = thread_index_in_threadgroup;
    uint token = thread_position_in_grid.x / EPILOGUE_THREADS;
    if (token >= ROWS) return;
    size_t streams_base = (size_t)token * H * D;
    size_t mix_base = (size_t)token * MIX_SIZE;
    size_t post_base = (size_t)token * H;
    size_t comb_base = (size_t)token * H * H;
    size_t collapsed_base = (size_t)token * D;
    threadgroup float pre_values[H];
    threadgroup float post_values[H];
    threadgroup float matrix[H * H];

SERIAL_SINKHORN_PLACEHOLDER    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Keep the original ordered four-stream collapse before BF16 rounding.
    // Contiguous four-element ownership matches MLX rms_single_row exactly.
    float thread_x[4];
    for (uint i = 0u; i < 4u; ++i) {
        uint dim = tid * 4u + i;
        float value = 0.0f;
        for (uint stream = 0u; stream < H; ++stream) {
            value += pre_values[stream] *
                float(streams[streams_base + stream * D + dim]);
        }
        T rounded = T(value);
        collapsed[collapsed_base + dim] = rounded;
        thread_x[i] = float(rounded);
    }
    {
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        threadgroup float local_sums[32];
        threadgroup float local_inv_mean[1];
        float acc = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        if (simd_group == 0u) local_sums[lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) local_sums[simd_group] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0u) {
            acc = simd_sum(local_sums[lane]);
            if (lane == 0u) {
                local_inv_mean[0] = metal::precise::rsqrt(
                    acc / uint(axis_size[0]) + norm_eps[0]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0u; i < 4u; ++i) {
            uint dim = tid * 4u + i;
            // MLX rounds normalized activation to T BEFORE weighted multiply.
            normalized[collapsed_base + dim] =
                norm_weight[dim] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
        }
    }
    for (uint index = tid; index < H; index += EPILOGUE_THREADS) {
        post[post_base + index] = post_values[index];
    }
    for (uint index = tid; index < H * H; index += EPILOGUE_THREADS) {
        comb[comb_base + index] = matrix[index];
    }
"""

_SERIAL_SINKHORN = r"""
    if (tid == 0u) {
        float scale_pre = float(hc_scale[0]);
        float scale_post = float(hc_scale[1]);
        float scale_comb = float(hc_scale[2]);
        float epsilon = float(sink_eps[0]);

        for (uint index = 0u; index < H; ++index) {
            float pre_x = float(mix[mix_base + index]) * scale_pre +
                float(hc_base[index]);
            pre_values[index] = 1.0f / (1.0f + metal::exp(-pre_x)) + epsilon;
            float post_x = float(mix[mix_base + H + index]) * scale_post +
                float(hc_base[H + index]);
            post_values[index] = 2.0f / (1.0f + metal::exp(-post_x));
        }

        for (uint row = 0u; row < H; ++row) {
            float row_max = -INFINITY;
            for (uint col = 0u; col < H; ++col) {
                uint index = row * H + col;
                float item = float(mix[mix_base + 2u * H + index]) * scale_comb +
                    float(hc_base[2u * H + index]);
                matrix[index] = item;
                row_max = metal::max(row_max, item);
            }
            float row_sum = 0.0f;
            for (uint col = 0u; col < H; ++col) {
                uint index = row * H + col;
                float item = metal::exp(matrix[index] - row_max);
                matrix[index] = item;
                row_sum += item;
            }
            for (uint col = 0u; col < H; ++col) {
                uint index = row * H + col;
                matrix[index] = matrix[index] / row_sum + epsilon;
            }
        }

        for (uint col = 0u; col < H; ++col) {
            float col_sum = 0.0f;
            for (uint row = 0u; row < H; ++row) {
                col_sum += matrix[row * H + col];
            }
            for (uint row = 0u; row < H; ++row) {
                uint index = row * H + col;
                matrix[index] /= col_sum + epsilon;
            }
        }

        for (uint iteration = 1u; iteration < ITERS; ++iteration) {
            for (uint row = 0u; row < H; ++row) {
                float row_sum = 0.0f;
                for (uint col = 0u; col < H; ++col) {
                    row_sum += matrix[row * H + col];
                }
                for (uint col = 0u; col < H; ++col) {
                    uint index = row * H + col;
                    matrix[index] /= row_sum + epsilon;
                }
            }
            for (uint col = 0u; col < H; ++col) {
                float col_sum = 0.0f;
                for (uint row = 0u; row < H; ++row) {
                    col_sum += matrix[row * H + col];
                }
                for (uint row = 0u; row < H; ++row) {
                    uint index = row * H + col;
                    matrix[index] /= col_sum + epsilon;
                }
            }
        }
    }
"""

_PARALLEL_SINKHORN = r"""
    // v3 (2026-10-10): the H x H = 4 x 4 Sinkhorn runs on 16 lanes of SIMD group 0 (lane = matrix element,
    // row sums over lane bits 0-1, column sums over bits 2-3) instead of one thread; lanes 16-31 mirror 0-15.
    // Same formulas, pairwise instead of serial summation order. The serial version cost ~30 us per call in a
    // dependent decode chain (x90 per token).
    if (simdgroup_index_in_threadgroup == 0u) {
        uint lane = thread_index_in_simdgroup;
        float scale_pre = float(hc_scale[0]);
        float scale_post = float(hc_scale[1]);
        float scale_comb = float(hc_scale[2]);
        float epsilon = float(sink_eps[0]);
        if (lane < H) {
            float pre_x = float(mix[mix_base + lane]) * scale_pre + float(hc_base[lane]);
            pre_values[lane] = 1.0f / (1.0f + metal::exp(-pre_x)) + epsilon;
            float post_x = float(mix[mix_base + H + lane]) * scale_post + float(hc_base[H + lane]);
            post_values[lane] = 2.0f / (1.0f + metal::exp(-post_x));
        }
        uint e = lane & 15u;
        float item = float(mix[mix_base + 2u * H + e]) * scale_comb + float(hc_base[2u * H + e]);
        float m = metal::max(item, simd_shuffle_xor(item, 1u));
        m = metal::max(m, simd_shuffle_xor(m, 2u));
        float ex = metal::exp(item - m);
        float rs = ex + simd_shuffle_xor(ex, 1u);
        rs = rs + simd_shuffle_xor(rs, 2u);
        float v = ex / rs + epsilon;
        float cs = v + simd_shuffle_xor(v, 4u);
        cs = cs + simd_shuffle_xor(cs, 8u);
        v = v / (cs + epsilon);
        for (uint iteration = 1u; iteration < ITERS; ++iteration) {
            rs = v + simd_shuffle_xor(v, 1u);
            rs = rs + simd_shuffle_xor(rs, 2u);
            v = v / (rs + epsilon);
            cs = v + simd_shuffle_xor(v, 4u);
            cs = cs + simd_shuffle_xor(cs, 8u);
            v = v / (cs + epsilon);
        }
        if (lane < 16u) matrix[lane] = v;
    }
"""

_SOURCE = _SOURCE_TEMPLATE.replace("SERIAL_SINKHORN_PLACEHOLDER", _SERIAL_SINKHORN[1:])
_SOURCE_V3 = _SOURCE_TEMPLATE.replace("SERIAL_SINKHORN_PLACEHOLDER", _PARALLEL_SINKHORN[1:])



def eligible(streams, hc_fn, hc_base, hc_scale, weight, *,
             rms_eps, sink_eps, norm_eps, iterations):
    """Preserve both stored F32 and hydrated BF16 coefficient arithmetic."""
    return (
        streams.shape == (1, 1, 4, 4096)
        and hc_fn.shape == (24, 16384)
        and hc_base.shape == (24,) and hc_scale.shape == (3,)
        and weight.shape == (4096,)
        and streams.dtype == hc_fn.dtype == weight.dtype == mx.bfloat16
        and hc_base.dtype == hc_scale.dtype
        and hc_base.dtype in (mx.float32, mx.bfloat16)
        and iterations == 20
        and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for v in (rms_eps, sink_eps, norm_eps))
    )


@lru_cache(maxsize=4)
def _kernel(capture=False, v3=False):
    # Dead Python siblings do not suppress CustomKernel output allocations.
    # The production specialization therefore never declares/stores collapse.
    store = "        collapsed[collapsed_base + dim] = rounded;\n"
    base = _SOURCE_V3 if v3 else _SOURCE
    assert base.count(store) == 1
    source = base if capture else base.replace(store, "")
    return mx.fast.metal_kernel(
        name=("vmlx_glm5_mhc_weighted_rms_bf16_4096_v3" if v3 else "vmlx_glm5_mhc_weighted_rms_bf16_4096_v2")
        + ("_capture" if capture else ""),
        input_names=["streams", "mix", "hc_base", "hc_scale", "sink_eps",
                     "norm_weight", "norm_eps", "axis_size"],
        output_names=["post", "comb"] + (["collapsed"] if capture else []) + ["normalized"],
        source=source,
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def glm5_mhc_norm_decode(streams, hc_fn, hc_base, hc_scale, weight, *,
                         rms_eps, sink_eps, norm_eps, iterations, enabled=False,
                         capture=False):
    """Return post/comb/normalized; capture adds collapse before normalized."""
    if not enabled or not eligible(
        streams, hc_fn, hc_base, hc_scale, weight, rms_eps=rms_eps,
        sink_eps=sink_eps, norm_eps=norm_eps, iterations=iterations
    ):
        return None
    # Invoke the existing projection unchanged: same grid, templates, inputs.
    v3 = glm5_mhc_v3_requested()
    pt = 1024 if v3 else 128          # v3: 16 loads/thread instead of 128 (39.5 -> 14.8 us dependent, 2026-10-10)
    mix = _projection_kernel()(
        inputs=[streams, hc_fn, _scalar(float(rms_eps))],
        template=[("FEATURES", 16384), ("MIX_SIZE", 24), ("ROWS", 1),
                  ("PROJECTION_THREADS", pt), ("PROJECTION_SIMDGROUPS", pt // 32)],
        grid=(pt * 24, 1, 1), threadgroup=(pt, 1, 1),
        output_shapes=[(1, 24)], output_dtypes=[mx.float32],
    )[0]
    return tuple(_kernel(capture, v3)(
        inputs=[streams, mix, hc_base, hc_scale, _scalar(float(sink_eps)),
                weight, _scalar(float(norm_eps)), _scalar(4096.0)],
        template=[("T", mx.bfloat16), ("H", 4), ("D", 4096), ("ROWS", 1),
                  ("MIX_SIZE", 24), ("ITERS", 20), ("EPILOGUE_THREADS", 1024)],
        grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(1, 1, 4), (1, 1, 4, 4)]
                      + ([(1, 1, 4096)] if capture else []) + [(1, 1, 4096)],
        output_dtypes=[mx.float32, mx.float32]
                      + ([mx.bfloat16] if capture else []) + [mx.bfloat16],
    ))


@lru_cache(maxsize=1)
def _compatible_runtime():
    try:
        return (importlib.metadata.version("mlx") in {"0.32.2", "0.32.3"}
                and mx.device_info().get("device_name") == "Apple M5 Max")
    except (importlib.metadata.PackageNotFoundError, RuntimeError, OSError):
        return False


_CALLS = 0


def try_glm5_hc_norm(streams, hc, norm):
    """Only entered by ordinary DecoderLayer; speculative/prefill stay stock."""
    if (os.environ.get("VMLX_GLM5_MHC_WEIGHTED_RMS", "1") != "1"
            or not affine_moe_ar_scope_active()
            or not hc._fused_decode or mx.default_device() != mx.gpu
            or not mx.metal.is_available() or not _compatible_runtime()):
        return None
    result = glm5_mhc_norm_decode(
        streams, hc.hc_fn, hc.hc_base, hc.hc_scale, norm.weight,
        rms_eps=hc.rms_eps, sink_eps=hc.eps, norm_eps=norm.eps,
        iterations=hc.iters, enabled=True,
    )
    if result is None:
        return None
    global _CALLS
    _CALLS += 1  # Graph construction count, NOT completed GPU execution.
    return result


def glm5_mhc_norm_status():
    # Requested is policy, not installation or completed GPU execution.
    return {"graph_calls": _CALLS,
            "requested": os.environ.get("VMLX_GLM5_MHC_WEIGHTED_RMS", "1") == "1"}
