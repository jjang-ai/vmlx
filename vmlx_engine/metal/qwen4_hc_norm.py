"""Guarded FP16 HC residual combine followed by grouped native-order RMS.

Fused boundary: the pending HC residual combine is read inside the norm
kernel. Reduction adapted from Apple MLX 0.32.2 (1f8e74e3),
mlx/backend/metal/kernels/rms_norm.metal and normalization.cpp: N_READS=4,
640 threads for axis2560, two SIMD sums, precise rsqrt, FP16 norm output.
The caller's affine norm weight multiply is AFTER that FP16 rounding.

Apple reduction license (MIT): Copyright (c) 2023-2024 Apple Inc.
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
import logging
import math
import os

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active


def hc_combine_norm_requested():
    return os.environ.get("VMLX_QWEN4_HC_COMBINE_NORM", "1") == "1"


@lru_cache(maxsize=1)
def _compatible_runtime():
    try:
        return (importlib.metadata.version("mlx") in ("0.32.2", "0.32.3")
                and mx.device_info().get("device_name") == "Apple M5 Max")
    except (importlib.metadata.PackageNotFoundError, RuntimeError, OSError):
        return False


def hc_combine_norm_eligible(residual, block, inject, weight, *, eps, group_size):
    """Metadata-only admission; does not inspect/evaluate tensor contents."""
    return (
        group_size == 2560
        and residual.shape == (1, 1, 10240)
        and block.shape == (1, 1, 2560)
        and inject.shape == (1, 1, 4)
        and weight.shape == (10240,)
        and all(x.dtype == mx.float16 for x in (residual, block, inject, weight))
        and isinstance(eps, (int, float)) and math.isfinite(eps) and eps >= 0
    )


_SOURCE = r'''
    {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr uint H = 2560;
    constexpr uint N_READS = 4;
    uint group = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint feature = lid * N_READS;
    uint base = group * H + feature;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    float acc = 0;
    float thread_x[N_READS];
    for (uint i = 0; i < N_READS; ++i) {
        // Two separate native elementwise FP16 rounding boundaries.
        half product = half(float(block[feature + i]) * float(inject[group]));
        half value = half(float(residual[base + i]) + float(product));
        combined[base + i] = value;
        thread_x[i] = float(value);
        acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    if (simd_group == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / uint(axis_size[0]) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < N_READS; ++i) {
        // rms_norm(x, None) outputs FP16 before GroupedRMSNorm's multiply.
        half normed = half(thread_x[i] * local_inv_mean[0]);
        weighted[base + i] = half(float(normed) * float(weight[base + i]));
    }
    }
'''


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_hc_combine_norm_f16_2560_v1",
        input_names=["residual", "block", "inject", "weight", "eps", "axis_size"],
        output_names=["combined", "weighted"],
        source=_SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=8)
def _epsilon(value):
    return mx.array([value], dtype=mx.float32)


@lru_cache(maxsize=1)
def _axis_size():
    # Keep the native runtime divisor rather than a compile-time reciprocal.
    return mx.array([2560], dtype=mx.uint32)


_OBSERVED = False


def hc_combine_norm(residual, block, inject, weight, *, eps, group_size, enabled):
    """Return (combined, weighted_norm), or None before any admitted dispatch.

    No evaluation, host readback, cache mutation or exception retry. An admitted
    kernel's errors propagate; the owning forward must not replay cache updates.
    """
    if (not enabled or not affine_moe_ar_scope_active()
            or not hc_combine_norm_eligible(residual, block, inject, weight,
                                           eps=eps, group_size=group_size)
            or mx.default_device() != mx.gpu or not mx.metal.is_available()
            or not _compatible_runtime()):
        return None
    result = _kernel()(
        inputs=[residual, block, inject, weight, _epsilon(float(eps)), _axis_size()],
        grid=(4 * 640, 1, 1), threadgroup=(640, 1, 1),
        output_shapes=[(1, 1, 10240), (1, 1, 10240)],
        output_dtypes=[mx.float16, mx.float16],
    )
    global _OBSERVED
    if not _OBSERVED:
        logging.getLogger(__name__).info(
            "Qwen HC combine/norm fusion graph: rows=1 streams=4 hidden=2560 "
            "dtype=float16 native_rms_reads=4 threads=640 stream=caller"
        )
        _OBSERVED = True
    return tuple(result)
