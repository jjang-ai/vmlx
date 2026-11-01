"""Guarded ordinary-AR FP16 GDN prework; no recurrence or cache mutation.

Fused GDN prework boundary. Arithmetic
derived from Apple MLX 0.32.2 depthwise_conv_1d, rms_single_row,
Sigmoid/LogAddExp and executing mlx-lm compute_g. MLX 0.32.3 uses its
native precise sigmoid expression while preserving the 0.32.2 body.
No upstream BF16 formula
is substituted. Coefficients may independently be BF16 or FP32; never cast to
admission. FP16 activation + BF16/FP32 dt_bias promotes to FP32 in MLX.

Apple arithmetic sources are MIT licensed, Copyright (c) 2023-2024 Apple Inc.
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
import os

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active


def gdn_prework_requested():
    return os.environ.get("VMLX_QWEN4_GDN_PREWORK", "1") == "1"


@lru_cache(maxsize=1)
def _compatible_runtime():
    try:
        return (importlib.metadata.version("mlx") in ("0.32.2", "0.32.3")
                and mx.device_info().get("device_name") == "Apple M5 Max")
    except (importlib.metadata.PackageNotFoundError, RuntimeError, OSError):
        return False


def gdn_prework_eligible(qkv, a, b, conv_state, weight, A_log, dt_bias, *,
                         mask=None, lengths=None, training=False,
                         incumbent_fused_conv=False):
    """Metadata only. Caller must also retain ordinary-AR scope admission."""
    expected = (
        (qkv, (1, 1, 10240), mx.float16),
        (a, (1, 1, 48), mx.float16), (b, (1, 1, 48), mx.float16),
        (conv_state, (1, 3, 10240), mx.float16),
        (weight, (10240, 4, 1), mx.float16),
    )
    return (mask is None and lengths is None and not training
            and not incumbent_fused_conv and all(
                x is not None and tuple(x.shape) == (48,)
                and x.dtype in (mx.bfloat16, mx.float32)
                for x in (A_log, dt_bias)) and all(
                x is not None and tuple(x.shape) == shape and x.dtype == dtype
                for x, shape, dtype in expected))


_SOURCE = r'''
    {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint head = threadgroup_position_in_grid.x; // 16 Q + 16 K + 48 V
    uint lane = thread_position_in_threadgroup.x;
    uint base = head * 128 + lane * 4;
    float values[4];
    float sum = 0;
    threadgroup float sums[32];
    threadgroup float inv[1];
    for (uint i = 0; i < 4; ++i) {
        uint c = base + i;
        float acc = 0;
        for (uint tap = 0; tap < 3; ++tap) {
            acc += float(history[tap * 10240 + c]) * float(weight[c * 4 + tap]);
        }
        acc += float(qkv[c]) * float(weight[c * 4 + 3]);
        half conv = half(acc); // native convolution materialization
        // nn.silu is compiled: native Sigmoid expression, then half multiply.
        half e = metal::exp(metal::abs(conv));
        half s = half(1) / (half(1) + e);
        half sigmoid = conv < 0 ? s : half(1) - s;
        half activated = conv * sigmoid;
        values[i] = float(activated);
        sum += values[i] * values[i];
        next_history[c] = history[10240 + c];
        next_history[10240 + c] = history[20480 + c];
        next_history[20480 + c] = qkv[c];
    }
    // Same second SIMD sum as native rms_single_row, even for one SIMD.
    sum = simd_sum(sum);
    sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sums[0] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(sum / axis[0] + eps[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint c = base + i;
        if (head < 32) {
            half rms = half(values[i] * inv[0]);
            half scale = scales[head < 16 ? 0 : 1];
            half scaled = scale * rms;
            if (head < 16) q[c] = scaled;
            else k[c - 2048] = scaled;
        } else v[c - 4096] = half(values[i]);
    }
    if (head < 48 && lane == 0) {
        // F16 + BF16/F32 promotes BEFORE softplus; A_log.astype(F32).
        float x = float(a[head]) + float(dt_bias[head]);
        float hi = metal::max(x, 0.0f), lo = metal::min(x, 0.0f);
        float sp;
        if (metal::isnan(x)) sp = metal::numeric_limits<float>::quiet_NaN();
        else if (lo == -INFINITY || hi == INFINITY) sp = hi;
        else {
            float ex = metal::exp(lo - hi);
            float xp1 = 1.0f + ex;
            float lp = xp1 == 1.0f ? ex : ex * (metal::log(xp1) / (xp1 - 1.0f));
            sp = hi + lp;
        }
        g[head] = metal::precise::exp(-metal::precise::exp(float(A_log[head])) * sp);
        // beta is an uncompiled unary primitive: precise exp, FP16 boundary.
        half be = metal::precise::exp(metal::abs(b[head]));
        half by = half(1) / (half(1) + be);
        beta[head] = b[head] < 0 ? by : half(1) - by;
    }
    }
'''


def _source_for_version(version):
    if version == "0.32.2":
        return _SOURCE
    if version != "0.32.3":
        raise ValueError("unqualified Qwen GDN prework MLX version")
    # MLX 0.32.3 Sigmoid uses precise::exp and its native expression
    # promotion. Half exponential/denominator temporaries change its bits.
    return _SOURCE.replace(
        "half e = metal::exp(metal::abs(conv));\n"
        "        half s = half(1) / (half(1) + e);",
        "auto s = 1 / (1 + metal::precise::exp(metal::abs(conv)));",
    ).replace(
        "half be = metal::precise::exp(metal::abs(b[head]));\n"
        "        half by = half(1) / (half(1) + be);",
        "auto by = 1 / (1 + metal::precise::exp(metal::abs(b[head])));",
    )


@lru_cache(maxsize=1)
def _kernel():
    version = importlib.metadata.version("mlx")
    source = _source_for_version(version)
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_gdn_prework_f16_f32coeff_v1_" + version.replace(".", "_"),
        input_names=["qkv", "a", "b", "history", "weight", "A_log", "dt_bias",
                     "axis", "eps", "scales"],
        output_names=["q", "k", "v", "next_history", "g", "beta"],
        source=source, ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _constants():
    return (mx.array([128], dtype=mx.uint32), mx.array([1e-6], dtype=mx.float32),
            mx.array([128 ** -1.0, 128 ** -0.5], dtype=mx.float16))


def gdn_prework(qkv, a, b, conv_state, weight, A_log, dt_bias, *, enabled=None,
                mask=None, lengths=None, training=False, incumbent_fused_conv=False):
    """Return q,k,v,next_conv_state,g,beta, or None without dispatch.

    Pure graph construction: no evaluation, state writes or exception retry.
    An admitted failure propagates. Call the incumbent gated_delta_kernel with
    returned q/k/v/g/beta and the untouched recurrent state; not gated_delta_update
    (which would recompute gates). No hook is installed by this module.
    """
    if enabled is None:
        enabled = gdn_prework_requested()
    if (not enabled or not affine_moe_ar_scope_active()
            or not gdn_prework_eligible(qkv, a, b, conv_state, weight, A_log, dt_bias,
                mask=mask, lengths=lengths, training=training,
                incumbent_fused_conv=incumbent_fused_conv)
            or mx.default_device() != mx.gpu or not mx.metal.is_available()
            or not _compatible_runtime()):
        return None
    return tuple(_kernel()(
        inputs=[qkv, a, b, conv_state, weight, A_log, dt_bias, *_constants()],
        grid=(80 * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, 16, 128), (1, 1, 16, 128), (1, 1, 48, 128),
                       (1, 3, 10240), (1, 1, 48), (1, 1, 48)],
        output_dtypes=[mx.float16, mx.float16, mx.float16, mx.float16,
                       mx.float32, mx.float16],
    ))


_ENGAGEMENT_LOGGED = False


def gdn_prework_update(qkv, a, b, conv_state, weight, A_log, dt_bias, state, *,
                       enabled, mask=None, lengths=None, training=False,
                       incumbent_fused_conv=False):
    """Construct prework + incumbent recurrence, without writing any cache.

    None means decline BEFORE kernel construction. Existing native state=None
    initialization is preserved after admission. Admitted failures propagate;
    caller must not catch and replay. The log attests graph selection, not GPU
    completion or performance, and performs no evaluation/host readback.
    """
    if not enabled or (state is not None and (
            tuple(state.shape) != (1, 48, 128, 128) or state.dtype != mx.float32)):
        return None
    values = gdn_prework(qkv, a, b, conv_state, weight, A_log, dt_bias,
                        enabled=enabled, mask=mask, lengths=lengths,
                        training=training, incumbent_fused_conv=incumbent_fused_conv)
    if values is None:
        return None
    from mlx_lm.models.gated_delta import gated_delta_kernel
    q, k, v, new_conv_state, g, beta = values
    if state is None:
        state = mx.zeros((1, 48, 128, 128), dtype=mx.float32)
    output, new_state = gated_delta_kernel(q, k, v, g, beta, state)
    global _ENGAGEMENT_LOGGED
    if not _ENGAGEMENT_LOGGED:
        logging.getLogger(__name__).info(
            "Qwen4 GDN prework: constructed ordinary-AR FP16 B1/S1 graph; "
            "native recurrence retained; A_log=%s dt_bias=%s",
            A_log.dtype, dt_bias.dtype,
        )
        _ENGAGEMENT_LOGGED = True
    return output, new_conv_state, new_state
