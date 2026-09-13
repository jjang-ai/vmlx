# SPDX-License-Identifier: Apache-2.0
# Metal reduction and precision structure adapted from mlx-vlm #2105.
# Copyright © 2025 Prince Canuma. Used under the MIT License.
#
# Ported from the local unified Qwen4 full-recurrence verify kernel.
# Only q/k normalization changes: this experimental component preserves the
# deployed oMLX RMS/scaling semantics. It is not a direct-L2 correction.
"""Experimental small-slab Qwen GDN recurrence with explicit prefix snapshots.

Default-off model integration; numerical and live admission are separate gates.
Projections stay outside this boundary and retain their actual quantization.
Derived from oMLX PR #3626 at 95d500b6f316a63bc8f4d02d2df70a6682371477.
The upstream BF16 route does not by itself qualify this FP16 component.
"""

from functools import cache
import os

import mlx.core as mx

NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
VALUE_DIM = 6144
CONV_DIM = 10240
CONV_KERNEL = 4
MAX_VERIFY_STEPS = 4  # Wider or different geometry is not admitted.
_THREADGROUP_Y_CANDIDATES = (32, 16, 8, 4)


def unified_gdn_verify_requested() -> bool:
    """Experimental Flash Next verifier; never enabled by a quant/model name."""
    return os.environ.get("VMLX_QWEN4_UNIFIED_GDN_VERIFY", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def unified_gdn_verify_eligible(
    qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
    recurrent_state, norm_weight, norm_eps,
) -> bool:
    """Only the measured FP16 geometry; no state/dtype coercion or allocation."""
    if (qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[2] != CONV_DIM
            or qkv.dtype != mx.float16 or not 3 <= qkv.shape[1] <= MAX_VERIFY_STEPS):
        return False
    steps = qkv.shape[1]
    expected = (
        (z, (1, steps, VALUE_DIM), qkv.dtype),
        (a, (1, steps, NUM_VALUE_HEADS), qkv.dtype),
        (b, (1, steps, NUM_VALUE_HEADS), qkv.dtype),
        (conv_state, (1, CONV_KERNEL - 1, CONV_DIM), qkv.dtype),
        (conv_weight, (CONV_DIM, CONV_KERNEL, 1), qkv.dtype),
        (A_log, (NUM_VALUE_HEADS,), qkv.dtype),
        (dt_bias, (NUM_VALUE_HEADS,), qkv.dtype),
        (recurrent_state, (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), mx.float32),
        (norm_weight, (VALUE_HEAD_DIM,), qkv.dtype),
    )
    return (norm_eps == 1e-6 and all(
        value is not None and tuple(value.shape) == shape and value.dtype == dtype
        for value, shape, dtype in expected
    ))


_HEADER = r"""
#include <metal_atomic>
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_log1p_fast(U x) {
  float xf = float(x);
  float xp1 = 1.0f + xf;
  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));
  return static_cast<U>(out);
}

// MLX utils.h has float and bfloat16 log1p overloads, but no half overload.
// A half input therefore keeps the log1p result in float until LogAddExp's
// final addition/cast. Rounding log1p to half first changes recurrent gates.
inline float mlx_log1p_fast(half x) {
  return mlx_log1p_fast<float>(float(x));
}

template <typename U>
inline U mlx_softplus_fast(U x) {
  if (metal::isnan(x))
    return metal::numeric_limits<U>::quiet_NaN();
  constexpr U inf = metal::numeric_limits<U>::infinity();
  U zero = static_cast<U>(0);
  U hi = metal::max(x, zero);
  U lo = metal::min(x, zero);
  return (lo == -inf || hi == inf)
      ? hi
      : (hi + mlx_log1p_fast(static_cast<U>(metal::exp(lo - hi))));
}

"""

_SOURCE = r"""
  const uint hv = threadgroup_position_in_grid.z;
  const uint hk = hv / RATIO;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty = thread_position_in_threadgroup.y;
  const uint tid = thread_index_in_threadgroup;

  constexpr int NT = 32 * TY;
  constexpr int NDK = DK / 32;
  constexpr int NDV = DV / TY;
  constexpr uint KD = (uint)(HK * DK);
  constexpr uint VD = (uint)(HV * DV);
  constexpr uint CD = 2u * KD + VD;
  constexpr uint KEEP = (uint)K - 1u;
  constexpr uint SNAPS = (uint)S - 1u;

  threadgroup float sq[DK];
  threadgroup float sk[DK];
  threadgroup T sq_squared[DK];
  threadgroup T sk_squared[DK];
  threadgroup float sv[DV];
  threadgroup float sy[DV];
  threadgroup float shr[4];

  device const float* si = recurrent_state + (size_t)hv * DV * DK;
  device float* so = recurrent_state_out + (size_t)hv * DV * DK;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i)
      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];
  }

  const bool owns_shared = (hv % RATIO) == 0u;

  // Convolution window bookkeeping is token independent: publish the final
  // window (the next conv cache) and every intermediate window the layer
  // records as a restore point.
  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
    uint part = idx / (uint)DK;
    uint d = idx - part * (uint)DK;
    uint c = part == 0u ? hk * DK + d
           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
    if (part == 2u || owns_shared) {
      for (uint tap = 0; tap < KEEP; ++tap) {
        uint row = (uint)S + tap;
        conv_state_out[(size_t)tap * CD + c] =
            row < KEEP ? conv_state[(size_t)row * CD + c]
                       : qkv[(size_t)(row - KEEP) * CD + c];
      }
      for (uint p = 1; p <= SNAPS; ++p) {
        for (uint tap = 0; tap < KEEP; ++tap) {
          uint row = p + tap;
          conv_snapshots[((size_t)(p - 1u) * KEEP + tap) * CD + c] =
              row < KEEP ? conv_state[(size_t)row * CD + c]
                         : qkv[(size_t)(row - KEEP) * CD + c];
        }
      }
    }
  }

  for (uint t = 0; t < (uint)S; ++t) {
    for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
      uint part = idx / (uint)DK;
      uint d = idx - part * (uint)DK;
      uint c = part == 0u ? hk * DK + d
             : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
      device const T* wc = conv_weight + (size_t)c * K;
      float acc = 0.0f;
      for (uint tap = 0; tap < (uint)K; ++tap) {
        uint row = t + tap;
        T xv = row < KEEP ? conv_state[(size_t)row * CD + c]
                          : qkv[(size_t)(row - KEEP) * CD + c];
        acc += float(xv) * float(wc[tap]);
      }
      T xb = static_cast<T>(acc);
      // nn.silu is reproduced by the fast sigmoid form on every finite bf16.
      T sl = xb * mlx_sigmoid_fast(xb);
      if (part == 0u) sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else sv[d] = float(sl);
    }

    if (tid == 0u) {
      T av = a[t * HV + hv] + dt_bias[hv];
      T sp = mlx_softplus_fast(av);
      shr[2] = metal::precise::exp(
          -metal::precise::exp(float(A_log[hv])) * float(sp));
      // mx.sigmoid on bf16 is the precise form on every finite bf16 input;
      // the fast form differs on one (x ~ -6.85), which real activations reach.
      shr[3] = float(mlx_sigmoid_precise(b[t * HV + hv]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Preserve oMLX generic RMS/scaling operation order exactly: four
    // FP32 square accumulations per SIMD lane, SIMD sum, epsilon after
    // mean, BF16 RMS materialization, then a separate BF16 scalar scale.
    if (simdgroup_index_in_threadgroup == 0u) {
      float pq = 0.0f, pk = 0.0f;
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) {
        float qv = sq[base + i], kv = sk[base + i];
        pq += qv * qv;
        pk += kv * kv;
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(pq / float(DK) + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk / float(DK) + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const T qscale = T(0.0078125f);
    const T kscale = T(0.08838834764831845f);
    for (uint d = tid; d < (uint)DK; d += NT) {
      const T qrms = T(1) * T(sq[d] * shr[0]);
      const T krms = T(1) * T(sk[d] * shr[1]);
      sq[d] = float(T(qscale * qrms));
      sk[d] = float(T(kscale * krms));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* state_dst =
        t < SNAPS ? state_snapshots + ((size_t)t * HV + hv) * DV * DK : so;
    for (int j = 0; j < NDV; ++j) {
      uint dv = ty + (uint)TY * (uint)j;
      float kv = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] * shr[2];
        kv += st[j][i] * sk[s];
      }
      kv = simd_sum(kv);
      float delta = (sv[dv] - kv) * shr[3];
      float out = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] + sk[s] * delta;
        out += st[j][i] * sq[s];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0u)
        sy[dv] = float(static_cast<T>(out));
      for (int i = 0; i < NDK; ++i)
        state_dst[(size_t)dv * DK + NDK * lane + i] = st[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0u) {
      float po = 0.0f;
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];
      po = simd_sum(po);
      if (lane == 0u)
        shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < (uint)DV; d += NT) {
      T normalized = static_cast<T>(sy[d] * shr[0]);
      normalized = norm_weight[d] * normalized;
      // float32 sigmoid of a bf16-valued gate: the precise form matches
      // mx.sigmoid on every finite bf16 input; the fast form differs on ~1%.
      float x = float(normalized) *
                mlx_sigmoid_precise<float>(float(z[t * VD + hv * DV + d]));
      output[t * VD + hv * DV + d] = static_cast<T>(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_unified_gdn_verify_rms",
        input_names=[
            "qkv",
            "z",
            "b",
            "a",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
            "norm_eps",
        ],
        output_names=[
            "output",
            "conv_state_out",
            "recurrent_state_out",
            "state_snapshots",
            "conv_snapshots",
        ],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def qwen4_unified_gdn_verify(
    qkv,
    z,
    b,
    a,
    conv_state,
    conv_weight,
    A_log,  # noqa: N803 - checkpoint ABI uses this canonical name
    dt_bias,
    recurrent_state,
    norm_weight,
    norm_eps: float,
    *,
    threadgroup_y: int,
):
    """Build the fused verify graph. Callers must run structural admission first.

    Returns ``(output, conv_state_out, recurrent_state_out, state_snapshots,
    conv_snapshots)``; ``state_snapshots[:, p]`` and ``conv_snapshots[:, p]``
    are the recurrent state and convolution window after ``p + 1`` tokens for
    ``p`` in ``range(S - 1)``.
    """
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; "
            f"expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    if qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[2] != CONV_DIM:
        raise ValueError("unsupported GDN verify input shape")
    if qkv.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError("unsupported GDN verify input dtype")
    steps = int(qkv.shape[1])
    expected = (
        (z, (1, steps, VALUE_DIM), qkv.dtype),
        (a, (1, steps, NUM_VALUE_HEADS), qkv.dtype),
        (b, (1, steps, NUM_VALUE_HEADS), qkv.dtype),
        (conv_state, (1, CONV_KERNEL - 1, CONV_DIM), qkv.dtype),
        (conv_weight, (CONV_DIM, CONV_KERNEL, 1), qkv.dtype),
        (A_log, (NUM_VALUE_HEADS,), qkv.dtype),
        (dt_bias, (NUM_VALUE_HEADS,), qkv.dtype),
        (recurrent_state, (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), mx.float32),
        (norm_weight, (VALUE_HEAD_DIM,), qkv.dtype),
    )
    if any(tuple(v.shape) != shape or v.dtype != dtype for v, shape, dtype in expected):
        raise ValueError("unsupported GDN verify parameter/state layout")
    if norm_eps != 1e-6:
        raise ValueError("unsupported GDN verify norm epsilon")
    # Keep the component's graph-construction surface bounded. Production
    # admission and end-to-end performance require separate evidence.
    if not 3 <= steps <= MAX_VERIFY_STEPS:
        raise ValueError(
            f"unsupported verify width {steps}; expected 3..{MAX_VERIFY_STEPS}"
        )
    outputs = _kernel()(
        inputs=[
            qkv,
            z,
            b,
            a,
            conv_state,
            conv_weight,
            A_log,
            dt_bias,
            recurrent_state,
            norm_weight,
            float(norm_eps),
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("S", steps),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, CONV_KERNEL - 1, CONV_DIM),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, mx.float32, qkv.dtype],
    )
    return tuple(outputs)
