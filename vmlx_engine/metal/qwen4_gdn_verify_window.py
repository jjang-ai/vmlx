# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused Gated-DeltaNet verify window for Qwen4 (Flash-Next) native MTP.

WHAT
----
One Metal dispatch evaluates a 3- or 4-token verify slab of a Qwen4 GDN layer
after its projections: causal conv + SiLU, the decay/beta gates, q/k RMS
scaling, the delta-rule recurrence and the sigmoid-gated RMS norm.  Besides the
slab output and the final conv window / recurrent state it emits every
intermediate restore point (state and conv window after 1..S-1 tokens), which
is what the MTP verifier needs to roll back to the accepted prefix without a
second pass.

ARITHMETIC
----------
The elementwise formulas are MLX's own (Sigmoid in ``unary_ops.h``, ``log1p``
in ``utils.h``, softplus as ``LogAddExp(x, 0)`` in ``binary_ops.h``).  Promotions
are kept where MLX applies them: the softplus input is ``a + dt_bias`` in the
promoted dtype of the two, the decay is formed in fp32 from those coefficients
without downcasting them, and the RMS reductions use four fp32 elements per lane
followed by ``simd_sum``.

Lever: ``VMLX_QWEN4_UNIFIED_GDN_VERIFY=1`` (default off).  The projections stay
outside this kernel with their native quantization.
"""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

KEY_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM          # 6144
CHANNELS = 2 * KEY_HEADS * HEAD_DIM + VALUE_WIDTH   # 10240
CONV_TAPS = 4
VERIFY_WIDTHS = (3, 4)
ROW_GROUPS = (32, 16, 8, 4)
_COEFFICIENT_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


def gdn_verify_window_requested() -> bool:
    raw = os.environ.get("VMLX_QWEN4_UNIFIED_GDN_VERIFY", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _coefficients_ok(A_log, dt_bias) -> bool:  # noqa: N803
    return all(
        c is not None and tuple(c.shape) == (VALUE_HEADS,) and c.dtype in _COEFFICIENT_DTYPES
        for c in (A_log, dt_bias)
    )


def _layout(steps: int, dtype):
    """(array index in the argument list, expected shape, expected dtype)."""
    return (
        (1, (1, steps, VALUE_WIDTH), dtype),                         # z
        (2, (1, steps, VALUE_HEADS), dtype),                         # b
        (3, (1, steps, VALUE_HEADS), dtype),                         # a
        (4, (1, CONV_TAPS - 1, CHANNELS), dtype),                    # conv window
        (5, (CHANNELS, CONV_TAPS, 1), dtype),                        # conv weight
        (8, (1, VALUE_HEADS, HEAD_DIM, HEAD_DIM), mx.float32),       # recurrent state
        (9, (HEAD_DIM,), dtype),                                     # norm weight
    )


def _arrays_match(arrays, steps, dtype) -> bool:
    for index, shape, want in _layout(steps, dtype):
        value = arrays[index]
        if value is None or tuple(value.shape) != shape or value.dtype != want:
            return False
    return True


def gdn_verify_window_eligible(
    qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,  # noqa: N803
    recurrent_state, norm_weight, norm_eps,
) -> bool:
    """Structural admission (fp16 activations); no allocation, never raises."""
    if qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[2] != CHANNELS:
        return False
    if qkv.dtype != mx.float16 or qkv.shape[1] not in VERIFY_WIDTHS:
        return False
    arrays = (qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
              recurrent_state, norm_weight)
    return (norm_eps == 1e-6 and _coefficients_ok(A_log, dt_bias)
            and _arrays_match(arrays, qkv.shape[1], qkv.dtype))


HEADER = r"""
// MLX Sigmoid (unary_ops.h): exponent in float, one rounding to U at the end.
template <typename U>
inline U gdnw_sigmoid_precise(U x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return static_cast<U>((x < 0) ? y : 1 - y);
}

// Same formula in U arithmetic with the fast exponent; equals nn.silu's gate.
template <typename U>
inline U gdnw_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

// MLX log1p (utils.h) evaluated in float.
inline float gdnw_log1p_float(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::max()) return xp1;
  if (xp1 == 1.0f) return x;
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

// float and bfloat16 inputs round the result back to their type, as MLX's
// overloads do; MLX has no half overload, so a half input keeps the float.
template <typename U>
inline U gdnw_log1p(U x) {
  return static_cast<U>(gdnw_log1p_float(float(x)));
}
inline float gdnw_log1p(half x) {
  return gdnw_log1p_float(float(x));
}

// nn.softplus(x) == LogAddExp(x, 0) (binary_ops.h).
template <typename U>
inline U gdnw_softplus(U x) {
  if (metal::isnan(x)) return metal::numeric_limits<U>::quiet_NaN();
  constexpr U inf = metal::numeric_limits<U>::infinity();
  const U zero = static_cast<U>(0);
  const U hi = metal::max(x, zero);
  const U lo = metal::min(x, zero);
  if (lo == -inf || hi == inf) return hi;
  return hi + gdnw_log1p(static_cast<U>(metal::exp(lo - hi)));
}

// Row X[row] of concat(conv window (WIN rows), slab (S rows)) at channel c.
template <typename U>
inline U gdnw_window_row(const device U* window, const device U* slab,
                         uint row, uint c, uint win, uint channels) {
  return row < win ? window[(size_t)row * channels + c]
                   : slab[(size_t)(row - win) * channels + c];
}
"""

SOURCE = r"""
  constexpr uint D = 128;
  constexpr uint QK_WIDTH = (uint)HK * D;
  constexpr uint WIDTH = 2u * QK_WIDTH + (uint)HV * D;
  constexpr uint WIN = (uint)TAPS - 1u;
  constexpr uint THREADS = 32u * (uint)RG;
  constexpr uint ROWS = D / (uint)RG;
  constexpr uint OWNED = 3u * D;     // q, k and v channels this head reads

  const uint head = threadgroup_position_in_grid.z;
  const uint key_head = head / ((uint)HV / (uint)HK);
  const uint lane = thread_index_in_simdgroup;
  const uint group = simdgroup_index_in_threadgroup;
  const uint flat = thread_index_in_threadgroup;
  const bool writes_qk = (head % ((uint)HV / (uint)HK)) == 0u;

  threadgroup float q_vec[D];
  threadgroup float k_vec[D];
  threadgroup float v_vec[D];
  threadgroup float y_vec[D];
  // 0 decay, 1 beta, 2 q inverse rms, 3 k inverse rms, 4 y inverse rms
  threadgroup float scalars[5];

  // channel of owned index: [0,D) q, [D,2D) k, [2D,3D) v
  #define GDNW_CHANNEL(idx) ((idx) < D ? key_head * D + (idx) \
      : ((idx) < 2u * D ? QK_WIDTH + key_head * D + (idx) - D \
                        : 2u * QK_WIDTH + head * D + (idx) - 2u * D))

  // Restore points and the next conv window are pure row copies.
  for (uint idx = flat; idx < OWNED; idx += THREADS) {
    if (idx < 2u * D && !writes_qk) continue;
    const uint c = GDNW_CHANNEL(idx);
    for (uint tap = 0; tap < WIN; ++tap) {
      conv_state_out[(size_t)tap * WIDTH + c] =
          gdnw_window_row(conv_state, qkv, (uint)S + tap, c, WIN, WIDTH);
      for (uint p = 1; p < (uint)S; ++p)
        conv_snapshots[((size_t)(p - 1u) * WIN + tap) * WIDTH + c] =
            gdnw_window_row(conv_state, qkv, p + tap, c, WIN, WIDTH);
    }
  }

  float state[ROWS][4];
  {
    device const float* src = recurrent_state + (size_t)head * D * D;
    for (uint j = 0; j < ROWS; ++j)
      for (uint i = 0; i < 4; ++i)
        state[j][i] = src[(size_t)(group + (uint)RG * j) * D + 4u * lane + i];
  }

  for (uint t = 0; t < (uint)S; ++t) {
    // causal conv over TAPS rows, fp32 accumulate, then SiLU in T
    for (uint idx = flat; idx < OWNED; idx += THREADS) {
      const uint c = GDNW_CHANNEL(idx);
      float acc = 0.0f;
      for (uint tap = 0; tap < (uint)TAPS; ++tap) {
        const T x = gdnw_window_row(conv_state, qkv, t + tap, c, WIN, WIDTH);
        acc += float(x) * float(conv_weight[(size_t)c * (uint)TAPS + tap]);
      }
      const T pre = static_cast<T>(acc);
      const float act = float(pre * gdnw_sigmoid_fast(pre));
      if (idx < D) q_vec[idx] = act;
      else if (idx < 2u * D) k_vec[idx - D] = act;
      else v_vec[idx - 2u * D] = act;
    }
    if (flat == 0u) {
      const GT shifted = static_cast<GT>(a[t * (uint)HV + head]) + static_cast<GT>(dt_bias[head]);
      const GT soft = gdnw_softplus(shifted);
      scalars[0] = metal::precise::exp(-metal::precise::exp(float(A_log[head])) * float(soft));
      scalars[1] = float(gdnw_sigmoid_precise(b[t * (uint)HV + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (group == 0u) {
      float q_sq = 0.0f, k_sq = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        const float qv = q_vec[4u * lane + i], kv = k_vec[4u * lane + i];
        q_sq += qv * qv;
        k_sq += kv * kv;
      }
      q_sq = simd_sum(q_sq);
      k_sq = simd_sum(k_sq);
      if (lane == 0u) {
        scalars[2] = metal::precise::rsqrt(q_sq / float(D) + 1.0e-6f);
        scalars[3] = metal::precise::rsqrt(k_sq / float(D) + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const T q_scale = T(0.0078125f);
      const T k_scale = T(0.08838834764831845f);
      for (uint d = flat; d < D; d += THREADS) {
        const T q_unit = T(1) * T(q_vec[d] * scalars[2]);
        const T k_unit = T(1) * T(k_vec[d] * scalars[3]);
        q_vec[d] = float(T(q_scale * q_unit));
        k_vec[d] = float(T(k_scale * k_unit));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      const float decay = scalars[0];
      const float beta_t = scalars[1];
      device float* dst = (t + 1u < (uint)S)
          ? state_snapshots + ((size_t)t * (uint)HV + head) * D * D
          : recurrent_state_out + (size_t)head * D * D;
      for (uint j = 0; j < ROWS; ++j) {
        const uint row = group + (uint)RG * j;
        float kv = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[j][i] = state[j][i] * decay;
          kv += state[j][i] * k_vec[4u * lane + i];
        }
        kv = simd_sum(kv);
        const float delta = (v_vec[row] - kv) * beta_t;
        float out = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[j][i] = state[j][i] + k_vec[4u * lane + i] * delta;
          out += state[j][i] * q_vec[4u * lane + i];
        }
        out = simd_sum(out);
        if (lane == 0u) y_vec[row] = float(static_cast<T>(out));
        for (uint i = 0; i < 4; ++i)
          dst[(size_t)row * D + 4u * lane + i] = state[j][i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (group == 0u) {
      float y_sq = 0.0f;
      for (uint i = 0; i < 4; ++i) y_sq += y_vec[4u * lane + i] * y_vec[4u * lane + i];
      y_sq = simd_sum(y_sq);
      if (lane == 0u) scalars[4] = metal::precise::rsqrt(y_sq / float(D) + norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = flat; d < D; d += THREADS) {
      T normed = static_cast<T>(y_vec[d] * scalars[4]);
      normed = norm_weight[d] * normed;
      const size_t at = (size_t)t * (uint)HV * D + head * D + d;
      output[at] = static_cast<T>(float(normed) * gdnw_sigmoid_precise<float>(float(z[at])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  #undef GDNW_CHANNEL
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_gdn_verify_window",
        input_names=["qkv", "z", "b", "a", "conv_state", "conv_weight", "A_log",
                     "dt_bias", "recurrent_state", "norm_weight", "norm_eps"],
        output_names=["output", "conv_state_out", "recurrent_state_out",
                      "state_snapshots", "conv_snapshots"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def gdn_verify_window(
    qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,  # noqa: N803
    recurrent_state, norm_weight, norm_eps: float, *, threadgroup_y: int,
):
    """Build the fused verify graph; run :func:`gdn_verify_window_eligible` first.

    Returns ``(output, conv_state_out, recurrent_state_out, state_snapshots,
    conv_snapshots)``.  Slot ``p`` of each snapshot array holds the state / conv
    window after ``p + 1`` tokens, for ``p`` in ``range(S - 1)``.
    """
    if threadgroup_y not in ROW_GROUPS:
        raise ValueError(f"unsupported threadgroup_y {threadgroup_y}; expected one of {ROW_GROUPS}")
    if qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[2] != CHANNELS:
        raise ValueError("unsupported GDN verify input shape")
    if qkv.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError("unsupported GDN verify input dtype")
    steps = int(qkv.shape[1])
    arrays = (qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
              recurrent_state, norm_weight)
    if not (_coefficients_ok(A_log, dt_bias) and _arrays_match(arrays, steps, qkv.dtype)):
        raise ValueError("unsupported GDN verify parameter/state layout")
    if norm_eps != 1e-6:
        raise ValueError("unsupported GDN verify norm epsilon")
    if steps not in VERIFY_WIDTHS:
        raise ValueError(f"unsupported verify width {steps}; expected one of {VERIFY_WIDTHS}")
    window = CONV_TAPS - 1
    return tuple(_kernel()(
        inputs=[qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
                recurrent_state, norm_weight, float(norm_eps)],
        template=[
            ("T", qkv.dtype),
            ("GT", mx.result_type(a.dtype, dt_bias.dtype)),
            ("HK", KEY_HEADS), ("HV", VALUE_HEADS), ("TAPS", CONV_TAPS),
            ("S", steps), ("RG", threadgroup_y),
        ],
        grid=(32, threadgroup_y, VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_WIDTH),
            (1, window, CHANNELS),
            (1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
            (1, steps - 1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
            (1, steps - 1, window, CHANNELS),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, mx.float32, qkv.dtype],
    ))
