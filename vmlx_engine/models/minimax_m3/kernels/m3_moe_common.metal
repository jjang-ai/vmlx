// SPDX-License-Identifier: Apache-2.0
// MiniMax-M3 / M3.1 fused MoE decode / verify kernels: shared helpers (header of K6 m3_router, K7 m3_moe_gate_up,
// K8 m3_moe_down). Loaded as the `header` of mx.fast.metal_kernel / MLXFast.metalKernel; every kernel body is in its
// own file. Row invariance: every quantity of verify row r is computed by threads (and in an order) that do not depend
// on how many rows share the launch, so a row computed alone (a decode step) and inside a 16-row verify are equal.
//
// The arithmetic follows today's unfused M3 path op for op so the fused path is bit-identical to it:
//   * jt_* : the JANGT v3 decode kernels (vmlx_engine/jangt/moe_kernels.py _gu_v3 / _down_v3): 16-weight lane
//     chunks, 12-bit trellis states, (s * MU & MASK) | ORB half2 codes, half2 fma chain per lane chunk, fp32 fold.
//   * aq4_* : MLX's one-row affine qmv (quantized.h load_vector / qdot / qmv_fast traversal, 4-bit, group 64).
//   * the router GEMV: MLX's one-row gemv (gemv.h GEMVKernel, BN = 8, SN = 32, TN = 4) and the Sigmoid functor.
// tests/test_minimax_m3_fused_moe.py checks each against the unfused ops bit for bit.

#ifndef SIMD_SIZE
#define SIMD_SIZE 32
#endif

// ------------------------------------------------------------------------------------------------ JANGT trellis
// One lane chunk = 16 weights = 8 trellis steps of 2 weights. KB2 = 2 * K (bits per weight x 2): 2 (K=1), 3 (1.5),
// 4 (2), 5 (2.5), 6 (3). New bits entering at step t >= 1: P1 (t odd) / P0 (t even) -- v2l.kpat / step_offsets.
constexpr constant uint JT_MASK2 = 2415890431u;   // MASK | MASK << 16
constexpr constant uint JT_ORB2 = 805318656u;     // ORB | ORB << 16

template <int KB2> struct jt_code;
template <> struct jt_code<2> { static constexpr constant int P0 = 2, P1 = 2; static constexpr constant uint MU = 590428815u; };
template <> struct jt_code<3> { static constexpr constant int P0 = 4, P1 = 2; static constexpr constant uint MU = 2124823417u; };
template <> struct jt_code<4> { static constexpr constant int P0 = 4, P1 = 4; static constexpr constant uint MU = 636231159u; };
template <> struct jt_code<5> { static constexpr constant int P0 = 6, P1 = 4; static constexpr constant uint MU = 636231159u; };
template <> struct jt_code<6> { static constexpr constant int P0 = 6, P1 = 6; static constexpr constant uint MU = 668265263u; };

template <int KB2>
constexpr int jt_lane_bits() { return 4 * (jt_code<KB2>::P0 + jt_code<KB2>::P1); }

template <int KB2>
constexpr int jt_step_offset(int j) { return (j / 2) * (jt_code<KB2>::P0 + jt_code<KB2>::P1) + (j & 1) * jt_code<KB2>::P1; }

// The lane's trellis window realigned into a0 / a1 (moe_kernels._window_load), then the 8 steps' half2 codes, each
// multiplied into the half2 accumulator in step order (moe_kernels._tdot3).
template <int KB2, int L>
inline half2 jt_tdot(const device uint32_t* wr, uint sh, const thread half2* xh, half2 acc) {
  constexpr int PL = jt_lane_bits<KB2>();
  constexpr int SPAN = jt_step_offset<KB2>(7) + L;
  constexpr uint MK = (1u << L) - 1u;
  constexpr uint MU = jt_code<KB2>::MU;
  uint a0, a1 = 0u;
  if (PL % 32 == 0) {
    a0 = wr[0];
    if (SPAN > 32) a1 = wr[1];
  } else if (SPAN <= 32) {
    uint w0 = wr[0], w1 = wr[1];
    a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)));
  } else {
    uint w0 = wr[0], w1 = wr[1], w2 = wr[2];
    a0 = sh == 0u ? w0 : ((w0 >> sh) | (w1 << (32u - sh)));
    a1 = sh == 0u ? w1 : ((w1 >> sh) | (w2 << (32u - sh)));
  }
#pragma clang loop unroll(full)
  for (int j = 0; j < 8; j++) {
    const int o = jt_step_offset<KB2>(j);
    uint ex;
    if (o + L <= 32) ex = o == 0 ? a0 : (a0 >> uint(o));
    else if (o >= 32) ex = o == 32 ? a1 : (a1 >> uint(o - 32));
    else ex = (a0 >> uint(o)) | (a1 << uint(32 - o));
    uint s = ex & MK;
    uint p = ((s * MU) & JT_MASK2) | JT_ORB2;
    acc = fma(xh[j], as_type<half2>(p), acc);
  }
  return acc;
}

// moe_kernels._xload3 (fp16 range guard on): 16 rotated inputs of this lane -> half2 xh[8], the guard's inverse
// scale xi, and the even / odd input sums.
inline void jt_xload(const device float* xp, thread half2* xh, thread float& xi, thread float& se, thread float& so) {
  float xt[16];
#pragma clang loop unroll(full)
  for (uint i = 0; i < 16u; i++) xt[i] = xp[i];
  float xm = 0.0f;
#pragma clang loop unroll(full)
  for (uint i = 0; i < 16u; i++) xm = metal::max(xm, metal::abs(xt[i]));
  float xs = xm > 1024.0f ? metal::exp2(-metal::ceil(metal::log2(xm * (1.0f / 1024.0f)))) : 1.0f;
  xi = 1.0f / xs;
#pragma clang loop unroll(full)
  for (uint i = 0; i < 8u; i++) {
    xh[i] = half2(half(xt[2u * i] * xs), half(xt[2u * i + 1u] * xs));
    se += xt[2u * i];
    so += xt[2u * i + 1u];
  }
}

// ------------------------------------------------------------------------------------------------ affine 4-bit qmv
// MLX quantized.h load_vector<T, float, 16, 4> and qdot<float, 16, 4> (bits = 4, values_per_thread = 16).
template <typename T>
inline float aq4_load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < 16; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline float aq4_qdot(const device uint8_t* w, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < 4; i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// ------------------------------------------------------------------------------------------------ SwiGLU-OAI
// minimax_m3.swiglu (mx.compile) on bf16 / fp16 values, op by op in the activation type T:
//   g = min(gate, limit); u = clip(up, -limit, limit); out = g * sigmoid(alpha * g) * (u + 1).
template <typename T>
inline T m3_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

template <typename T>
inline T m3_swiglu_oai(T up, T gate, float alpha, float limit) {
  T g = metal::min(gate, static_cast<T>(limit));
  T u = metal::min(up, static_cast<T>(limit));
  u = metal::max(u, static_cast<T>(-limit));
  T a = static_cast<T>(alpha) * g;
  T og = g * m3_sigmoid<T>(a);
  return og * (u + static_cast<T>(1));
}
