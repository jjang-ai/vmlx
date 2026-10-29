// SPDX-License-Identifier: Apache-2.0
// MiniMax-M3 "flat" dense projections (decode AND DSpark verify, 1..16 rows): header of flat_prep / flat_mm.
//
// Every projection runs ONE 16-row matrix-unit tile (Metal 4 MetalPerformancePrimitives matmul2d) per weight group,
// whatever the row count: a row's result never depends on how many rows share the launch, so a one-row decode step
// and the same row inside a 16-row verify are bit-identical, and 1..16 rows read the weights once at nearly the same
// cost. The matrix unit multiplies the activation (bf16; half for MXFP8) by the RAW integer / fp8 codes of one weight
// group; the group's scale (and affine bias, through the per-group activation sums XS) is applied in fp32 after the
// dot:  y[m, n] = sum_g s[n, g] * (A_g . Q_g)[m, n] + b[n, g] * XS[m, g].
// This is NOT MLX's qmv arithmetic (different summation order; activations rounded to bf16 as the stock unfused path
// rounds a norm output); decode and verify switch to it together (VMLX_M3_FLAT_DENSE=1).
//
// Created by Jinho Jang (eric@jangq.ai).

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// e8m0 scale byte -> float (MLX fp8_e8m0; 0 -> 2^-127).
METAL_FUNC float flat_e8m0(uint s) {
  return as_type<float>(s == 0u ? 0x400000u : (s << 23));
}

// The (row, column) of each destination element this thread holds: P's layout for a 16 x NT fp32 destination.
template <typename PT, int CAP>
METAL_FUNC void flat_layout(thread PT& P, int R, thread int* mi, thread int* ni, thread bool* live) {
  const int cap = int(P.get_capacity());
  for (int i = 0; i < CAP; ++i) {
    mi[i] = 0;
    ni[i] = 0;
    live[i] = false;
    if (i < cap && P.is_valid_element(i)) {
      auto id = P.get_multidimensional_index(i);
      ni[i] = int(id[0]);
      mi[i] = int(id[1]);
      live[i] = mi[i] < R;
    }
  }
}

// One simdgroup accumulates its weight groups [g0, g1) of an NT-column weight tile (columns n0 .. n0 + NT - 1 of
// the segment) into C (P's layout): FMT 0 affine 4-bit g64 (bf16 A, raw codes), 2 affine 8-bit g64, 3 affine 6-bit
// g64 (codes widened to bytes in `stage`, NT == 32), 1 MXFP8 (half A, fp8 e4m3 codes, e8m0 scale per 32).
// C[i] = fma(s, P[i], fma(b, XS, C[i])) per group in group order (affine); fma(s, P[i], C[i]) (MXFP8).
template <int FMT, int K, int NT, typename ST>
METAL_FUNC void flat_acc(const device bfloat16_t* A, const device half* A16, const device float* XS,
                         const device uint8_t* W, const device ST* S, const device ST* B, int n0, int R, int g0,
                         int g1, thread float* C, thread int* mi, thread int* ni, thread bool* live,
                         threadgroup uint8_t* stage, ushort lane) {
  constexpr int CAP = 16 * NT / 32;
  if (FMT == 1) {
    constexpr int GS = 32, KG = K / GS;
    constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A16, dextents<int32_t, 2>(K, R));
    tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline> tB(
        (device uchar*)W + (size_t)n0 * K, dextents<int32_t, 2>(K, NT));
    auto a0 = tA.slice(0, 0);
    auto b0 = tB.slice(0, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    flat_layout<decltype(P), CAP>(P, R, mi, ni, live);
    const device uint8_t* Sb = (const device uint8_t*)S;
    for (int g = g0; g < g1; ++g) {
      auto a = tA.slice(g * GS, 0);
      auto b = tB.slice(g * GS, 0);
      op.run(a, b, P);
      for (int i = 0; i < CAP; ++i) {
        if (live[i]) C[i] = fma(flat_e8m0(uint(Sb[(size_t)(n0 + ni[i]) * KG + g])), P[i], C[i]);
      }
    }
  } else {
    constexpr int GS = 64, KG = K / GS;
    constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)A, dextents<int32_t, 2>(K, R));
    if (FMT == 0) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB(
          (device uchar*)W + (size_t)n0 * (K / 2), dextents<int32_t, 2>(K, NT));
      auto a0 = tA.slice(0, 0);
      auto b0 = tB.slice(0, 0);
      auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
      flat_layout<decltype(P), CAP>(P, R, mi, ni, live);
      for (int g = g0; g < g1; ++g) {
        auto a = tA.slice(g * GS, 0);
        auto b = tB.slice(g * GS, 0);
        op.run(a, b, P);
        for (int i = 0; i < CAP; ++i) {
          if (live[i]) {
            const size_t sb = (size_t)(n0 + ni[i]) * KG + g;
            C[i] = fma(float(S[sb]), P[i], fma(float(B[sb]), XS[mi[i] * KG + g], C[i]));
          }
        }
      }
    } else if (FMT == 2) {
      tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> tB(
          (device uint8_t*)W + (size_t)n0 * K, dextents<int32_t, 2>(K, NT));
      auto a0 = tA.slice(0, 0);
      auto b0 = tB.slice(0, 0);
      auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
      flat_layout<decltype(P), CAP>(P, R, mi, ni, live);
      for (int g = g0; g < g1; ++g) {
        auto a = tA.slice(g * GS, 0);
        auto b = tB.slice(g * GS, 0);
        op.run(a, b, P);
        for (int i = 0; i < CAP; ++i) {
          if (live[i]) {
            const size_t sb = (size_t)(n0 + ni[i]) * KG + g;
            C[i] = fma(float(S[sb]), P[i], fma(float(B[sb]), XS[mi[i] * KG + g], C[i]));
          }
        }
      }
    } else {
      static_assert(FMT != 3 || NT == 32, "6-bit: one column per lane");
      tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> tB(stage, dextents<int32_t, 2>(GS, NT));
      auto a0 = tA.slice(0, 0);
      auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB), float>();
      flat_layout<decltype(P), CAP>(P, R, mi, ni, live);
      // 6-bit codes: value k of a row at bits [6k, 6k + 6) of its little-endian byte stream (MLX packing)
      const device uint8_t* wrow = W + (size_t)(n0 + int(lane)) * (K * 6 / 8);
      threadgroup uint* st32 = (threadgroup uint*)stage + lane * (GS / 4);
      for (int g = g0; g < g1; ++g) {
        const device uint8_t* src = wrow + (size_t)g * (GS * 6 / 8);
        for (int t = 0; t < GS / 4; ++t) {
          const uint b0_ = src[3 * t], b1_ = src[3 * t + 1], b2_ = src[3 * t + 2];
          const uint v0 = b0_ & 63u, v1 = (b0_ >> 6) | ((b1_ & 15u) << 2), v2 = (b1_ >> 4) | ((b2_ & 3u) << 4);
          const uint v3 = b2_ >> 2;
          st32[t] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tA.slice(g * GS, 0);
        op.run(a, tB, P);
        simdgroup_barrier(mem_flags::mem_threadgroup);  // the op has read the stage before the next widening
        for (int i = 0; i < CAP; ++i) {
          if (live[i]) {
            const size_t sb = (size_t)(n0 + ni[i]) * KG + g;
            C[i] = fma(float(S[sb]), P[i], fma(float(B[sb]), XS[mi[i] * KG + g], C[i]));
          }
        }
      }
    }
  }
}
