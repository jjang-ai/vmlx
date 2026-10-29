// SPDX-License-Identifier: Apache-2.0
// flat_prep: the activation operand of the flat projections (flat_mm) for R rows (1 .. 16), one threadgroup per row.
//
// Host-prepended constants:
//   using xt = <input dtype>;  using nwt = <norm weight dtype (any type when NORM == 0)>;
//   constexpr int K = input width (multiple of 64), NORM = 1: GemmaRMSNorm first (x * rsqrt(mean(x^2) + EPS) *
//   (1 + w), all fp32), H16 = 1: also the half copy (the MXFP8 operand), constexpr float EPS.
// Inputs (names): X [R, K] xt (row-contiguous), NW [K] nwt (any >= 8-element array when NORM == 0).
// Outputs: A [R, K] bfloat16 = the (normalized) rows rounded once to bf16; A16 [R, K] half = half(A) (exact for
//   bf16 values in half's range; shape (1,) when H16 == 0); XS [max(8, R K / 64)] float (row m, group g at m K / 64 + g) = per 64-value group, the sum of
//   that group's bf16 values (lane pairs, then simd_sum: a fixed order per row).
// Grid (256, R, 1), threadgroup (256, 1, 1). Every row is computed by its own threadgroup with a fixed order.

  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint r = threadgroup_position_in_grid.y;
  const device xt* xr = X + (size_t)r * K;
  threadgroup float red[8];
  float inv = 1.0f;
  if (NORM) {
    float ss = 0.0f;
    for (uint i = tid; i < uint(K); i += 256u) {
      const float v = float(xr[i]);
      ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += red[i];
    inv = metal::precise::rsqrt(tot / float(K) + EPS);
  }
  for (uint g = sg; g < uint(K / 64); g += 8u) {
    const uint i0 = g * 64u + 2u * lane;
    float v0 = float(xr[i0]);
    float v1 = float(xr[i0 + 1u]);
    if (NORM) {
      v0 = (v0 * inv) * (1.0f + float(NW[i0]));
      v1 = (v1 * inv) * (1.0f + float(NW[i0 + 1u]));
    }
    const bfloat16_t b0 = static_cast<bfloat16_t>(v0);
    const bfloat16_t b1 = static_cast<bfloat16_t>(v1);
    A[(size_t)r * K + i0] = b0;
    A[(size_t)r * K + i0 + 1u] = b1;
    if (H16) {
      A16[(size_t)r * K + i0] = half(float(b0));
      A16[(size_t)r * K + i0 + 1u] = half(float(b1));
    }
    const float s = simd_sum(float(b0) + float(b1));
    if (lane == 0u) XS[(size_t)r * (K / 64) + g] = s;
  }
