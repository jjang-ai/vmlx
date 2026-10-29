// SPDX-License-Identifier: Apache-2.0
// xprep: the per-row input chunks of the multi-row projection builds (attn_qkv_pre / attn_oproj_pre), computed ONCE
// per row with m3a_load_chunk in the projection kernels' own block / lane order: one simdgroup per row.
//
// Host-prepended constants:
//   using xt = <input dtype>;  using nwt = <norm weight dtype (any when NORM == 0)>;
//   constexpr int KIN = input width (a multiple of 512), NORM = 1: the input RMSNorm's x * (1 + w) and its inverse
//   RMS, BITS_A = place-value pre-division of xa (4: divided, as m3a_load_chunk does for 4-bit codes; 8: none),
//   TWO = 1: also xb, the undivided chunks (the MXFP8 index projections' input); constexpr float EPS
// Inputs (names): x [R, KIN] xt, nw [KIN] nwt (any >= 8-element array when NORM == 0).
// Outputs: xa [R, KIN] float, xb [R, KIN] float (shape (1,) when TWO == 0), xs [R, KIN / 16] float (chunk sums,
//   chunk = block * 32 + lane), inv [max(8, R)] float (rsqrt(mean(x^2) + EPS) exactly as attn_qkv's epilogue, NORM).
// Grid (32, R, 1), threadgroup (32, 1, 1).

  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int VPT = 16;
  constexpr int BLK = 512;
  float ss = 0.0f;
  for (int k = 0; k < KIN; k += BLK) {
    float xq[VPT];
    float s;
    m3a_load_chunk<xt, nwt, VPT, BITS_A, NORM != 0>(x + (size_t)r * KIN, nw, k + int(lane) * VPT, xq, s, ss);
    device float4* a4 = (device float4*)(xa + (size_t)r * KIN + k + int(lane) * VPT);
    for (int i = 0; i < VPT / 4; ++i) a4[i] = float4(xq[4 * i], xq[4 * i + 1], xq[4 * i + 2], xq[4 * i + 3]);
    if (TWO) {
      float xq8[VPT];
      float s8, ss8 = 0.0f;
      m3a_load_chunk<xt, nwt, VPT, 8, NORM != 0>(x + (size_t)r * KIN, nw, k + int(lane) * VPT, xq8, s8, ss8);
      device float4* b4 = (device float4*)(xb + (size_t)r * KIN + k + int(lane) * VPT);
      for (int i = 0; i < VPT / 4; ++i) b4[i] = float4(xq8[4 * i], xq8[4 * i + 1], xq8[4 * i + 2], xq8[4 * i + 3]);
    }
    xs[(size_t)r * (KIN / VPT) + k / VPT + int(lane)] = s;
  }
  if (NORM) {
    const float iv = metal::precise::rsqrt(simd_sum(ss) / float(KIN) + EPS);
    if (lane == 0) inv[r] = iv;
  }
