// SPDX-License-Identifier: Apache-2.0
// E1 prep: the per-row input chunks of the multi-row projection builds (attn_qkv_e1 / attn_oproj_e1), computed ONCE
// per row with m3a_load_chunk in the projections' own block / lane order (one simdgroup per row), so the builds read
// exactly the values the one-row builds compute in their loops.
//
// Host-prepended constants:
//   using xt = <input dtype>;  using nwt = <norm weight dtype (float when NORM == 0)>;
//   constexpr int KIN = input width (a multiple of 512);  NORM = 1: the input RMSNorm's x * (1 + w) and its inverse
//   RMS (K1), 0: none (K5's o_proj input);  TWO = 1: also xb, the chunks WITHOUT the 4-bit place-value division (the
//   MXFP8 index projections' input);  constexpr float EPS (any when NORM == 0)
// Inputs (names): x [R, KIN] xt, nw [KIN] nwt (any >= 8-element array when NORM == 0).
// Outputs (RP = R rounded up to whole row tiles of the builds; rows R .. RP - 1 are written as zeros): xa [RP, KIN]
//   float (chunks divided by the 4-bit place values, as m3a_load_chunk<.., 4, ..>), xb [RP, KIN] float (TWO; else
//   shape (1,)), xs [RP, KIN / 16] float (chunk sums, chunk = block * 32 + lane), inv [max(8, RP)] float (NORM:
//   rsqrt(mean(x^2) + EPS) exactly as attn_qkv's epilogue computes it).
// Grid (32, RP, 1), threadgroup (32, 1, 1).

  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int VPT = 16;
  constexpr int BLK = 512;
  if (r >= int(x_shape[0])) {
    for (int k = 0; k < KIN; k += BLK) {
      device float4* a4 = (device float4*)(xa + (size_t)r * KIN + k + int(lane) * VPT);
      for (int i = 0; i < VPT / 4; ++i) a4[i] = float4(0.0f);
      if (TWO) {
        device float4* b4 = (device float4*)(xb + (size_t)r * KIN + k + int(lane) * VPT);
        for (int i = 0; i < VPT / 4; ++i) b4[i] = float4(0.0f);
      }
      xs[(size_t)r * (KIN / VPT) + k / VPT + int(lane)] = 0.0f;
    }
    if (NORM && lane == 0) inv[r] = 0.0f;
    return;
  }
  float ss = 0.0f;
  for (int k = 0; k < KIN; k += BLK) {
    float xq[VPT];
    float sm;
    m3a_load_chunk<xt, nwt, VPT, 4, NORM != 0>(x + (size_t)r * KIN, nw, k + int(lane) * VPT, xq, sm, ss);
    device float4* a4 = (device float4*)(xa + (size_t)r * KIN + k + int(lane) * VPT);
    for (int i = 0; i < VPT / 4; ++i) a4[i] = float4(xq[4 * i], xq[4 * i + 1], xq[4 * i + 2], xq[4 * i + 3]);
    if (TWO) {
      float xq8[VPT];
      float s8, ss8 = 0.0f;
      m3a_load_chunk<xt, nwt, VPT, 8, NORM != 0>(x + (size_t)r * KIN, nw, k + int(lane) * VPT, xq8, s8, ss8);
      device float4* b4 = (device float4*)(xb + (size_t)r * KIN + k + int(lane) * VPT);
      for (int i = 0; i < VPT / 4; ++i) b4[i] = float4(xq8[4 * i], xq8[4 * i + 1], xq8[4 * i + 2], xq8[4 * i + 3]);
    }
    xs[(size_t)r * (KIN / VPT) + k / VPT + int(lane)] = sm;
  }
  if (NORM) {
    const float iv = metal::precise::rsqrt(simd_sum(ss) / float(KIN) + EPS);
    if (lane == 0) inv[r] = iv;
  }
