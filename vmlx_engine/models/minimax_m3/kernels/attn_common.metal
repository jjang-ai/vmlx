// SPDX-License-Identifier: Apache-2.0
// MiniMax-M3 fused attention (decode / DSpark verify, 1..16 rows): helpers shared by the kernel bodies.
//
// This file is the `header` of every attn_*.metal kernel (MLX mx.fast.metal_kernel / MLXFast.metalKernel). Each body
// file is a kernel BODY: the host prepends `using` / `constexpr` lines (listed at the top of each body file) and lets
// MLX generate the signature from the input / output names. Python: vmlx_engine/models/minimax_m3/kernels/__init__.py.
//
// Row invariance (all kernels): every output row is computed by code whose arithmetic does not depend on how many
// rows share the launch or on which slot the row occupies, so a row computed alone (a decode step) is bit-identical
// to the same row inside a 16-row DSpark verify. The projections (K1, K5) put row tiles on the grid: a threadgroup
// computes one (weight tile, RT-row tile) pair, the row tiles of a weight tile adjacent so the tile's repeated reads
// hit the GPU's shared cache; one row runs RT = 1, several rows RT = 2, and an output element's arithmetic does not
// depend on RT or on the row's slot (m3a_rowsN).
//
// Created by Jinho Jang (eric@jangq.ai).

#include <metal_stdlib>
using namespace metal;

// fp8 e4m3 byte -> float, exact (MLX fp8_e4m3: shift into an fp16, rebias by 2^8).
METAL_FUNC float m3a_fp8(uint b) {
  half h = as_type<half>(ushort((b & 127u) << 7));
  h *= half(256.0h);
  return float((b & 128u) ? -h : h);
}

// e8m0 scale byte -> float (MLX fp8_e8m0; 0 -> 2^-127).
METAL_FUNC float m3a_e8m0(uint s) {
  return as_type<float>(s == 0u ? 0x400000u : (s << 23));
}

// Exclusive prefix sum of v over the threads of a threadgroup of NSG simdgroups, in thread order; total = the sum.
// Contains threadgroup barriers (every thread calls it); scratch holds NSG values.
template <int NSG>
METAL_FUNC uint m3a_tg_exclusive_scan(uint v, uint lane, int sg, threadgroup uint* scratch, thread uint& total) {
  const uint excl = simd_prefix_exclusive_sum(v);
  const uint sgsum = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = sgsum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = 0u;
  total = 0u;
  for (int i = 0; i < NSG; ++i) {
    const uint t = scratch[i];
    before += i < sg ? t : 0u;
    total += t;
  }
  return before + excl;
}

// Order-preserving uint key of a float (larger float -> larger key; NaN above +inf).
METAL_FUNC uint m3a_fkey(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ------------------------------------------------------------------------------------------- projection chunks
// A lane owns VPT consecutive inputs of every BLK = 32 * VPT wide block of the input (16 for affine 4-bit and MXFP8,
// 8 for affine 8-bit). For one row, one block:
//   m3a_load_chunk: x' = x * (1 + nw) (NORM; fp32) or x; ss += x^2 (raw x, NORM); xs = sum(x') in order; xp = x'
//     pre-divided by the 4-bit code's place value (MLX's trick: the codes are masked, not shifted; exact).
//   m3a_codes / m3a_fp8_codes: the lane's VPT codes of one output row as floats.
//   m3a_dot: dot(xp, codes), four products per partial sum, partial sums added in order.
// An output element accumulates res += scale * dot + bias * xs (affine) or res += scale * dot (MXFP8) over the
// blocks, then simd_sum over the lanes.
template <typename XT, typename NWT, int VPT, int BITS, bool NORM>
METAL_FUNC void m3a_load_chunk(const device XT* xr, const device NWT* nw, int xo, thread float* xp, thread float& xs,
                               thread float& ss) {
  xs = 0.0f;
  for (int i = 0; i < VPT; ++i) {
    const float xv = float(xr[xo + i]);
    if (NORM) ss = fma(xv, xv, ss);
    const float xn = NORM ? xv * (1.0f + float(nw[xo + i])) : xv;
    xs += xn;
    if (BITS == 4) {
      const int sh = i & 3;
      xp[i] = sh == 0 ? xn : (sh == 1 ? xn * 0.0625f : (sh == 2 ? xn * 0.00390625f : xn * 0.000244140625f));
    } else {
      xp[i] = xn;
    }
  }
}

template <int BITS, int VPT>
METAL_FUNC void m3a_codes(const device uint8_t* wo, thread float* wf) {
  const uint2 wv = *((const device uint2*)wo);
  if (BITS == 4) {
    for (int i = 0; i < 4; ++i) {
      const uint h16 = (i < 2 ? (wv.x >> (16 * i)) : (wv.y >> (16 * (i - 2)))) & 0xffffu;
      wf[4 * i + 0] = float(h16 & 0x000fu);
      wf[4 * i + 1] = float(h16 & 0x00f0u);
      wf[4 * i + 2] = float(h16 & 0x0f00u);
      wf[4 * i + 3] = float(h16 & 0xf000u);
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      wf[i] = float((wv.x >> (8 * i)) & 0xffu);
      wf[4 + i] = float((wv.y >> (8 * i)) & 0xffu);
    }
  }
}

METAL_FUNC void m3a_fp8_codes(const device uint8_t* wo, thread float* wf) {
  const uint4 wv = *((const device uint4*)wo);
  for (int i = 0; i < 4; ++i) {
    wf[i] = m3a_fp8((wv.x >> (8 * i)) & 0xffu);
    wf[4 + i] = m3a_fp8((wv.y >> (8 * i)) & 0xffu);
    wf[8 + i] = m3a_fp8((wv.z >> (8 * i)) & 0xffu);
    wf[12 + i] = m3a_fp8((wv.w >> (8 * i)) & 0xffu);
  }
}

template <int VPT>
METAL_FUNC float m3a_dot(const thread float* xp, const thread float* wf) {
  float d = 0.0f;
  for (int i = 0; i < VPT; i += 4) {
    d += xp[i] * wf[i] + xp[i + 1] * wf[i + 1] + xp[i + 2] * wf[i + 2] + xp[i + 3] * wf[i + 3];
  }
  return d;
}

// ------------------------------------------------------------------------------------ RT rows, registers
// One simdgroup, RPS output rows starting at local row lrow (FMT 0 = affine BITS / GS with scales + biases of type
// ST, 1 = MXFP8), rows r0 .. r0 + RT - 1 of x ([*, KIN] row-contiguous; rows >= R skipped): each lane's chunk of
// every row in registers, every output row's codes dequantized once for the RT rows. An output element's arithmetic
// is the same sequence for every RT and every slot: per block m3a_load_chunk, m3a_codes / m3a_fp8_codes, m3a_dot,
// res += s * dot + b * xs (affine) or res += s * dot (MXFP8); res / ss are this lane's partials (the caller
// simd_sum's them).
template <typename XT, typename NWT, typename ST, int FMT, int BITS, int GS, int KIN, int RPS, int RT, bool NORM>
METAL_FUNC void m3a_rowsN(const device XT* x, const device NWT* nw, const device uint8_t* w, const device ST* sc,
                          const device ST* bi, int lrow, int r0, int R, uint lane, thread float (&res)[RT][RPS],
                          thread float (&ss)[RT]) {
  constexpr int VPT = (FMT == 1 || BITS == 4) ? 16 : 8;
  constexpr int BLK = VPT * 32;
  constexpr int ROWB = FMT == 1 ? KIN : KIN * BITS / 8;
  constexpr int NG = KIN / (FMT == 1 ? 32 : GS);
  static_assert(KIN % BLK == 0, "shape");
  for (int t = 0; t < RT; ++t) {
    ss[t] = 0.0f;
    for (int o = 0; o < RPS; ++o) res[t][o] = 0.0f;
  }
  const device uint8_t* wl = w + (size_t)lrow * ROWB + lane * (VPT * (FMT == 1 ? 8 : BITS) / 8);
  const int gl = (int(lane) * VPT) / (FMT == 1 ? 32 : GS);
  for (int k = 0; k < KIN; k += BLK) {
    float xp[RT][VPT];
    float xs[RT];
    for (int t = 0; t < RT; ++t) {
      if (r0 + t < R) {
        m3a_load_chunk<XT, NWT, VPT, (FMT == 1 ? 8 : BITS), NORM>(x + (size_t)(r0 + t) * KIN, nw, k + int(lane) * VPT,
                                                                  xp[t], xs[t], ss[t]);
      }
    }
    const int gi = k / (FMT == 1 ? 32 : GS) + gl;
    for (int o = 0; o < RPS; ++o) {
      float wf[VPT];
      float s, b = 0.0f;
      if (FMT == 1) {
        m3a_fp8_codes(wl + (size_t)o * ROWB + k, wf);
        s = m3a_e8m0(uint(((const device uint8_t*)sc)[(size_t)(lrow + o) * NG + gi]));
      } else {
        m3a_codes<BITS, VPT>(wl + (size_t)o * ROWB + k * BITS / 8, wf);
        s = float(sc[(size_t)(lrow + o) * NG + gi]);
        b = float(bi[(size_t)(lrow + o) * NG + gi]);
      }
      for (int t = 0; t < RT; ++t) {
        if (r0 + t < R) {
          if (FMT == 1) {
            res[t][o] += s * m3a_dot<VPT>(xp[t], wf);
          } else {
            res[t][o] += s * m3a_dot<VPT>(xp[t], wf) + b * xs[t];
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------ E1: multi-row builds, codes decoded once per tile
// DSpark verify windows of 2 .. 16 rows (VMLX_M3_EXACT_FLAT, fused_attn.py). The one-row builds above are untouched;
// these compute every output element with exactly their per-element sequence, so a verify row equals its decode step
// bit for bit, while each weight word is read and decoded ONCE per tile of TR rows instead of once per 2-row tile:
//   * attn_xprep runs m3a_load_chunk once per (row, block, lane) and stores x' (xp), the chunk sums (xs) and the
//     inverse RMS (K1) in the projections' own block / lane order;
//   * m3e_rows walks the weights: per block and output row, its 16 codes are decoded one four-value group at a time
//     into half registers (4-bit codes times their place value and fp8 e4m3 values are exact in half; float(half) is
//     the float m3a_codes / m3a_fp8_codes give) and applied to every row of the tile with m3a_dot's statement
//     d += x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3, d starting at 0 per block; then res += s * d + b * xs (affine) or
//     res += s * d (MXFP8) per block, simd_sum per output -- the m3a_rowsN sequence, element by element.
// Registers bound the tile: TR rows x O outputs of d and res per lane (4 x 6 / 4 x 4 measured best on the M5 Max;
// 4 x 8 / 8 x 4 spill and run 2x slower).

// Four codes (group gi4 of a lane's 16) of one output row: affine 4-bit from its uint2 (wv[0]), MXFP8 from its 16
// bytes (wv[0], wv[1]).
template <int FMT>
METAL_FUNC void m3e_codes4(const thread uint2* wv, int gi4, thread half* c) {
  if (FMT == 1) {
    const uint2 w2 = wv[gi4 >> 1];
    const uint wd = (gi4 & 1) ? w2.y : w2.x;
    for (int j = 0; j < 4; ++j) {
      const uint bb = (wd >> (8 * j)) & 0xffu;
      half h = as_type<half>(ushort((bb & 127u) << 7));
      h *= half(256.0h);
      c[j] = (bb & 128u) ? -h : h;
    }
  } else {
    const uint h16 = (gi4 < 2 ? (wv[0].x >> (16 * gi4)) : (wv[0].y >> (16 * (gi4 - 2)))) & 0xffffu;
    c[0] = half(h16 & 0x000fu);
    c[1] = half(h16 & 0x00f0u);
    c[2] = half(h16 & 0x0f00u);
    c[3] = half(h16 & 0xf000u);
  }
}

// Partials res[TR][O] (this lane's, before simd_sum) of output rows lrow .. lrow + O - 1 for rows r0 .. r0 + TR - 1 of
// the prepared input (xp [*, KIN] f32, xs [*, KIN / 16] f32, padded by attn_xprep to whole tiles: every tile row is
// computed, the caller stores the valid ones; a per-row guard here cost 10-15 %). FMT 0: affine 4-bit, group GS,
// scales + biases ST; FMT 1: MXFP8 (e8m0 scale bytes per 32, no bias).
template <typename ST, int FMT, int GS, int KIN, int O, int TR>
METAL_FUNC void m3e_rows(const device float* xp, const device float* xsb, const device uint8_t* w,
                         const device ST* sc, const device ST* bi, int lrow, int r0, uint lane,
                         thread float (&res)[TR][O]) {
  constexpr int VPT = 16;
  constexpr int BLK = 512;
  constexpr int ROWB = FMT == 1 ? KIN : KIN / 2;
  constexpr int NG = KIN / (FMT == 1 ? 32 : GS);
  static_assert(KIN % BLK == 0, "shape");
  for (int t = 0; t < TR; ++t)
    for (int o = 0; o < O; ++o) res[t][o] = 0.0f;
  const device uint8_t* wl = w + (size_t)lrow * ROWB + lane * (FMT == 1 ? 16 : 8);
  const int gl = (int(lane) * VPT) / (FMT == 1 ? 32 : GS);
  const device float* xr0 = xp + (size_t)r0 * KIN + int(lane) * VPT;
  for (int k = 0; k < KIN; k += BLK) {
    const int gi = k / (FMT == 1 ? 32 : GS) + gl;
    constexpr int WN = FMT == 1 ? 2 : 1;
    uint2 wv[O][WN];
    float s[O], b[O];
    for (int o = 0; o < O; ++o) {
      if (FMT == 1) {
        const uint4 v4 = *((const device uint4*)(wl + (size_t)o * ROWB + k));
        wv[o][0] = v4.xy;
        wv[o][WN - 1] = v4.zw;
        s[o] = m3a_e8m0(uint(((const device uint8_t*)sc)[(size_t)(lrow + o) * NG + gi]));
        b[o] = 0.0f;
      } else {
        wv[o][0] = *((const device uint2*)(wl + (size_t)o * ROWB + k / 2));
        s[o] = float(sc[(size_t)(lrow + o) * NG + gi]);
        b[o] = float(bi[(size_t)(lrow + o) * NG + gi]);
      }
    }
    float d[TR][O];
    for (int t = 0; t < TR; ++t)
      for (int o = 0; o < O; ++o) d[t][o] = 0.0f;
    for (int i = 0; i < 4; ++i) {
      half c[O][4];
      for (int o = 0; o < O; ++o) m3e_codes4<FMT>(wv[o], i, c[o]);
      for (int t = 0; t < TR; ++t) {
        const float4 x4 = ((const device float4*)(xr0 + (size_t)t * KIN + k))[i];
        for (int o = 0; o < O; ++o) {
          d[t][o] += x4.x * float(c[o][0]) + x4.y * float(c[o][1]) + x4.z * float(c[o][2]) + x4.w * float(c[o][3]);
        }
      }
    }
    for (int t = 0; t < TR; ++t) {
      if (FMT == 1) {
        for (int o = 0; o < O; ++o) res[t][o] += s[o] * d[t][o];
      } else {
        const float xs = xsb[(size_t)(r0 + t) * (KIN / VPT) + k / VPT + int(lane)];
        for (int o = 0; o < O; ++o) res[t][o] += s[o] * d[t][o] + b[o] * xs;
      }
    }
  }
}
