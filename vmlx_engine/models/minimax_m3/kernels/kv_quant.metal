// SPDX-License-Identifier: Apache-2.0
// MiniMax-M3 KV cache formats: packed-row encode (one simdgroup per row) and decode helpers.
//
// Appended to attn_common.metal to form the header of every attn_* / kvq_* kernel (kernels/__init__.py header()).
//
// A quantized cache lane holds one PACKED ROW of RB bytes per (token, head): the codes, then the scales. A token's
// row depends on that token's D values only (groups run along the head dim), so a row quantized alone (a decode step)
// is bit-identical to the same row quantized inside a prefill chunk or a DSpark verify, and slicing / trimming /
// concatenating lanes along the token axis never touches a row's bytes.
//
//   F  name     codes                        scales (after the codes)                                   RB (D = 128)
//   0  bf16     (unquantized: the lane holds the nominal dtype)                                          -
//   1  mxfp8    D x e4m3 bytes               D / 32 e8m0 bytes: 2^n, n the smallest with amax 2^-n <= 448 132
//                                            (no element ever saturates)
//   2  nvfp4    D / 2 bytes, 2 e2m1 codes    D / 16 e4m3 block scales, then 1 row exponent byte k + 127:   76
//               (dim 2j low nibble)          value = e2m1 * s_block * 2^k, k chosen per row so the largest
//                                            block scale lands in [224, 448] (e4m3 normal range for every
//                                            block within 2^14 of the row max); padded to 4 bytes
//   3  affine8  D bytes                      D / 64 bf16 scales, then D / 64 bf16 biases: value =         136
//                                            bf16(fma(s, q, b)), q in 0 .. 255
//   4  affine4  D / 2 bytes (dim 2j low)     same, q in 0 .. 15                                            72
//
// Rounding: e4m3 / e2m1 codes round to nearest, ties to even; mxfp8 / nvfp4 decode exactly (a code times a power of
// two, or times an e4m3 scale, fits a bf16), affine decodes round once to the nominal dtype. Divisions use
// metal::precise::divide so the encoders do not depend on fast-math reciprocal choices. RB is a multiple of 4.
//
// Created by Jinho Jang (eric@jangq.ai).

constexpr int m3q_code_bytes(int F, int D) { return (F == 1 || F == 3) ? D : D / 2; }
constexpr int m3q_meta_bytes(int F, int D) {
  return F == 1 ? D / 32 : (F == 2 ? D / 16 + 1 : 4 * (D / 64));
}
constexpr int m3q_row_bytes(int F, int D) {
  return F == 0 ? 0 : ((m3q_code_bytes(F, D) + m3q_meta_bytes(F, D) + 3) / 4) * 4;
}

// 2^i as a float, i in [-126, 127] (a normal number).
METAL_FUNC float m3q_pow2(int i) { return as_type<float>(uint(i + 127) << 23); }

// float -> e4m3 byte, round to nearest even, |x| >= 448 saturates to 448, NaN -> 0x7f.
METAL_FUNC uint m3q_e4m3(float x) {
  const uint u = as_type<uint>(x);
  const uint s = (u >> 24) & 0x80u;
  const uint a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return s | 0x7fu;
  if (a >= 0x43e00000u) return s | 0x7eu;                     // 448
  if (a < 0x3c800000u) {                                      // below 2^-6: subnormal codes m * 2^-9 (m = 8 is 2^-6)
    return s | uint(metal::rint(as_type<float>(a) * 512.0f));
  }
  const uint mant = a & 0x7fffffu;
  uint q = (((a >> 23) - 120u) << 3) | (mant >> 20);         // exponent rebiased 127 -> 7, top 3 mantissa bits
  const uint rest = mant & 0xfffffu;
  q += (rest > 0x80000u || (rest == 0x80000u && (q & 1u))) ? 1u : 0u;
  return s | min(q, 0x7eu);
}

// float -> e2m1 nibble (grid 0, 0.5, 1, 1.5, 2, 3, 4, 6), round to nearest even, saturating at 6.
METAL_FUNC uint m3q_e2m1(float x) {
  const uint s = (as_type<uint>(x) >> 28) & 0x8u;
  const float a = metal::fabs(x);
  uint m;
  if (a > 5.0f) m = 7u;
  else if (a >= 3.5f) m = 6u;
  else if (a > 2.5f) m = 5u;
  else if (a >= 1.75f) m = 4u;
  else if (a > 1.25f) m = 3u;
  else if (a >= 0.75f) m = 2u;
  else if (a > 0.25f) m = 1u;
  else m = 0u;
  return s | m;
}

// e2m1 nibble -> float (exact).
METAL_FUNC float m3q_fp4(uint c) {
  const uint e = (c >> 1) & 3u;
  const float h = (c & 1u) ? 1.5f : 1.0f;
  const float v = e == 0u ? ((c & 1u) ? 0.5f : 0.0f) : h * float(1u << (e - 1u));
  return (c & 8u) ? -v : v;
}

// mxfp8 group exponent: the smallest n in [-127, 127] with amax * 2^-n <= 448 (amax 0 / NaN -> -127).
METAL_FUNC int m3q_mx_exp(float amax) {
  if (!(amax > 0.0f)) return -127;
  int e = int((as_type<uint>(amax) >> 23) & 0xffu) - 127;
  e = min(e, 127);
  int n = e - 8;
  if (n < -127) return -127;                                  // amax < 2^-118: amax * 2^127 < 512, and below 448
  if (amax * m3q_pow2(-n) > 448.0f) n += 1;
  return min(n, 127);
}

// nvfp4 row exponent: the smallest k in [-126, 127] with ramax * 2^-k <= 6 * 448.
METAL_FUNC int m3q_nv_exp(float ramax) {
  if (!(ramax > 0.0f)) return -126;
  int e = int((as_type<uint>(ramax) >> 23) & 0xffu) - 127;
  e = min(e, 127);
  int k = e - 11;
  if (k < -126) return -126;                                  // ramax * 2^126 < 2048
  if (ramax * m3q_pow2(-k) > 2688.0f) k += 1;
  return min(k, 127);
}

// Encode one row held in the simdgroup layout (lane l holds v[e] = dim l + 32 e, e < D / 32) into row[0 .. RB).
// Every lane of the simdgroup must call it (simd reductions). Writes every byte of the row.
template <int F, int D>
METAL_FUNC void m3q_store_row(thread float* v, uint lane, device uint8_t* row) {
  constexpr int DPL = D / 32;
  constexpr int CB = m3q_code_bytes(F, D);
  constexpr int RB = m3q_row_bytes(F, D);
  if constexpr (F == 1) {
    for (int e = 0; e < DPL; ++e) {
      const int n = m3q_mx_exp(simd_max(metal::fabs(v[e])));
      row[lane + 32 * e] = uint8_t(m3q_e4m3(v[e] * m3q_pow2(-n)));
      if (lane == 0) row[CB + e] = uint8_t(n + 127);
    }
    for (int i = CB + DPL + int(lane); i < RB; i += 32) row[i] = 0;
  } else if constexpr (F == 2) {
    float am = 0.0f;
    for (int e = 0; e < DPL; ++e) am = metal::max(am, metal::fabs(v[e]));
    const int k = m3q_nv_exp(simd_max(am));
    const float ik = m3q_pow2(-k);
    for (int e = 0; e < DPL; ++e) {
      const float x = v[e] * ik;
      float bm = metal::fabs(x);
      bm = metal::max(bm, simd_shuffle_xor(bm, ushort(8)));
      bm = metal::max(bm, simd_shuffle_xor(bm, ushort(4)));
      bm = metal::max(bm, simd_shuffle_xor(bm, ushort(2)));
      bm = metal::max(bm, simd_shuffle_xor(bm, ushort(1)));
      const uint sb = m3q_e4m3(metal::precise::divide(bm, 6.0f));
      const float s = m3a_fp8(sb);
      const uint c = s > 0.0f ? m3q_e2m1(metal::precise::divide(x, s)) : 0u;
      const uint hi = simd_shuffle_down(c, ushort(1));
      if ((lane & 1u) == 0u) row[(lane + 32 * e) / 2] = uint8_t(c | (hi << 4));
      if ((lane & 15u) == 0u) row[CB + 2 * e + int(lane / 16u)] = uint8_t(sb);
    }
    if (lane == 0) row[CB + D / 16] = uint8_t(k + 127);
    for (int i = CB + D / 16 + 1 + int(lane); i < RB; i += 32) row[i] = 0;
  } else {
    constexpr int LV = F == 3 ? 255 : 15;
    constexpr int NG = D / 64;
    for (int g = 0; g < NG; ++g) {
      const float mn = simd_min(metal::min(v[2 * g], v[2 * g + 1]));
      const float mxv = simd_max(metal::max(v[2 * g], v[2 * g + 1]));
      const bfloat16_t sb = bfloat16_t(metal::precise::divide(mxv - mn, float(LV)));
      const bfloat16_t bb = bfloat16_t(mn);
      const float s = float(sb), b = float(bb);
      for (int h = 0; h < 2; ++h) {
        const int e = 2 * g + h;
        uint q = 0u;
        if (s > 0.0f) q = uint(metal::clamp(metal::rint(metal::precise::divide(v[e] - b, s)), 0.0f, float(LV)));
        if constexpr (F == 3) {
          row[lane + 32 * e] = uint8_t(q);
        } else {
          const uint hi = simd_shuffle_down(q, ushort(1));
          if ((lane & 1u) == 0u) row[(lane + 32 * e) / 2] = uint8_t(q | (hi << 4));
        }
      }
      if (lane == 0) {
        const ushort sbits = as_type<ushort>(sb), bbits = as_type<ushort>(bb);
        row[CB + 2 * g] = uint8_t(sbits & 255u);
        row[CB + 2 * g + 1] = uint8_t(sbits >> 8);
        row[CB + 2 * NG + 2 * g] = uint8_t(bbits & 255u);
        row[CB + 2 * NG + 2 * g + 1] = uint8_t(bbits >> 8);
      }
    }
    for (int i = CB + 4 * NG + int(lane); i < RB; i += 32) row[i] = 0;
  }
}

// N (2, 4 or 8) code bytes (8-bit codes) or N nibbles (4-bit codes) starting at byte p, as one little-endian word
// (p aligned to the load: 8-bit codes 4 / 2 bytes, 4-bit codes N / 2 bytes).
template <int N, bool NIBBLES>
METAL_FUNC uint m3q_word(const device uint8_t* p, int w) {
  if constexpr (NIBBLES) {
    if constexpr (N == 8) return *((const device uint*)p);
    else if constexpr (N == 4) return uint(*((const device ushort*)p));
    else return uint(*p);
  } else {
    if constexpr (N >= 4) return *((const device uint*)(p + 4 * w));
    else return uint(*((const device ushort*)p));
  }
}

// Decode N (2, 4 or 8) consecutive dims d0 .. d0 + N - 1 of a packed row (d0 a multiple of N; the row 4-byte
// aligned) into float values rounded to the nominal type TN.
template <int F, int D, typename TN, int N>
METAL_FUNC void m3q_load(const device uint8_t* row, int d0, thread float* out) {
  constexpr int CB = m3q_code_bytes(F, D);
  constexpr int W = N >= 4 ? 4 : N;                         // 8-bit codes per word
  if constexpr (F == 1) {
    const float s = m3a_e8m0(uint(row[CB + d0 / 32]));
    for (int w = 0; w < N / W; ++w) {
      const uint c = m3q_word<N, false>(row + d0, w);
      for (int i = 0; i < W; ++i) out[W * w + i] = float(TN(m3a_fp8((c >> (8 * i)) & 255u) * s));
    }
  } else if constexpr (F == 2) {
    const float s = m3a_fp8(uint(row[CB + d0 / 16])) * m3q_pow2(int(row[CB + D / 16]) - 127);
    const uint c = m3q_word<N, true>(row + d0 / 2, 0);
    for (int i = 0; i < N; ++i) out[i] = float(TN(m3q_fp4((c >> (4 * i)) & 15u) * s));
  } else {
    constexpr int NG = D / 64;
    const int g = d0 / 64;
    const float s = float(as_type<bfloat16_t>(*((const device ushort*)(row + CB + 2 * g))));
    const float b = float(as_type<bfloat16_t>(*((const device ushort*)(row + CB + 2 * NG + 2 * g))));
    if constexpr (F == 3) {
      for (int w = 0; w < N / W; ++w) {
        const uint c = m3q_word<N, false>(row + d0, w);
        for (int i = 0; i < W; ++i) out[W * w + i] = float(TN(metal::fma(s, float((c >> (8 * i)) & 255u), b)));
      }
    } else {
      const uint c = m3q_word<N, true>(row + d0 / 2, 0);
      for (int i = 0; i < N; ++i) out[i] = float(TN(metal::fma(s, float((c >> (4 * i)) & 15u), b)));
    }
  }
}
