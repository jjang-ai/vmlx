// SPDX-License-Identifier: Apache-2.0
// K7 m3_moe_gate_up: the routed experts' (JANGT) gate / up + SwiGLU-OAI of every (row, slot), AND the shared
// expert's (affine 4-bit, group 64) gate / up + SwiGLU-OAI of every row, in one launch. Also picks the routing:
// top-KS of sigmoid + correction bias per row (group_expert_select, n_group = topk_group = 1), normalized and scaled.
//
// Template: T (activation dtype), D (hidden = gate/up input), I (expert intermediate), E (experts), ROWS (1..16),
//   KS (experts per token), KB2 (2 x JANGT bits per weight of gate/up: 2 / 3 / 4 / 5 / 6), L (trellis state bits,
//   12), W (uint32 words per gate/up row), NOUT (gate/up outlier columns, 0 = none), ROUTED (1: the routed experts'
//   gate / up here; 0: only the routing -- inds / scores -- and the shared expert, h not written), SHARED (1: this launch also
//   computes the shared expert's gate / up; 0: the caller runs it, hs is not written), ALPHA / LIMIT / RSF (the
//   SwiGLU-OAI alpha 1.702, limit 7.0 and the routed scaling factor 2.0 as float32 bit patterns, read with
//   as_type<float>: compile-time constants, as the v3 kernels' literals -- the compiler folds alpha into fast::exp, so
//   a runtime alpha rounds differently).
// Inputs:  fc [16] f32 = (mu0, mu1, isd0, isd1: moe_kernels code_stats of KB2; zero padding), xr [ROWS, D] f32 (K6), x [ROWS, D] T, sig [ROWS, E] f32 (K6), bias [E] f32 (e_score_correction_bias),
//   wg / wu [E, I, W] u32, sg / su_ [E, I] f32 (JANGT gate / up), ocol [E, max(NOUT,1)] i32,
//   woutg / woutu [E, I, max(NOUT,1)] f16, sgw / suw [I, D / 8] u32, sgs / sgb / sus / sub [I, D / 64] T (shared).
// Outputs: h [ROWS * KS, I] f32 (routed SwiGLU-OAI, slot t * KS + j), hs [ROWS, I] T (shared SwiGLU-OAI),
//   inds [ROWS, KS] u32, scores [ROWS, KS] f32 (written by the row tile 0 threadgroups).
// Grid: (64, ROWS * KS + SHARED * ROWS, I / 8), threadgroup (64, 1, 1): 2 simdgroups x 4 output rows of tile
//   grid.z; grid.y = z < ROWS * KS is the (row, slot) z of the routed part (moe_kernels._gu_v3 per slot, bit for bit),
//   z = ROWS * KS + t the shared expert of row t (MLX one-row qmv_fast per output row, then MLX's bf16 rounding and
//   the compiled SwiGLU op by op). The slot is the faster grid axis (as v3's slot_fastest): consecutive threadgroups
//   read the same weight tile for different slots, so experts picked by several rows come from cache.

  const uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint row0 = threadgroup_position_in_grid.z * 8u + sgi * 4u;
  const uint z = threadgroup_position_in_grid.y;
  const float alpha = as_type<float>(ALPHA), limit = as_type<float>(LIMIT);
  if (z < uint(ROWS * KS)) {
    // ---- routing of row t (stable descending order of sigmoid + bias: argpartition's order)
    const uint t = z / uint(KS), slot = z % uint(KS);
    constexpr int PER = (E + 31) / 32;
    float cv[PER];
    for (int i = 0; i < PER; i++) {
      const uint c = lane + 32u * uint(i);
      cv[i] = c < uint(E) ? sig[(size_t)t * E + c] + bias[c] : -INFINITY;
    }
    // KS rounds of a simdgroup arg-max (ties to the lower expert id); every lane keeps the picks it needs as scalars
    // (no dynamically indexed thread arrays: the slot's expert, its sigmoid, and the running denominator).
    uint e = 0u; float o_slot = 0.0f, den = 0.0f;
    for (int j = 0; j < KS; j++) {
      float bv = -INFINITY; uint bi = 0xffffffffu;
      for (int i = 0; i < PER; i++) {
        const uint c = lane + 32u * uint(i);
        if (c < uint(E) && (cv[i] > bv || (cv[i] == bv && c < bi))) { bv = cv[i]; bi = c; }
      }
      for (ushort d = 16; d >= 1; d >>= 1) {
        float ov = simd_shuffle_xor(bv, d); uint oi = simd_shuffle_xor(bi, d);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      const float oj = sig[(size_t)t * E + bi];
      den = j == 0 ? oj : den + oj;                 // scores.sum(-1): ((o0 + o1) + o2) + o3
      if (uint(j) == slot) { e = bi; o_slot = oj; }
      for (int i = 0; i < PER; i++) if (lane + 32u * uint(i) == bi) cv[i] = -INFINITY;
    }
    if (threadgroup_position_in_grid.z == 0u && sgi == 0u && lane == 0u) {
      inds[(size_t)t * KS + slot] = e;
      float sc = o_slot / den;
      scores[(size_t)t * KS + slot] = sc * as_type<float>(RSF);
    }
    if (ROUTED == 0) return;                   // the routed experts run elsewhere (jangt fast_decode); routing only
    // ---- moe_kernels._gu_v3 for slot z (rot_out off, out float)
    constexpr int PL = jt_lane_bits<KB2>();
    const uint lb = lane * uint(PL); const uint sh = lb & 31u;
    const device uint32_t* wp = wg + ((size_t)e * I + row0) * W + (lb >> 5);
    const device uint32_t* up = wu + ((size_t)e * I + row0) * W + (lb >> 5);
    const device float* xp = xr + (size_t)t * D + lane * 16u;
    float g0[4], g1[4], u0[4], u1[4];
#pragma clang loop unroll(full)
    for (uint r = 0; r < 4u; r++) { g0[r] = 0.0f; g1[r] = 0.0f; u0[r] = 0.0f; u1[r] = 0.0f; }
    float se = 0.0f, so = 0.0f;
    constexpr int FULL = D / 512, TAIL = D % 512;
    for (int b = 0; b < FULL + (TAIL ? 1 : 0); b++) {
      if (b == FULL && lane >= uint(TAIL / 16)) break;
      half2 xh[8]; float xi;
      jt_xload(xp, xh, xi, se, so);
#pragma clang loop unroll(full)
      for (uint r = 0; r < 4u; r++) {
        half2 ag = jt_tdot<KB2, L>(wp + r * W, sh, xh, half2(0.0h));
        half2 au = jt_tdot<KB2, L>(up + r * W, sh, xh, half2(0.0h));
        g0[r] += float(ag.x) * xi; g1[r] += float(ag.y) * xi; u0[r] += float(au.x) * xi; u1[r] += float(au.y) * xi;
      }
      wp += PL; up += PL; xp += 512u;
    }
    float og[4] = {0.0f, 0.0f, 0.0f, 0.0f}, ou[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (NOUT > 0) {
      if (lane < uint(NOUT)) {
        int c = ocol[(size_t)e * NOUT + lane];
        if (c >= 0) {
          float xo = float(x[(size_t)t * D + uint(c)]);
#pragma clang loop unroll(full)
          for (uint r = 0; r < 4u; r++) {
            og[r] = xo * float(woutg[((size_t)e * I + row0 + r) * NOUT + lane]);
            ou[r] = xo * float(woutu[((size_t)e * I + row0 + r) * NOUT + lane]);
          }
        }
      }
#pragma clang loop unroll(full)
      for (uint r = 0; r < 4u; r++) { og[r] = simd_sum(og[r]); ou[r] = simd_sum(ou[r]); }
    }
    float SE = simd_sum(se), SO = simd_sum(so);
    const float mu0 = fc[0], mu1 = fc[1], isd0 = fc[2], isd1 = fc[3];
#pragma clang loop unroll(full)
    for (uint r = 0; r < 4u; r++) {
      float G0 = simd_sum(g0[r]), G1 = simd_sum(g1[r]), U0 = simd_sum(u0[r]), U1 = simd_sum(u1[r]);
      if (lane == 0u) {
        float g = ((G0 - mu0 * SE) * isd0 + (G1 - mu1 * SO) * isd1) * sg[(size_t)e * I + row0 + r];
        float u = ((U0 - mu0 * SE) * isd0 + (U1 - mu1 * SO) * isd1) * su_[(size_t)e * I + row0 + r];
        if (NOUT > 0) { g += og[r]; u += ou[r]; }
        g = metal::min(g, limit); u = metal::clamp(u, -limit, limit);
        h[(size_t)z * I + row0 + r] = float((g / (1.0f + metal::fast::exp(-(alpha * g)))) * (u + 1.0f));
      }
    }
    return;
  }
  if (SHARED == 0) return;
  // ---- shared expert of row t = z - ROWS * KS: MLX qmv_fast (4-bit, group 64) for gate and up, 4 output rows per
  // simdgroup. One row per threadgroup (rows run in parallel; the tile's weights are shared through the cache).
  // (Several rows per threadgroup with [rows][4] accumulator arrays miscompiled here: rows other than one per pass
  // came back wrong when both gate and up were live.)
  constexpr int in_w = D / 2;            // bytes per weight row
  constexpr int in_g = D / 64;           // groups per row
  const uint t = z - uint(ROWS * KS);
  const device uint8_t* gw = (const device uint8_t*)sgw + row0 * in_w + lane * 8u;
  const device uint8_t* uw = (const device uint8_t*)suw + row0 * in_w + lane * 8u;
  const device T* gs = sgs + row0 * in_g + lane / 4u;
  const device T* gb = sgb + row0 * in_g + lane / 4u;
  const device T* us = sus + row0 * in_g + lane / 4u;
  const device T* ub = sub + row0 * in_g + lane / 4u;
  const device T* xs = x + (size_t)t * D + lane * 16u;
  float rg[4] = {0.0f, 0.0f, 0.0f, 0.0f}, ru[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < D; k += 512) {
    float x_thread[16];
    float sum = aq4_load_vector<T>(xs, x_thread);
#pragma clang loop unroll(full)
    for (int o = 0; o < 4; o++) {
      rg[o] += aq4_qdot(gw + o * in_w, x_thread, float(gs[o * in_g]), float(gb[o * in_g]), sum);
      ru[o] += aq4_qdot(uw + o * in_w, x_thread, float(us[o * in_g]), float(ub[o * in_g]), sum);
    }
    gw += 256; uw += 256; gs += 8; gb += 8; us += 8; ub += 8; xs += 512;
  }
#pragma clang loop unroll(full)
  for (int o = 0; o < 4; o++) {
    T gv = static_cast<T>(simd_sum(rg[o]));
    T uv = static_cast<T>(simd_sum(ru[o]));
    if (lane == 0u) hs[(size_t)t * I + row0 + o] = m3_swiglu_oai<T>(uv, gv, alpha, limit);
  }
