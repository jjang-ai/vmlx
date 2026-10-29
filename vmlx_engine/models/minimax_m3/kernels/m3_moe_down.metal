// SPDX-License-Identifier: Apache-2.0
// K8 m3_moe_down: the routed experts' (JANGT) down projection with the routing-weighted combine, the shared expert's
// (affine 4-bit, group 64) down projection, and the residual add, in one launch; output = the new residual stream.
//
// Template: T (activation dtype), D (hidden = down output), I (expert intermediate = down input), E, ROWS (1..16),
//   KS, KB2 (2 x JANGT bits per weight of down), L (12), W (uint32 words per down row), NOUT (down outlier columns,
//   0 = none).
// Inputs:  fc [4] f32 = (mu0, mu1, isd0, isd1: moe_kernels code_stats of KB2), hr [ROWS * KS, I] f32 (K7's h rotated with each slot's expert column scales: moe_kernels.rotate_rows(h,
//   sv_dn, inds)), h [ROWS * KS, I] f32 (K7), inds / scores [ROWS, KS] (K7), wd [E, D, W] u32, sd [E, D] f32,
//   ocol [E, max(NOUT,1)] i32, wout [E, D, max(NOUT,1)] f16, hs [ROWS, I] T (K7), sdw [D, I / 8] u32,
//   sds / sdb [D, I / 64] T, resid [ROWS, D] T (the residual stream before the MoE block).
// Output:  out [ROWS, D] T = resid + (T(routed) + T(shared)) -- JTMixedSwitchGLU.routed rounds the weighted sum to T,
//   the shared expert's qmv output is T, SparseMoeBlock adds them in T, the decoder layer adds the residual in T.
// Grid: (64, D / 8, ROWS), threadgroup (64, 1, 1): 2 simdgroups x 4 output rows of row t. Routed part =
//   moe_kernels._down_v3 for row t (slots in routing order, acc = fma(w, y, acc) from 0); shared part = MLX one-row
//   qmv_fast per output row.

  const uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u;
  const uint t = threadgroup_position_in_grid.z;
  constexpr int PL = jt_lane_bits<KB2>();
  const uint lb = lane * uint(PL); const uint sh = lb & 31u;
  const float mu0 = fc[0], mu1 = fc[1], isd0 = fc[2], isd1 = fc[3];
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  constexpr int FULL = I / 512, TAIL = I % 512;
  for (uint sl = 0; sl < uint(KS); sl++) {
    const uint ts = t * uint(KS) + sl;
    const uint e = inds[ts];
    const float wgt = float(scores[ts]);
    float od[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (NOUT > 0) {
      if (lane < uint(NOUT)) {
        int c = ocol[(size_t)e * NOUT + lane];
        if (c >= 0) {
          float ho = float(h[(size_t)ts * I + uint(c)]);
#pragma clang loop unroll(full)
          for (uint r = 0; r < 4u; r++) od[r] = ho * float(wout[((size_t)e * D + row0 + r) * NOUT + lane]);
        }
      }
#pragma clang loop unroll(full)
      for (uint r = 0; r < 4u; r++) od[r] = simd_sum(od[r]);
    }
    const device uint32_t* wp = wd + ((size_t)e * D + row0) * W + (lb >> 5);
    const device float* xp = hr + (size_t)ts * I + lane * 16u;
    float d0[4], d1[4];
#pragma clang loop unroll(full)
    for (uint r = 0; r < 4u; r++) { d0[r] = 0.0f; d1[r] = 0.0f; }
    float se = 0.0f, so = 0.0f;
    for (int b = 0; b < FULL + (TAIL ? 1 : 0); b++) {
      if (b == FULL && lane >= uint(TAIL / 16)) break;
      half2 xh[8]; float xi;
      jt_xload(xp, xh, xi, se, so);
#pragma clang loop unroll(full)
      for (uint r = 0; r < 4u; r++) {
        half2 ad = jt_tdot<KB2, L>(wp + r * W, sh, xh, half2(0.0h));
        d0[r] += float(ad.x) * xi; d1[r] += float(ad.y) * xi;
      }
      wp += PL; xp += 512u;
    }
    float SE = simd_sum(se), SO = simd_sum(so);
#pragma clang loop unroll(full)
    for (uint r = 0; r < 4u; r++) {
      float D0 = simd_sum(d0[r]), D1 = simd_sum(d1[r]);
      float y = ((D0 - mu0 * SE) * isd0 + (D1 - mu1 * SO) * isd1) * sd[(size_t)e * D + row0 + r];
      if (NOUT > 0) y += od[r];
      acc[r] = fma(wgt, y, acc[r]);
    }
  }
  // ---- shared expert down: MLX qmv_fast (4-bit, group 64), one row
  constexpr int in_w = I / 2;
  constexpr int in_g = I / 64;
  const device uint8_t* ws = (const device uint8_t*)sdw + row0 * in_w + lane * 8u;
  const device T* sc = sds + row0 * in_g + lane / 4u;
  const device T* bi = sdb + row0 * in_g + lane / 4u;
  const device T* xs = hs + (size_t)t * I + lane * 16u;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < I; k += 512) {
    float x_thread[16];
    float sum = aq4_load_vector<T>(xs, x_thread);
#pragma clang loop unroll(full)
    for (int o = 0; o < 4; o++) res[o] += aq4_qdot(ws + o * in_w, x_thread, float(sc[o * in_g]), float(bi[o * in_g]), sum);
    ws += 256; sc += 8; bi += 8; xs += 512;
  }
#pragma clang loop unroll(full)
  for (uint r = 0; r < 4u; r++) {
    T shv = static_cast<T>(simd_sum(res[r]));
    if (lane == 0u) {
      T yr = static_cast<T>(acc[r]);
      T mlp = yr + shv;
      out[(size_t)t * D + row0 + r] = resid[(size_t)t * D + row0 + r] + mlp;
    }
  }
