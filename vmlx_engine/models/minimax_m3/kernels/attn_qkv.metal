// SPDX-License-Identifier: Apache-2.0
// K1: input RMSNorm (GemmaRMSNorm, 1 + w in fp32) fused into ONE projection launch producing q | k | v (affine) and
// index q | index k (MXFP8 or affine) for R rows (1 .. 16), fp32 output.
//
// Host-prepended constants:
//   using xt   = <activation dtype>;        // x: the residual stream before the norm, [R, KIN] row-contiguous
//   using nwt  = <input_layernorm.weight dtype>;
//   using sat  = <q / k / v scales+biases dtype>;
//   using sbt  = <index q / k scales+biases dtype (affine B); float for MXFP8>;
//   constexpr int KIN = hidden size;  NQ / NK / NV = q / k / v output rows;  NIQ / NIK = index q / k rows
//   constexpr int BITS_A, GS_A        // q / k / v: affine bits (4 / 8) and group size (32 / 64 / 128)
//   constexpr int FMT_B, BITS_B, GS_B // index: 1 = MXFP8 (pass 8 / 32), 2 = affine BITS_B / GS_B
//   constexpr int NSG, RPS            // simdgroups per threadgroup, output rows per simdgroup (2, 4: MLX qmv's shape)
//   constexpr int RT                  // rows per threadgroup (1 for one row; the host's choice for several)
//   constexpr float EPS               // rms_norm_eps
// Inputs (names): x, nw, q_w, q_s, q_b, k_w, k_s, k_b, v_w, v_s, v_b, iq_w, iq_s, iq_b, ik_w, ik_s, ik_b
//   (MXFP8 has no biases: pass any array of >= 8 elements). Output: y [R, NQ + NK + NV + NIQ + NIK] float32.
// Grid: (32 * NSG * NTG * ceil(R / RT), 1, 1), threadgroup (32 * NSG, 1, 1), NTG = (NQ + NK + NV + NIQ + NIK) /
//   (NSG * RPS): threadgroup id = tile * ceil(R / RT) + row tile (a weight tile's row tiles adjacent, so its repeated
//   reads hit the shared cache). Every segment's row count must be a multiple of NSG * RPS.
//
// y[r, j] = rsqrt(mean(x_r^2) + EPS) * sum_i W[j, i] * x_r[i] * (1 + nw[i]): the sum of squares is accumulated on the
// fly while the weight walk reads x_r (no normalized copy of x), every product in fp32.

  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = int(x_shape[0]);
  const int NRT = (R + RT - 1) / RT;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;  // weight tile
  const int r0 = (int(threadgroup_position_in_grid.x) - tg * NRT) * RT;
  constexpr int NA = NQ + NK + NV;
  constexpr int NTOT = NA + NIQ + NIK;
  const int row0 = tg * (NSG * RPS);
  if (row0 >= NTOT) return;
  const int orow = row0 + sg * RPS;

  const device uint8_t* w;
  const device sat* sa = q_s;
  const device sat* ba = q_b;
  const device sbt* sb = (const device sbt*)iq_s;
  const device sbt* bb = (const device sbt*)iq_b;
  int lrow;
  if (row0 < NQ) {
    w = (const device uint8_t*)q_w; lrow = orow;
  } else if (row0 < NQ + NK) {
    w = (const device uint8_t*)k_w; sa = k_s; ba = k_b; lrow = orow - NQ;
  } else if (row0 < NA) {
    w = (const device uint8_t*)v_w; sa = v_s; ba = v_b; lrow = orow - NQ - NK;
  } else if (row0 < NA + NIQ) {
    w = (const device uint8_t*)iq_w; lrow = orow - NA;
  } else {
    w = (const device uint8_t*)ik_w; sb = (const device sbt*)ik_s; bb = (const device sbt*)ik_b; lrow = orow - NA - NIQ;
  }
  float res[RT][RPS];
  float ss[RT];
  if (row0 < NA) {
    m3a_rowsN<xt, nwt, sat, 0, BITS_A, GS_A, KIN, RPS, RT, true>(x, nw, w, sa, ba, lrow, r0, R, lane, res, ss);
  } else if (FMT_B == 1) {
    m3a_rowsN<xt, nwt, sbt, 1, 8, 32, KIN, RPS, RT, true>(x, nw, w, sb, bb, lrow, r0, R, lane, res, ss);
  } else {
    m3a_rowsN<xt, nwt, sbt, 0, BITS_B, GS_B, KIN, RPS, RT, true>(x, nw, w, sb, bb, lrow, r0, R, lane, res, ss);
  }
  for (int t = 0; t < RT; ++t) {
    if (r0 + t >= R) break;
    const float inv = metal::precise::rsqrt(simd_sum(ss[t]) / float(KIN) + EPS);
    for (int o = 0; o < RPS; ++o) {
      const float v = simd_sum(res[t][o]) * inv;
      if (int(lane) == o) y[(size_t)(r0 + t) * NTOT + orow + o] = v;
    }
  }
