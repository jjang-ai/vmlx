// SPDX-License-Identifier: Apache-2.0
// K1 multi-row build on precomputed input chunks (attn_xprep): q | k | v (affine) and index q | k (MXFP8 or affine 4)
// for R rows (2 .. 16), fp32 output, bit-identical per row to attn_qkv's one-row build (same per-element sequence;
// the norm multiply, place-value division, chunk sums and the inverse RMS come from attn_xprep, which computes them
// with the same m3a_load_chunk calls in the same order).
//
// Host-prepended constants: as attn_qkv (xt unused here), but VPT must be 16 for both formats (affine 4-bit, MXFP8).
// Inputs (names): xa [R, KIN] f32, xb [R, KIN] f32 (MXFP8 index input; xa again for affine-4 index), xs [R, KIN / 16]
//   f32, inv [>= R] f32, then q_w .. ik_b as attn_qkv. Output: y [R, NQ + NK + NV + NIQ + NIK] float32.
// Grid / threadgroup: as attn_qkv's multi-row build.

  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = int(xa_shape[0]);
  const int NRT = (R + RT - 1) / RT;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;
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
  if (row0 < NA) {
    m3a_rowsP<sat, 0, BITS_A, GS_A, KIN, RPS, RT>(xa, xs, w, sa, ba, lrow, r0, R, lane, res);
  } else if (FMT_B == 1) {
    m3a_rowsP<sbt, 1, 8, 32, KIN, RPS, RT>(xb, xs, w, sb, bb, lrow, r0, R, lane, res);
  } else {
    m3a_rowsP<sbt, 0, BITS_B, GS_B, KIN, RPS, RT>(xa, xs, w, sb, bb, lrow, r0, R, lane, res);
  }
  for (int t = 0; t < RT; ++t) {
    if (r0 + t >= R) break;
    const float iv = inv[r0 + t];
    for (int o = 0; o < RPS; ++o) {
      const float v = simd_sum(res[t][o]) * iv;
      if (int(lane) == o) y[(size_t)(r0 + t) * NTOT + orow + o] = v;
    }
  }
