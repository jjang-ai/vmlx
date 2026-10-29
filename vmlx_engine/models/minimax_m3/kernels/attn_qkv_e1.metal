// SPDX-License-Identifier: Apache-2.0
// K1 multi-row build (E1; DSpark verify, 2 .. 16 rows): q | k | v (affine 4-bit) and index q | index k (MXFP8 or affine
// 4-bit) of the input-normed rows, fp32 output, every element bit-identical to attn_qkv's one-row build (m3e_rows runs
// its per-element sequence on attn_xprep's chunks; the inverse RMS comes from attn_xprep, computed as attn_qkv does).
// Each weight word is read and decoded once per tile of TR rows.
//
// Host-prepended constants:
//   using sat = <q / k / v scales+biases dtype>;  using sbt = <index scales+biases dtype (affine B); float for MXFP8>;
//   constexpr int KIN, NQ, NK, NV, NIQ, NIK, GS_A, FMT_B (1 = MXFP8, 2 = affine 4-bit), GS_B  (as attn_qkv; the q / k / v
//   and affine index projections must be 4-bit)
//   constexpr int NSG, O, TR      // simdgroups / threadgroup, output rows / simdgroup, rows / tile
// Inputs (names): xa [RP, KIN] f32, xb [RP, KIN] f32 (attn_xprep's undivided chunks for an MXFP8 index; xa again for
//   an affine one), xs [RP, KIN / 16] f32, inv [>= RP] f32 (RP = NRT * TR rows, attn_xprep zero-pads), rv [R] (any
//   dtype: only its length, the valid row count, is read), then q_w, q_s, q_b, k_w, k_s, k_b, v_w, v_s, v_b, iq_w,
//   iq_s, iq_b, ik_w, ik_s, ik_b as attn_qkv (MXFP8: any >= 8-element arrays for the biases).
// Output: y [R, NQ + NK + NV + NIQ + NIK] float32.
// Grid: (32 * NSG * NTG * NRT, 1, 1), threadgroup (32 * NSG, 1, 1); NTG = NTOT / (NSG * O), NRT = RP / TR;
//   threadgroup id = tile * NRT + row tile (a weight tile's row tiles adjacent). Every segment a multiple of NSG * O.

  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = int(rv_shape[0]);
  const int NRT = int(xa_shape[0]) / TR;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;
  const int r0 = (int(threadgroup_position_in_grid.x) - tg * NRT) * TR;
  constexpr int NA = NQ + NK + NV;
  constexpr int NTOT = NA + NIQ + NIK;
  const int row0 = tg * (NSG * O);
  if (row0 >= NTOT) return;
  const int orow = row0 + sg * O;

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
  float res[TR][O];
  if (row0 < NA) {
    m3e_rows<sat, 0, GS_A, KIN, O, TR>(xa, xs, w, sa, ba, lrow, r0, lane, res);
  } else if (FMT_B == 1) {
    m3e_rows<sbt, 1, 32, KIN, O, TR>(xb, xs, w, sb, bb, lrow, r0, lane, res);
  } else {
    m3e_rows<sbt, 0, GS_B, KIN, O, TR>(xa, xs, w, sb, bb, lrow, r0, lane, res);
  }
  for (int t = 0; t < TR; ++t) {
    if (r0 + t >= R) break;
    const float iv = inv[r0 + t];
    for (int o = 0; o < O; ++o) {
      const float v = simd_sum(res[t][o]) * iv;
      if (int(lane) == o) y[(size_t)(r0 + t) * NTOT + orow + o] = v;
    }
  }
