// SPDX-License-Identifier: Apache-2.0
// K5 multi-row build on precomputed input chunks (attn_xprep with NORM 0 on the K4 output): o_proj (affine 4-bit) +
// residual add for R rows (2 .. 16), bit-identical per row to attn_oproj's one-row build.
//
// Host-prepended constants: as attn_oproj (SSP must be 0).
// Inputs (names): xa [R, KIN] f32 (attn_xprep of o), xs [R, KIN / 16] f32, xr [R, NOUT] xt, w / s / b.
// Output: h [R, NOUT] xt. Grid / threadgroup: as attn_oproj's multi-row build.

  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = int(xa_shape[0]);
  const int NRT = (R + RT - 1) / RT;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;
  const int r0 = (int(threadgroup_position_in_grid.x) - tg * NRT) * RT;
  constexpr int COLS = NSG * RPS;
  const int col = tg * COLS + sg * RPS;
  float res[RT][RPS];
  m3a_rowsP<st, 0, BITS, GS, KIN, RPS, RT>(xa, xs, (const device uint8_t*)w, s, b, col, r0, R, lane, res);
  for (int t = 0; t < RT; ++t) {
    if (r0 + t >= R) break;
    for (int c = 0; c < RPS; ++c) {
      const float hv = float(xt(simd_sum(res[t][c]) + float(xr[(size_t)(r0 + t) * NOUT + col + c])));
      if (int(lane) == c) h[(size_t)(r0 + t) * NOUT + col + c] = xt(hv);
    }
  }
