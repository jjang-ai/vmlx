// SPDX-License-Identifier: Apache-2.0
// K5 multi-row build (E1; DSpark verify, 2 .. 16 rows): o_proj (affine 4-bit) + residual add, every element
// bit-identical to attn_oproj's one-row build (m3e_rows on attn_xprep's chunks of the K4 output, NORM 0; the same
// residual add and one rounding). Each weight word is read and decoded once per tile of TR rows.
//
// Host-prepended constants:
//   using xt = <residual dtype>;  using st = <o_proj scales+biases dtype>;
//   constexpr int KIN = q heads * head dim, NOUT = hidden size, GS;  NSG, O, TR as attn_qkv_e1
// Inputs (names): xa [RP, KIN] f32 (attn_xprep of the K4 output; RP = NRT * TR rows, zero-padded), xs [RP, KIN / 16]
//   f32, xr [R, NOUT] xt (the residual stream before attention; R = the valid rows), w / s / b (o_proj weight /
//   scales / biases).
// Output: h [R, NOUT] xt = xr + o_proj(o).
// Grid: (32 * NSG * (NOUT / (NSG * O)) * NRT, 1, 1), threadgroup (32 * NSG, 1, 1), NRT = RP / TR; threadgroup id =
//   column tile * NRT + row tile.

  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = int(xr_shape[0]);
  const int NRT = int(xa_shape[0]) / TR;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;
  const int r0 = (int(threadgroup_position_in_grid.x) - tg * NRT) * TR;
  const int col = (tg * NSG + sg) * O;
  float res[TR][O];
  m3e_rows<st, 0, GS, KIN, O, TR>(xa, xs, (const device uint8_t*)w, s, b, col, r0, lane, res);
  for (int t = 0; t < TR; ++t) {
    if (r0 + t >= R) break;
    for (int c = 0; c < O; ++c) {
      const float hv = float(xt(simd_sum(res[t][c]) + float(xr[(size_t)(r0 + t) * NOUT + col + c])));
      if (int(lane) == c) h[(size_t)(r0 + t) * NOUT + col + c] = xt(hv);
    }
  }
