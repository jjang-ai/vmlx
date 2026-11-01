// SPDX-License-Identifier: Apache-2.0
// K5: o_proj (affine 4 / 8-bit) + residual add for R rows (1 .. 16), one rounding to the residual dtype; optionally
// the per-threadgroup partial sums of squares of the new residual rows (the post-attention RMSNorm's input).
//
// Host-prepended constants:
//   using xt = <residual dtype>;  using st = <o_proj scales+biases dtype>;
//   constexpr int KIN = q heads * head dim (o_proj input), NOUT = hidden size, BITS, GS (affine)
//   constexpr int NSG, RPS, RT   // simdgroups / threadgroup, output columns / simdgroup, rows / threadgroup:
//                                // one row 2, 4, 1; several rows 4, 8, 2 (fused_attn.py)
//   constexpr int SSP            // 1: write the sum-of-squares partials
// Inputs (names): o [R, KIN] float32 (K4 output), xr [R, NOUT] xt (the residual stream before attention),
//   w / s / b (o_proj weight / scales / biases).
// Outputs: h [R, NOUT] xt = xr + o_proj(o) (fp32 sum, one rounding); ssp [R, NOUT / 8] float32 when SSP: for every
//   8 consecutive columns, sum of h (rounded) squared, one fma per column in column order (a consumer adds a row's
//   NOUT / 8 partials in order); else a 1-element dummy.
// Grid: (32 * NSG * (NOUT / (NSG * RPS)) * ceil(R / RT), 1, 1), threadgroup (32 * NSG, 1, 1), threadgroup id =
//   column tile * ceil(R / RT) + row tile; NOUT a multiple of NSG * RPS (and of 8).

  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  const int R = int(o_shape[0]);
  const int NRT = (R + RT - 1) / RT;
  const int tg = int(threadgroup_position_in_grid.x) / NRT;  // column tile
  const int r0 = (int(threadgroup_position_in_grid.x) - tg * NRT) * RT;
  constexpr int COLS = NSG * RPS;
  const int col = tg * COLS + sg * RPS;
  threadgroup float hcol[RT * COLS];

  float res[RT][RPS];
  float ss[RT];
  m3a_rowsN<float, float, st, 0, BITS, GS, KIN, RPS, RT, false>(o, o, (const device uint8_t*)w, s, b, col, r0, R,
                                                               lane, res, ss);
  for (int t = 0; t < RT; ++t) {
    if (r0 + t >= R) break;
    for (int c = 0; c < RPS; ++c) {
      const float hv = float(xt(simd_sum(res[t][c]) + float(xr[(size_t)(r0 + t) * NOUT + col + c])));
      if (int(lane) == c) {
        h[(size_t)(r0 + t) * NOUT + col + c] = xt(hv);
        if (SSP) hcol[t * COLS + sg * RPS + c] = hv;
      }
    }
  }
  if (SSP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int G8 = COLS / 8;
    if (tid < RT * G8) {
      const int t = tid / G8;
      const int gg = tid - t * G8;
      if (r0 + t < R) {
        float acc = 0.0f;
        for (int c = 0; c < 8; ++c) acc = fma(hcol[t * COLS + gg * 8 + c], hcol[t * COLS + gg * 8 + c], acc);
        ssp[(size_t)(r0 + t) * (NOUT / 8) + tg * G8 + gg] = acc;
      }
    }
  }
