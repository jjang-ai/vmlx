// SPDX-License-Identifier: Apache-2.0
// KVQ-Q: encode K / V / index-key rows into packed cache rows (kv_quant.metal formats), one simdgroup per row.
// The unfused (stock) append path's encoder; the fused attention's K2 encodes with the same m3q_store_row, so a row
// gets the same bytes from either path.
//
// Host-prepended constants: using tn = <input dtype>;  constexpr int F = format (1 .. 4), D = head dim (mult. of 64)
// Inputs (names): x [B, H, L, D] (any strides, x_strides), out: y [B, H, L, RB] uint8 (row-contiguous).
// Grid: (32 * B * H * L, 1, 1) rounded up to the threadgroup, threadgroup (128, 1, 1).

  constexpr int DPL = D / 32;
  constexpr int RB = m3q_row_bytes(F, D);
  const uint lane = thread_index_in_simdgroup;
  const int H = int(x_shape[1]);
  const int L = int(x_shape[2]);
  const int rows = int(x_shape[0]) * H * L;
  const int r = int(thread_position_in_grid.x / 32u);
  if (r >= rows) return;
  const int b = r / (H * L);
  const int h = (r / L) % H;
  const int t = r % L;
  const device tn* src = x + (size_t)b * x_strides[0] + (size_t)h * x_strides[1] + (size_t)t * x_strides[2];
  float v[DPL];
  for (int e = 0; e < DPL; ++e) v[e] = float(src[(size_t)(int(lane) + 32 * e) * x_strides[3]]);
  m3q_store_row<F, D>(v, lane, y + (size_t)r * RB);
