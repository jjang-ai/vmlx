// SPDX-License-Identifier: Apache-2.0
// KVQ-D: decode packed cache rows (kv_quant.metal formats) into the nominal dtype, 8 dims per thread. The unfused
// (stock) read path: prefill chunks and the unfused decode / verify attention read the lanes through this.
//
// Host-prepended constants: using tn = <output dtype>;  constexpr int F = format (1 .. 4), D = head dim
// Inputs (names): lane [B, H, n, RB] uint8 (strided view, the last axis contiguous: lane_strides[3] == 1).
// Output: out [B, H, n, D] tn.  Grid: (B * H * n * D / 8, 1, 1) rounded up, threadgroup (256, 1, 1).

  constexpr int CH = D / 8;
  const int H = int(lane_shape[1]);
  const int n = int(lane_shape[2]);
  const size_t rows = (size_t)lane_shape[0] * H * n;
  const size_t gid = thread_position_in_grid.x;
  const size_t r = gid / CH;
  if (r >= rows) return;
  const int c = int(gid - r * CH);
  const int b = int(r / ((size_t)H * n));
  const int h = int((r / n) % H);
  const int t = int(r % n);
  const device uint8_t* row = lane + (size_t)b * lane_strides[0] + (size_t)h * lane_strides[1] + (size_t)t * lane_strides[2];
  float o[8];
  m3q_load<F, D, tn, 8>(row, 8 * c, o);
  device tn* dst = out + r * D + 8 * c;
  for (int i = 0; i < 8; ++i) dst[i] = tn(o[i]);
