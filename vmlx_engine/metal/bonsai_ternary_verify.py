"""Experimental exact 2-bit affine verifier for 2-3 activation rows.

The half2 decoder and thread layout are adapted from
https://github.com/ddalcu/mlx-serve/blob/v26.9.5/src/qmv2.zig
Copyright (c) 2026 David Dalcu, MIT License. Permission is granted to use,
copy, modify, and distribute this code, provided this copyright and license
notice are included. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. See the upstream MIT license for the
complete permission and liability terms.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
inline void h2dec(uint u, thread half2* q) {
  uint u6 = u >> 6;
  q[0] = as_type<half2>((u  & 0x00030003u) | 0x64006400u) - half2(1025.0h);
  q[1] = as_type<half2>((u  & 0x000C000Cu) | 0x5C005C00u) - half2(257.0h);
  q[2] = as_type<half2>((u  & 0x00300030u) | 0x54005400u) - half2(65.0h);
  q[3] = as_type<half2>((u  & 0x00C000C0u) | 0x4C004C00u) - half2(17.0h);
  q[4] = as_type<half2>((u  & 0x03000300u) | 0x44004400u) - half2(5.0h);
  q[5] = as_type<half2>((u6 & 0x00300030u) | 0x54005400u) - half2(65.0h);
  q[6] = as_type<half2>((u6 & 0x00C000C0u) | 0x4C004C00u) - half2(17.0h);
  q[7] = as_type<half2>((u6 & 0x03000300u) | 0x44004400u) - half2(5.0h);
}
"""
_SOURCE = r"""
constexpr int R = ROWS, G = GROUPS;
constexpr int KW = K / 16, KG = K / 128;
uint lane = thread_index_in_simdgroup;
uint sgi = simdgroup_index_in_threadgroup;
int grow = (threadgroup_position_in_grid.y * G + sgi) * R;
const device uint* wq = w + grow * KW + lane;
const device half* sp = scales + grow * KG + lane / 8;
float res[R][M];
for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) res[r][m] = 0;
for (int k = 0; k < K; k += 512) {
  uint wv[R]; float sc[R];
  for (int r = 0; r < R; ++r) { wv[r] = wq[r * KW]; sc[r] = float(sp[r * KG]); }
  half2 q[R][8];
  for (int r = 0; r < R; ++r) h2dec(wv[r], q[r]);
  for (int m = 0; m < M; ++m) {
    const device half* xr = x + m * K + k + lane * 16;
    half2 xh[8];
    for (int j = 0; j < 8; ++j) xh[j] = half2(xr[j], xr[j + 8]);
    for (int r = 0; r < R; ++r) {
      float2 acc = float2(q[r][0]) * float2(xh[0]);
      for (int j = 1; j < 8; ++j) acc = fma(float2(q[r][j]), float2(xh[j]), acc);
      res[r][m] += sc[r] * (acc.x + acc.y);
    }
  }
  wq += 32; sp += 4;
}
for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) {
  float v = simd_sum(res[r][m]);
  if (lane == 0) y[m * N + grow + r] = half(v);
}
"""
@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_bonsai_ternary_verify_m2m3_v1",
        input_names=["x", "w", "scales"],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
    )


def maybe_ternary_verify(x: mx.array, linear) -> mx.array | None:
    """Return a verifier projection only for the proven exact ternary layout."""
    if x.ndim < 2 or x.dtype != mx.float16:
        return None
    m = 1
    for dim in x.shape[:-1]:
        m *= int(dim)
    if m not in (2, 3) or linear.bits != 2 or linear.group_size != 128:
        return None
    if linear.scales.dtype != mx.float16 or linear.biases.dtype != mx.float16:
        return None
    k = int(x.shape[-1])
    n = int(linear.weight.shape[0])
    r, g = (2, 4) if m == 2 and os.environ.get("VMLX_BONSAI_VERIFY_R2G4", "1") == "1" else (4, 8)
    if k % 512 or n % (r * g) or int(linear.weight.shape[1]) * 16 != k:
        return None
    y = _kernel()(
        inputs=[x.reshape(m, k), linear.weight, linear.scales],
        template=[("K", k), ("N", n), ("M", m), ("ROWS", r), ("GROUPS", g)],
        grid=(g * 32, n // (r * g), 1),
        threadgroup=(g * 32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]
    if "bias" in linear:
        y = y + linear.bias
    return y.reshape((*x.shape[:-1], n))
