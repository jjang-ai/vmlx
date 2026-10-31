"""Row-exact multi-row float32 GEMV for speculative verify (the MiniMax-M3 fp32 MoE routers).

A verify of T rows ran each fp32 router (``x @ W.T``, W [E, D] float32, x bf16 -> float32) as T one-row calls: MLX's
multi-row matmul reduces in another order than its one-row gemv, so a batched call would not equal the decode steps.
One launch here computes every row with the one-row gemv's arithmetic (MLX gemv.h GEMVKernel<float, BM=1, BN=8,
SM=1, SN=32, TM=4, TN=4>, the configuration MLX dispatches for K >= 16 * N: lane l of simdgroup s accumulates columns
(s * 32 + l) * 4 + i * 1024 .. + 3 with fma in column order, a shuffle-down tree, then the 8 simdgroup partials in
order), reading each weight block once for all rows. Admitted per (rows, K, N, x dtype) only after a bit-equality
check against stock one-row calls; anything else returns None (the caller keeps one-row calls).

Lever: VMLX_ROW_EXACT_GEMV (default on; read per call).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

MAX_ROWS = 16
_ADMITTED: dict = {}
STATS = {"calls": 0, "admitted": 0, "rejected": 0}

_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint n = threadgroup_position_in_grid.x;
  threadgroup float part[8 * ROWS];
  const device float* wrow = w + (size_t)n * K;
  float result[ROWS];
#pragma clang loop unroll(full)
  for (int r = 0; r < ROWS; r++) result[r] = 0.0f;
  int bn = int(sg * 32u + lane) * 4;
  constexpr int blockN = 1024;
  for (int i = 0; i < K / blockN; ++i) {
    float inter[4];
#pragma clang loop unroll(full)
    for (int tn = 0; tn < 4; tn++) inter[tn] = wrow[bn + tn];
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; r++) {
      float v_coeff[4];
#pragma clang loop unroll(full)
      for (int tn = 0; tn < 4; tn++) v_coeff[tn] = static_cast<float>(x[(size_t)r * K + bn + tn]);
#pragma clang loop unroll(full)
      for (int tn = 0; tn < 4; tn++) result[r] += inter[tn] * v_coeff[tn];
    }
    bn += blockN;
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < ROWS; r++) {
#pragma clang loop unroll(full)
    for (ushort sn = 16; sn >= 1; sn >>= 1) result[r] += simd_shuffle_down(result[r], sn);
    if (lane == 0u) part[sg * ROWS + r] = result[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < uint(ROWS)) {
    float acc = part[lane];
    for (int s = 1; s < 8; s++) acc += part[s * ROWS + lane];
    y[(size_t)lane * N + n] = acc;
  }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(name="vmlx_row_exact_gemv_f32", input_names=["x", "w"], output_names=["y"],
                                source=_SOURCE)


def enabled() -> bool:
    return os.environ.get("VMLX_ROW_EXACT_GEMV", "1").strip().lower() not in {"0", "false", "no", "off"}


def _launch(x2d, weight):
    rows, K = x2d.shape
    N = int(weight.shape[0])
    return _kernel()(inputs=[x2d, weight], template=[("K", K), ("N", N), ("ROWS", rows)],
                     grid=(256 * N, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, N)],
                     output_dtypes=[mx.float32])[0]


def row_exact_gemv(x, weight):
    """``x @ weight.T`` (float32 weight, float32 result) for 2..16 rows, every row bit-equal to its one-row call, or
    None (unsupported shape / dtype, or the shape failed admission)."""
    if not enabled() or x.ndim < 2 or weight.ndim != 2 or weight.dtype != mx.float32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    K, N = int(x.shape[-1]), int(weight.shape[0])
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    if not 2 <= rows <= MAX_ROWS or K % 1024 or K < 16 * N or int(weight.shape[1]) != K:
        return None
    key = (rows, K, N, str(x.dtype))
    ok = _ADMITTED.get(key)
    if ok is None:
        probe = (mx.random.normal((rows, K), key=mx.random.key(97)) * 0.5
                 + mx.random.uniform(-2, 2, (rows, 1), key=mx.random.key(98))).astype(x.dtype)
        want = mx.concatenate([probe[r:r + 1] @ weight.T for r in range(rows)], axis=0)
        ok = bool(mx.array_equal(_launch(probe, weight), want).item())
        _ADMITTED[key] = ok
        STATS["admitted" if ok else "rejected"] += 1
        (logger.info if ok else logger.warning)("row-exact gemv %s rows=%d K=%d N=%d x=%s",
                                                "admitted" if ok else "REJECTED", rows, K, N, x.dtype)
    if not ok:
        return None
    STATS["calls"] += 1
    return _launch(mx.contiguous(x.reshape(rows, K)), weight).reshape(*x.shape[:-1], N)
