"""Multi-row mxfp8 QMV: 2..8 activation rows through an mxfp8 Linear with each weight read and decoded once.

Why: a speculative verify forward feeds 2..8 token rows through every dense projection. MLX's mxfp8
``quantized_matmul`` handles M >= 2 with ``fp_qmv_wide`` up to a small M and with the tiled ``qmm`` above it;
measured on DeepSeek-V4.1 projections (M5 Max, 40 layers): 6 rows cost 1.6-2.3x the 1-row call (wo_b 5120x8192:
3.05 -> 7.04 ms, wq_b 32768x1024: 2.48 -> 4.33 ms) although a 1-row mxfp8 qmv is bandwidth-bound and the extra rows
need no extra weight bytes.

How: MLX's ``fp_qmv_fast_impl`` (the M=1 kernel for K % 256 == 0, N % 8 == 0) with every x row handled inside the same
threadgroup: per 256-value block each lane loads its 8 activations of every row, then for each of its 4 output rows
decodes the 8 E4M3 weights and the E8M0 group scale ONCE (exact values, as MLX's fp8_e4m3 / fp8_e8m0 conversions) and
runs MLX qdot's per-row sequence on them (accum += x * w in order, scale * accum, += into the row's result, simd_sum).
So each row equals the M=1 call bit for bit, and verify rows match 1-row decode on these projections. Every
(K, N, rows, dtype) is admitted only after a bit-equality self-check against stock M=1 calls; a shape that does not
match keeps the stock path (logged once).

Layouts (MLX mxfp8): weight uint32 [.., N, K/4] (E4M3 bytes), scales uint8 [.., N, K/32] (E8M0). x [B, M, K] with
weights either shared (one matrix) or batched with x (B matrices: DeepSeek-V4.1's grouped wo_a).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

MAX_ROWS = 8
_ADMITTED: dict[tuple, bool] = {}
STATS = {"calls": 0, "admitted_shapes": 0, "rejected_shapes": 0}

_HEADER = r"""
inline float e4m3_value(uint8_t b) {
  // MLX fp8_e4m3::operator float16_t, then float: exact
  uint16_t v = b & 127;
  uint16_t sign_bit = ((uint16_t)((b >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(converted * 256.0);
}
inline float e8m0_value(uint8_t b) {
  // MLX fp8_e8m0::operator float
  uint32_t out = (b == 0 ? 0x400000 : (static_cast<uint16_t>(b) << 23));
  return as_type<float>(out);
}
"""

_SOURCE = r"""
  constexpr int K = KDIM;
  constexpr int N = NDIM;
  constexpr int values_per_thread = 8;            // pack_factor (4) * packs_per_thread (2), 8-bit
  constexpr int block_size = values_per_thread * 32;
  constexpr int group_size = 32;
  constexpr int RPS = RPSV;                        // output rows per simdgroup (does not change any row's arithmetic)
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint b = tid.z;

  const device uint8_t* ws = (const device uint8_t*)w + (WBATCH ? (size_t)b * N * K : 0);
  const device uint8_t* sc = scales + (WBATCH ? (size_t)b * N * (K / group_size) : 0);
  const int out_row = tid.y * (NSG * RPS) + simd_gid * RPS;
  ws += (size_t)out_row * K + simd_lid * values_per_thread;
  sc += (size_t)out_row * (K / group_size) + simd_lid / (group_size / values_per_thread);
  const device T* xs = x + (size_t)b * ROWS * K + simd_lid * values_per_thread;

  XT x_thread[ROWS][values_per_thread];
  float result[ROWS][RPS];
  for (int r = 0; r < ROWS; r++)
    for (int row = 0; row < RPS; row++) result[r][row] = 0.0f;

  for (int k = 0; k < K; k += block_size) {
    for (int r = 0; r < ROWS; r++) {
      for (int i = 0; i < values_per_thread; i++) x_thread[r][i] = xs[r * K + i];
    }
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * K;
      float s = e8m0_value(sc[row * (K / group_size)]);
      float wv[values_per_thread];
      for (int i = 0; i < values_per_thread; i++) wv[i] = e4m3_value(wl[i]);
      for (int r = 0; r < ROWS; r++) {
        float accum = 0.0f;
        for (int i = 0; i < values_per_thread; i++) accum += float(x_thread[r][i]) * wv[i];
        result[r][row] += s * accum;
      }
    }
    ws += block_size;
    sc += block_size / group_size;
    xs += block_size;
  }
  for (int r = 0; r < ROWS; r++) {
    for (int row = 0; row < RPS; row++) {
      float v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[((size_t)b * ROWS + r) * N + out_row + row] = static_cast<T>(v);
      }
    }
  }
"""


def fp8_rows_requested() -> bool:
    """VMLX_FP8_ROWS_QMV=0 keeps MLX's multi-row kernels (read per call: in-process A/B)."""
    return os.environ.get("VMLX_FP8_ROWS_QMV", "1").strip().lower() not in {"0", "false", "no", "off"}


# (rows per simdgroup, simdgroups per threadgroup, x registers as the input dtype) per row count; none of them changes a
# row's arithmetic (each output row is still its own 8-value qdot chain + simd_sum), only register pressure / occupancy.
# Measured on DeepSeek-V4.1 projections (M5 Max, 40 weight copies, median of 8; stock = mx.quantized_matmul mxfp8):
#   6 rows: wq_b 32768x1024 3.19 vs 4.33 ms, wo_b 5120x8192 4.24-4.46 vs 7.01, shared w1/w2 1.31/1.24 vs 1.86/1.71,
#   wq_a 0.83 vs 0.94; 4 rows: wq_b 2.59 vs 2.86, wo_b 3.31 vs 4.04. At 2 rows MLX's fp_qmv_wide is as fast and at 8
#   rows every config here is slower (register pressure), so those row counts keep the stock kernel (None).
CONFIG = {2: None, 3: (2, 2, False), 4: (2, 2, False), 5: (2, 2, True), 6: (2, 2, True), 7: (2, 2, True), 8: None}


@lru_cache(maxsize=None)
def _kernel(K: int, N: int, rows: int, wbatch: bool, rps: int, nsg: int, xt_native: bool):
    src = _SOURCE.replace("KDIM", str(K)).replace("NDIM", str(N)).replace("ROWS", str(rows)) \
        .replace("WBATCH", "true" if wbatch else "false").replace("RPSV", str(rps)).replace("NSG", str(nsg)) \
        .replace("XT x_thread", "T x_thread" if xt_native else "float x_thread")
    return mx.fast.metal_kernel(name=f"vmlx_fp8_rows_qmv_k{K}_n{N}_r{rows}_{'wb' if wbatch else 'ws'}_p{rps}_g{nsg}_{int(xt_native)}",
                                input_names=["x", "w", "scales"], output_names=["y"], source=src, header=_HEADER)


def _launch(x3, w, scales, wbatch: bool, config=None):
    B, M, K = x3.shape
    N = w.shape[-2]
    rps, nsg, xt_native = config or CONFIG[M]
    if N % (rps * nsg):
        rps, nsg = 1, 1
    kern = _kernel(K, N, M, wbatch, rps, nsg, bool(xt_native) and x3.dtype != mx.float32)
    return kern(inputs=[x3, w, scales], template=[("T", x3.dtype)], grid=(32, nsg * (N // (rps * nsg)), B),
                threadgroup=(32, nsg, 1), output_shapes=[(B, M, N)], output_dtypes=[x3.dtype])[0]


def _admissible_shape(x3, w, scales) -> bool:
    B, M, K = x3.shape
    if not (2 <= M <= MAX_ROWS) or x3.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if w.dtype != mx.uint32 or scales.dtype != mx.uint8 or K % 256 or w.shape[-1] * 4 != K:
        return False
    if scales.shape[-1] * 32 != K or scales.shape[-2] != w.shape[-2]:
        return False
    return w.ndim == 2 or (w.ndim == 3 and w.shape[0] == B)


def _self_check(x3, w, scales, wbatch: bool) -> bool:
    """Bit equality of every row against stock M=1 mxfp8 calls on these weights (random rows, the call's dtype)."""
    B, M, K = x3.shape
    probe = (mx.random.normal((B, M, K), key=mx.random.key(1234)) * 2.0).astype(x3.dtype)
    got = _launch(probe, w, scales, wbatch)
    for r in range(M):
        ref = mx.quantized_matmul(probe[:, r:r + 1], w, scales, None, transpose=True, group_size=32, bits=8, mode="mxfp8")
        if not mx.array_equal(got[:, r:r + 1], ref).item():
            return False
    return True


def fp8_rows_qmv(x, w, scales):
    """x [..., M, K] (M in 2..8) @ mxfp8 weight [N, K/4] (or [B, N, K/4] batched with x [B, M, K]) -> [..., M, N], or
    None when the shape is not admitted (the caller keeps mx.quantized_matmul)."""
    if not fp8_rows_requested() or x.ndim < 2:
        return None
    lead = x.shape[:-2]
    M, K = x.shape[-2], x.shape[-1]
    if w.ndim == 3:
        if x.ndim != 3 or x.shape[0] != w.shape[0]:
            return None
        x3 = x
    else:
        x3 = x.reshape(-1, M, K) if lead else x[None]
        if x3.shape[0] != 1:      # several shared-weight matrices of rows: fold them into rows when they fit
            return None
    if not _admissible_shape(x3, w, scales) or CONFIG.get(M) is None:
        return None
    wbatch = w.ndim == 3
    key = (K, int(w.shape[-2]), M, str(x3.dtype), wbatch)
    ok = _ADMITTED.get(key)
    if ok is None:
        ok = _self_check(x3, w, scales, wbatch)
        _ADMITTED[key] = ok
        STATS["admitted_shapes" if ok else "rejected_shapes"] += 1
        if not ok:
            logger.info("fp8 rows QMV: shape %s not bit-equal to the stock M=1 kernel; keeping mx.quantized_matmul", key)
    if not ok:
        return None
    STATS["calls"] += 1
    y = _launch(x3, w, scales, wbatch)
    return y.reshape(*lead, M, -1) if lead else y[0]
