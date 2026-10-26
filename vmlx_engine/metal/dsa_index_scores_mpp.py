# SPDX-License-Identifier: Apache-2.0
"""DSA indexer pool scores on the M5 matrix units: out[q, k] = sum_h w[q, h] * relu(scale * Q[q, h, :] . K[k, :]).

WHY (DeepSeek-V4.1 long-context prefill, 2026-10-26)
----------------------------------------------------
The fp32 fused kernel (``dsa_index_scores.py``) runs its q.k products as 8x8 ``simdgroup_float8x8`` MMAs: 61 ms per
index layer per 2048-query chunk at 100k keys on the M5 Ultra (~27 TFLOPS), and ``simdgroup_half8x8`` fragments were
no faster (63 ms). The release computes these scores from bf16 operands, so the same function can run on the matrix
units (Metal 4 ``mpp::tensor_ops::matmul2d``): bf16 x bf16 products are exact in fp32 and the op accumulates in fp32.

HOW
---
One threadgroup of NSG simdgroups owns a TQ x TK (queries x keys) output tile. Per head h it runs one cooperative
[TQ x D] @ [D x TK] ``matmul2d`` into a destination cooperative tensor, then applies scale, relu and the head weight
w[q, h] (staged in threadgroup memory) element-wise in registers and adds into a second cooperative tensor that holds
the tile's running head sum. The [S, T] fp32 tile is written once; there is no [S, H, T] intermediate in device memory.
The last row/column tile is clamped to end at S / T (it overlaps its neighbour; overlapping cells are recomputed with
the same arithmetic), and per-threadgroup base pointers keep every tensor offset far below 2^31 at 1M keys.

Tile shape (M5 Ultra, S=2048, T=100k, H=32, D=128, bf16): TQ x TK x NSG = 16x256x4 14 ms (~119 TFLOPS); 32x128x4
18 ms; 16x512x8 16 ms; 16x128x4 22 ms; 64x128x4 43 ms; 128x128x4 75 ms. Wider per-thread tiles spill (16x512x2 and
16x1024x4 compute wrong results on this compiler, so the shape is fixed and the self-check in ``available()`` runs it).

ARITHMETIC
----------
Not bit-identical to the fp32 kernel: the operands are rounded to bf16 (exact for callers whose operands are bf16
values, as DeepSeek-V4.1's are) and the op's dot-product order is its own. Heads are summed 0..H-1 in fp32.
"""
from __future__ import annotations

import hashlib
import os

import mlx.core as mx

TQ, TK, NSG = 16, 256, 4

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_SOURCE = r"""
  const int S = int(q_shape[1]);
  const int T = int(k_shape[1]);
  const int c0 = min(int(threadgroup_position_in_grid.x) * TK, T - TK);
  const int r0 = min(int(threadgroup_position_in_grid.y) * TQ, S - TQ);
  const uint tid = thread_position_in_threadgroup.x;
  const float sc = scale[0];

  threadgroup float ws[TQ * H];                        // this tile's head weights, [TQ][H]
  for (uint i = tid; i < TQ * H; i += NSG * 32) ws[i] = float(w[(size_t)r0 * H + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc = matmul2d_descriptor(TQ, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  // Q rows r0.. as a [TQ][H*D] matrix (head h is columns h*D..h*D+D), K rows c0.. as [TK][D] (transposed operand).
  tensor<device elem_t, dextents<int32_t, 2>, tensor_inline> tQ(
      (device elem_t*)q + (size_t)r0 * (H * D), dextents<int32_t, 2>(H * D, TQ));
  tensor<device elem_t, dextents<int32_t, 2>, tensor_inline> tK(
      (device elem_t*)k + (size_t)c0 * D, dextents<int32_t, 2>(D, TK));
  auto b = tK.slice<D, TK>(0, 0);
  auto a0 = tQ.slice<D, TQ>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b), float>();
  #pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;

  for (int h = 0; h < H; ++h) {
    auto a = tQ.slice<D, TQ>(h * D, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    #pragma unroll
    for (uint16_t i = 0; i < P.get_capacity(); ++i) {
      if (P.is_valid_element(i)) {
        const auto idx = P.get_multidimensional_index(i);      // (key column, query row) within the tile
        const float s = HAS_SCALE ? P[i] * sc : P[i];
        acc[i] += metal::max(s, 0.0f) * ws[idx[1] * H + h];
      }
    }
  }

  // The tile's [TQ][TK] block of the [S][T] output (row stride T).
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(
      out + (size_t)r0 * T + c0, dextents<int32_t, 2>(TK, TQ), array<int32_t, 2>{1, T});
  acc.store(tO);
"""

_COMPILED: dict = {}


def _kernel(elem: str, H: int, D: int, has_scale: bool):
    key = (elem, H, D, has_scale)
    run = _COMPILED.get(key)
    if run is None:
        consts = (f"  using elem_t = {elem};\n"
                  + "".join(f"  constexpr int {n} = {int(v)};\n"
                            for n, v in (("H", H), ("D", D), ("TQ", TQ), ("TK", TK), ("NSG", NSG),
                                         ("HAS_SCALE", int(has_scale)))))
        source = consts + _SOURCE
        name = f"vmlx_dsa_mpp_index_scores_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"
        run = _COMPILED[key] = mx.fast.metal_kernel(
            name=name, input_names=["q", "k", "w", "scale"], output_names=["out"], source=source, header=_HEADER)
    return run


_ELEM = {mx.bfloat16: "bfloat", mx.float16: "half"}


def _launch(q, k, w, scale: float):
    S, H, D = (int(d) for d in q.shape[1:])
    T = int(k.shape[1])
    run = _kernel(_ELEM[q.dtype], H, D, scale != 1.0)
    return run(
        inputs=[q, k, w, mx.array([float(scale)], dtype=mx.float32)],
        grid=(-(-T // TK) * 32 * NSG, -(-S // TQ), 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(1, S, T)],
        output_dtypes=[mx.float32],
    )[0]


def mpp_index_requested() -> bool:
    return os.environ.get("VMLX_DSA_MPP_INDEX", "1").strip().lower() not in {"0", "false", "off", "no"}


_AVAILABLE = None


def available() -> bool:
    """Metal 4 tensor ops compile here and this kernel matches an fp32 reference (checked once per process),
    including clamped row/column tails."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            if mx.default_device() != mx.gpu or not mx.metal.is_available():
                raise RuntimeError("no Metal GPU")
            key = mx.random.key(7)
            k1, k2, k3 = mx.random.split(key, 3)
            S, T, H, D = 40, 600, 4, 128
            q = mx.random.normal((1, S, H, D), key=k1).astype(mx.bfloat16)
            kk = mx.random.normal((1, T, D), key=k2).astype(mx.bfloat16)
            w = mx.random.normal((1, S, H), key=k3)
            got = _launch(q, kk, w, 0.5)
            s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), kk.astype(mx.float32), stream=mx.cpu)
            ref = mx.sum(mx.maximum(s * 0.5, 0.0) * w[..., None], axis=2, stream=mx.cpu)
            err = (mx.abs(got - ref).max() / mx.maximum(mx.abs(ref).max(), 1e-6)).item()
            _AVAILABLE = bool(err < 1e-4)
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def dsa_pool_index_scores_mpp(q: mx.array, pool_keys: mx.array, head_w: mx.array, scale: float, *,
                              enabled: bool = True):
    """q (1, S, H, D), pool_keys (1, P, D) — both bf16 or both fp16 — and head_w (1, S, H) fp32/bf16/fp16 ->
    (1, S, P) fp32 = sum_h head_w * relu(scale * q . pool_key), on the matrix units; None when not admitted (callers
    fall back to ``dsa_pool_index_scores`` / the stock path).

    Admitted: batch 1, D a multiple of 32, 1 <= H <= 256, S >= 1, P >= 1, scale > 0, Metal 4 tensor ops present and
    passing ``available()``. S below one 16-row tile or P below one 256-key tile is zero-padded to one tile."""
    if not enabled or not mpp_index_requested():
        return None
    if q.ndim != 4 or pool_keys.ndim != 3 or head_w.ndim != 3:
        return None
    B, S, H, D = (int(d) for d in q.shape)
    if B != 1 or pool_keys.shape[0] != 1 or int(pool_keys.shape[2]) != D or tuple(head_w.shape) != (1, S, H):
        return None
    P = int(pool_keys.shape[1])
    if (q.dtype not in _ELEM or pool_keys.dtype != q.dtype
            or head_w.dtype not in (mx.float32, mx.bfloat16, mx.float16)):
        return None
    if D % 32 or D > 512 or not 1 <= H <= 256 or S < 1 or P < 1 or not float(scale) > 0.0:
        return None
    if not available():
        return None
    pad_s, pad_p = max(0, TQ - S), max(0, TK - P)
    if pad_s or pad_p:
        qq = mx.pad(q, [(0, 0), (0, pad_s), (0, 0), (0, 0)]) if pad_s else q
        ww = mx.pad(head_w, [(0, 0), (0, pad_s), (0, 0)]) if pad_s else head_w
        kk = mx.pad(pool_keys, [(0, 0), (0, pad_p), (0, 0)]) if pad_p else pool_keys
        return _launch(qq, kk, ww, float(scale))[:, :S, :P]
    return _launch(q, pool_keys, head_w, float(scale))
