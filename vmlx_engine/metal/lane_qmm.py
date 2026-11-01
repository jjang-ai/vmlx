# SPDX-License-Identifier: Apache-2.0
"""Affine quantized matmul for 1..128 rows on the M5 matrix units whose output rows do not depend on the row count.

Why (Qwen3.8-27B DFlash2 verify)
--------------------------------
MLX's ``quantized_matmul`` changes kernel with the row count (vector kernels for a few rows, a tiled kernel from 6
rows); on the 27B JANG_4D target one row costs 43 ms, 5 rows 53, 6 rows 69, 8 rows 79 and 12 rows 115. Verify cost
should be ~flat in the number of drafted rows. This module always runs a 16- or 32-row tile on the matrix units
(Metal 4 ``mpp::tensor_ops::matmul2d``), so 1..16 rows cost about the same, and since the per-row arithmetic never
depends on how many rows share the call, a verify row's bits equal the one-row decode's.

How
---
* The matrix unit multiplies the bf16 rows by the weight's RAW unsigned codes, one quantization group (G inputs) at a
  time, in fp32. The affine dequant is folded in afterwards per group:
  ``C = fma(scale[g, n], P[m, n], fma(bias[g, n], rowsum[g, m], C))`` with ``rowsum[g, m]`` the fp32 sum of the
  row's G inputs of group g. That is ``x @ (scale * code + bias)^T`` reassociated per group.
* 4-bit codes go to the unit straight from the packed weight (``uint4b_format``). 2- and 3-bit codes are re-packed to
  nibbles and 5-, 6- and 8-bit codes to bytes in threadgroup memory first.
* The groups are cut into SK contiguous slices (``split_k``, a property of the weight shape only), one simdgroup
  each; slice results are added in slice order.
* Scales and biases are packed once per weight as (K/G, N, 2) bf16 pairs (``pack_scales``).

Trade-offs: not bit-identical to MLX's own quantized kernels (another summation order), but identical across row
counts. Affine weights with bf16 scales, G = 64 (or 32 for 4-bit), K % 64 == 0, N % 4 == 0 (``supports``); needs
Metal 4 tensor ops (M5-class GPU), callers check ``available()``.
"""
from __future__ import annotations

import os
import threading

import mlx.core as mx
import mlx.nn as nn

MAX_ROWS = 128
# Output width from which a projection counts as a vocabulary head (lm_head / tied embedding) and stays on MLX.
VOCAB_HEAD_MIN_N = 65536
BITS = (2, 3, 4, 5, 6, 8)
COLS = 32           # output columns per threadgroup (one matrix-unit tile)
ROW_TILE = 16       # rows of the smallest matrix-unit tile
MAX_ROW_BLOCK = 32  # rows per threadgroup

_MPP_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# Per-group fp32 row sums of x: rowsum[g, m] = sum of x[m, g*G : (g+1)*G], left to right. The row count is padded
# to whole row blocks (rows[1]) and the padding rows are 0, so the matmul reads them without bounds checks.
_ROWSUM_BODY = """
  const int M = rows[0];
  const int RS = rows[1];
  const uint g = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (g >= uint(K / G) || int(m) >= RS) {
    return;
  }
  float total = 0.0f;
  if (int(m) < M) {
    const device bfloat* src = (const device bfloat*)X + m * K + g * G;
    for (int i = 0; i < G; i++) {
      total += float(src[i]);
    }
  }
  rowsum[g * RS + m] = total;
"""

# Main kernel. Baked constants: N, K, G, BITS, SK (K slices), ROW_BLOCK (16 or 32), TILED.
_MATMUL_BODY = """
  constexpr int GROUPS = K / G;
  constexpr int STAGE_WORDS = BITS == 4 ? 0 : (BITS < 4 ? 8 : 16);  // per column: 64 codes as nibbles or bytes
  const int M = rows[0];
  const int RS = rows[1];  // row stride of rowsum (rows padded to whole row blocks)
  const uint lane = thread_index_in_simdgroup;
  const uint slice = simdgroup_index_in_threadgroup;
  const int tile = threadgroup_position_in_grid.x;
  const int col0 = tile * COLS;
  const int row0 = threadgroup_position_in_grid.y * ROW_BLOCK;
  const int g_first = (int(slice) * GROUPS) / SK;
  const int g_stop = ((int(slice) + 1) * GROUPS) / SK;

  constexpr auto desc = matmul2d_descriptor(ROW_BLOCK, COLS, G, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> unit;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> xs_view(
      (device bfloat*)X + int64_t(row0) * K, dextents<int32_t, 2>(K, M - row0));

  // Code views: 4-bit straight from the packed weight, other widths from this slice's staging area.
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> w_view((device uchar*)Wq, dextents<int32_t, 2>(K, N));
  threadgroup uint staging[STAGE_WORDS > 0 ? SK * COLS * STAGE_WORDS : 1];
  threadgroup uint* stage = staging + slice * COLS * STAGE_WORDS;
  using stage_elem = metal::conditional_t<(BITS < 4), uint4b_format, uint8_t>;
  tensor<threadgroup stage_elem, dextents<int32_t, 2>, tensor_inline> stage_view(
      (threadgroup uchar*)stage, dextents<int32_t, 2>(64, COLS));

  auto probe = unit.template get_destination_cooperative_tensor<decltype(xs_view.slice(0, 0)), decltype(stage_view), float>();
  constexpr int ELEMS = ROW_BLOCK * COLS / 32;  // fp32 results per lane (probe.get_capacity())
  float acc[ELEMS];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < ELEMS; i++) {
    acc[i] = 0.0f;
  }
  // Coordinates of this thread's elements: runs of 4 consecutive columns on one row.
  short run_col[ELEMS / 4];
  short run_row[ELEMS / 4];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < ELEMS / 4; r++) {
    const auto at = probe.get_multidimensional_index(r * 4);
    run_col[r] = at[0];
    run_row[r] = at[1];
  }
  const device uint4* pairs = (const device uint4*)SBt;  // 4 columns of (scale, bias) bf16 pairs per load

  // Fold group g's products into the accumulators: acc = fma(scale, P, fma(bias, rowsum, acc)).
  auto fold = [&](int g, thread auto& prod) {
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < ELEMS / 4; r++) {
      const int col = col0 + run_col[r];
      const int row = row0 + run_row[r];
      const uint4 raw = (N % COLS == 0 || col < N) ? pairs[(g * N + col) / 4] : uint4(0);
      const vec<bfloat, 8> sb = as_type<vec<bfloat, 8>>(raw);
      const float rs = rowsum[g * RS + row];
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 4; j++) {
        const int i = r * 4 + j;
        acc[i] = fma(float(sb[2 * j]), prod[i], fma(float(sb[2 * j + 1]), rs, acc[i]));
      }
    }
  };

  for (int g = g_first; g < g_stop; g++) {
    auto x_g = xs_view.slice(g * G, 0);
    if constexpr (BITS == 4) {
      auto w_g = w_view.slice(g * G, col0);
      auto prod = unit.template get_destination_cooperative_tensor<decltype(x_g), decltype(w_g), float>();
      if constexpr (TILED) {
        // One (column tile, group) block is contiguous: COLS columns x G codes.
        tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> w_tile(
            (device uchar*)Wq + (int64_t(tile) * GROUPS + g) * (COLS * G / 2), dextents<int32_t, 2>(G, COLS));
        unit.run(x_g, w_tile, prod);
      } else {
        unit.run(x_g, w_g, prod);
      }
      fold(g, prod);
    } else {
      // Lane = one output column: unpack its 64 codes of group g into the staging area.
      constexpr int SRC_WORDS = 2 * BITS;  // 64 codes x BITS bits
      constexpr int OUT_BITS = BITS < 4 ? 4 : 8;
      constexpr int PER_WORD = 32 / OUT_BITS;
      constexpr uint CODE_MASK = (1u << BITS) - 1u;
      const int col = col0 + int(lane);
      uint src[SRC_WORDS];
      const device uint* words = (const device uint*)Wq;
      const device uint* from = TILED ? words + ((int64_t(tile) * GROUPS + g) * COLS + lane) * SRC_WORDS
                                      : words + int64_t(col) * (K * BITS / 32) + g * SRC_WORDS;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < SRC_WORDS; i++) {
        src[i] = (N % COLS == 0 || col < N) ? from[i] : 0u;
      }
      _Pragma("clang loop unroll(full)")
      for (int w = 0; w < STAGE_WORDS; w++) {
        // The PER_WORD codes of output word w are one contiguous field of the source bits.
        constexpr int FIELD = PER_WORD * BITS;
        const int bit = w * FIELD;
        const int i = bit / 32;
        const int shift = bit % 32;
        uint packed = src[i] >> shift;
        if (shift + FIELD > 32) {
          packed |= src[i + 1] << (32 - shift);
        }
        if (FIELD < 32) {
          packed &= (1u << FIELD) - 1u;
        }
        // Spread the field so code e starts at bit e * OUT_BITS: repeatedly move the upper half of every block of
        // codes up to its output position (blocks of PER_WORD codes, then PER_WORD / 2, ...).
        if (BITS != OUT_BITS) {
          // SPREAD_KEEP_<level>: per block, the low (codes moved) * BITS bits (computed on the host, see _spread_masks).
          constexpr int LEVELS = PER_WORD == 8 ? 3 : 2;  // log2(PER_WORD)
          constexpr uint keeps[3] = {SPREAD_KEEP_0, SPREAD_KEEP_1, SPREAD_KEEP_2};
          _Pragma("clang loop unroll(full)")
          for (int level = 0; level < LEVELS; level++) {
            const int span = PER_WORD >> (level + 1);   // codes moved per block at this level
            packed = (packed & keeps[level]) | (((packed >> (span * BITS)) & keeps[level]) << (span * OUT_BITS));
          }
        }
        stage[lane * STAGE_WORDS + w] = packed;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      auto prod = unit.template get_destination_cooperative_tensor<decltype(x_g), decltype(stage_view), float>();
      unit.run(x_g, stage_view, prod);
      simdgroup_barrier(mem_flags::mem_threadgroup);  // the unit has read the stage before it is overwritten
      fold(g, prod);
    }
  }

  // Add the K slices in slice order, 16 accumulators at a time (bounded threadgroup memory), and store.
  constexpr int CHUNK = 16;
  threadgroup float spill[(SK > 1 ? SK - 1 : 1) * CHUNK * 32];
  _Pragma("clang loop unroll(full)")
  for (int c0 = 0; c0 < ELEMS; c0 += CHUNK) {
    if (SK > 1) {
      if (slice > 0) {
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < CHUNK; i++) {
          spill[((slice - 1) * CHUNK + i) * 32 + lane] = acc[c0 + i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (slice == 0) {
        _Pragma("clang loop unroll(full)")
        for (int s = 1; s < SK; s++) {
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < CHUNK; i++) {
            acc[c0 + i] += spill[((s - 1) * CHUNK + i) * 32 + lane];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (slice == 0) {
      _Pragma("clang loop unroll(full)")
      for (int r = c0 / 4; r < (c0 + CHUNK) / 4; r++) {
        const int col = col0 + run_col[r];
        const int row = row0 + run_row[r];
        if (row < M && col < N) {
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < 4; j++) {
            Y[int64_t(row) * N + col + j] = static_cast<bfloat>(acc[r * 4 + j]);
          }
        }
      }
    }
  }
"""


def _spread_masks(bits: int) -> list[int]:
    """Masks of the code-spreading steps in the matmul kernel for ``bits``-bit codes widened to nibbles (2, 3 bits)
    or bytes (5, 6, 8): at level l, blocks of 2 * span codes (span = codes per word >> (l + 1)) sit at a stride of
    2 * span output slots; the mask keeps each block's low span codes."""
    out_bits = 4 if bits < 4 else 8
    per_word = 32 // out_bits
    masks = []
    for level in range(3):
        span = per_word >> (level + 1)
        keep = 0
        if span >= 1:
            for blk in range(1 << level):
                keep |= ((1 << (span * bits)) - 1) << (blk * 2 * span * out_bits)
        masks.append(keep & 0xFFFFFFFF)
    return masks


def _source(body: str, consts: dict) -> str:
    text = "".join(f"  constexpr int {k} = {int(v)};\n" for k, v in consts.items())
    if "BITS" in consts:
        text += "".join(f"  constexpr uint SPREAD_KEEP_{i} = {m}u;\n" for i, m in enumerate(_spread_masks(consts["BITS"])))
    return text + body


_KERNELS: dict = {}
_KERNELS_LOCK = threading.Lock()


def _compiled(kind: str, consts: dict):
    key = (kind, tuple(sorted(consts.items())))
    kern = _KERNELS.get(key)
    if kern is None:
        with _KERNELS_LOCK:
            kern = _KERNELS.get(key)
            if kern is None:
                tag = "_".join(f"{k.lower()}{int(v)}" for k, v in sorted(consts.items()))
                if kind == "rowsum":
                    kern = mx.fast.metal_kernel(name=f"vmlx_lane_rowsum_{tag}", input_names=["X", "rows"],
                                                output_names=["rowsum"], source=_source(_ROWSUM_BODY, consts))
                else:
                    kern = mx.fast.metal_kernel(name=f"vmlx_lane_matmul_{tag}",
                                                input_names=["X", "rowsum", "Wq", "SBt", "rows"],
                                                output_names=["Y"], source=_source(_MATMUL_BODY, consts),
                                                header=_MPP_HEADER)
                _KERNELS[key] = kern
    return kern


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count. Doubles while the grid stays under
    1024 threadgroups and every slice keeps at least 8 groups of 64, up to 8."""
    tiles = -(-n // COLS)
    sk = 1
    while sk < 8 and tiles * sk < 1024 and (k // 64) // (sk * 2) >= 8:
        sk *= 2
    return sk


def tile_weight(weight: mx.array, *, bits: int, group: int = 64) -> mx.array:
    """MLX's packed (N, K*bits/32) weight regrouped as [N/32][K/group][32 columns x one group's words] (same bytes),
    so one (column tile, group) block is contiguous."""
    n, kw, w = int(weight.shape[0]), int(weight.shape[1]), group * bits // 32
    return mx.contiguous(weight.reshape(n // COLS, COLS, kw // w, w).transpose(0, 2, 1, 3).reshape(n, kw))


def untile_weight(weight: mx.array, *, bits: int, group: int = 64) -> mx.array:
    """Inverse of ``tile_weight``: MLX's packed layout again."""
    n, kw, w = int(weight.shape[0]), int(weight.shape[1]), group * bits // 32
    return mx.contiguous(weight.reshape(n // COLS, kw // w, COLS, w).transpose(0, 2, 1, 3).reshape(n, kw))


def pack_scales(scales: mx.array, biases: mx.array) -> mx.array:
    """(N, K/G) scales and biases -> (K/G, N, 2) bf16 (scale, bias) pairs, group-major."""
    return mx.stack([scales.T, biases.T], axis=-1).astype(mx.bfloat16)


def _groups_ok(bits: int, group: int) -> bool:
    return bits in BITS and group in ((32, 64) if bits == 4 else (64,))


def supports(weight: mx.array, scales: mx.array, x: mx.array, bits: int, group_size: int, mode: str = "affine") -> bool:
    if mode != "affine" or not _groups_ok(bits, group_size):
        return False
    if x.dtype != mx.bfloat16 or scales.dtype != mx.bfloat16 or weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    k = int(x.shape[-1])
    return k % 64 == 0 and int(weight.shape[1]) * 32 == k * bits and int(weight.shape[0]) % 4 == 0


_AVAILABLE = None


def available() -> bool:
    """Metal 4 tensor ops compile here and give the right answer (checked once per process)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            w = mx.random.normal((64, 128)).astype(mx.bfloat16)
            q, s, b = mx.quantize(w, group_size=64, bits=4)
            x = mx.random.normal((3, 128)).astype(mx.bfloat16)
            y = lane_matmul(x, q, pack_scales(s, b), bits=4, group=64)
            ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=4).astype(mx.float32).T
            _AVAILABLE = bool(mx.allclose(y.astype(mx.float32), ref, atol=0.25, rtol=0.05).item())
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


class _Launch:
    """Everything one (N, K, M, bits, group, tiled) call needs, built once: per-call graph building is then a couple
    of kernel invocations (it otherwise costs several microseconds per projection, hundreds of times per step)."""

    __slots__ = ("rowsum", "rowsum_kw", "matmul", "matmul_kw", "rows", "rowsum_key")

    def __init__(self, N: int, K: int, M: int, bits: int, group: int, tiled: bool) -> None:
        if not _groups_ok(bits, group):
            raise ValueError(f"lane matmul: unsupported {bits}-bit weights with groups of {group}")
        mp = ROW_TILE * -(-M // ROW_TILE)
        groups = K // group
        block = min(mp, MAX_ROW_BLOCK)
        rp = block * -(-mp // block)  # rows padded to whole row blocks: the matmul's rowsum reads need no guard
        self.rows = mx.array([M, rp], dtype=mx.int32)
        self.rowsum = _compiled("rowsum", {"K": K, "G": group})
        self.rowsum_kw = dict(grid=(groups, rp, 1), threadgroup=(min(groups, 256), 1, 1),
                              output_shapes=[(groups, rp)], output_dtypes=[mx.float32])
        sk = split_k(N, K)
        self.matmul = _compiled("matmul", {"N": N, "K": K, "G": group, "BITS": bits, "SK": sk, "COLS": COLS,
                                           "ROW_BLOCK": block, "TILED": int(tiled)})
        self.matmul_kw = dict(grid=(-(-N // COLS) * 32 * sk, -(-mp // block), 1), threadgroup=(32 * sk, 1, 1),
                              output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])
        self.rowsum_key = (K, group, rp)


_LAUNCHES: dict = {}

# The row sums depend only on x, K, G and the padded row count, not on the weight. Projections fed the SAME
# activation (q/k/v, gate/up, a GDN layer's in-projections) run back to back, so the last row sums are kept with a
# reference to their x and reused when the next call passes that same array object (holding x keeps its id from being
# recycled). One slot per thread; identical bits since it is the same kernel on the same x.
# VMLX_LANE_SHARE_XSUM=0 recomputes them for every call (A/B).
_SHARE_XSUM = os.environ.get("VMLX_LANE_SHARE_XSUM", "1") != "0"
_XS_LAST = threading.local()


def lane_matmul(x: mx.array, weight: mx.array, sbt: mx.array, *, bits: int, group: int = 64,
                tiled: bool = False) -> mx.array:
    """x (..., K) bf16 times the packed ``weight`` (N, K*bits/32) transposed; at most MAX_ROWS rows."""
    K = int(x.shape[-1])
    N = int(weight.shape[0])
    lead = x.shape[:-1]
    M = 1
    for d in lead:
        M *= int(d)
    if M > MAX_ROWS:
        raise ValueError(f"lane_matmul takes at most {MAX_ROWS} rows, got {M}")
    key = (N, K, M, bits, group, bool(tiled))
    launch = _LAUNCHES.get(key)
    if launch is None:
        launch = _LAUNCHES[key] = _Launch(N, K, M, bits, group, bool(tiled))
    x2 = x.reshape(M, K)
    last = getattr(_XS_LAST, "slot", None) if _SHARE_XSUM else None
    if last is not None and last[0] is x and last[1] == launch.rowsum_key:
        rowsum = last[2]
    else:
        rowsum = launch.rowsum(inputs=[x2, launch.rows], **launch.rowsum_kw)[0]
        if _SHARE_XSUM:
            _XS_LAST.slot = (x, launch.rowsum_key, rowsum)
    y = launch.matmul(inputs=[x2, rowsum, weight, sbt, launch.rows], **launch.matmul_kw)[0]
    return y.reshape(*lead, N)


# -- routing a model's projections -------------------------------------------------------------------------------
#
# Installed per module by class swap, never by patching nn.QuantizedLinear.__call__: drafters and every other model
# in the process keep MLX's kernels. Calls of <= MAX_ROWS bf16 rows (decode, DFlash2 verify, short resume deltas)
# take the lane kernel; wider calls (prefill chunks) and other dtypes take MLX's quantized_matmul on the MLX-layout
# weight (rebuilt per call when tiled: one weight copy per call, ~1-2 % of a 2048-row chunk's compute).


class LaneQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, x: mx.array) -> mx.array:
        rows = 1
        for d in x.shape[:-1]:
            rows *= int(d)
        tiled = getattr(self, "_lane_tiled", False)
        if rows <= MAX_ROWS and x.dtype == mx.bfloat16:
            y = lane_matmul(x, self["weight"], self._lane_sbt, bits=self.bits, group=self.group_size, tiled=tiled)
        else:
            weight = untile_weight(self["weight"], bits=self.bits, group=self.group_size) if tiled else self["weight"]
            if "scales" in self:
                scales, biases = self["scales"], self["biases"]
            else:  # dropped at install: rebuilt per call from the packed pairs (same bf16 values)
                sbt = self._lane_sbt
                scales, biases = mx.contiguous(sbt[..., 0].T), mx.contiguous(sbt[..., 1].T)
            y = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                    group_size=self.group_size, bits=self.bits)
        if "bias" in self:
            y = y + self["bias"]
        return y


def _eligible(module) -> bool:
    w, s = module["weight"], module["scales"]
    k = int(w.shape[1]) * 32 // int(module.bits) if w.ndim == 2 else 0
    return (type(module) is nn.QuantizedLinear and getattr(module, "mode", "affine") == "affine"
            and _groups_ok(module.bits, module.group_size)
            and s.dtype == mx.bfloat16 and w.dtype == mx.uint32 and w.ndim == 2
            and k % 64 == 0 and int(w.shape[0]) % 4 == 0)


def _count(table: dict, key: str) -> None:
    table[key] = table.get(key, 0) + 1


def install(model, *, tile: bool = True) -> dict:
    """Route ``model``'s plain affine QuantizedLinear layers through the lane matmul.

    Returns counts {"lane": n, "tiled": n, "skipped": {kind: n}} for the load log.
    """
    if not available():
        return {"lane": 0, "tiled": 0, "skipped": {"no Metal 4 tensor ops": 1}}
    counts = {"lane": 0, "tiled": 0, "skipped": {}}
    keep_vocab = os.environ.get("VMLX_LANE_VOCAB_HEAD", "0") == "1"
    keep_scales = os.environ.get("VMLX_LANE_KEEP_SCALES", "0") == "1"
    pending, pending_bytes = [], 0
    for _name, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear) or isinstance(module, LaneQuantizedLinear):
            continue
        if not keep_vocab and int(module["weight"].shape[0]) >= VOCAB_HEAD_MIN_N:
            # Vocabulary-sized heads stay on MLX: measured 2026-10-11 (M5 Max, N=248,320 K=5,120 8-bit g64) the lane
            # kernel is slower at every row count the DFlash2 lane uses: 1 row 3.46 vs 2.54 ms, 5 rows 3.95 vs 2.62,
            # 8 rows 3.95 vs 3.98, 16 rows 3.95 vs 3.57. VMLX_LANE_VOCAB_HEAD=1 routes them too (A/B).
            _count(counts["skipped"], "vocab-sized head")
            continue
        if not _eligible(module):
            _count(counts["skipped"],
                   f"{type(module).__name__} {module.bits}-bit g{module.group_size} {module['scales'].dtype}")
            continue
        sbt = pack_scales(module["scales"], module["biases"])
        object.__setattr__(module, "_lane_sbt", sbt)
        pending.append(sbt)
        pending_bytes += sbt.nbytes
        if module["biases"].dtype == mx.bfloat16 and not keep_scales:
            # The packed pairs hold exactly the bf16 scales and biases, so the originals would be a second resident
            # copy (Qwen3.8 27B JANG_4D: 1.42 GiB) that only the > MAX_ROWS fallback reads; it rebuilds them per call
            # instead (transient, ~1/16 of the weight bytes, same values). Non-bf16 biases are kept (the pairs hold a
            # bf16 rounding of them). VMLX_LANE_KEEP_SCALES=1 keeps them.
            module.pop("scales")
            module.pop("biases")
        w = module["weight"]
        if tile and int(w.shape[0]) % COLS == 0 and (module.bits == 4 or module.group_size == 64):
            module.weight = tile_weight(w, bits=module.bits, group=module.group_size)
            object.__setattr__(module, "_lane_tiled", True)
            pending.append(module["weight"])
            pending_bytes += 2 * w.nbytes  # the old layout lives until this batch is evaluated
            counts["tiled"] += 1
        module.__class__ = LaneQuantizedLinear
        counts["lane"] += 1
        if pending_bytes >= 2 * 1024**3:
            mx.eval(pending)
            pending, pending_bytes = [], 0
    if pending:
        mx.eval(pending)
    mx.clear_cache()
    return counts
