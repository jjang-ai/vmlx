# SPDX-License-Identifier: Apache-2.0
"""Opt-in Gated-DeltaNet prefill for Qwen4 with token-tiled staging.

WHAT
----
A drop-in for mlx-lm's ``gated_delta_update`` on long prefill chunks.  The
numbers it produces are mlx-lm's scalar-gated recurrence, bit for bit: every
value row keeps a 128-wide fp32 state split four elements per SIMD lane, and
each token does

    s *= g;  kv = simd_sum(lane-local sum of s * k);  d = (v - kv) * beta;
    s += k * d;  y = simd_sum(lane-local sum of s * q)

in exactly that order.  Only the memory schedule differs.

WHY
---
mlx-lm's kernel runs one 32-thread row per value dimension and every row walks
the whole sequence reading k and q straight from device memory, so a 128-row
value head reads the same key/query rows 128 times.  Here one threadgroup owns
a 32-row slice of a value head.  Per tile of TB tokens it copies the shared
key/query rows (and its v columns, g and beta) into threadgroup memory once,
then every simdgroup walks the tile from there.

Lever: ``VMLX_QWEN4_GDN_BLOCKED_PREFILL=1`` (default off).  Tile length:
``VMLX_QWEN4_GDN_BLOCKED_PREFILL_TB`` in {16, 32, 48}; default 32, or 16 for
float32 inputs (32 float32 tokens of keys and queries alone fill the 32 KiB
threadgroup budget).  Masked calls, decode/verify widths (< 16 tokens) and any
other geometry go to mlx-lm unchanged.
"""

from __future__ import annotations

import os
from typing import Tuple

import mlx.core as mx

TOKEN_TILES = (16, 32, 48)
KEY_DIM = 128            # four fp32 state elements per lane, 32 lanes
ROWS_PER_SIMDGROUP = 4
SIMDGROUPS = 8
SLICE_ROWS = ROWS_PER_SIMDGROUP * SIMDGROUPS   # value rows per threadgroup
MIN_TOKENS = 16

_LEVER = "VMLX_QWEN4_GDN_BLOCKED_PREFILL"
_TILE_LEVER = "VMLX_QWEN4_GDN_BLOCKED_PREFILL_TB"

_BODY = r"""
  #define GDNT_UNROLL _Pragma("clang loop unroll(full)")
  const uint lane = thread_index_in_simdgroup;
  const uint group = simdgroup_index_in_threadgroup;
  const uint tid = thread_position_in_threadgroup.x;
  const int slice = int(threadgroup_position_in_grid.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int batch = int(threadgroup_position_in_grid.z);
  const int key_head = head / (HV / HK);
  const int steps = int(T);

  constexpr int THREADS = 32 * NSG;
  constexpr int SLICE = ROWS * NSG;
  const int my_rows = slice * SLICE + int(group) * ROWS;
  const int my_cols = int(lane) * 4;

  threadgroup InT keys_tile[TB][128];
  threadgroup InT queries_tile[TB][128];
  threadgroup InT values_tile[TB][SLICE];
  threadgroup float decay_tile[TB];
  threadgroup float beta_tile[TB];

  float s[ROWS][4];
  const size_t state_base = ((size_t)batch * HV + head) * DV;
  GDNT_UNROLL for (int r = 0; r < ROWS; ++r) {
    device const float* src = state_in + (state_base + my_rows + r) * 128 + my_cols;
    GDNT_UNROLL for (int i = 0; i < 4; ++i) s[r][i] = src[i];
  }

  for (int first = 0; first < steps; first += TB) {
    const int count = min(TB, steps - first);
    // four contiguous elements per copy (rows are 128 wide, so offsets stay 4-aligned)
    for (int idx = int(tid); idx < count * 32; idx += THREADS) {
      const int t = idx / 32, c = (idx % 32) * 4;
      const size_t at = (((size_t)batch * steps + first + t) * HK + key_head) * 128 + c;
      *(threadgroup vec<InT, 4>*)&keys_tile[t][c] = *(const device vec<InT, 4>*)(k + at);
      *(threadgroup vec<InT, 4>*)&queries_tile[t][c] = *(const device vec<InT, 4>*)(q + at);
    }
    for (int idx = int(tid); idx < count * SLICE; idx += THREADS) {
      const int t = idx / SLICE, c = idx % SLICE;
      values_tile[t][c] =
          v[(((size_t)batch * steps + first + t) * HV + head) * DV + slice * SLICE + c];
    }
    for (int idx = int(tid); idx < count; idx += THREADS) {
      const size_t at = ((size_t)batch * steps + first + idx) * HV + head;
      decay_tile[idx] = g[at];
      beta_tile[idx] = beta[at];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < count; ++t) {
      const float decay = decay_tile[t];
      const float b = beta_tile[t];
      // this lane's four key/query elements, shared by all of its rows
      const vec<float, 4> kk = vec<float, 4>(*(threadgroup const vec<InT, 4>*)&keys_tile[t][my_cols]);
      const vec<float, 4> qq = vec<float, 4>(*(threadgroup const vec<InT, 4>*)&queries_tile[t][my_cols]);
      // ROWS value rows side by side in one vector; each component is one
      // row's scalar recurrence in mlx-lm's element order
      vec<float, ROWS> kv = 0.0f;
      GDNT_UNROLL for (int i = 0; i < 4; ++i) {
        GDNT_UNROLL for (int r = 0; r < ROWS; ++r) {
          s[r][i] = s[r][i] * decay;
          kv[r] += s[r][i] * kk[i];
        }
      }
      kv = simd_sum(kv);
      vec<float, ROWS> d;
      GDNT_UNROLL for (int r = 0; r < ROWS; ++r)
        d[r] = (values_tile[t][int(group) * ROWS + r] - kv[r]) * b;
      vec<float, ROWS> o = 0.0f;
      GDNT_UNROLL for (int i = 0; i < 4; ++i) {
        GDNT_UNROLL for (int r = 0; r < ROWS; ++r) {
          s[r][i] = s[r][i] + kk[i] * d[r];
          o[r] += s[r][i] * qq[i];
        }
      }
      o = simd_sum(o);
      if (lane == 0) {
        device InT* y_row =
            y + (((size_t)batch * steps + first + t) * HV + head) * DV + my_rows;
        GDNT_UNROLL for (int r = 0; r < ROWS; ++r) y_row[r] = static_cast<InT>(o[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  GDNT_UNROLL for (int r = 0; r < ROWS; ++r) {
    device float* dst = state_out + (state_base + my_rows + r) * 128 + my_cols;
    GDNT_UNROLL for (int i = 0; i < 4; ++i) dst[i] = s[r][i];
  }
"""

_kernels: dict[int, object] = {}


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def tiled_prefill_requested() -> bool:
    return _truthy(_LEVER)


def resolve_token_tile(block_t=None, input_dtype=None) -> int:
    """Tile length from the argument, then the env lever, then the dtype default."""
    if block_t is None:
        block_t = os.environ.get(_TILE_LEVER)
    if block_t is None:
        block_t = 16 if input_dtype == mx.float32 else 32
    block_t = int(block_t)
    if block_t not in TOKEN_TILES:
        raise ValueError(f"{_TILE_LEVER} must be one of {TOKEN_TILES}, got {block_t}")
    if input_dtype == mx.float32 and block_t > 16:
        raise ValueError(
            "float32 inputs need a token tile of 16 to fit 32 KiB of threadgroup memory"
        )
    return block_t


def _kernel(tile: int):
    kernel = _kernels.get(tile)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"vmlx_qwen4_gdn_tiled_prefill_t{tile}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=_BODY,
        )
        _kernels[tile] = kernel
    return kernel


def kernel_geometry_ok(q: mx.array, v: mx.array, g: mx.array, mask, state) -> bool:
    """Shapes and dtypes the staged kernel is written for."""
    if mask is not None or g.ndim != 3 or q.ndim != 4 or v.ndim != 4:
        return False
    _, _, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if key_dim != KEY_DIM or value_dim % SLICE_ROWS or key_heads <= 0:
        return False
    if value_heads % key_heads:
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    return state is None or state.dtype == mx.float32


def gated_delta_tiled(q, k, v, g, beta, state=None, block_t=None):
    """Run the staged recurrence; ``g``/``beta`` are cast to float32 here."""
    batch, steps, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if state is None:
        state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
    tile = resolve_token_tile(block_t, q.dtype)
    y, state_out = _kernel(tile)(
        inputs=[q, k, v, g.astype(mx.float32), beta.astype(mx.float32), state, steps],
        template=[
            ("InT", q.dtype), ("HK", key_heads), ("HV", value_heads), ("DV", value_dim),
            ("TB", tile), ("ROWS", ROWS_PER_SIMDGROUP), ("NSG", SIMDGROUPS),
        ],
        grid=(32 * SIMDGROUPS * (value_dim // SLICE_ROWS), value_heads, batch),
        threadgroup=(32 * SIMDGROUPS, 1, 1),
        output_shapes=[(batch, steps, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, mx.float32],
    )
    return y, state_out


def qwen4_gated_delta_prefill(
    q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True  # noqa: N803
) -> Tuple[mx.array, mx.array]:
    """mlx-lm ``gated_delta_update`` with the staged kernel for long unmasked chunks."""
    from mlx_lm.models.gated_delta import compute_g, gated_delta_update

    if (
        tiled_prefill_requested() and use_kernel and mask is None
        and mx.default_device() == mx.gpu and mx.metal.is_available()
        and q.ndim == 4 and v.ndim == 4 and q.shape[1] >= MIN_TOKENS
        and k.shape == q.shape and v.shape[:2] == q.shape[:2]
        and q.dtype == k.dtype == v.dtype
        and a.shape == b.shape == v.shape[:3]
        and A_log.shape == dt_bias.shape == (v.shape[2],)
    ):
        g = compute_g(A_log, a, dt_bias)
        state_shape = (q.shape[0], v.shape[2], v.shape[3], q.shape[3])
        if (
            kernel_geometry_ok(q, v, g, mask, state)
            and v.shape[2] > 0 and v.shape[3] > 0
            and (state is None or tuple(state.shape) == state_shape)
        ):
            return gated_delta_tiled(q, k, v, g, mx.sigmoid(b), state)
    return gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel
    )


__all__ = [
    "TOKEN_TILES",
    "gated_delta_tiled",
    "kernel_geometry_ok",
    "qwen4_gated_delta_prefill",
    "resolve_token_tile",
    "tiled_prefill_requested",
]
