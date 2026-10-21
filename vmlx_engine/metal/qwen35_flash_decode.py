# SPDX-License-Identifier: Apache-2.0
"""Long-context Qwen3.5 decode / native-MTP verify attention on the split-key flash kernel (sdpa_flash_decode).

At the Ornith-1.5-122B geometry (32 query heads over 2 KV heads, head_dim 256, bf16 KV) MLX's one-row vector SDPA
streams the cache at ~205 GB/s at 128k keys, and the shipped row-exact verify attends rows in PAIRS, each pair
re-reading every key it may see. The flash kernel processes all 16 query heads of a KV head (and up to 8 rows) as
simdgroup-matrix bands over one read of each key. Synthetic M5 Max, ms per attention layer (flash_bench.py):

    keys     1 row MLX -> flash     4-row verify pairs -> flash
    16k      0.274 -> 0.256         0.750 -> 0.518
    128k     1.214 -> 0.930         4.404 -> 2.307
    200k     1.814 -> 1.142         6.975 -> 3.178

Exactness contract (greedy native MTP == AR): the route is chosen per ROW from that row's own key count, identically
for a one-row decode and for every verify row. A row that sees fewer than ``min_keys()`` keys keeps today's path
(MLX vector SDPA / row pairs); a row at or above it runs the flash kernel, and verify rows are grouped by
``chunk_size(own key count)`` so each group's call uses the chunk its rows' decode steps used. Inside the kernel a
qrow's arithmetic depends only on the chunk size: keys past a row's causal end are masked to exact zeros, so a verify
row equals its one-row decode bit for bit (flash_bench rows==decode at 16k / 128k / 200k; the 32,771 and 65,539
failures it found were the chunk-step straddle this grouping removes).

Scope: query heads / KV heads == 16 and head_dim == 256 (Ornith). Other Qwen3.5 geometries (27B GQA 6, MiniCPM-V
4.7 / 35B GQA 8, whose DSpark verify has its own attention) are untouched until measured.
VMLX_QWEN35_FLASH_DECODE_MIN_KEYS=<n> moves the threshold (0 = off, A/B only).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import mlx.core as mx

from .sdpa_flash_decode import MAX_ROWS, chunk_size, sdpa_flash

# One-row flash beats MLX's vector kernel from ~16k keys at this geometry (8k: 0.235 vs 0.226 ms, 16k: 0.256 vs 0.274).
DEFAULT_MIN_KEYS = 16384
_GQA = 16
_HEAD_DIM = 256


def min_keys() -> int:
    raw = os.environ.get("VMLX_QWEN35_FLASH_DECODE_MIN_KEYS", "").strip().lower()
    if not raw:
        return DEFAULT_MIN_KEYS
    if raw in {"off", "false", "no"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MIN_KEYS


def _geometry(queries, keys, values) -> bool:
    if not (isinstance(keys, mx.array) and isinstance(values, mx.array)):
        return False  # quantized KV storage keeps the stock path
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4 or int(queries.shape[0]) != 1:
        return False
    heads, kv_heads = int(queries.shape[1]), int(keys.shape[1])
    return (
        kv_heads > 0 and heads == _GQA * kv_heads
        and int(queries.shape[3]) == _HEAD_DIM and int(values.shape[3]) == _HEAD_DIM
        and keys.dtype in (mx.bfloat16, mx.float16) and keys.dtype == values.dtype == queries.dtype
    )


def _flash(queries, keys, values, scale: float) -> Optional[mx.array]:
    rows = int(queries.shape[2])
    out = sdpa_flash(queries, keys, values, scale, "causal" if rows > 1 else None, None)
    if out is None:
        return None
    # (B, L, H * Dv) -> (B, H, L, Dv)
    return out.reshape(1, rows, int(queries.shape[1]), int(values.shape[3])).transpose(0, 2, 1, 3)


def decode_attention(queries, keys, values, scale: float, mask) -> Optional[mx.array]:
    """One-row decode over a long cache, or None for the caller's own SDPA."""
    # One query over every cached key: "causal" masks nothing, so it routes like no mask (the verify rows below see
    # the same keys through their own causal mask).
    if int(queries.shape[2]) != 1 or not (mask is None or (isinstance(mask, str) and mask == "causal")):
        return None
    threshold = min_keys()
    if threshold <= 0 or not _geometry(queries, keys, values) or int(keys.shape[2]) < threshold:
        return None
    return _flash(queries, keys, values, scale)


def verify_attention(
    queries, keys, values, scale: float, attend_pairs: Callable[[mx.array, mx.array, mx.array], mx.array],
) -> mx.array:
    """Native-MTP verify rows (the last ``S`` positions of ``keys``), each attended exactly like its decode step.

    ``attend_pairs(q, k, v)`` is the shipped row-exact path (row_exact_qmv.attend_rows_in_pairs over the model's SDPA)
    for rows treated as the last ``q.shape[2]`` positions of the ``k`` / ``v`` it is given.
    """
    rows = int(queries.shape[2])
    threshold = min_keys()
    if threshold <= 0 or rows > MAX_ROWS or not _geometry(queries, keys, values):
        return attend_pairs(queries, keys, values)
    start = int(keys.shape[2]) - rows
    counts = [start + r + 1 for r in range(rows)]
    if counts[-1] < threshold:
        return attend_pairs(queries, keys, values)
    low = sum(1 for c in counts if c < threshold)
    parts = []
    if low:
        end = counts[low - 1]
        parts.append(attend_pairs(queries[:, :, :low], keys[:, :, :end], values[:, :, :end]))
    first = low
    while first < rows:
        chunk = chunk_size(counts[first])
        last = first
        while last + 1 < rows and chunk_size(counts[last + 1]) == chunk:
            last += 1
        end = counts[last]
        q, k, v = queries[:, :, first:last + 1], keys[:, :, :end], values[:, :, :end]
        out = _flash(q, k, v, scale)
        # sdpa_flash declines only on shape / dtype, which a one-row decode at these counts shares: it fell back too.
        parts.append(out if out is not None else attend_pairs(q, k, v))
        first = last + 1
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
