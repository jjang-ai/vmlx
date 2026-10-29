# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 prompt boundary without a copy: a marker instead of a prompt-cache snapshot.

The generator used to clone the whole prompt cache right after prefill (``_clone_admissible_prompt_cache_snapshot``)
so the prefix cache could store it once the request finished. On MiniMax-M3.1 (95.5 GiB with its drafter) a 65k-token
prompt's clone is 8.2 GiB of new GPU memory held for the whole decode; it raised [METAL] Insufficient Memory and failed
the request (2026-10-28), and the headroom guard that followed simply stopped caching long prompts.

An M3 cache does not need the copy. Its K / V / index lanes (``MiniMaxM3SparseCache``) and the dense layers' K / V
(stock ``KVCache``) are append-only: decode appends rows at the logical end, DSpark verification appends draft rows
and trims back to the accepted length, capacity growth copies the existing rows into a larger buffer. Rows below the
prompt length are written by the prefill and never again, so the live cache read at the terminal store holds exactly
the rows a clone taken after prefill would hold.

The marker keeps the live cache OBJECTS and the row count, never arrays: an array view of a live buffer held during
decode stops MLX from donating that buffer to the cache's slice update, so the first decode step copies every lane in
full and the copy lives as long as the view (measured: +1x the K/V/index buffers). The views are taken only by
``views()``, at the terminal store, after the request stopped writing; they alias the live buffers (no new memory) and
the SSD block store copies them block by block under a bounded in-flight budget.

``VMLX_M3_ZERO_COPY_SNAPSHOT=0`` restores the clone (A/B and control).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

ENV = "VMLX_M3_ZERO_COPY_SNAPSHOT"
INFLIGHT_ENV = "VMLX_M3_STORE_MAX_INFLIGHT_MB"
DEFAULT_INFLIGHT_MB = 256


def zero_copy_enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


def store_inflight_bytes() -> int:
    """Cap on the bytes a streamed M3 store has handed to the SSD writer and not yet written (default 256 MB)."""
    try:
        mb = float(os.environ.get(INFLIGHT_ENV, DEFAULT_INFLIGHT_MB))
    except (TypeError, ValueError):
        mb = float(DEFAULT_INFLIGHT_MB)
    return max(1, int(mb * 1024 * 1024))


def _layer_kind(layer: Any) -> Optional[str]:
    name = type(layer).__name__
    if name == "MiniMaxM3SparseCache":
        return "sparse"
    from mlx_lm.models.cache import KVCache

    if type(layer) is KVCache:
        return "dense"
    return None


class M3PromptBoundary:
    """The live M3 cache objects and the number of prompt rows they held when the prefill ended."""

    __slots__ = ("layers", "rows")

    def __init__(self, layers: List[Any], rows: int):
        self.layers = list(layers)
        self.rows = int(rows)

    def __bool__(self) -> bool:  # a marker is a snapshot that exists
        return True

    def __repr__(self) -> str:
        return f"M3PromptBoundary(layers={len(self.layers)}, rows={self.rows})"

    def nbytes(self, rows: Optional[int] = None) -> int:
        """Bytes of the first ``rows`` (default: all prompt rows) on every lane: what a copy would cost."""
        rows = self.rows if rows is None else int(rows)
        total = 0
        for layer in self.layers:
            for name in ("keys", "values", "idx_keys"):
                value = getattr(layer, name, None)
                if value is None:
                    continue
                shape = tuple(int(s) for s in value.shape)
                per_row = value.nbytes // max(1, shape[2]) if len(shape) == 4 and shape[2] else 0
                total += per_row * rows
        return int(total)

    def views(self, rows: Optional[int] = None) -> Optional[List[Any]]:
        """New cache objects whose arrays are views of rows [0, rows) of the live layers (no copy); None when any
        layer no longer holds those rows on every lane. Take them only after the request stopped writing."""
        from .prefill_geometry import cache_offset, truncated_cache

        rows = self.rows if rows is None else int(rows)
        if rows <= 0 or rows > self.rows:
            return None
        offset = cache_offset(self.layers)
        if offset is None or offset < rows:
            return None
        return truncated_cache(self.layers, rows)


def capture(cache: Optional[List[Any]]) -> Optional[M3PromptBoundary]:
    """A marker for an M3 cache list (sparse layers and stock KVCache dense layers only, every lane at the same
    length); None for anything else (the caller keeps its clone path)."""
    if not cache:
        return None
    kinds = [_layer_kind(layer) for layer in cache]
    if any(kind is None for kind in kinds) or "sparse" not in kinds:
        return None
    from .prefill_geometry import cache_offset

    rows = cache_offset(cache)
    if rows is None or rows <= 0:
        return None
    return M3PromptBoundary(cache, rows)
