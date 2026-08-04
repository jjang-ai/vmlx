# SPDX-License-Identifier: Apache-2.0
"""Bounded DSV4 indexer bypass for contexts below sparse-selection use.

The request-level guard is enabled by default and can be disabled with
``VMLX_DSV4_SKIP_UNUSED_INDEXER=0``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)
_PATCHED_CLASS: type | None = None


def _enabled() -> bool:
    return os.environ.get("VMLX_DSV4_SKIP_UNUSED_INDEXER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_dsv4_indexer_skip(model: Any) -> int:
    global _PATCHED_CLASS
    if not _enabled():
        return 0

    indexers = [
        module
        for _name, module in list(getattr(model, "named_modules", lambda: [])())
        if hasattr(module, "index_topk")
        and hasattr(module, "update_pool")
        and hasattr(module, "compressor")
    ]
    if not indexers:
        return 0
    indexer_class = type(indexers[0])
    if _PATCHED_CLASS is indexer_class:
        return len(indexers)
    if getattr(indexer_class, "_vmlx_dsv4_indexer_skip", False):
        _PATCHED_CLASS = indexer_class
        return len(indexers)
    if _PATCHED_CLASS is not None:
        logger.info(
            "DSV4 unused-indexer bypass skipped: multiple indexer classes in process"
        )
        return 0
    if any(type(module) is not indexer_class for module in indexers):
        return 0
    original_update_pool = indexer_class.update_pool

    def _skip_update(self: Any, x: mx.array, rope: Any, cache: Any, start_pos: int):
        if (
            not _enabled()
            or cache is None
            or not hasattr(cache, "_branch_state")
            or getattr(x, "ndim", 0) < 2
            or int(x.shape[1]) != 1
            or not bool(getattr(cache, "_vmlx_dsv4_indexer_skip_decode", False))
        ):
            return original_update_pool(self, x, rope, cache, start_pos)
        try:
            main_state = cache._branch_state("compressor_state")
            main_pooled = main_state.get("pooled")
            indexer_state = cache._branch_state("indexer_state")
            if main_pooled is None:
                indexer_state["buffer_kv"] = None
                indexer_state["buffer_gate"] = None
                indexer_state["pooled"] = None
                return mx.zeros((int(x.shape[0]), 0, int(self.head_dim)), dtype=x.dtype)
            if getattr(main_pooled, "ndim", 0) < 2:
                raise ValueError("compressed pool has no sequence dimension")
            rows = int(main_pooled.shape[1])
            zeros = mx.zeros((int(x.shape[0]), rows, int(self.head_dim)), dtype=x.dtype)
            indexer_state["buffer_kv"] = None
            indexer_state["buffer_gate"] = None
            indexer_state["pooled"] = zeros
            return zeros
        except Exception as exc:
            logger.warning(
                "DSV4 unused-indexer bypass fell back to native update_pool: %s",
                exc,
            )
            return original_update_pool(self, x, rope, cache, start_pos)

    indexer_class._vmlx_dsv4_indexer_original_update_pool = original_update_pool
    indexer_class.update_pool = _skip_update
    indexer_class._vmlx_dsv4_indexer_skip = True
    _PATCHED_CLASS = indexer_class
    logger.info(
        "DSV4 unused-indexer bypass candidate enabled for %d modules", len(indexers)
    )
    return len(indexers)


__all__ = ["install_dsv4_indexer_skip"]
