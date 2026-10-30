# SPDX-License-Identifier: Apache-2.0
"""Recurrent-cache state for speculative verify forwards.

A GatedDeltaNet layer cannot be trimmed like a KV cache: its conv and SSM
state fold every row in. A verify forward over ``[confirmed, d1..dn]`` therefore
leaves two things on the layer's cache instance:

* ``rollback_state`` — the ``(conv, ssm)`` pair right after the confirmed row
  (a class-level ``None`` slot on ``ArraysCache`` makes the attribute always
  readable);
* a partial-restore record ``(n, restore)`` where ``restore(accepted)`` rebuilds
  the state after ``accepted`` of the ``n`` draft rows by re-running only that
  layer's short recurrent step. That is what lets a depth-2/3 chain keep an
  accepted prefix instead of falling back to depth 1.

Only the owning layer arms a cache instance, and it does so on every forward
(an SSD restore builds fresh instances). Nothing here reaches ``cache.state``,
so serialized caches are unchanged.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Callable, Tuple

from ._marks import is_marked, mark

logger = logging.getLogger(__name__)

_slot_ready = False

_RECORD = "_vmlx_partial_restore"


def install_rollback_slot() -> bool:
    """Give ``ArraysCache`` a ``rollback_state = None`` default (idempotent)."""
    global _slot_ready
    if _slot_ready:
        return True
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mlx_lm.models.cache not importable; no rollback slot")
        return False
    if hasattr(ArraysCache, "rollback_state") and not is_marked(ArraysCache):
        # The installed mlx-lm provides the slot itself.
        mark(ArraysCache, "upstream")
    else:
        ArraysCache.rollback_state = None
        mark(ArraysCache, "patch")
    _slot_ready = True
    return True


def _allowed(cache: Any, count: Any) -> bool:
    record = getattr(cache, _RECORD, None)
    return record is not None and isinstance(count, int) and 0 < count <= record[0]


def _settle(cache: Any) -> None:
    setattr(cache, _RECORD, None)
    cache.rollback_state = None


def _restore(cache: Any, count: int) -> bool:
    if not _allowed(cache, count):
        return False
    drafts, rebuild = getattr(cache, _RECORD)
    # Build the accepted-prefix state before touching the cache. An error here
    # must fail the request: resuming from a half-restored layer set is unsafe.
    conv, ssm = rebuild(drafts - count)
    cache[0], cache[1] = conv, ssm
    cache.advance(-count)
    _settle(cache)
    return True


def arm_partial_rollback(cache: Any) -> None:
    """Expose the partial-rollback protocol on ``cache`` and clear any record."""
    cache.supports_partial_rollback = True
    cache.can_rollback_speculative = MethodType(_allowed, cache)
    cache.rollback_speculative = MethodType(_restore, cache)
    cache.commit_speculative = MethodType(_settle, cache)
    _settle(cache)


def remember_partial_rollback(
    cache: Any, drafts: int, rebuild: Callable[[int], Tuple[Any, Any]]
) -> None:
    setattr(cache, _RECORD, (drafts, rebuild))
