# SPDX-License-Identifier: Apache-2.0
"""Attach ``rollback_state`` slot to ``mlx_lm.models.cache.ArraysCache``.

The vMLX native-MTP verifier snapshots ``(conv_state, ssm_state)`` after the
confirmed prefix of a draft/verify forward. If a draft token rejects, this slot
lets the runtime restore the hybrid state without touching accepted history.
The slot is inert when no MTP path runs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Attach ``rollback_state = None`` to ``ArraysCache`` (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mlx_lm.models.cache not importable; skipping rollback_state")
        return False

    if hasattr(ArraysCache, "rollback_state") and not hasattr(
        ArraysCache, "_omlx_rollback_attached"
    ):
        # The installed runtime may already expose this natively.
        _PATCHED = True
        ArraysCache._omlx_rollback_attached = "upstream"
        return True

    ArraysCache.rollback_state = None
    ArraysCache._omlx_rollback_attached = "patch"
    _PATCHED = True
    return True
