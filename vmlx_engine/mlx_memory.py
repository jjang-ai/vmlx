"""MLX memory-cache cleanup helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def clear_mlx_memory_cache(
    *,
    mx: Any | None = None,
    log: logging.Logger | None = None,
) -> str | None:
    """Clear MLX's allocator cache using the API available in this MLX build.

    MLX 0.31.x removed/does not expose ``mx.clear_memory_cache()``. Prefer the
    current top-level cache API and keep the deprecated Metal API only as an
    older-build fallback.
    """

    active_log = log or logger
    if mx is None:
        try:
            import mlx.core as mx  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask request flow
            active_log.warning("Unable to import MLX for memory cache cleanup: %s", exc)
            return None

    if callable(getattr(mx, "clear_cache", None)):
        try:
            mx.clear_cache()
            return "mx.clear_cache"
        except Exception as exc:  # noqa: BLE001 - log API drift/failure
            active_log.warning(
                "MLX memory cache cleanup via mx.clear_cache failed: %s",
                exc,
            )
            return None

    metal = getattr(mx, "metal", None)
    if metal is not None and callable(getattr(metal, "clear_cache", None)):
        try:
            metal.clear_cache()
            return "mx.metal.clear_cache"
        except Exception as exc:  # noqa: BLE001 - log API drift/failure
            active_log.warning(
                "MLX memory cache cleanup via mx.metal.clear_cache failed: %s",
                exc,
            )
            return None

    active_log.warning("No known MLX memory cache clearing API available")
    return None
