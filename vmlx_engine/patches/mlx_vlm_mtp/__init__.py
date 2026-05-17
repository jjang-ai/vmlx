# SPDX-License-Identifier: Apache-2.0
"""vMLX-owned native MTP adapters for mlx-vlm model copies."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_mlx_vlm_mtp_patch() -> bool:
    """Install native MTP runtime methods on supported mlx-vlm model copies."""
    global _PATCHED
    if _PATCHED:
        return True

    from . import qwen35_vl

    qwen_ok = qwen35_vl.apply()
    if not qwen_ok:
        logger.warning("mlx-vlm Qwen3.5/3.6 MTP adapter failed to apply")
        return False

    _PATCHED = True
    logger.info("mlx-vlm native MTP adapters applied")
    return True
