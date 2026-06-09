# SPDX-License-Identifier: Apache-2.0
"""Runtime shims for upstream mlx_vlm fixes not yet present in pinned wheels."""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def install() -> None:
    """Install all mlx_vlm compatibility patches once."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_gemma4_video_processor_hf_kwargs()
    _APPLIED = True


def _patch_gemma4_video_processor_hf_kwargs() -> None:
    """Backport mlx-vlm#1321: ignore unused HF video processor config keys."""
    try:
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4VideoProcessor
    except Exception as exc:
        logger.debug("mlx_vlm_compat: Gemma4VideoProcessor unavailable: %s", exc)
        return

    original_init = getattr(Gemma4VideoProcessor, "__init__", None)
    if original_init is None or getattr(
        original_init, "_vmlx_gemma4_video_hf_kwargs_patch", False
    ):
        return

    signature = inspect.signature(original_init)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return
    accepted = set(signature.parameters) - {"self"}

    def _vmlx_gemma4_video_init(self, *args, **kwargs):
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in accepted
        }
        return original_init(self, *args, **filtered_kwargs)

    _vmlx_gemma4_video_init._vmlx_gemma4_video_hf_kwargs_patch = True  # type: ignore[attr-defined]
    Gemma4VideoProcessor.__init__ = _vmlx_gemma4_video_init
