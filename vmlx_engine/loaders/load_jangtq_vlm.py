# SPDX-License-Identifier: Apache-2.0
"""JANGTQ VLM loader — re-export of ``jang_tools.load_jangtq_vlm``.

Matches ``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.1 module layout.
Handles the shared VLM skeleton build (mlx_vlm's Model + vision tower +
processor) plus JANGTQ hydration (TQ kernel replacement, compile-friendly
MoE, MLA bit-width fix, wired-limit auto-tune).

For Kimi K2.6 specifically, import ``load_jangtq_kimi_vlm`` instead — it
layers the Kimi-required MODEL_REMAPPING patch, lower VL wired_limit,
and vision/language command-buffer split on top of this base loader.
"""

from __future__ import annotations

try:
    from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
except ImportError as _ie:  # pragma: no cover
    raise ImportError(
        "vmlx_engine.loaders.load_jangtq_vlm requires `jang_tools` in "
        "the active Python environment."
    ) from _ie

__all__ = ["load_jangtq_vlm_model"]
