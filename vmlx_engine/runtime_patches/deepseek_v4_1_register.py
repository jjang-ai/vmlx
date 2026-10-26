# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4.1-Flash runtime patch installer — registers ``jang_tools.dsv4_1.mlx_model``
into ``mlx_lm.models.deepseek_v41`` so ``mlx_lm.utils.load_model()`` can resolve
``model_type="deepseek_v41"`` to the custom MLX implementation (V4-family MLA head_dim=512,
mHC, sqrtsoftplus routing, compressor/indexer, plus Engram n-gram taps, a vision tower, and
dSpark MTP heads).

Import this at ``vmlx_engine`` package init so the registration happens before any model-load
path walks ``config.json.model_type``. Idempotent (``jang_tools.dsv4_1.mlx_register`` reassigns
``sys.modules`` — calling twice is harmless).
"""

from __future__ import annotations


def register() -> bool:
    try:
        from jang_tools.dsv4_1 import mlx_register  # noqa: F401  (side effect: register)
        return True
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "DeepSeek V4.1 runtime patch: jang_tools.dsv4_1 unavailable. "
            "DSV4.1 bundles will fail to load; reinstall vMLX from latest DMG."
        )
        return False


# Auto-install on package import so callers never forget.
register()
