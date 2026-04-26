# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 runtime patch installer — registers ``jang_tools.dsv4.mlx_model``
into ``mlx_lm.models.deepseek_v4`` so ``mlx_lm.utils.load_model()`` can
resolve ``model_type="deepseek_v4"`` to the custom MLX implementation with
MLA head_dim=512, mHC, sqrtsoftplus routing, and compressor/indexer state.

Import this at ``vmlx_engine`` package init so the registration happens
before ``jang_loader._load_jang_v2()`` tries to instantiate a DSV4 skeleton.

Idempotent: ``jang_tools.dsv4.mlx_register`` calls ``sys.modules.setdefault``
under the hood (via reassignment — calling twice is harmless).

Cross-reference: research/DSV4-RUNTIME-ARCHITECTURE.md §3 (class list),
§5 (bundle cheat sheet), and §7.1 (AutoTokenizer fallback explained —
our PreTrainedTokenizerFast path in ``jang_tools.load_jangtq`` already
handles this).
"""

from __future__ import annotations


def register() -> bool:
    """Install DSV4 model_type into mlx_lm's dispatch table.

    Returns True on success, False if ``jang_tools.dsv4`` is not available
    (older jang-tools pre-dsv4 release). Logs a warning but does not raise —
    the caller will hit an ImportError later when attempting to actually load
    a DSV4 bundle, with a clearer user-facing message.
    """
    try:
        from jang_tools.dsv4 import mlx_register  # noqa: F401
        return True
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "DeepSeek V4 runtime patch: jang_tools.dsv4 unavailable. "
            "DSV4 bundles will fail to load; reinstall vMLX from latest DMG."
        )
        return False


# Auto-install on package import so callers never forget.
register()
