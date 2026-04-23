# SPDX-License-Identifier: Apache-2.0
"""Text JANGTQ loader — re-export of ``jang_tools.load_jangtq``.

Matches the module path prescribed in
``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.1 so docs like §1.3's

    from vmlx_engine.loaders.load_jangtq import load_jangtq_model

resolve even though the production code lives in ``jang_tools`` (bundled
into vMLX's Python runtime). Keeping this as a re-export means the
P3/P15/P17/P18 TurboQuant Metal kernels stay in one place — we can't
silently drift a second copy.
"""

from __future__ import annotations

try:
    from jang_tools.load_jangtq import load_jangtq_model
    from jang_tools.load_jangtq import (  # noqa: F401 — public surface
        _hydrate_jangtq_model,
    )
except ImportError as _ie:  # pragma: no cover
    raise ImportError(
        "vmlx_engine.loaders.load_jangtq requires `jang_tools` in the "
        "active Python environment. Install it via `pip install jang-tools` "
        "or use vMLX's bundled Python which ships it by default."
    ) from _ie

__all__ = ["load_jangtq_model"]
