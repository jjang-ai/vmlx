# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned MiniMax-M3 (minimax_m3_vl) text runtime.

The JANG_2L bundle declares ``model_type=minimax_m3_vl``. mlx-lm has no native
package for it, so vMLX vendors the compact MLX runtime (minimax_m3.py + cache.py)
and installs it under the ``mlx_lm.models.minimax_m3_vl`` namespace at load time,
so the standard loader path (``mlx_lm.load`` / engine model resolution) finds it.

Idempotent; defers to upstream if mlx-lm ever ships native support.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE = "mlx_lm.models.minimax_m3_vl"
_VENDORED = Path(__file__).resolve().parent / "minimax_m3.py"


def minimax_m3_runtime_available() -> bool:
    if _PACKAGE in sys.modules:
        return True
    if importlib.util.find_spec(_PACKAGE) is not None:
        return True
    return _VENDORED.is_file()


def register_minimax_m3_runtime() -> bool:
    """Install the vendored minimax_m3 module under the mlx-lm namespace."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        importlib.import_module(_PACKAGE)
        _REGISTERED = True
        logger.debug("minimax_m3_vl runtime already provided by mlx-lm")
        return False
    except ModuleNotFoundError:
        pass
    if not _VENDORED.is_file():
        return False
    spec = importlib.util.spec_from_file_location(_PACKAGE, _VENDORED)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = mod
    spec.loader.exec_module(mod)
    _REGISTERED = True
    logger.info("Registered vendored minimax_m3_vl runtime (%s)", _VENDORED)
    return True
