# SPDX-License-Identifier: Apache-2.0
"""Register the vMLX-owned ERNIE-4.5 MoE runtime under ``mlx_lm.models.ernie4_5_moe``.

Unlike glm5_next this does NOT defer to upstream when mlx-lm ships the module, because
upstream mlx-lm's ``ernie4_5_moe.py`` drops the router selection bias
(``e_score_correction_bias``) and therefore routes ~29% of (token, layer) pairs differently
from the reference implementation (see ernie4_5_moe.py header). The vendored module is
installed over the upstream one unconditionally; idempotent.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE = "mlx_lm.models.ernie4_5_moe"
_VENDORED = Path(__file__).resolve().parent / "ernie4_5_moe.py"


def ernie4_5_runtime_available() -> bool:
    return _VENDORED.is_file()


def register_ernie4_5_runtime() -> bool:
    """Install the vendored module into ``sys.modules`` (over upstream if present)."""
    global _REGISTERED
    if _REGISTERED and sys.modules.get(_PACKAGE) is not None:
        return True
    if not _VENDORED.is_file():
        return False
    sys.modules.pop(_PACKAGE, None)
    spec = importlib.util.spec_from_file_location(_PACKAGE, _VENDORED)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(_PACKAGE, None)
        raise
    try:
        parent = importlib.import_module("mlx_lm.models")
        setattr(parent, "ernie4_5_moe", mod)
    except Exception:
        pass
    _REGISTERED = True
    logger.info("Registered vendored ernie4_5_moe runtime (%s), overriding upstream", _VENDORED)
    return True
