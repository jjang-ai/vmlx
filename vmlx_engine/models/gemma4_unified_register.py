# SPDX-License-Identifier: Apache-2.0
"""Register the source-owned Gemma 4 Unified VLM/audio runtime.

Gemma 4 12B Unified artifacts use ``model_type=gemma4_unified`` with an
encoder-free early-fusion image/audio wrapper. The local mlx-vlm build used by
vMLX does not always ship that package, so vMLX vendors the compact runtime and
registers it under the upstream ``mlx_vlm.models.gemma4_unified`` namespace at
VLM-load time.

The registration is idempotent and defers to upstream when mlx-vlm eventually
ships native support.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False
_PACKAGE_NAME = "mlx_vlm.models.gemma4_unified"
_VENDORED_DIR = Path(__file__).resolve().parent / "gemma4_unified"


def gemma4_unified_runtime_available() -> bool:
    """Return True when upstream or vMLX-vendored Gemma 4 Unified runtime exists."""
    if _PACKAGE_NAME in sys.modules:
        return True
    if importlib.util.find_spec(_PACKAGE_NAME) is not None:
        return True
    return (_VENDORED_DIR / "__init__.py").is_file()


def register_gemma4_unified_runtime() -> bool:
    """Install vendored Gemma 4 Unified modules into ``sys.modules`` if needed."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        importlib.import_module(_PACKAGE_NAME)
        _REGISTERED = True
        logger.debug("Gemma 4 Unified runtime already provided by mlx-vlm")
        return False
    except ModuleNotFoundError:
        pass

    init_path = _VENDORED_DIR / "__init__.py"
    if not init_path.is_file():
        return False

    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(_VENDORED_DIR)],
    )
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    try:
        parent = importlib.import_module("mlx_vlm.models")
        setattr(parent, "gemma4_unified", module)
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PACKAGE_NAME, None)
        raise

    _REGISTERED = True
    logger.info(
        "Registered source-owned Gemma 4 Unified VLM/audio runtime under "
        "mlx_vlm.models.gemma4_unified"
    )
    return True


_register_gemma4_unified_runtime = register_gemma4_unified_runtime
