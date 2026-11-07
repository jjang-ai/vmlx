# SPDX-License-Identifier: Apache-2.0
"""Resolve the vendored Ricursive runtime before generic or JANG model loading."""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from threading import RLock

_PACKAGE = "mlx_lm.models.ricursive"
_LOCK = RLock()


def register_ricursive_runtime() -> bool:
    """Register once; never hide a broken upstream import or leave a partial module."""
    with _LOCK:
        if _PACKAGE in sys.modules:
            return False
        try:
            importlib.import_module(_PACKAGE)
            return False
        except ModuleNotFoundError as exc:
            if exc.name != _PACKAGE:
                raise
        path = Path(__file__).with_name("ricursive.py")
        spec = importlib.util.spec_from_file_location(_PACKAGE, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Ricursive runtime from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_PACKAGE, None)
            raise
        logging.getLogger("vmlx_engine").info("Registered Ricursive runtime: %s", path)
        return True


def ricursive_runtime_available() -> bool:
    return _PACKAGE in sys.modules or Path(__file__).with_name("ricursive.py").is_file()


def ensure_ricursive_runtime_registered(model_path, *, config=None) -> bool:
    """Route from config identity, never folder names or quant labels."""
    if config is None:
        try:
            config = json.loads((Path(model_path) / "config.json").read_text())
        except (OSError, ValueError, TypeError):
            return False
    if not isinstance(config, dict) or config.get("model_type") != "ricursive":
        return False
    try:
        register_ricursive_runtime()
    except Exception as exc:
        raise RuntimeError(f"Ricursive runtime registration failed for {model_path}: {exc}") from exc
    return True
