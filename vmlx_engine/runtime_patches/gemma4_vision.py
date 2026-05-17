# SPDX-License-Identifier: Apache-2.0
"""Runtime guard for Gemma 4 VLM pixel-value list inputs.

The release bundle patches ``mlx_vlm.models.gemma4.vision`` in place before
signing. PyPI/source users still import upstream ``mlx_vlm`` directly, so this
module installs a lazy import hook and normalizes mixed numpy/MLX pixel lists
before upstream Gemma 4 vision code concatenates them.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from typing import Any

_TARGET = "mlx_vlm.models.gemma4.vision"
_PATCH_MARKER = "_vmlx_gemma4_pixel_values_patch"


def _patch_module(module: Any) -> None:
    vision_model = getattr(module, "VisionModel", None)
    if vision_model is None:
        return
    original = getattr(vision_model, "__call__", None)
    if original is None or getattr(original, _PATCH_MARKER, False):
        return
    try:
        import inspect

        src = inspect.getsource(original)
        if "mlxstudio#88" in src and "isinstance(v, mx.array)" in src:
            setattr(original, _PATCH_MARKER, True)
            return
    except Exception:
        pass

    def _vmlx_gemma4_call(self, pixel_values):
        import mlx.core as mx

        if isinstance(pixel_values, list):
            # mlxstudio#88: multi-image processors can hand us a Python list
            # containing numpy arrays and MLX arrays. mx.concatenate only
            # accepts mx.array inputs, so coerce each element first.
            pixel_values = [
                v if isinstance(v, mx.array) else mx.array(v)
                for v in pixel_values
            ]
            pixel_values = mx.concatenate(pixel_values, axis=0)
        elif not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        return original(self, pixel_values)

    setattr(_vmlx_gemma4_call, _PATCH_MARKER, True)
    vision_model.__call__ = _vmlx_gemma4_call


class _Gemma4VisionPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader):
        self._wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        if create is not None:
            return create(spec)
        return None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_module(module)


class _Gemma4VisionPatchFinder(importlib.abc.MetaPathFinder):
    _vmlx_gemma4_vision_patch_finder = True

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if not isinstance(spec.loader, _Gemma4VisionPatchLoader):
            spec.loader = _Gemma4VisionPatchLoader(spec.loader)
        return spec


def install() -> None:
    module = sys.modules.get(_TARGET)
    if module is not None:
        _patch_module(module)
    if not any(
        getattr(finder, "_vmlx_gemma4_vision_patch_finder", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _Gemma4VisionPatchFinder())
