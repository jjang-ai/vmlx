# SPDX-License-Identifier: Apache-2.0
"""Runtime guard for Gemma 4 VLM pixel-value list inputs.

Older ``mlx-vlm`` releases concatenate a mixed numpy/MLX list directly. Newer
releases process each image independently and already coerce every item. Keep
the compatibility wrapper for the older implementation, but never replace the
new native different-sized-image path with the legacy concatenate behavior.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from functools import wraps
from typing import Any

_TARGET = "mlx_vlm.models.gemma4.vision"
_PATCH_MARKER = "_vmlx_gemma4_pixel_values_patch"


def _patch_image_budget(module: Any) -> None:
    """Match the legacy vision tower's pooling grid to larger image inputs.

    The processor can emit up to 1120 soft tokens, while the older tower fixes
    both patch positions and pooling at its configured default (usually 280).
    Scope the larger grid to this synchronous forward, including failures.
    The original list path calls each image separately, preserving its shape.
    """
    import inspect

    model = getattr(module, "VisionModel", None)
    if model is None:
        return
    original = model.__call__
    if getattr(original, "_vmlx_gemma4_dynamic_budget", False):
        return
    try:
        # Only adapt the known fixed-grid implementation, not a future native
        # implementation with a different position/pooling contract.
        if "real_positions[: self.max_patches]" not in inspect.getsource(model):
            return
    except (OSError, TypeError):
        return

    @wraps(original)
    def forward(self, pixel_values):
        shape = getattr(pixel_values, "shape", ())
        if len(shape) != 4:
            return original(self, pixel_values)
        height, width = map(int, shape[-2:])
        patch = int(self.patch_size)
        pool = int(self.pooling_kernel_size)
        patches = (height // patch) * (width // patch)
        if patches <= self.max_patches:
            return original(self, pixel_values)
        if height % (patch * pool) or width % (patch * pool):
            raise ValueError("Gemma 4 image dimensions must align with its pooling grid")
        previous = (self.max_patches, self.default_output_length,
                    self.pooler.default_output_length)
        try:
            self.max_patches = patches
            self.default_output_length = patches // (pool * pool)
            self.pooler.default_output_length = self.default_output_length
            return original(self, pixel_values)
        finally:
            (self.max_patches, self.default_output_length,
             self.pooler.default_output_length) = previous

    forward._vmlx_gemma4_dynamic_budget = True
    model.__call__ = forward


def _has_native_mixed_list_support(source: str) -> bool:
    return all(
        needle in source
        for needle in (
            "if isinstance(pixel_values, list):",
            "if not isinstance(img, mx.array):",
            "img = mx.array(img)",
        )
    )


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
        if (
            "mlxstudio#88" in src
            and "isinstance(v, mx.array)" in src
        ) or _has_native_mixed_list_support(src):
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
        _patch_image_budget(module)


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
        _patch_image_budget(module)
    if not any(
        getattr(finder, "_vmlx_gemma4_vision_patch_finder", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _Gemma4VisionPatchFinder())
