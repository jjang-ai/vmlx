# SPDX-License-Identifier: Apache-2.0
"""Runtime guard for Gemma 4 tiny RGB image preprocessing.

Upstream ``mlx_vlm.models.gemma4.processing_gemma4`` asks transformers to infer
the channel dimension after PIL images have become numpy arrays. Very small RGB
images such as 1x1 produce shape ``(1, 1, 3)``, which transformers treats as an
ambiguous channel-first image. Gemma4 then transposes it to ``(1, 3, 1)`` and
Pillow rejects the array during resize.

vMLX API image inputs are file/PIL/data-URL images, so the natural array layout
is channel-last. Bias only ambiguous 3D RGB/RGBA-like shapes back to
``ChannelDimension.LAST`` and leave every other upstream inference path alone.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from typing import Any

_TARGET = "mlx_vlm.models.gemma4.processing_gemma4"
_PATCH_MARKER = "_vmlx_gemma4_tiny_rgb_patch"


def _patch_module(module: Any) -> None:
    original = getattr(module, "infer_channel_dimension_format", None)
    channel_dimension = getattr(module, "ChannelDimension", None)
    if original is None or channel_dimension is None:
        return
    if getattr(original, _PATCH_MARKER, False):
        return

    def _vmlx_infer_channel_dimension_format(image, *args, **kwargs):
        shape = tuple(getattr(image, "shape", ()) or ())
        if len(shape) == 3:
            first = int(shape[0])
            last = int(shape[-1])
            valid_channels = {1, 3, 4}
            if first in valid_channels and last in valid_channels:
                return channel_dimension.LAST
        return original(image, *args, **kwargs)

    setattr(_vmlx_infer_channel_dimension_format, _PATCH_MARKER, True)
    module.infer_channel_dimension_format = _vmlx_infer_channel_dimension_format


class _Gemma4ProcessingPatchLoader(importlib.abc.Loader):
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


class _Gemma4ProcessingPatchFinder(importlib.abc.MetaPathFinder):
    _vmlx_gemma4_processing_patch_finder = True

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if not isinstance(spec.loader, _Gemma4ProcessingPatchLoader):
            spec.loader = _Gemma4ProcessingPatchLoader(spec.loader)
        return spec


def install() -> None:
    module = sys.modules.get(_TARGET)
    if module is not None:
        _patch_module(module)
    if not any(
        getattr(finder, "_vmlx_gemma4_processing_patch_finder", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _Gemma4ProcessingPatchFinder())
