"""Runtime compatibility patches for upstream mlx_vlm bugs.

These are monkey-patches applied once at import time. They wrap known-broken
methods so callers don't need to fork vendored files.

Patches applied
---------------

* Qwen3-VL ``VisionModel.rot_pos_emb`` — upstream types ``grid_thw`` as
  ``mx.array`` but the caller sometimes passes a numpy ``ndarray`` (seen on
  Qwen3.5-35B-A3B bf16, issue #69). ``mx.max(ndarray)`` raises
  ``TypeError: max(): incompatible function arguments``. Coerce on entry.
* Qwen3-VL ``VisionModel.__call__`` — same ``grid_thw`` typing issue when
  ``fast_pos_embed_interpolate`` iterates over a numpy array.
* Qwen3.5/3.6 VL ``Model.sanitize`` — HF-native 3D patch-embed weights can
  arrive as ``(out, channels, temporal, height, width)`` while MLX Conv3D
  expects channels-last ``(out, temporal, height, width, channels)``.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)
_applied = False


def apply() -> None:
    """Apply all mlx_vlm compat patches (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True
    _patch_qwen3_vl_grid_thw()
    _patch_qwen35_patch_embed_layout()


def _patch_qwen3_vl_grid_thw() -> None:
    try:
        import mlx.core as mx
        from mlx_vlm.models.qwen3_vl import vision as _qv
    except ImportError:
        return

    VisionModel = getattr(_qv, "VisionModel", None)
    if VisionModel is None:
        return

    def _as_mx(x):
        if isinstance(x, mx.array):
            return x
        try:
            return mx.array(x)
        except Exception:
            return x

    orig_rot = VisionModel.rot_pos_emb
    if not getattr(orig_rot, "_vmlx_patched", False):
        def rot_pos_emb(self, grid_thw):
            return orig_rot(self, _as_mx(grid_thw))
        rot_pos_emb._vmlx_patched = True  # type: ignore[attr-defined]
        VisionModel.rot_pos_emb = rot_pos_emb  # type: ignore[assignment]

    orig_call = VisionModel.__call__
    if not getattr(orig_call, "_vmlx_patched", False):
        def __call__(self, hidden_states, grid_thw, **kwargs):
            return orig_call(self, hidden_states, _as_mx(grid_thw), **kwargs)
        __call__._vmlx_patched = True  # type: ignore[attr-defined]
        VisionModel.__call__ = __call__  # type: ignore[assignment]

    _logger.debug("mlx_vlm_compat: patched Qwen3-VL VisionModel grid_thw coercion")


def _qwen35_patch_embed_to_mlx_layout(key, value):
    if (
        str(key).endswith("patch_embed.proj.weight")
        and getattr(value, "ndim", None) == 5
        and int(value.shape[1]) in (1, 3)
        and int(value.shape[-1]) not in (1, 3)
    ):
        return value.transpose(0, 2, 3, 4, 1)
    return value


def _patch_qwen35_patch_embed_layout() -> None:
    try:
        from mlx_vlm.models.qwen3_5 import qwen3_5 as _qwen_vl
    except ImportError:
        _qwen_vl = None
    try:
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as _qwen_moe_vl
    except ImportError:
        _qwen_moe_vl = None

    for module in (_qwen_vl, _qwen_moe_vl):
        Model = getattr(module, "Model", None)
        if Model is None:
            continue
        original = getattr(Model, "sanitize", None)
        if original is None or getattr(original, "_vmlx_patch_embed_layout", False):
            continue

        def sanitize(self, weights, _original=original):
            fixed = {}
            for key, value in weights.items():
                fixed[key] = _qwen35_patch_embed_to_mlx_layout(key, value)
            return _original(self, fixed)

        sanitize._vmlx_patch_embed_layout = True  # type: ignore[attr-defined]
        Model.sanitize = sanitize  # type: ignore[assignment]

    _logger.debug("mlx_vlm_compat: patched Qwen3.5/3.6 patch_embed layout")
