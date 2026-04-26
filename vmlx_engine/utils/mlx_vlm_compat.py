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
