# SPDX-License-Identifier: Apache-2.0
"""Exact N-1 rotating-window capture for mixed full/sliding-window caches.

A mixed-SWA prompt store needs the cache exactly at the N-1 prompt boundary.
Full-attention ``KVCache`` layers are append-only, so the live cache can be
sliced back to it at store time. ``RotatingKVCache`` windows cannot: decode
rolls the ring and evicts rows. The text lane therefore used to re-run the
whole prompt (a "deferred clean prompt-boundary re-prefill") after every
turn. The MLLM lane already avoids that with an end-of-prefill capture; this
module applies the same math to the text lane, where the capture runs right
after the N-1 prefill, so each ring is already exactly at the boundary.

Retained rows match the MLLM capture: the window at the boundary plus at most
two cache blocks of the causal concat overhang, which the block store needs
to cut exact rotating checkpoints at the blocks before a terminal partial
block. ``keep`` (attention-sink) rows are preserved. The capture declines
rather than guessing whenever the buffer is not in temporal order (a
single-token update rolled it in place), rings disagree on the offset, or the
window is not fully present. A declined capture keeps the clean re-prefill.
"""

import logging
import os
from copy import copy as _shallow_copy
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def inline_boundary_enabled() -> bool:
    """``VMLX_MIXED_SWA_INLINE_BOUNDARY=0`` restores the clean re-prefill."""
    raw = os.environ.get("VMLX_MIXED_SWA_INLINE_BOUNDARY", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def inline_boundary_verify_enabled() -> bool:
    """Opt-in diagnostic: also run the clean re-prefill and compare tensors."""
    raw = os.environ.get("VMLX_MIXED_SWA_INLINE_VERIFY", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def capture_rotating_boundary(
    cache: Any,
    *,
    block_size: int = 64,
    log_id: str = "?",
) -> Tuple[Optional[Tuple[int, Dict[int, Any]]], str]:
    """Snapshot every ``RotatingKVCache`` layer at its current boundary.

    Returns ``((boundary, {layer_idx: snapshot}), "captured")`` or
    ``(None, reason)``. ``boundary`` is the shared ring offset (absolute
    prompt position), which callers must check against their store key.
    """
    import mlx.core as mx

    if not cache:
        return None, "no_cache"
    rotating = [
        (idx, layer)
        for idx, layer in enumerate(cache)
        if type(layer).__name__ == "RotatingKVCache"
    ]
    if not rotating:
        return None, "no_rotating_layers"
    offsets = {int(getattr(layer, "offset", 0) or 0) for _, layer in rotating}
    if len(offsets) != 1:
        return None, "rotating_offsets_disagree"
    boundary = offsets.pop()
    if boundary <= 0:
        return None, "empty_boundary"
    retain_blocks = max(1, int(block_size or 64))

    snap_map: Dict[int, Any] = {}
    materialize = []
    for layer_idx, layer in rotating:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            return None, f"layer_{layer_idx}_no_buffer"
        idx = int(getattr(layer, "_idx", -1))
        max_size = int(getattr(layer, "max_size", 0) or 0)
        keep = int(getattr(layer, "keep", 0) or 0)
        physical = int(keys.shape[2])
        required = min(boundary, max_size)
        if idx != physical or max_size <= 0 or required <= keep or physical < required:
            logger.info(
                "mixed-SWA boundary capture declined for %s: layer %d "
                "offset=%d physical=%d idx=%d required=%d keep=%d — not a "
                "temporal post-prefill buffer covering the N-1 window",
                log_id, layer_idx, boundary, physical, idx, required, keep,
            )
            return None, "not_temporal_window"
        retained = min(physical, required + 2 * retain_blocks)
        if retained == physical:
            snap_k = keys[..., :physical, :]
            snap_v = values[..., :physical, :]
        elif keep > 0:
            recent = retained - keep
            if recent <= 0:
                return None, "keep_not_covered"
            snap_k = mx.concatenate(
                [keys[..., :keep, :], keys[..., physical - recent : physical, :]],
                axis=2,
            )
            snap_v = mx.concatenate(
                [values[..., :keep, :], values[..., physical - recent : physical, :]],
                axis=2,
            )
        else:
            snap_k = keys[..., physical - retained : physical, :]
            snap_v = values[..., physical - retained : physical, :]
        # Break the lazy reference to the full prefill buffer so the retained
        # copy is window-sized, not prompt-sized.
        snap_k = mx.contiguous(snap_k)
        snap_v = mx.contiguous(snap_v)
        materialize.extend((snap_k, snap_v))
        snap = _shallow_copy(layer)
        snap.keys = snap_k
        snap.values = snap_v
        snap.offset = boundary
        snap._idx = int(snap_k.shape[2])
        snap_map[layer_idx] = snap
    if materialize:
        mx.eval(*materialize)
    return (boundary, snap_map), "captured"


def _window_rows(layer: Any) -> Tuple[Any, Any]:
    """The causal window a RotatingKVCache attends from, in temporal order."""
    keys, values = layer.keys, layer.values
    physical = int(keys.shape[2])
    max_size = int(getattr(layer, "max_size", physical) or physical)
    keep = int(getattr(layer, "keep", 0) or 0)
    offset = int(getattr(layer, "offset", physical) or physical)
    idx = int(getattr(layer, "_idx", physical))
    if idx != physical and hasattr(layer, "_temporal_order"):
        keys = layer._temporal_order(keys)
        values = layer._temporal_order(values)
        physical = int(keys.shape[2])
    window = min(offset, max_size)
    if window >= physical:
        return keys, values
    if keep > 0:
        import mlx.core as mx

        recent = window - keep
        return (
            mx.concatenate([keys[..., :keep, :], keys[..., physical - recent :, :]], axis=2),
            mx.concatenate([values[..., :keep, :], values[..., physical - recent :, :]], axis=2),
        )
    return keys[..., physical - window :, :], values[..., physical - window :, :]


def compare_boundary_caches(inline: Any, reference: Any) -> Dict[str, Any]:
    """Compare an inline-assembled boundary cache with a clean re-prefill.

    Full-attention layers compare ``[:offset]``; rotating layers compare the
    temporal causal window (overhang lengths may differ legitimately). Returns
    ``{"exact": bool, "max_abs": float, "layers": n, "mismatch_layers": [...]}``.
    """
    import mlx.core as mx

    result: Dict[str, Any] = {"exact": True, "max_abs": 0.0, "layers": 0, "mismatch_layers": []}
    if not inline or not reference or len(inline) != len(reference):
        result.update(exact=False, error="layer_count")
        return result
    for idx, (a, b) in enumerate(zip(inline, reference)):
        result["layers"] += 1
        if int(getattr(a, "offset", -1)) != int(getattr(b, "offset", -2)):
            result["exact"] = False
            result["mismatch_layers"].append((idx, "offset"))
            continue
        if type(a).__name__ == "RotatingKVCache" or type(b).__name__ == "RotatingKVCache":
            ak, av = _window_rows(a)
            bk, bv = _window_rows(b)
        else:
            off = int(a.offset)
            ak, av = a.keys[..., :off, :], a.values[..., :off, :]
            bk, bv = b.keys[..., :off, :], b.values[..., :off, :]
        if ak.shape != bk.shape or av.shape != bv.shape:
            result["exact"] = False
            result["mismatch_layers"].append((idx, "shape"))
            continue
        diff = max(
            float(mx.max(mx.abs(ak.astype(mx.float32) - bk.astype(mx.float32))).item()),
            float(mx.max(mx.abs(av.astype(mx.float32) - bv.astype(mx.float32))).item()),
        )
        result["max_abs"] = max(result["max_abs"], diff)
        if diff != 0.0:
            result["exact"] = False
            result["mismatch_layers"].append((idx, round(diff, 6)))
    result["mismatch_layers"] = result["mismatch_layers"][:8]
    return result
