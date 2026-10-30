# SPDX-License-Identifier: Apache-2.0
"""Contain MLX sorted-RHS NAX row overflow without rechunking model prefill.

MLX #3922 fixes a pre-clamp signed-short conversion in the sorted expert
kernel. Affected builds leave rows unwritten above 32768 expanded expert rows.
The independent ragged-K defect (#3887) is not addressed here.

Each sorted call is bounded below the overflow. We qualify the ROW defect
independently, on the executing device;
a ragged-K canary cannot establish whether the row fix is present.
No quantization metadata, expert choices, reduction axes, or cache state change.
"""
from __future__ import annotations

import functools
import logging
import threading

import mlx.core as mx

_LOG = logging.getLogger(__name__)
_MAX_ROWS = 32768
_ORIGINAL = None
_ROW_DEFECT: bool | None = None
_LOCK = threading.Lock()
_COUNTERS = {"protected_calls": 0, "segments": 0, "unsorted_fallback_calls": 0}


def _row_overflow_present() -> bool:
    """One small deterministic row-boundary canary, never at import time.

    Constant quantized weights have an exactly representable answer. Unlike
    a random-dot-product tolerance, this catches finite unwritten output too.
    Explicit aranges/fills do not consume or reseed the generation RNG.
    """
    global _ROW_DEFECT
    if _ROW_DEFECT is not None:
        return _ROW_DEFECT
    with _LOCK:
        if _ROW_DEFECT is not None:
            return _ROW_DEFECT
        try:
            rows, experts, width = 32769, 8, 64
            indices = mx.minimum(mx.arange(rows) // 4096, experts - 1).astype(mx.uint32)
            weight = mx.broadcast_to(
                ((mx.arange(experts) + 1) / width)[:, None, None],
                (experts, width, width),
            ).astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
            inputs = mx.ones((rows, 1, width), dtype=mx.bfloat16)
            output = _ORIGINAL(
                inputs, packed, scales, biases, rhs_indices=indices,
                group_size=64, bits=4, sorted_indices=True,
            )
            expected = (indices + 1).astype(mx.bfloat16)[:, None, None]
            matches = mx.all(output == expected)
            mx.eval(matches)
            _ROW_DEFECT = not bool(matches.item())
            if _ROW_DEFECT:
                _LOG.warning(
                    "MLX sorted expert matmul row-overflow canary failed: "
                    "bounding affected calls to %d expanded rows; "
                    "model prefill shape and quantization remain unchanged",
                    _MAX_ROWS,
                )
        except (RuntimeError, ValueError, TypeError) as exc:
            # A failed qualification is not proof the unsafe kernel is sound.
            # The bounded calls preserve gather semantics on healthy builds too.
            _ROW_DEFECT = True
            _LOG.warning("MLX row-overflow qualification unavailable (%s); using bounded sorted calls", exc)
    return _ROW_DEFECT


def _on_gpu(stream) -> bool:
    device = mx.default_device() if stream is None else getattr(stream, "device", stream)
    return device == mx.gpu and mx.metal.is_available()


def _bounds(rows: int):
    count = (rows + _MAX_ROWS - 1) // _MAX_ROWS
    size = (rows + count - 1) // count
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def _gather(x, w, /, scales, biases=None, lhs_indices=None, rhs_indices=None,
            transpose=True, group_size=None, bits=None, mode="affine", *,
            sorted_indices=False, stream=None):
    kwargs = dict(lhs_indices=lhs_indices, rhs_indices=rhs_indices,
                  transpose=transpose, group_size=group_size, bits=bits, mode=mode,
                  sorted_indices=sorted_indices, stream=stream)
    if (not sorted_indices or lhs_indices is not None or rhs_indices is None
            or not transpose or x.ndim < 2
            or rhs_indices.size * x.shape[-2] <= _MAX_ROWS
            or not _on_gpu(stream) or not _row_overflow_present()):
        return _ORIGINAL(x, w, scales, biases, **kwargs)

    _COUNTERS["protected_calls"] += 1
    # This is the layout produced by sorted SwitchGLU. Slicing this independent
    # matrix-row dimension does not split KDA/DSA attention or its native state.
    if (x.ndim == 3 and x.shape[1] == 1 and rhs_indices.ndim == 1
            and x.shape[0] == rhs_indices.shape[0]):
        pieces = []
        for start, end in _bounds(x.shape[0]):
            pieces.append(_ORIGINAL(
                x[start:end], w, scales, biases,
                **dict(kwargs, rhs_indices=rhs_indices[start:end]),
            ))
        _COUNTERS["segments"] += len(pieces)
        return mx.concatenate(pieces, axis=0, stream=stream)

    # sorted_indices is a scheduling hint. Unsupported broadcast layouts keep
    # their original shapes/indices; never guess a flattening or reorder them.
    _COUNTERS["unsorted_fallback_calls"] += 1
    return _ORIGINAL(x, w, scales, biases, **dict(kwargs, sorted_indices=False))


def status() -> dict:
    return {"row_defect_observed": _ROW_DEFECT, "max_sorted_rows": _MAX_ROWS,
            **_COUNTERS}


def install() -> bool:
    """Install once without allocating GPU state on the importing thread."""
    global _ORIGINAL
    if getattr(mx.gather_qmm, "_vmlx_sorted_row_guard", False):
        return False
    _ORIGINAL = mx.gather_qmm
    wrapped = functools.wraps(_ORIGINAL)(_gather)
    wrapped._vmlx_sorted_row_guard = True
    mx.gather_qmm = wrapped
    return True
