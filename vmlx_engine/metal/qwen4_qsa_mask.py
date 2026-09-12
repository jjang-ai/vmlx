# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact expansion of an already selected Qwen QSA block mask.

This is the additive *index* mask, not a complete causal attention mask.
Future tail tokens stay visible here; QSAAttention applies causality later.
Selection, score arithmetic, pooled keys and native cache state are untouched.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED: set[tuple[int, int]] = set()


def qsa_mask_requested() -> bool:
    return os.environ.get("VMLX_QWEN4_QSA_MASK", "0") == "1"


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_additive_block_mask",
        input_names=["hits", "counts", "length"],
        output_names=["mask"],
        ensure_row_contiguous=True,
        source=r"""
            const uint i = thread_position_in_grid.x;
            const uint n = length[0];
            const uint seq = hits_shape[1];
            const uint blocks = hits_shape[2];
            if (i >= uint(hits_shape[0]) * seq * n) return;
            const uint row = i / n;
            const uint token = i % n;
            const uint complete = uint(counts[row % seq]);
            const uint block = token / 4u;
            const bool selected = block < blocks && block < complete &&
                                  hits[row * blocks + block];
            const bool tail = token >= complete * 4u;
            mask[i] = selected || tail ? 0.0f : -INFINITY;
        """,
    )


def qsa_block_mask(hits, counts, *, ratio: int, key_length: int, enabled: bool):
    """Return the exact F32 index mask, or None for the unchanged stock path.

The first eligible launch is materialized to catch compilation/device errors
before adoption. Later asynchronous execution errors propagate normally.
Length is a runtime scalar, never a per-growing-token shader specialization.
Only a fresh mask is written, so compilation fallback cannot append cache twice.
"""
    global _FAILED
    if not enabled or _FAILED or hits.ndim != 3:
        return None
    batch, rows, blocks = hits.shape
    if (
        not 1 <= batch <= 2
        or not 1 <= rows <= 4
        or hits.dtype != mx.bool_
        or counts.dtype != mx.int32
        or counts.shape != (rows,)
        or ratio != 4
        or not 2048 < key_length <= 131072
        or blocks != key_length // ratio
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None
    try:
        result = _kernel()(
            inputs=[hits, counts, mx.array([key_length], dtype=mx.uint32)],
            grid=(batch * rows * key_length, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, 1, rows, key_length)],
            output_dtypes=[mx.float32],
        )[0]
        shape = (batch, rows)
        if shape not in _OBSERVED:
            mx.eval(result)
            _OBSERVED.add(shape)
            logger.info(
                "Qwen QSA additive mask active: batch=%d rows=%d tokens=%d "
                "ratio=4 output=float32 selection=unchanged causal=separate",
                batch, rows, key_length,
            )
        return result
    except (RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning("Qwen QSA mask disabled after launch failure: %s", exc)
        return None
