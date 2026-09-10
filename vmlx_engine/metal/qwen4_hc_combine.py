"""Opt-in Qwen single-row residual combine with the stock rounding boundary.

Unlike a fused multiply-add, the product is rounded to the input dtype before
adding the residual. Projection, normalization and reduction math is untouched.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def exact_hc_combine_requested() -> bool:
    return os.environ.get("VMLX_QWEN4_EXACT_HC_COMBINE", "0") == "1"


@lru_cache(maxsize=16)
def _kernel(streams: int, hidden: int):
    return mx.fast.metal_kernel(
        name=f"vmlx_qwen4_exact_hc_combine_h{streams}_d{hidden}",
        input_names=["residual", "block", "inject"],
        output_names=["output"],
        source=f"""
            {{
            #pragma clang fp contract(off)
            #pragma clang fp reassociate(off)
            uint i = thread_position_in_grid.x;
            if (i >= {streams * hidden}u) return;
            uint stream = i / {hidden}u;
            uint feature = i % {hidden}u;
            T product = T(float(block[feature]) * float(inject[stream]));
            output[i] = T(float(residual[i]) + float(product));
            }}
        """,
    )


def exact_hc_combine(residual, block, inject, *, enabled: bool):
    """Return the exact one-row candidate, or None for the unchanged stock path."""
    if not enabled or residual.ndim != 3 or block.ndim != 3 or inject.ndim != 3:
        return None
    if residual.shape[:2] != (1, 1) or block.shape[:2] != (1, 1) or inject.shape[:2] != (1, 1):
        return None
    streams, hidden = int(inject.shape[-1]), int(block.shape[-1])
    if streams <= 0 or hidden <= 0 or residual.shape[-1] != streams * hidden:
        return None
    if residual.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return None
    if block.dtype != residual.dtype or inject.dtype != residual.dtype:
        return None
    result = _kernel(streams, hidden)(
        inputs=[residual, block, inject],
        template=[("T", residual.dtype)],
        grid=(streams * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(residual.shape)],
        output_dtypes=[residual.dtype],
    )[0]
    global _OBSERVED
    if not _OBSERVED:
        logging.getLogger(__name__).info(
            "Qwen exact HC combine active: streams=%d hidden=%d dtype=%s",
            streams, hidden, residual.dtype,
        )
        _OBSERVED = True
    return result
