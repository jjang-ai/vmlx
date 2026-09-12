# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM decode-router scheduling, with the stock FP32 arithmetic.

Only post-matmul pointwise/selection work is compiled. Router weight storage,
its FP32 matmul, shared/routed projections and architecture cache state remain
owned by the model. No persistent FP32 copy or quantization rewrite is made.
"""

from functools import lru_cache
import logging
import math
import os

import mlx.core as mx

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False


def router_compile_requested() -> bool:
    return os.environ.get("VMLX_GLM5_ROUTER_COMPILE", "0") == "1"


@lru_cache(maxsize=64)
def _compiled_post(top_k: int, norm_topk: bool):
    def post(logits, correction_bias):
        scores = mx.sigmoid(logits)
        choice = scores + correction_bias
        idx = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
        weights = mx.take_along_axis(scores, idx, axis=-1)
        if norm_topk:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return idx, weights

    return mx.compile(post)


def glm5_router_post(logits, correction_bias, *, top_k, norm_topk, scaling,
                     enabled: bool):
    """Return stock-contract (indices, weights), or None for the stock path.

The small-row bound is a scheduling qualification, not a context/output cap.
Long prefill, batching, unsupported dtypes and compile errors keep the original
implementation. The helper owns no model, input or native cache arrays.
"""
    global _FAILED, _OBSERVED
    if not enabled or _FAILED:
        return None
    if (
        logits.ndim != 3 or logits.shape[0] != 1
        or not 1 <= logits.shape[1] <= 8
        or logits.dtype != mx.float32
        or correction_bias.dtype != mx.float32
        or correction_bias.shape != (logits.shape[-1],)
        or not isinstance(top_k, int) or isinstance(top_k, bool)
        or not 1 <= top_k <= logits.shape[-1]
        or not isinstance(norm_topk, bool)
        or not isinstance(scaling, (float, int))
        or not math.isfinite(scaling)
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None
    try:
        indices, weights = _compiled_post(top_k, norm_topk)(
            logits, correction_bias
        )
        # MLX's compiled embedded scalar constants can round differently from
        # the stock eager scalar conversion (and canonicalize negative zero).
        # Preserve this producer boundary instead of rounding model settings.
        outputs = indices, weights * scaling
        if not _OBSERVED:
            # Resolve the first launch before declaring execution observed.
            # No cache/model mutation precedes this safe fallback boundary.
            mx.eval(outputs)
            _OBSERVED = True
            logger.info(
                "GLM compiled router active: rows=%d experts=%d top_k=%d "
                "norm_topk=%s scaling=%s logits=fp32 weight_storage=unchanged "
                "scope=base_text_small_rows",
                logits.shape[1], logits.shape[-1], top_k, norm_topk,
                float(scaling).hex(),
            )
        return outputs
    except (RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning("GLM compiled router disabled after launch failure: %s", exc)
        return None
