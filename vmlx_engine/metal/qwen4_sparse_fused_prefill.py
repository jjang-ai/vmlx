# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused sparse QSA prefill (online softmax) for Qwen4 (Flash-Next).

``VMLX_QWEN4_SPARSE_FUSED_PREFILL=1`` routes long fp16 prefill chunks to
``qwen4_qsa_sparse_kernels.sparse_online_attention``, which never materializes
a score tensor.  Its contract is MLX's CURRENT fused masked attention
(``force_fused=True``), not the unfused score path used by the
``VMLX_QWEN4_PREFILL_DIRECT`` lane.

Admission is narrow and proven: only the measured MLX build / GPU, fp16,
256-4096 query rows and 32k-128k logical context; a bit-exact preflight against
fused SDPA runs once before the first dispatch and a failure disables the lane
for the process.  Decode, verification and other shapes keep stock attention.
"""

import logging
import os
import threading
from functools import lru_cache

import mlx.core as mx

from . import qwen4_qsa_direct_prefill as direct
from . import qwen4_qsa_sparse_kernels as kernels

logger = logging.getLogger(__name__)
MATH_ABI = kernels.MATH_ABI
MIN_CONTEXT = 32768
MAX_CONTEXT = 131072
MAX_QUERY_ROWS = 4096
DISPATCH_COUNT = 0
_state = "unproven"
_lock = threading.Lock()
_first_dispatch = False


def enabled():
    return os.environ.get("VMLX_QWEN4_SPARSE_FUSED_PREFILL", "0") == "1"


@lru_cache(maxsize=1)
def _hardware_allowed():
    try:
        return (
            mx.__version__ == "0.32.2"
            and mx.device_info().get("architecture") == "applegpu_g17s"
        )
    except (AttributeError, RuntimeError):
        return False


def _call(q, k, v, ids, valid, *, pos_start, total_tokens, scale):
    selected = direct.selected_block_buffer(ids, valid, pos_start=pos_start)
    return kernels.sparse_online_attention(q, k, v, selected, pos_start, scale)


def _preflight():
    # Explicit RNG keys leave the model sampling generator untouched. Padded
    # logical views, scattered complete blocks, and 0/1/2/3-token tails.
    rows, total = 256, 8195
    offset = total - rows
    q = mx.random.normal((1, 24, rows, 256), key=mx.random.key(704)).astype(mx.float16)
    k = mx.random.normal((1, 2, total + 19, 256), key=mx.random.key(705)).astype(mx.float16)[:, :, :total]
    v = mx.random.normal((1, 2, total + 19, 256), key=mx.random.key(706)).astype(mx.float16)[:, :, :total]
    positions = mx.arange(offset, total)
    complete = (positions + 1) // 4
    ids = (mx.arange(512)[None] * complete[:, None] // 512).astype(mx.int32)
    valid = mx.ones(ids.shape, dtype=mx.bool_)
    chosen = mx.put_along_axis(mx.zeros((rows, total // 4), dtype=mx.bool_), ids,
                               mx.array(True), axis=-1)
    keep = mx.concatenate([mx.repeat(chosen, 4, axis=-1),
                           mx.zeros((rows, total % 4), dtype=mx.bool_)], axis=-1)
    tokens = mx.arange(total)[None]
    mask = mx.where((keep | (tokens >= complete[:, None] * 4))
                    & (tokens <= positions[:, None]), 0, -mx.inf).astype(q.dtype)[None, None]
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=mask, force_fused=True)
    got = _call(q, k, v, ids, valid, pos_start=offset, total_tokens=total, scale=0.0625)
    equal = mx.array_equal(ref, got)
    mx.eval(equal)
    if not bool(equal):
        diff = got.astype(mx.float32) - ref.astype(mx.float32)
        raise RuntimeError(f"current-SDPA mismatch: max_abs={float(mx.max(mx.abs(diff)))}")


def ready():
    global _state
    if not enabled() or not _hardware_allowed() or mx.default_device() != mx.gpu:
        return False
    if direct.lane_unavailable_reason(require_enabled=False) is not None:
        return False
    if _state != "unproven":
        return _state == "ready"
    with _lock:
        if _state == "unproven":
            try:
                _preflight()
                _state = "ready"
            except Exception:
                _state = "failed"
                logger.warning("QSA sparse fused prefill disabled after numerical preflight",
                               exc_info=True)
    return _state == "ready"


def supported(q, k, v, ids, valid, *, pos_start, total_tokens, scale):
    if not enabled() or q.dtype != mx.float16 or not 256 <= q.shape[2] <= MAX_QUERY_ROWS:
        return False
    if not MIN_CONTEXT <= total_tokens <= MAX_CONTEXT:
        return False
    reason = direct.direct_unsupported_reason(
        q, k, v, ids, valid, pos_start=pos_start, total_tokens=total_tokens,
        scale=scale, require_enabled=False,
    )
    return reason is None and ready()


def attention(q, k, v, ids, valid, *, pos_start, total_tokens, scale):
    global DISPATCH_COUNT, _first_dispatch, _state
    if not supported(q, k, v, ids, valid, pos_start=pos_start,
                     total_tokens=total_tokens, scale=scale):
        raise ValueError("unsupported sparse fused QSA prefill")
    out = _call(q, k, v, ids, valid, pos_start=pos_start,
                total_tokens=total_tokens, scale=scale)
    if not _first_dispatch:
        try:
            mx.eval(out)
        except Exception:
            _state = "failed"
            # Cache has already advanced: propagate, never retry this chunk
            # on a different path with the same mutated cache.
            raise
        _first_dispatch = True
        logger.info("QSA prefill dispatch path=sparse_online rows=%d context=%d dtype=%s math=%s",
                    q.shape[2], total_tokens, q.dtype, MATH_ABI)
    DISPATCH_COUNT += 1
    return out
