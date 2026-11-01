# SPDX-License-Identifier: Apache-2.0
"""Opt-in sparse-score QSA prefill attention for Qwen4 (Flash-Next).

``VMLX_QWEN4_PREFILL_DIRECT=1`` lets long prefill chunks compute QK^T only for
the keys the QSA indexer selected (``qwen4_qsa_sparse_kernels.sparse_scores``)
instead of the whole context.  Softmax (precise) and the probability-value
product are MLX's own ops on the dense score layout, so the result equals
MLX's stock masked attention bit for bit; storing scores densely is what keeps
the downstream rounding identical.

Safety properties:

* Fail closed.  Any call outside the measured geometry returns a reason and the
  model keeps stock attention.  An explicit consumer call that is unsupported
  raises instead of silently computing something else.
* Proven before use.  The first readiness check runs a numerical preflight per
  dtype (float16 and bfloat16) against ``mx.fast.scaled_dot_product_attention``
  with the equivalent dense mask and requires bit equality.  Any failure retires
  the lane for the process (terminal), so traffic stays on stock attention.
* No silent fallback after dispatch.  The first real dispatch per dtype is
  evaluated eagerly; if it raises, the lane is retired and the error propagates
  (the KV cache has already advanced, so retrying another path would be wrong).

Selection ABI: the kernel ignores ``valid`` and reads the first
``min(512, (pos + 1) // 4)`` ids of each row, which must be strictly ascending
block ids.  Both Qwen4 selectors emit that prefix
(``tests/test_qsa_selector_prefix_contract.py``).
``VMLX_QWEN4_PREFILL_DIRECT_VALIDATE=1`` re-checks it per call (synchronizes).
"""

from __future__ import annotations

import logging
import math
import operator
import os
import threading

import mlx.core as mx

from . import qwen4_qsa_sparse_kernels as kernels

logger = logging.getLogger(__name__)

BLOCK = kernels.BLOCK
TOP_K = kernels.TOP_K
SCALE = 0.0625            # 1/sqrt(256), exact in binary
MAX_CONTEXT = 1 << 20
DTYPES = (mx.float16, mx.bfloat16)

_UNPROVEN, _PROVEN, _FAILED = "unproven", "proven", "failed"


class _LaneState:
    """Process-wide readiness.  FAILED is terminal; proofs are per dtype."""

    def __init__(self):
        self.status = _UNPROVEN
        self.proven: frozenset[str] = frozenset()
        self.lock = threading.RLock()

    def fail(self):
        self.status = _FAILED
        self.proven = frozenset()

    def prove(self, dtype):
        if self.status == _FAILED:
            return
        self.proven = self.proven | {str(dtype)}
        if all(str(d) in self.proven for d in DTYPES):
            self.status = _PROVEN

    def has(self, dtype) -> bool:
        return str(dtype) in self.proven


LANE = _LaneState()


def _flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def direct_prefill_requested() -> bool:
    return _flag("VMLX_QWEN4_PREFILL_DIRECT")


def _on_gpu() -> bool:
    try:
        return mx.metal.is_available() and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def lane_unavailable_reason(*, require_enabled: bool = True) -> str | None:
    """Why the lane cannot serve this process at all (ignores call geometry)."""
    if LANE.status == _FAILED:
        return "the sparse-score lane was retired after a failed check in this process"
    if require_enabled and not direct_prefill_requested():
        return "VMLX_QWEN4_PREFILL_DIRECT is off"
    if not kernels.matrix_units_available():
        return "the GPU has no NAX matrix units for the sparse-score kernel"
    return None


def _host_int(value) -> int | None:
    if isinstance(value, (bool, mx.array)):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def direct_unsupported_reason(
    queries, keys, values, block_ids, block_valid, *,
    pos_start, total_tokens, scale,
    compress_ratio: int = BLOCK, block_topk: int = TOP_K,
    require_enabled: bool = True,
) -> str | None:
    """``None`` when the sparse kernels accept this call, else the reason."""
    reason = lane_unavailable_reason(require_enabled=require_enabled)
    if reason is not None:
        return reason
    if not _on_gpu():
        return "the active MLX device is not an available Metal GPU"
    tensors = (queries, keys, values, block_ids, block_valid)
    if not all(isinstance(t, mx.array) for t in tensors):
        return "all tensor inputs must be MLX arrays"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "Q, K, and V must be rank four"
    if block_ids.ndim != 2 or block_valid.ndim != 2:
        return "block ids and validity must be rank two"
    batch, heads, rows, dim = (int(x) for x in queries.shape)
    if (batch, heads, dim) != (1, kernels.QUERY_HEADS, kernels.HEAD_DIM):
        return "Q must have shape [1, 24, S, 256]"
    if rows < 16:
        return "at least 16 query rows are required"
    k_batch, kv_heads, key_len, k_dim = (int(x) for x in keys.shape)
    if (k_batch, kv_heads, k_dim) != (1, kernels.KV_HEADS, kernels.HEAD_DIM):
        return "K must have shape [1, 2, T, 256]"
    if tuple(values.shape) != tuple(keys.shape):
        return "V must have the same shape as K"
    if queries.dtype not in DTYPES:
        return "Q must be float16 or bfloat16"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "Q, K, and V dtypes must match"
    if block_ids.dtype != mx.int32 or block_valid.dtype != mx.bool_:
        return "block ids must be int32 and validity must be bool"
    if tuple(block_ids.shape) != (rows, TOP_K) or tuple(block_valid.shape) != (rows, TOP_K):
        return "block ids and validity must have shape [S, 512]"
    if int(compress_ratio) != BLOCK:
        return "the kernels expand exactly four-token blocks"
    if int(block_topk) != TOP_K:
        return "the kernels read exactly 512 selected blocks"
    start, total = _host_int(pos_start), _host_int(total_tokens)
    if start is None or total is None:
        return "pos_start and total_tokens must be host integers"
    if isinstance(scale, (bool, mx.array)) or not isinstance(scale, (int, float)):
        return "scale must be a numeric host scalar"
    if start < 0 or total <= 0:
        return "positions must describe a non-empty non-negative suffix"
    if start + rows != total:
        return "Q must be the suffix ending exactly at total_tokens"
    if key_len != total:
        return "K/V must be the logical cache prefix with length == total_tokens"
    if total > MAX_CONTEXT:
        return "the logical token count exceeds the production context limit"
    if total // BLOCK <= TOP_K:
        return "the context has not crossed the dense/sparse boundary"
    if not math.isfinite(float(scale)) or float(scale) != SCALE:
        return "scale must equal 1/sqrt(256)"
    return None


def direct_supported(queries, keys, values, block_ids, block_valid, *,
                     pos_start, total_tokens, scale,
                     compress_ratio: int = BLOCK, block_topk: int = TOP_K) -> bool:
    return direct_unsupported_reason(
        queries, keys, values, block_ids, block_valid, pos_start=pos_start,
        total_tokens=total_tokens, scale=scale, compress_ratio=compress_ratio,
        block_topk=block_topk,
    ) is None


def _selection_violation(block_ids, block_valid, *, pos_start, compress_ratio) -> str | None:
    """Check the valid-prefix selection contract (synchronizes)."""
    rows, slots = (int(x) for x in block_ids.shape)
    visible = (mx.arange(pos_start, pos_start + rows, dtype=mx.int32) + 1) // int(compress_ratio)
    count = block_valid.astype(mx.int32).sum(axis=-1)
    both = block_valid[:, 1:] & block_valid[:, :-1]
    checks = (
        ("block_valid is not a per-row prefix",
         mx.all((mx.arange(slots, dtype=mx.int32)[None] < count[:, None]) == block_valid)),
        ("valid count != min(512, complete blocks)",
         mx.all(count == mx.minimum(visible, slots))),
        ("valid block ids are not strictly ascending",
         mx.all(~both | (block_ids[:, 1:] > block_ids[:, :-1]))),
        ("a valid block id is negative or not fully visible",
         mx.all(~block_valid | ((block_ids >= 0) & (block_ids < visible[:, None])))),
    )
    mx.eval([c for _, c in checks])
    for message, ok in checks:
        if not bool(ok.item()):
            return message
    return None


def selected_block_buffer(block_ids, block_valid, *, pos_start, compress_ratio: int = BLOCK,
                          validate: bool | None = None) -> mx.array:
    """``[S, 512]`` int32 ids -> contiguous ``[1, 1, S, 512]`` uint32 kernel buffer."""
    if block_ids.dtype != mx.int32:
        raise ValueError("block ids must be int32")
    if block_valid.dtype != mx.bool_:
        raise ValueError("block validity must be bool")
    if block_ids.ndim != 2 or tuple(block_valid.shape) != tuple(block_ids.shape):
        raise ValueError("block ids and validity must be matching [S, 512]")
    if int(block_ids.shape[1]) != TOP_K:
        raise ValueError("block ids must have exactly 512 slots")
    if validate is None:
        validate = _flag("VMLX_QWEN4_PREFILL_DIRECT_VALIDATE")
    if validate:
        problem = _selection_violation(block_ids, block_valid, pos_start=pos_start,
                                       compress_ratio=compress_ratio)
        if problem is not None:
            raise ValueError("QSA sparse selection contract violated: " + problem)
    return mx.contiguous(block_ids.astype(mx.uint32)[None, None])


def direct_attention(queries, keys, values, block_ids, block_valid, *,
                     pos_start, total_tokens, scale,
                     compress_ratio: int = BLOCK, block_topk: int = TOP_K) -> mx.array:
    """Sparse-score attention for one prefill chunk; ``[1, 24, S, 256]`` out."""
    reason = direct_unsupported_reason(
        queries, keys, values, block_ids, block_valid, pos_start=pos_start,
        total_tokens=total_tokens, scale=scale, compress_ratio=compress_ratio,
        block_topk=block_topk,
    )
    if reason is not None:
        raise ValueError(f"unsupported QSA prefill direct call: {reason}")
    ids = selected_block_buffer(block_ids, block_valid, pos_start=int(pos_start),
                                compress_ratio=int(compress_ratio))
    scores = kernels.sparse_scores(queries, keys, ids, int(pos_start), float(scale))
    weights = mx.softmax(scores, axis=-1, precise=True)
    out = (weights @ values[:, :, None]).reshape(queries.shape)
    return _first_dispatch_check(out, queries.dtype)


def _first_dispatch_check(out, dtype):
    """Evaluate the first dispatch of a dtype eagerly; retire the lane on error."""
    if LANE.status == _FAILED or LANE.has(dtype):
        return out
    with LANE.lock:
        if LANE.status == _FAILED or LANE.has(dtype):
            return out
        try:
            mx.eval(out)
        except Exception:
            LANE.fail()
            logger.warning("%s: first %s sparse-score dispatch failed; lane retired",
                           __name__, dtype)
            raise
        # Running is not proof of identical arithmetic: only the preflight
        # records a dtype as proven.
    return out


def _dense_mask(rows, total, pos_start, block_ids, dtype):
    positions = mx.arange(pos_start, total)
    tokens = mx.arange(total)
    complete = (positions + 1) // BLOCK
    chosen = mx.put_along_axis(mx.zeros((rows, total // BLOCK), dtype=mx.bool_), block_ids,
                               mx.array(True), axis=-1)
    keep = mx.repeat(chosen, BLOCK, axis=-1)
    if total % BLOCK:
        keep = mx.concatenate([keep, mx.zeros((rows, total % BLOCK), dtype=mx.bool_)], axis=-1)
    keep = (keep | (tokens[None] >= complete[:, None] * BLOCK)) & (tokens[None] <= positions[:, None])
    return mx.where(keep, 0, -mx.inf).astype(dtype)[None, None]


def _preflight_one(dtype) -> bool:
    """One real call vs stock masked SDPA; bit equality or the lane retires."""
    if LANE.status == _FAILED:
        return False
    if LANE.has(dtype):
        return True
    with LANE.lock:
        if LANE.status == _FAILED:
            return False
        if LANE.has(dtype):
            return True
        rows, total = 256, 2304
        start = total - rows
        try:
            # explicit keys: the sampling RNG is left alone
            q = mx.random.normal((1, kernels.QUERY_HEADS, rows, kernels.HEAD_DIM),
                                 key=mx.random.key(3701)).astype(dtype)
            k = mx.random.normal((1, kernels.KV_HEADS, total, kernels.HEAD_DIM),
                                 key=mx.random.key(3702)).astype(dtype)
            v = mx.random.normal((1, kernels.KV_HEADS, total, kernels.HEAD_DIM),
                                 key=mx.random.key(3703)).astype(dtype)
            ids = mx.contiguous(mx.broadcast_to(mx.arange(TOP_K, dtype=mx.int32)[None], (rows, TOP_K)))
            valid = mx.ones((rows, TOP_K), dtype=mx.bool_)
            got = direct_attention(q, k, v, ids, valid, pos_start=start,
                                   total_tokens=total, scale=SCALE)
            want = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=SCALE, mask=_dense_mask(rows, total, start, ids, dtype))
            mx.eval(got, want)
            if not bool(mx.array_equal(got, want)):
                raise RuntimeError("sparse-score attention differs from stock SDPA on this runtime")
        except Exception as exc:
            LANE.fail()
            logger.warning("%s: %s sparse-score preflight failed; lane retired: %s",
                           __name__, dtype, exc)
            return False
        LANE.prove(dtype)
        return True


def direct_preflight(*, dtype=None) -> bool:
    """Prove one dtype (or all) before traffic.  Synchronizes; not for the hot path."""
    if lane_unavailable_reason() is not None or not _on_gpu():
        return False
    if dtype is None:
        return all(_preflight_one(d) for d in DTYPES)
    return _preflight_one(dtype)


def direct_ready() -> bool:
    """Whether this process may route prefill chunks to the sparse-score lane."""
    if lane_unavailable_reason() is not None:
        return False
    if LANE.status == _PROVEN:
        return True
    if not _on_gpu():
        return False
    return all(_preflight_one(d) for d in DTYPES)


__all__ = [
    "LANE",
    "direct_attention",
    "direct_preflight",
    "direct_prefill_requested",
    "direct_ready",
    "direct_supported",
    "direct_unsupported_reason",
    "lane_unavailable_reason",
    "selected_block_buffer",
]
