"""Opt-in Qwen4 verification with stock low-precision intermediates.

For a certified sparse QSA mask, compute only selected query-key scores.
Preserve the stock softmax positions and probability-value reduction order.
Scores and probabilities retain the upstream dtype; fused vector SDPA has
different rounding and is not used here. Full-model qualification is required.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)
_logged_dispatches: set[tuple[str, int, str]] = set()


def _log_dispatch(q, k, path: str) -> None:
    # Shape/dtype metadata only: do not eval, synchronize, or read tensor data.
    # Context is reported on first use, not keyed, so growing history cannot
    # turn this into a per-token log or an unbounded set.
    key = (path, q.shape[2], str(q.dtype))
    if key not in _logged_dispatches:
        _logged_dispatches.add(key)
        logger.info(
            "QSA verification dispatch path=%s rows=%d context=%d dtype=%s",
            path, q.shape[2], k.shape[2], q.dtype,
        )


def qwen4_verify_sdpa(
    q, k, v, mask, *, scale, selected_token_bound=None,
    selected_four_token_block_bound=None,
):
    """Return grouped verification, or None to retain normal dispatch.

    selected_token_bound is supplied only by the built-in QSA indexer call
    site, which guarantees at most that many finite mask entries per row.
    selected_four_token_block_bound additionally certifies the number of
    four-token-aligned blocks containing finite entries. Without that
    stronger contract, PV retains the complete key/value sequence.
    """
    if os.environ.get("VMLX_QWEN4_VERIFY_SDPA", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != 1
        or k.shape[0] != 1
        or q.shape[1] != 24
        or k.shape[1] != 2
        or q.shape[2] not in (3, 4)
        or q.shape[3] != 256
        or k.shape[3] != 256
        or k.shape[2] < 8192
        or q.dtype not in (mx.float16, mx.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or (
            mask is not None
            and (mask.shape != (1, 1, q.shape[2], k.shape[2]) or mask.dtype != q.dtype)
        )
        or mx.default_device() != mx.gpu
    ):
        return None
    batch, heads, rows, dim = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    scaled = q * mx.array(scale, dtype=q.dtype)
    if (
        mask is not None
        and isinstance(selected_token_bound, int)
        and 0 < selected_token_bound < k.shape[2]
    ):
        # All finite entries are retained, including the incomplete causal
        # tail. Extra slots remain masked; sort restores absolute token order.
        indices = mx.sort(
            mx.argpartition(-mask[0, 0], kth=selected_token_bound - 1, axis=-1)[
                :, :selected_token_bound
            ],
            axis=-1,
        )
        selected_keys = mx.take(k, indices, axis=2)
        queries = scaled.reshape(batch, kv_heads, groups, rows, dim).transpose(
            0, 1, 3, 2, 4
        )
        # Four query heads preserve MLX's 32-lane QK reduction tree. A
        # 12-head matrix instead selects a different reduction and rounding.
        scores = (
            queries.reshape(batch, kv_heads, rows, groups // 4, 4, dim)
            @ mx.swapaxes(selected_keys[:, :, :, None], -1, -2)
        ).reshape(batch, kv_heads, rows, groups, selected_token_bound)
        dense_scores = mx.put_along_axis(
            mx.full(
                (batch, kv_heads, rows, groups, k.shape[2]),
                -mx.inf,
                dtype=q.dtype,
            ),
            indices[None, None, :, None, :],
            scores,
            axis=-1,
        )
        # Softmax must retain absolute positions. PV must retain complete
        # absolute 16-token reduction groups, not just the final K % 16.
        probabilities = mx.softmax(
            dense_scores + mask[0, 0][None, None, :, None, :],
            axis=-1,
            precise=True,
        )
        if (
            type(selected_four_token_block_bound) is int
            and selected_four_token_block_bound > 0
        ):
            total = k.shape[2]
            bulk = total // 16 * 16
            # Repacking four-token blocks into different groups changes the
            # reduction tree on M5 even when the compacted K has the same
            # remainder. Keep every lane of each selected absolute group,
            # including its masked zeros, and append the original tail.
            # At most one 16-token group is needed per selected four-token
            # block, so the caller's bound remains a conservative capacity.
            block_count = min(
                selected_four_token_block_bound,
                bulk // 16,
            )
            if block_count * 16 < bulk:
                keep = mx.any(
                    mx.isfinite(mask[0, 0, :, :bulk]).reshape(rows, bulk // 16, 16),
                    axis=-1,
                )
                blocks = mx.sort(
                    mx.argpartition(
                        -keep.astype(mx.int32), kth=block_count - 1, axis=-1
                    )[:, :block_count],
                    axis=-1,
                )
                pv_indices = (blocks[:, :, None] * 16 + mx.arange(16)).reshape(rows, -1)
                if bulk < total:
                    tail = mx.broadcast_to(
                        mx.arange(bulk, total)[None], (rows, total - bulk)
                    )
                    pv_indices = mx.concatenate([pv_indices, tail], axis=-1)
                compact_probabilities = mx.take_along_axis(
                    probabilities, pv_indices[None, None, :, None, :], axis=-1
                )
                compact_values = mx.take(v, pv_indices, axis=2)
                _log_dispatch(q, k, "sparse_qk_compact_pv")
                return (compact_probabilities @ compact_values).transpose(
                    0, 1, 3, 2, 4
                ).reshape(batch, heads, rows, dim)
        _log_dispatch(q, k, "sparse_qk_dense_pv")
        return (
            probabilities.transpose(0, 1, 3, 2, 4) @ v[:, :, None]
        ).reshape(batch, heads, rows, dim)
    _log_dispatch(q, k, "stock_sdpa")
    return mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=scale)
