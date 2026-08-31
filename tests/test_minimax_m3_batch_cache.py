# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Qwen/JANG sparse-cache continuous batching."""

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.mllm_batch_generator import (  # noqa: E402
    _ensure_batch_cache,
    _merge_caches,
)
from vmlx_engine.models.minimax_m3.cache import (  # noqa: E402
    BatchMiniMaxM3SparseCache,
    MiniMaxM3SparseCache,
)


def _fill_sparse(length: int, base: float) -> MiniMaxM3SparseCache:
    cache = MiniMaxM3SparseCache()
    for token in range(length):
        value = base + token
        keys = mx.full((1, 2, 1, 4), value)
        values = mx.full((1, 2, 1, 4), -value)
        idx_keys = mx.full((1, 1, 1, 3), value + 0.5)
        cache.update_and_fetch(keys, values)
        cache.update_index(idx_keys)
    mx.eval(cache.keys, cache.values, cache.idx_keys)
    return cache


def _assert_row(batch, row: int, original: MiniMaxM3SparseCache) -> None:
    extracted = batch.extract(row)
    mx.eval(extracted.keys, extracted.values, extracted.idx_keys)
    assert isinstance(extracted, MiniMaxM3SparseCache)
    assert extracted.offset == original.offset
    assert bool(mx.all(extracted.keys == original.keys[..., : original.offset, :]))
    assert bool(mx.all(extracted.values == original.values[..., : original.offset, :]))
    assert bool(
        mx.all(extracted.idx_keys == original.idx_keys[..., : original.offset, :])
    )


def test_merge_preserves_index_lane_and_decode_append():
    long = _fill_sparse(5, 10.0)
    short = _fill_sparse(3, 100.0)

    merged = BatchMiniMaxM3SparseCache.merge([long, short])

    assert merged.keys.shape == (2, 2, 5, 4)
    assert merged.idx_keys.shape == (2, 1, 5, 3)
    assert merged.left_padding.tolist() == [0, 2]
    _assert_row(merged, 0, long)
    _assert_row(merged, 1, short)

    merged.update_and_fetch(
        mx.full((2, 2, 1, 4), 999.0),
        mx.full((2, 2, 1, 4), -999.0),
    )
    history = merged.update_index(mx.full((2, 1, 1, 3), 999.5))
    mx.eval(history)
    assert history.shape == (2, 1, 6, 3)
    assert bool(mx.all(history[:, :, -1, :] == 999.5))


def test_runtime_merge_and_single_promotion_keep_sparse_type():
    first = _fill_sparse(4, 1.0)
    second = _fill_sparse(2, 20.0)

    merged = _merge_caches([[first], [second]])[0]
    promoted = _ensure_batch_cache([first])[0]

    assert isinstance(merged, BatchMiniMaxM3SparseCache)
    assert isinstance(promoted, BatchMiniMaxM3SparseCache)
    _assert_row(merged, 0, first)
    _assert_row(merged, 1, second)
    _assert_row(promoted, 0, first)


def test_filter_keeps_index_alignment_after_padding_shift():
    first = _fill_sparse(5, 1.0)
    middle = _fill_sparse(2, 20.0)
    last = _fill_sparse(3, 40.0)
    merged = BatchMiniMaxM3SparseCache.merge([first, middle, last])
    # Force MLX's 256-token capacity reserve before filtering; allocated length
    # is then intentionally much larger than the logical cache length.
    merged.update_and_fetch(
        mx.full((3, 2, 1, 4), 99.0),
        mx.full((3, 2, 1, 4), -99.0),
    )
    merged.update_index(mx.full((3, 1, 1, 3), 99.5))

    merged.filter(mx.array([1, 2], mx.int32))

    assert merged.left_padding.tolist() == [1, 0]
    assert merged._idx == 4
    assert merged.idx_keys.shape[2] > merged._idx
    for row, original in ((0, middle), (1, last)):
        extracted = merged.extract(row)
        assert extracted.offset == original.offset + 1
        assert bool(
            mx.all(
                extracted.idx_keys[..., : original.offset, :]
                == original.idx_keys[..., : original.offset, :]
            )
        )
        assert bool(mx.all(extracted.idx_keys[..., -1, :] == 99.5))


def test_extend_keeps_both_sparse_batches_extractable():
    first = _fill_sparse(5, 1.0)
    second = _fill_sparse(2, 20.0)
    third = _fill_sparse(4, 40.0)
    left = BatchMiniMaxM3SparseCache.merge([first, second])
    right = BatchMiniMaxM3SparseCache.merge([third])

    left.extend(right)

    assert left.keys.shape[0] == 3
    assert left.idx_keys.shape[0] == 3
    _assert_row(left, 0, first)
    _assert_row(left, 1, second)
    _assert_row(left, 2, third)


def test_trim_rewinds_both_append_offsets():
    merged = BatchMiniMaxM3SparseCache.merge([_fill_sparse(4, 1.0)])

    assert merged.trim(2) == 2
    merged.update_and_fetch(
        mx.full((1, 2, 1, 4), 77.0),
        mx.full((1, 2, 1, 4), -77.0),
    )
    history = merged.update_index(mx.full((1, 1, 1, 3), 77.5))
    mx.eval(history)

    assert history.shape[2] == 3
    assert bool(mx.all(history[:, :, -1, :] == 77.5))
