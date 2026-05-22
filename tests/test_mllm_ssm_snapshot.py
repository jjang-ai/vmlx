# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMBatchGenerator._snapshot_ssm_per_row / _restore_ssm_rows.

Foundation for batched speculative decoding rollback in hybrid SSM/ATT
models (issue #134/#135 follow-up). When a batched verify forward produces
partial-reject rows, those rows' SSM state must revert to pre-verify while
fully-accepted rows keep their advanced state. These primitives provide that
selective restore via per-row concatenate.

Tests do NOT require a real model or BatchMambaCache instance — they use
minimal cache fakes that mimic the .cache / .is_trimmable() / .extract()
contract used by the snapshot/restore code.

Run:
    .venv/bin/python -m pytest tests/test_mllm_ssm_snapshot.py -v
"""

from __future__ import annotations

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Cache fakes
# ---------------------------------------------------------------------------


class _FakeSSMCache:
    """Mimics BatchMambaCache enough for snapshot/restore.

    - is_trimmable() = False (SSM layers can't be trimmed via offset)
    - .cache: list of mx.array of shape (B, ...) — per-batch SSM state
    - .extract(idx) returns a single-row stand-in (object with .cache attr)
    """

    def __init__(self, cache_arrays):
        # cache_arrays: list of mx.array, each shape (B, ...)
        self.cache = list(cache_arrays)

    def is_trimmable(self) -> bool:
        return False

    def extract(self, idx: int):
        # Mirror BatchMambaCache.extract: return an object whose .cache holds
        # per-row slices.
        return _FakeMambaCache(
            [mx.contiguous(a[idx : idx + 1]) if a is not None else None
             for a in self.cache]
        )


class _FakeMambaCache:
    """Single-row companion type returned by _FakeSSMCache.extract."""

    def __init__(self, cache_arrays):
        self.cache = list(cache_arrays)


class _FakeKVCache:
    """Mimics BatchKVCache: trimmable, has keys/values."""

    def __init__(self, B: int, seqlen: int, n_heads: int = 2, head_dim: int = 4):
        self.keys = mx.zeros((B, n_heads, seqlen, head_dim))
        self.values = mx.zeros((B, n_heads, seqlen, head_dim))
        self.offset = mx.array([seqlen] * B)

    def is_trimmable(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def _import_helpers():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    return (
        MLLMBatchGenerator._snapshot_ssm_per_row,
        MLLMBatchGenerator._restore_ssm_rows,
    )


def test_snapshot_skips_trimmable_layers():
    """KV cache layers must NOT be snapshotted — they use offset rewind instead."""
    snapshot_fn, _ = _import_helpers()
    kv = _FakeKVCache(B=2, seqlen=4)
    snap = snapshot_fn([kv])
    assert snap == {}


def test_snapshot_skips_empty_ssm_cache():
    """Layer with cache=None or empty list is skipped (not yet allocated)."""
    snapshot_fn, _ = _import_helpers()
    empty_ssm = _FakeSSMCache(cache_arrays=[])
    snap = snapshot_fn([empty_ssm])
    assert snap == {}


def test_snapshot_captures_per_row_state():
    """Each row gets its own snapshot via .extract(idx)."""
    snapshot_fn, _ = _import_helpers()
    B = 3
    state_arr = mx.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # shape (3, 2)
    ssm = _FakeSSMCache(cache_arrays=[state_arr])
    snap = snapshot_fn([ssm])

    assert 0 in snap  # layer index 0 captured
    assert len(snap[0]) == B  # one snapshot per row
    # Row 0 snapshot should hold [[1.0, 1.0]]
    row0 = snap[0][0]
    assert hasattr(row0, "cache")
    assert row0.cache[0].tolist() == [[1.0, 1.0]]
    row2 = snap[0][2]
    assert row2.cache[0].tolist() == [[3.0, 3.0]]


def test_snapshot_mixed_layers():
    """Mix of KV (skipped) + SSM (captured)."""
    snapshot_fn, _ = _import_helpers()
    kv = _FakeKVCache(B=2, seqlen=4)
    ssm = _FakeSSMCache(cache_arrays=[mx.array([[1.0], [2.0]])])
    cache = [kv, ssm, kv]
    snap = snapshot_fn(cache)

    assert 0 not in snap  # kv skipped
    assert 1 in snap      # ssm captured
    assert 2 not in snap  # kv skipped


# ---------------------------------------------------------------------------
# Restore tests
# ---------------------------------------------------------------------------


def test_restore_empty_snapshot_is_noop():
    _, restore_fn = _import_helpers()
    ssm = _FakeSSMCache(cache_arrays=[mx.array([[1.0], [2.0]])])
    original = ssm.cache[0].tolist()
    restore_fn([ssm], {}, [0, 1])
    assert ssm.cache[0].tolist() == original


def test_restore_empty_row_indices_is_noop():
    _, restore_fn = _import_helpers()
    ssm = _FakeSSMCache(cache_arrays=[mx.array([[1.0], [2.0]])])
    snap = {0: [_FakeMambaCache([mx.array([[99.0]])]),
                _FakeMambaCache([mx.array([[99.0]])])]}
    original = ssm.cache[0].tolist()
    restore_fn([ssm], snap, [])
    assert ssm.cache[0].tolist() == original


def test_restore_single_row_keeps_other_row_current():
    """Critical correctness test: restoring row 0 must NOT clobber row 1."""
    snapshot_fn, restore_fn = _import_helpers()

    pre_verify = mx.array([[1.0], [2.0]])  # rows: pre-verify
    ssm = _FakeSSMCache(cache_arrays=[pre_verify])

    # Snapshot pre-verify
    snap = snapshot_fn([ssm])

    # Simulate verify forward: state advances on both rows
    ssm.cache = [mx.array([[10.0], [20.0]])]  # post-verify

    # Restore only row 0
    restore_fn([ssm], snap, [0])

    # Row 0 should be back to pre-verify (1.0); row 1 should stay at 20.0
    result = ssm.cache[0].tolist()
    assert result == [[1.0], [20.0]], (
        f"row 0 should restore to 1.0, row 1 should stay at 20.0, got {result}"
    )


def test_restore_multiple_rows():
    snapshot_fn, restore_fn = _import_helpers()

    pre = mx.array([[1.0], [2.0], [3.0], [4.0]])  # 4 rows
    ssm = _FakeSSMCache(cache_arrays=[pre])
    snap = snapshot_fn([ssm])
    ssm.cache = [mx.array([[10.0], [20.0], [30.0], [40.0]])]

    # Restore rows 0 and 2
    restore_fn([ssm], snap, [0, 2])

    result = ssm.cache[0].tolist()
    assert result == [[1.0], [20.0], [3.0], [40.0]]


def test_restore_all_rows_equals_full_revert():
    snapshot_fn, restore_fn = _import_helpers()

    pre = mx.array([[1.0], [2.0]])
    ssm = _FakeSSMCache(cache_arrays=[pre])
    snap = snapshot_fn([ssm])
    ssm.cache = [mx.array([[99.0], [99.0]])]

    restore_fn([ssm], snap, [0, 1])
    assert ssm.cache[0].tolist() == [[1.0], [2.0]]


def test_restore_multi_array_layer():
    """SSM layers often have multiple state arrays (e.g. Mamba conv+ssm states)."""
    snapshot_fn, restore_fn = _import_helpers()

    a1 = mx.array([[1.0], [2.0]])
    a2 = mx.array([[10.0, 11.0], [20.0, 21.0]])
    ssm = _FakeSSMCache(cache_arrays=[a1, a2])
    snap = snapshot_fn([ssm])

    # Simulate verify
    ssm.cache = [
        mx.array([[99.0], [99.0]]),
        mx.array([[88.0, 88.0], [88.0, 88.0]]),
    ]
    restore_fn([ssm], snap, [0])

    # Row 0 of both arrays restored; row 1 keeps post-verify
    assert ssm.cache[0].tolist() == [[1.0], [99.0]]
    assert ssm.cache[1].tolist() == [[10.0, 11.0], [88.0, 88.0]]


def test_restore_preserves_kv_layers():
    """Restore must touch only SSM layers — KV layers (trimmable) untouched."""
    snapshot_fn, restore_fn = _import_helpers()

    kv = _FakeKVCache(B=2, seqlen=4)
    kv_keys_id = id(kv.keys)
    ssm = _FakeSSMCache(cache_arrays=[mx.array([[1.0], [2.0]])])
    cache = [kv, ssm]
    snap = snapshot_fn(cache)
    ssm.cache = [mx.array([[99.0], [99.0]])]
    restore_fn(cache, snap, [0])

    # KV layer keys array must be the same object (untouched)
    assert id(kv.keys) == kv_keys_id


# ---------------------------------------------------------------------------
# Round-trip: snapshot → mutate → restore some rows → verify shape preserved
# ---------------------------------------------------------------------------


def test_round_trip_shape_preserved():
    """Snapshot then restore selectively; final cache shape unchanged."""
    snapshot_fn, restore_fn = _import_helpers()

    pre = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    ssm = _FakeSSMCache(cache_arrays=[pre])
    snap = snapshot_fn([ssm])
    ssm.cache = [mx.array([[7.0, 8.0], [9.0, 0.0], [1.0, 2.0]])]
    restore_fn([ssm], snap, [1])

    assert ssm.cache[0].shape == (3, 2)
    assert ssm.cache[0].tolist() == [[7.0, 8.0], [3.0, 4.0], [1.0, 2.0]]
