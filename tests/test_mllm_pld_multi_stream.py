# SPDX-License-Identifier: Apache-2.0
"""Multi-stream (B>1) per-row replay validation tests (PR #152).

PR #150 C.5 added per-row replay forward + write-back primitives. B=1 was
validated live; the B>1 path uses concatenate-per-row in _writeback_kv_row
and _writeback_ssm_row but was preserved without live validation. These
tests exercise the B>1 paths with minimal mocks to catch obvious bugs in
per-row offset bookkeeping and other-row preservation.

For LIVE multi-stream validation (4 concurrent prompts on a real model),
use tests/benchmark/test_pld_byte_equality_mllm.py with --max-num-seqs 4
and 4 distinct prompts.

Run:
    .venv/bin/python -m pytest tests/test_mllm_pld_multi_stream.py -v
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
import pytest


def _import_gen():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    return MLLMBatchGenerator


# ---------------------------------------------------------------------------
# Minimal fakes (subset of test_mllm_per_row_replay.py shapes, B>1 sized)
# ---------------------------------------------------------------------------


class _FakeKVLayer:
    """Mimics BatchKVCache for B>1 testing."""

    def __init__(self, B: int, max_seq: int, n_heads: int = 2, head_dim: int = 4,
                 offsets=None, value_seed=0):
        # Each row gets a distinct value pattern to detect cross-row corruption
        rows_k = []
        rows_v = []
        for r in range(B):
            row_val = float(value_seed + r * 10)
            rows_k.append(mx.full((1, n_heads, max_seq, head_dim), row_val))
            rows_v.append(mx.full((1, n_heads, max_seq, head_dim), row_val + 0.5))
        self.keys = mx.concatenate(rows_k, axis=0)
        self.values = mx.concatenate(rows_v, axis=0)
        self.offset = mx.array(offsets if offsets is not None else [max_seq] * B)

    def is_trimmable(self) -> bool:
        return True


class _FakeSSMLayer:
    """Mimics BatchMambaCache for B>1 testing."""

    def __init__(self, B: int, value_per_row=None):
        if value_per_row is None:
            value_per_row = [float(r) + 1.0 for r in range(B)]
        rows = [mx.array([[v]]) for v in value_per_row]
        self.cache = [mx.concatenate(rows, axis=0)]

    def is_trimmable(self) -> bool:
        return False


class _FakeSoloKV:
    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset


class _FakeSoloSSM:
    def __init__(self, cache_arrays):
        self.cache = list(cache_arrays)


# ---------------------------------------------------------------------------
# _writeback_kv_row B>1 — preserves other rows
# ---------------------------------------------------------------------------


def test_writeback_kv_b2_preserves_other_rows():
    """B=2: write back row 0 only. Row 1's tensor slice must be unchanged."""
    Gen = _import_gen()
    # B=2: row 0 value=5.0, row 1 value=15.0
    batch_layer = _FakeKVLayer(B=2, max_seq=5, value_seed=5)
    # Snapshot row 1 BEFORE the writeback
    row_1_before = batch_layer.keys[1:2].tolist()

    # Solo state for row 0 with value=99
    solo = _FakeSoloKV(
        mx.full((1, 2, 5, 4), 99.0),
        mx.full((1, 2, 5, 4), 99.0),
        offset=5,
    )
    Gen._writeback_kv_row(batch_layer, solo, row_idx=0)

    # Row 1 must be byte-identical to before
    row_1_after = batch_layer.keys[1:2].tolist()
    assert row_1_after == row_1_before, (
        "Row 1 was corrupted by row 0 writeback"
    )
    # Row 0 must have been replaced with solo values
    assert batch_layer.keys[0:1].tolist() == solo.keys.tolist()


def test_writeback_kv_b4_only_target_row_changes():
    """B=4: write back row 2 only. Rows 0, 1, 3 unchanged."""
    Gen = _import_gen()
    batch_layer = _FakeKVLayer(B=4, max_seq=6, value_seed=1)
    rows_before = [batch_layer.keys[r:r+1].tolist() for r in range(4)]

    solo = _FakeSoloKV(
        mx.full((1, 2, 6, 4), 88.0),
        mx.full((1, 2, 6, 4), 88.0),
        offset=6,
    )
    Gen._writeback_kv_row(batch_layer, solo, row_idx=2)

    for r in [0, 1, 3]:
        assert batch_layer.keys[r:r+1].tolist() == rows_before[r], (
            f"Row {r} was corrupted by row 2 writeback"
        )
    # Row 2 replaced
    assert batch_layer.keys[2:3, 0, 0, 0].item() == 88.0


# ---------------------------------------------------------------------------
# Per-row offset bookkeeping
# ---------------------------------------------------------------------------


def test_per_row_offset_diverges_after_partial_rollback():
    """After per-row writeback with differing offsets, batch_layer.offset
    must be an mx.array with per-row distinct values."""
    Gen = _import_gen()
    batch_layer = _FakeKVLayer(B=2, max_seq=10, offsets=[10, 10])
    solo = _FakeSoloKV(
        mx.full((1, 2, 7, 4), 99.0),
        mx.full((1, 2, 7, 4), 99.0),
        offset=7,
    )
    Gen._writeback_kv_row(batch_layer, solo, row_idx=0)

    offsets = batch_layer.offset.tolist()
    # Row 0 should be at new_seq=7 (rolled back from 10)
    # Row 1 should still be at 10 (untouched)
    assert offsets == [7, 10], (
        f"Per-row offsets should diverge after partial rollback: {offsets}"
    )


def test_per_row_offset_b4_mixed_no_change_on_other_rows():
    """B=4 mixed accept: write back row 1 to offset 5, others stay at 10."""
    Gen = _import_gen()
    batch_layer = _FakeKVLayer(B=4, max_seq=10, offsets=[10, 10, 10, 10])
    solo = _FakeSoloKV(
        mx.full((1, 2, 5, 4), 50.0),
        mx.full((1, 2, 5, 4), 50.0),
        offset=5,
    )
    Gen._writeback_kv_row(batch_layer, solo, row_idx=1)

    offsets = batch_layer.offset.tolist()
    assert offsets == [10, 5, 10, 10]


# ---------------------------------------------------------------------------
# SSM writeback B>1 — preserves other rows
# ---------------------------------------------------------------------------


def test_writeback_ssm_b4_other_rows_preserved():
    """B=4 SSM writeback to row 2; rows 0/1/3 unchanged."""
    Gen = _import_gen()
    # Each row has distinct SSM value
    batch_layer = _FakeSSMLayer(B=4, value_per_row=[1.0, 2.0, 3.0, 4.0])
    rows_before = [batch_layer.cache[0][r:r+1].tolist() for r in range(4)]

    solo = _FakeSoloSSM([mx.array([[99.0]])])
    Gen._writeback_ssm_row(batch_layer, solo, row_idx=2)

    # Row 2 replaced
    assert batch_layer.cache[0][2:3].tolist() == [[99.0]]
    # Others unchanged
    for r in [0, 1, 3]:
        assert batch_layer.cache[0][r:r+1].tolist() == rows_before[r]


# ---------------------------------------------------------------------------
# Per-row snapshot — captures distinct state per row
# ---------------------------------------------------------------------------


def test_snapshot_per_row_captures_distinct_state_b4():
    """B=4 SSM snapshot must produce 4 distinct row snapshots."""
    Gen = _import_gen()
    # Need extract() method on the layer for snapshot to work
    class _SSMWithExtract(_FakeSSMLayer):
        def extract(self, idx):
            return _FakeSoloSSM(
                [mx.contiguous(a[idx:idx+1]) for a in self.cache]
            )

    layer = _SSMWithExtract(B=4, value_per_row=[1.0, 2.0, 3.0, 4.0])
    snap = Gen._snapshot_ssm_per_row([layer])

    assert 0 in snap
    assert len(snap[0]) == 4
    # Each row snapshot has its distinct value
    for r in range(4):
        assert snap[0][r].cache[0].tolist() == [[float(r + 1)]]


# ---------------------------------------------------------------------------
# Per-row restore — selective rows only
# ---------------------------------------------------------------------------


def test_restore_subset_of_rows_b4():
    """B=4: restore rows 0 and 3 only; rows 1, 2 keep current state."""
    Gen = _import_gen()
    class _SSMWithExtract(_FakeSSMLayer):
        def extract(self, idx):
            return _FakeSoloSSM(
                [mx.contiguous(a[idx:idx+1]) for a in self.cache]
            )

    layer = _SSMWithExtract(B=4, value_per_row=[1.0, 2.0, 3.0, 4.0])
    snap = Gen._snapshot_ssm_per_row([layer])

    # Mutate all rows to simulate post-verify state
    layer.cache = [mx.array([[10.0], [20.0], [30.0], [40.0]])]

    # Restore rows 0 and 3 only
    Gen._restore_ssm_rows([layer], snap, [0, 3])

    result = layer.cache[0].tolist()
    # Row 0 restored to 1.0, row 1 stays at 20.0, row 2 stays at 30.0, row 3 restored to 4.0
    assert result == [[1.0], [20.0], [30.0], [4.0]]
