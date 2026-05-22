# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMBatchGenerator per-row writeback + replay helpers (C.5).

Per-row replay is the correctness fix for hybrid SSM rollback in
_step_speculative: after partial/full reject, replay [seed, drafts...]
through a solo cache extracted from snapshot + pre-verify KV, then write
the advanced single-row state back into the batch cache. Recovers the
accepted drafts on partial accept (full PLD gain in hybrid path).

These tests exercise _writeback_kv_row, _writeback_ssm_row, and
_per_row_replay_forward in isolation using mock layers + a mock model.

Run:
    .venv/bin/python -m pytest tests/test_mllm_per_row_replay.py -v
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeKVLayer:
    def __init__(self, B: int, max_seq: int, n_heads: int = 2, head_dim: int = 4,
                 offsets=None, value_seed=0):
        # Fill keys/values with deterministic values so writeback can be verified
        shape = (B, n_heads, max_seq, head_dim)
        self.keys = mx.full(shape, float(value_seed))
        self.values = mx.full(shape, float(value_seed + 1))
        self.offset = mx.array(offsets if offsets is not None else [max_seq] * B)

    def is_trimmable(self) -> bool:
        return True


class _FakeSSMLayer:
    def __init__(self, state_arr):
        self.cache = [state_arr]

    def is_trimmable(self) -> bool:
        return False


class _FakeSoloKV:
    """Mimics mlx_lm.models.cache.KVCache for the writeback test."""

    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset


class _FakeSoloSSM:
    def __init__(self, cache_arrays):
        self.cache = list(cache_arrays)


# ---------------------------------------------------------------------------
# _writeback_kv_row
# ---------------------------------------------------------------------------


def _import_gen():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    return MLLMBatchGenerator


def test_writeback_kv_row_simple():
    """Write a row's KVCache state into a batch with same max_seq."""
    Gen = _import_gen()
    # Batch: B=3, max_seq=5, fill=7
    batch_layer = _FakeKVLayer(B=3, max_seq=5, value_seed=7)
    # Solo state: row to write with value 99, offset=5
    solo_keys = mx.full((1, 2, 5, 4), 99.0)
    solo_values = mx.full((1, 2, 5, 4), 99.0)
    solo = _FakeSoloKV(solo_keys, solo_values, offset=5)

    Gen._writeback_kv_row(batch_layer, solo, row_idx=1)

    # Row 1 should be 99.0, rows 0 and 2 should be 7.0
    keys_list = batch_layer.keys.tolist()
    assert keys_list[0][0][0][0] == 7.0
    assert keys_list[1][0][0][0] == 99.0
    assert keys_list[2][0][0][0] == 7.0
    # Offset[1] updated
    assert batch_layer.offset.tolist()[1] == 5


def test_writeback_kv_row_grows_batch():
    """Solo state longer than batch → batch grows max_seq."""
    Gen = _import_gen()
    batch_layer = _FakeKVLayer(B=2, max_seq=3, value_seed=1)
    # Solo state with offset=5 (larger than batch's max_seq=3)
    solo_keys = mx.full((1, 2, 5, 4), 99.0)
    solo_values = mx.full((1, 2, 5, 4), 99.0)
    solo = _FakeSoloKV(solo_keys, solo_values, offset=5)

    Gen._writeback_kv_row(batch_layer, solo, row_idx=0)

    # Batch should have grown to 5
    assert batch_layer.keys.shape[2] == 5
    assert batch_layer.offset.tolist()[0] == 5
    # Row 1 (untouched) should still have original value in positions 0..2
    keys_list = batch_layer.keys.tolist()
    assert keys_list[1][0][0][0] == 1.0


def test_writeback_kv_row_shorter_solo_padded():
    """Solo state shorter than batch → solo padded for concat."""
    Gen = _import_gen()
    batch_layer = _FakeKVLayer(B=2, max_seq=10, value_seed=1)
    # Solo state with seq=3 (shorter)
    solo_keys = mx.full((1, 2, 3, 4), 50.0)
    solo_values = mx.full((1, 2, 3, 4), 50.0)
    solo = _FakeSoloKV(solo_keys, solo_values, offset=3)

    Gen._writeback_kv_row(batch_layer, solo, row_idx=1)

    # Offset[1] = 3 (the solo's offset)
    assert batch_layer.offset.tolist()[1] == 3
    # Row 1 first 3 positions = 50, last 7 = 0 (padded)
    keys_list = batch_layer.keys.tolist()
    assert keys_list[1][0][0][0] == 50.0
    assert keys_list[1][0][2][0] == 50.0
    assert keys_list[1][0][3][0] == 0.0  # padded
    # Row 0 untouched
    assert keys_list[0][0][0][0] == 1.0


# ---------------------------------------------------------------------------
# _writeback_ssm_row
# ---------------------------------------------------------------------------


def test_writeback_ssm_row_replaces_target_row_only():
    """SSM writeback puts solo state into target row; others untouched."""
    Gen = _import_gen()
    ssm_state = mx.array([[1.0], [2.0], [3.0]])  # 3 rows
    batch_layer = _FakeSSMLayer(ssm_state)

    solo = _FakeSoloSSM([mx.array([[99.0]])])

    Gen._writeback_ssm_row(batch_layer, solo, row_idx=1)

    assert batch_layer.cache[0].tolist() == [[1.0], [99.0], [3.0]]


def test_writeback_ssm_row_no_op_on_empty_solo():
    Gen = _import_gen()
    ssm_state = mx.array([[1.0], [2.0]])
    batch_layer = _FakeSSMLayer(ssm_state)

    Gen._writeback_ssm_row(batch_layer, None, row_idx=0)
    assert batch_layer.cache[0].tolist() == [[1.0], [2.0]]


# ---------------------------------------------------------------------------
# _per_row_replay_forward
# ---------------------------------------------------------------------------


class _MockReplayModel:
    """Captures replay forward calls + advances solo KV by T positions."""

    def __init__(self):
        self.calls: List = []

    def __call__(self, input_tokens, cache=None):
        self.calls.append((input_tokens, cache))
        B, T = input_tokens.shape
        # Simulate model advancing KV layers' offset by T
        if cache is not None:
            for layer in cache:
                if layer is None:
                    continue
                if hasattr(layer, "is_trimmable") and layer.is_trimmable():
                    if hasattr(layer, "offset"):
                        if isinstance(layer.offset, mx.array):
                            layer.offset = layer.offset + T
                        else:
                            layer.offset = int(layer.offset) + T
                    # Also grow keys/values to reflect new length
                    if hasattr(layer, "keys") and layer.keys is not None:
                        cur = layer.keys.shape[2]
                        new_pad = mx.zeros(
                            (layer.keys.shape[0], layer.keys.shape[1], T,
                             layer.keys.shape[3]),
                            dtype=layer.keys.dtype,
                        )
                        layer.keys = mx.concatenate([layer.keys, new_pad], axis=2)
                        layer.values = mx.concatenate([layer.values, new_pad], axis=2)
        # Return logits (B, T, V) — content doesn't matter for replay tests
        return mx.zeros((B, T, 100))


def test_replay_forward_advances_solo_and_writes_back():
    """End-to-end: snapshot row, replay [seed, d0], writeback to batch."""
    Gen = _import_gen()

    # Build a fake generator with the helpers
    gen = Gen.__new__(Gen)
    gen.language_model = _MockReplayModel()

    # Batch state: B=2, max_seq=10, post-verify offsets [13, 13] (advanced by K+1=3)
    kv = _FakeKVLayer(B=2, max_seq=13, value_seed=5, offsets=[13, 13])
    ssm = _FakeSSMLayer(mx.array([[10.0], [20.0]]))  # post-verify state
    batch_cache = [kv, ssm]

    # Pre-verify SSM snapshot (state at N=10, before verify)
    snapshot = {1: [_FakeSoloSSM([mx.array([[1.0]])]),
                    _FakeSoloSSM([mx.array([[2.0]])])]}
    # Pre-verify KV offset (current 13 - advance 3 = 10)
    pre_verify_offsets = {0: 10}

    # Replay row 0 with [seed=5, d0=3] (n_accept=1, partial accept)
    success = gen._per_row_replay_forward(
        batch_cache=batch_cache,
        snapshot=snapshot,
        row_idx=0,
        replay_tokens=[5, 3],
        pre_verify_offsets=pre_verify_offsets,
    )

    assert success is True
    # Model was called once with replay_tokens shape (1, 2)
    assert len(gen.language_model.calls) == 1
    inp, _ = gen.language_model.calls[0]
    assert inp.shape == (1, 2)
    # KV row 0 offset = pre_verify + T = 10 + 2 = 12
    assert batch_cache[0].offset.tolist()[0] == 12
    # KV row 1 (untouched by replay) still at 13
    assert batch_cache[0].offset.tolist()[1] == 13
    # SSM row 0 written back from solo (mock model didn't modify SSM state,
    # so solo's value = snapshot value = 1.0); row 1 untouched at 20.0
    ssm_after = batch_cache[1].cache[0].tolist()
    assert ssm_after[0][0] == 1.0  # row 0 from snapshot (writeback)
    assert ssm_after[1][0] == 20.0  # row 1 unchanged


def test_replay_forward_returns_false_on_model_exception():
    Gen = _import_gen()

    class _RaisingModel:
        def __call__(self, inp, cache=None):
            raise RuntimeError("simulated OOM")

    gen = Gen.__new__(Gen)
    gen.language_model = _RaisingModel()

    kv = _FakeKVLayer(B=1, max_seq=10, value_seed=1, offsets=[10])
    ssm = _FakeSSMLayer(mx.array([[5.0]]))
    batch_cache = [kv, ssm]
    snapshot = {1: [_FakeSoloSSM([mx.array([[1.0]])])]}
    pre_verify_offsets = {0: 8}

    success = gen._per_row_replay_forward(
        batch_cache=batch_cache,
        snapshot=snapshot,
        row_idx=0,
        replay_tokens=[5, 3],
        pre_verify_offsets=pre_verify_offsets,
    )

    assert success is False


def test_replay_forward_empty_tokens_returns_false():
    Gen = _import_gen()
    gen = Gen.__new__(Gen)
    gen.language_model = _MockReplayModel()
    success = gen._per_row_replay_forward(
        batch_cache=[], snapshot={}, row_idx=0, replay_tokens=[],
        pre_verify_offsets={},
    )
    assert success is False
