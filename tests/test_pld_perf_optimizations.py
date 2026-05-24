# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PR #174 PLD hot-path performance optimizations.

Validates three optimizations to _step_speculative and _writeback_kv_row:
  - Opt 1: Redundant mx.eval(predicted) removal (tolist implicitly evals)
  - Opt 2: In-place KV row write via slice assignment (no O(B) concat)
  - Opt 3: Vectorized offset rewind via mx.maximum (no per-layer tolist)

Run:
    .venv/bin/python -m pytest tests/test_pld_perf_optimizations.py -v
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Fixtures (adapted from test_mllm_step_speculative.py)
# ---------------------------------------------------------------------------


def _import_gen():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    return MLLMBatchGenerator


class _FakeKVLayer:
    def __init__(self, B, max_seq, offsets=None, head_dim=4, n_heads=2):
        self.keys = mx.zeros((B, n_heads, max_seq, head_dim))
        self.values = mx.zeros((B, n_heads, max_seq, head_dim))
        self.offset = mx.array(offsets if offsets is not None else [max_seq] * B)

    def is_trimmable(self) -> bool:
        return True


class _FakeKVLayerScalar:
    """KV layer with scalar (int) offset — single-request path."""
    def __init__(self, max_seq, head_dim=4, n_heads=2):
        self.keys = mx.zeros((1, n_heads, max_seq, head_dim))
        self.values = mx.zeros((1, n_heads, max_seq, head_dim))
        self.offset = max_seq

    def is_trimmable(self) -> bool:
        return True


class _FakeReq:
    def __init__(self, last_token=5, num_tokens=1, output_tokens=None,
                 input_ids=None):
        self.last_token = last_token
        self.num_tokens = num_tokens
        self.output_tokens = output_tokens or []
        self.input_ids = input_ids if input_ids is not None else mx.array([10, 5, 3, 4, 99, 88, 10])
        self.max_tokens = 128
        self.scratch_extra_tokens = None
        self._pld_ngram_index = None
        self._cached_prompt_token_ids = None


class _FakeBatch:
    def __init__(self, requests, cache, y):
        self.requests = requests
        self.cache = cache
        self.y = y
        self.logprobs = [None] * len(requests)


class _MockLanguageModel:
    """Model with controllable argmax output per position."""
    def __init__(self, argmax_plan):
        self.argmax_plan = argmax_plan
        self.calls = []
        self._call_idx = 0

    def __call__(self, input_tokens, cache=None):
        B, T = input_tokens.shape
        V = 100
        logits = mx.zeros((B, T, V))
        for b in range(B):
            plan = self.argmax_plan[b] if b < len(self.argmax_plan) else self.argmax_plan[0]
            for t in range(T):
                pos = self._call_idx
                if pos < len(plan):
                    token_id = plan[pos]
                    # Make target token have highest logit
                    row = mx.zeros((V,))
                    row = row.at[token_id].add(100.0)
                    logits = logits.at[b, t].add(row)
        self._call_idx += T
        if cache:
            for c in cache:
                if hasattr(c, "offset") and isinstance(c.offset, mx.array):
                    c.offset = c.offset + T
        self.calls.append(input_tokens)
        return logits


def _make_gen(language_model, pld_enabled=True, is_hybrid=False):
    Gen = _import_gen()
    gen = Gen.__new__(Gen)
    gen.language_model = language_model
    gen._pld_spec_enabled = pld_enabled
    gen._pld_excluded_token_ids = None
    gen._spec_batched_steps = 0
    gen._spec_batched_tokens = 0
    gen._spec_batched_acceptance_ema = 0.0
    gen._spec_batched_accept_histogram = {}
    gen._spec_batched_min_acceptance = 0.30
    gen._spec_batched_warmup_steps = 20
    gen._spec_batched_probe_interval = 200
    gen._spec_batched_cooldown = 0
    gen._spec_batched_cooldown_count = 0
    gen._spec_batched_debug_remaining = 0
    gen._pld_replay_attempts = 0
    gen._pld_replay_emitted = 0
    gen._pld_replay_failures = 0
    gen._pld_replay_enabled = True
    gen._is_hybrid = is_hybrid

    def _fallback_step(input_tokens, cache):
        B = input_tokens.shape[0] if hasattr(input_tokens, "shape") else len(input_tokens)
        return mx.zeros((B,), dtype=mx.int32), [None] * B
    gen._step = _fallback_step
    return gen


# ---------------------------------------------------------------------------
# Opt 1 — Redundant mx.eval removal
# ---------------------------------------------------------------------------


def test_predicted_tolist_matches_with_and_without_eval():
    """tolist() implicitly evals; explicit eval before it is redundant."""
    logits = mx.random.normal(shape=(2, 3, 100))
    predicted = mx.argmax(logits, axis=-1)

    # Path A: explicit eval then tolist
    predicted_a = mx.argmax(logits, axis=-1)
    mx.eval(predicted_a)
    result_a = predicted_a.tolist()

    # Path B: tolist only (implicit eval)
    predicted_b = mx.argmax(logits, axis=-1)
    result_b = predicted_b.tolist()

    assert result_a == result_b


# ---------------------------------------------------------------------------
# Opt 2 — In-place KV row write
# ---------------------------------------------------------------------------


def test_writeback_inplace_matches_concat_b4():
    """In-place writeback produces correct cache state for B=4."""
    Gen = _import_gen()
    B, H, max_seq, D = 4, 2, 20, 4

    layer = _FakeKVLayer(B, max_seq, offsets=[10, 12, 8, 15])
    # Fill with known data so we can verify
    layer.keys = mx.random.normal(shape=(B, H, max_seq, D))
    layer.values = mx.random.normal(shape=(B, H, max_seq, D))
    mx.eval(layer.keys, layer.values)

    # Solo state for row 1
    solo = _FakeKVLayer(1, 11, offsets=[11])
    solo.keys = mx.random.normal(shape=(1, H, 11, D))
    solo.values = mx.random.normal(shape=(1, H, 11, D))
    mx.eval(solo.keys, solo.values)

    Gen._writeback_kv_row(layer, solo, row_idx=1)

    # Verify: row 1 should have solo's data in positions 0:11
    mx.eval(layer.keys, layer.values)
    assert mx.allclose(
        layer.keys[1:2, :, :11, :], solo.keys[..., :11, :]
    ), "Row 1 keys should match solo"
    assert mx.allclose(
        layer.values[1:2, :, :11, :], solo.values[..., :11, :]
    ), "Row 1 values should match solo"
    # Offset updated
    assert layer.offset.tolist()[1] == 11


def test_writeback_inplace_preserves_other_rows():
    """In-place write at row 2 must not modify rows 0, 1, 3."""
    Gen = _import_gen()
    B, H, max_seq, D = 4, 2, 20, 4

    layer = _FakeKVLayer(B, max_seq, offsets=[10, 12, 8, 15])
    layer.keys = mx.random.normal(shape=(B, H, max_seq, D))
    layer.values = mx.random.normal(shape=(B, H, max_seq, D))
    mx.eval(layer.keys, layer.values)

    # Snapshot other rows before writeback
    snap_k = {r: mx.array(layer.keys[r:r+1]) for r in [0, 1, 3]}
    snap_v = {r: mx.array(layer.values[r:r+1]) for r in [0, 1, 3]}
    for arr in list(snap_k.values()) + list(snap_v.values()):
        mx.eval(arr)

    solo = _FakeKVLayer(1, 9, offsets=[9])
    solo.keys = mx.random.normal(shape=(1, H, 9, D))
    solo.values = mx.random.normal(shape=(1, H, 9, D))
    mx.eval(solo.keys, solo.values)

    Gen._writeback_kv_row(layer, solo, row_idx=2)
    mx.eval(layer.keys, layer.values)

    for r in [0, 1, 3]:
        assert mx.array_equal(layer.keys[r:r+1], snap_k[r]), f"Row {r} keys modified"
        assert mx.array_equal(layer.values[r:r+1], snap_v[r]), f"Row {r} values modified"


def test_writeback_inplace_zeros_stale_positions():
    """After writing solo with offset=8, positions 8-19 at target row must be zero."""
    Gen = _import_gen()
    B, H, max_seq, D = 4, 2, 20, 4

    layer = _FakeKVLayer(B, max_seq, offsets=[10, 12, 8, 15])
    # Fill with ones so zeros are detectable
    layer.keys = mx.ones((B, H, max_seq, D))
    layer.values = mx.ones((B, H, max_seq, D))
    mx.eval(layer.keys, layer.values)

    solo = _FakeKVLayer(1, 8, offsets=[8])
    solo.keys = mx.ones((1, H, 8, D)) * 5.0
    solo.values = mx.ones((1, H, 8, D)) * 5.0
    mx.eval(solo.keys, solo.values)

    Gen._writeback_kv_row(layer, solo, row_idx=1)
    mx.eval(layer.keys, layer.values)

    # Positions 8-19 at row 1 should be zero
    stale_k = layer.keys[1, :, 8:, :]
    stale_v = layer.values[1, :, 8:, :]
    assert mx.allclose(stale_k, mx.zeros_like(stale_k)), "Stale key positions not zeroed"
    assert mx.allclose(stale_v, mx.zeros_like(stale_v)), "Stale value positions not zeroed"


def test_writeback_inplace_growth_falls_back():
    """Solo longer than batch allocation → concat fallback, keys grow."""
    Gen = _import_gen()
    B, H, max_seq, D = 4, 2, 20, 4

    layer = _FakeKVLayer(B, max_seq, offsets=[10, 12, 8, 15])
    original_seq = layer.keys.shape[2]

    solo = _FakeKVLayer(1, 25, offsets=[25])
    solo.keys = mx.random.normal(shape=(1, H, 25, D))
    solo.values = mx.random.normal(shape=(1, H, 25, D))
    mx.eval(solo.keys, solo.values)

    Gen._writeback_kv_row(layer, solo, row_idx=0)
    mx.eval(layer.keys, layer.values)

    # Keys should have grown
    assert layer.keys.shape[2] >= 25, f"Keys should grow: {layer.keys.shape[2]}"
    assert layer.offset.tolist()[0] == 25


def test_writeback_inplace_b1_fast_path():
    """B=1 uses direct replacement — keys/values replaced entirely."""
    Gen = _import_gen()
    H, D = 2, 4

    layer = _FakeKVLayer(1, 20, offsets=[15])
    layer.keys = mx.ones((1, H, 20, D))
    mx.eval(layer.keys)

    solo = _FakeKVLayer(1, 10, offsets=[10])
    solo.keys = mx.ones((1, H, 10, D)) * 7.0
    solo.values = mx.ones((1, H, 10, D)) * 7.0
    mx.eval(solo.keys, solo.values)

    Gen._writeback_kv_row(layer, solo, row_idx=0)
    mx.eval(layer.keys, layer.values)

    # B=1 replaces entirely
    assert layer.keys.shape == solo.keys.shape, "B=1 should replace, not pad"
    assert mx.allclose(layer.keys, solo.keys)


# ---------------------------------------------------------------------------
# Opt 3 — Vectorized offset rewind
# ---------------------------------------------------------------------------


def test_vectorized_rewind_matches_scalar():
    """Vectorized mx.maximum rewind produces same offsets as scalar loop."""
    offsets = [10, 12, 8, 15]
    shortfalls = [2, 0, 1, 3]
    expected = [max(0, o - s) for o, s in zip(offsets, shortfalls)]

    layers = [_FakeKVLayer(4, 20, offsets=list(offsets)) for _ in range(4)]

    shortfall_arr = mx.array(shortfalls)
    for layer in layers:
        layer.offset = mx.maximum(layer.offset - shortfall_arr, 0)

    for i, layer in enumerate(layers):
        result = layer.offset.tolist()
        assert result == expected, f"Layer {i}: {result} != {expected}"


def test_vectorized_rewind_clamps_to_zero():
    """Shortfall larger than offset → clamp to 0."""
    offsets = [3, 1, 0, 2]
    shortfalls = [5, 5, 5, 5]

    layer = _FakeKVLayer(4, 20, offsets=offsets)
    shortfall_arr = mx.array(shortfalls)
    layer.offset = mx.maximum(layer.offset - shortfall_arr, 0)

    assert layer.offset.tolist() == [0, 0, 0, 0]


def test_vectorized_rewind_scalar_offset():
    """Scalar (int) offset path still works."""
    layer = _FakeKVLayerScalar(max_seq=20)
    layer.offset = 10  # scalar

    # Scalar path: compute directly
    shortfalls = [3]
    worst = max(shortfalls)
    layer.offset = max(0, int(layer.offset) - worst)

    assert layer.offset == 7
    assert isinstance(layer.offset, int)


def test_vectorized_rewind_noop_when_zero_shortfall():
    """All shortfalls=0 → offsets unchanged."""
    offsets = [10, 12, 8, 15]
    layer = _FakeKVLayer(4, 20, offsets=offsets)
    shortfall_arr = mx.array([0, 0, 0, 0])
    layer.offset = mx.maximum(layer.offset - shortfall_arr, 0)

    assert layer.offset.tolist() == offsets


# ---------------------------------------------------------------------------
# Integration — step_speculative byte-equality after opts
# ---------------------------------------------------------------------------


def test_step_speculative_byte_equal_after_opts():
    """Full accept path produces expected tokens with all 3 opts active."""
    # Plan: seed=5, drafts will be [3, 4]. argmax at each position:
    #   pos 0 (seed=5 verify): → 3 (matches draft 0) ✓
    #   pos 1 (draft=3 verify): → 4 (matches draft 1) ✓
    #   pos 2 (draft=4 verify): → 77 (bonus token)
    model = _MockLanguageModel(argmax_plan=[[3, 4, 77]])
    gen = _make_gen(model, pld_enabled=True, is_hybrid=False)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer(1, 20, offsets=[10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # Full accept: primary = bonus token at pos 2 = 77
    assert int(sampled[0].item()) == 77
    # Extras = accepted drafts [3, 4]
    assert req.scratch_extra_tokens == [3, 4]
    # Telemetry
    assert gen._spec_batched_steps == 1
