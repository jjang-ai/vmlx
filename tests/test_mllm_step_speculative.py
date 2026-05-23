# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLMBatchGenerator._step_speculative (issue #134 + #135).

Verifies in-batch PLD speculative decoding logic:
  - Per-row accept/reject via greedy argmax
  - Per-row KV offset rewind on partial/full reject
  - Per-row SSM snapshot/restore on hybrid models
  - Fallback to _step when any request has no drafts
  - Conservative correction-only emission on hybrid partial-accept (per C.3
    simplification; full per-row replay is a future enhancement)

Uses minimal mocks rather than constructing a real MLLMBatchGenerator
(which requires a model + processor). The tests exercise the helper logic
by attaching only the state attributes `_step_speculative` reads.

Run:
    .venv/bin/python -m pytest tests/test_mllm_step_speculative.py -v
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Mock model + cache fakes (minimal, enough for _step_speculative path)
# ---------------------------------------------------------------------------


class _MockLanguageModel:
    """Mock language model returning controllable logits.

    Records calls so tests can assert behaviour. Logits shape (B, T, V) where
    V is a small toy vocab (e.g. 100). The argmax of each position is
    configurable per call so tests can drive accept/reject.

    UPDATED for sequential verify (PR #172): _step_speculative now calls
    the model K+1 times with T=1. The model tracks a global position
    counter `_seq_pos` across calls within one verify cycle, returning
    argmax_plan[i][_seq_pos] for the single position each call.
    Counter resets when input length T > 1 (legacy multi-token tests).
    """

    def __init__(self, argmax_plan):
        # argmax_plan: list of lists — argmax_plan[i][j] is the predicted
        # token id at row i, position j of the most recent verify cycle.
        self.argmax_plan = argmax_plan
        self.calls: List[Tuple[mx.array, object]] = []
        self._seq_pos = 0  # incremented across T=1 calls

    def __call__(self, input_tokens, cache=None):
        self.calls.append((input_tokens, cache))
        B, T = input_tokens.shape
        # Simulate real model behaviour: cache layers' offsets advance by T.
        if cache is not None:
            for layer in cache:
                if hasattr(layer, "is_trimmable") and layer.is_trimmable():
                    if hasattr(layer, "offset"):
                        if isinstance(layer.offset, mx.array):
                            layer.offset = layer.offset + T
                        else:
                            try:
                                layer.offset = int(layer.offset) + T
                            except Exception:
                                pass
        V = 100
        logits_data = []
        # Determine which positions of argmax_plan to use:
        # - T == 1: return position [_seq_pos] from plan; increment counter
        # - T > 1 (legacy): return positions [0..T-1] from plan; no counter use
        if T == 1:
            pos_indices = [self._seq_pos]
            self._seq_pos += 1
        else:
            pos_indices = list(range(T))
            self._seq_pos = T  # reset for any subsequent T=1 calls
        for i in range(B):
            row = []
            for plan_j in pos_indices:
                vec = [0.0] * V
                if (
                    i < len(self.argmax_plan)
                    and plan_j < len(self.argmax_plan[i])
                ):
                    target = self.argmax_plan[i][plan_j]
                    if 0 <= target < V:
                        vec[target] = 100.0
                row.append(vec)
            logits_data.append(row)
        return mx.array(logits_data)


class _FakeKVLayer:
    """Mimics BatchKVCache: trimmable, has offset as mx.array, no SSM state."""

    def __init__(self, offsets):
        self.keys = mx.zeros((len(offsets), 2, max(offsets) if offsets else 1, 4))
        self.values = mx.zeros((len(offsets), 2, max(offsets) if offsets else 1, 4))
        self.offset = mx.array(offsets)

    def is_trimmable(self) -> bool:
        return True


class _FakeSSMLayer:
    """Non-trimmable layer with per-row state in .cache."""

    def __init__(self, state_arr):
        self.cache = [state_arr]

    def is_trimmable(self) -> bool:
        return False

    def extract(self, idx: int):
        return _FakeMambaCache(
            [mx.contiguous(a[idx : idx + 1]) if a is not None else None
             for a in self.cache]
        )


class _FakeMambaCache:
    def __init__(self, cache_arrays):
        self.cache = list(cache_arrays)


class _FakeBatch:
    """Mimics MLLMBatch with .requests, .cache, .y, .logprobs."""

    def __init__(self, requests, cache, y):
        self.requests = requests
        self.cache = cache
        self.y = y
        self.logprobs = [None] * len(requests)


class _FakeReq:
    def __init__(self, last_token, num_tokens=1, output_tokens=None,
                 input_ids=None, max_tokens=128):
        self.last_token = last_token
        self.num_tokens = num_tokens
        self.output_tokens = output_tokens or []
        self.input_ids = input_ids
        self.max_tokens = max_tokens
        self.scratch_extra_tokens = None
        self._pld_ngram_index = None
        self._cached_prompt_token_ids = None


# ---------------------------------------------------------------------------
# Build a partial _step_speculative test environment
# ---------------------------------------------------------------------------


def _make_gen(language_model, pld_enabled=True, is_hybrid=False,
              excluded_token_ids=None):
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.language_model = language_model
    gen._pld_spec_enabled = pld_enabled
    gen._pld_excluded_token_ids = excluded_token_ids
    gen._spec_batched_steps = 0
    gen._spec_batched_tokens = 0
    gen._spec_batched_acceptance_ema = 0.0
    gen._pld_replay_attempts = 0
    gen._pld_replay_emitted = 0
    gen._pld_replay_failures = 0
    gen._pld_replay_enabled = True
    gen._is_hybrid = is_hybrid
    # Minimal _step fallback that just returns a constant
    def _fallback_step(input_tokens, cache):
        B = input_tokens.shape[0] if hasattr(input_tokens, 'shape') else len(input_tokens)
        return mx.zeros((B,), dtype=mx.int32), [None] * B
    gen._step = _fallback_step
    return gen


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


def test_step_speculative_falls_back_when_no_drafts():
    """No prompt → no n-gram match → no drafts → fall back to _step."""
    model = _MockLanguageModel(argmax_plan=[[42, 43]])
    gen = _make_gen(model, pld_enabled=True)

    req = _FakeReq(last_token=5, num_tokens=1, output_tokens=[],
                   input_ids=mx.array([1, 2]))  # too short for n-gram
    cache = [_FakeKVLayer([10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, logprobs = gen._step_speculative(batch, K=2)
    # Fallback path called — model was NOT invoked through speculative path
    assert len(model.calls) == 0


def test_step_speculative_falls_back_in_prefill():
    """req.num_tokens == 0 → in prefill → fall back."""
    model = _MockLanguageModel(argmax_plan=[[42]])
    gen = _make_gen(model, pld_enabled=True)
    req = _FakeReq(last_token=None, num_tokens=0)
    cache = [_FakeKVLayer([10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    gen._step_speculative(batch, K=2)
    assert len(model.calls) == 0  # model not called via spec path


def test_step_speculative_k_zero_falls_back():
    model = _MockLanguageModel(argmax_plan=[[42]])
    gen = _make_gen(model)
    req = _FakeReq(last_token=5, num_tokens=1,
                   input_ids=mx.array([1, 2, 3, 4, 1, 2]))
    batch = _FakeBatch([req], [_FakeKVLayer([10])], y=mx.array([5]))

    gen._step_speculative(batch, K=0)
    assert len(model.calls) == 0


# ---------------------------------------------------------------------------
# Full accept path
# ---------------------------------------------------------------------------


def test_full_accept_pure_attention():
    """K=2, both drafts match argmax → emit primary=bonus + extras=[d0,d1]."""
    # Prompt: [1, 2, 3, 4, 5, 1, 2]. Bigram [1,2] match at idx 0. drafts=[3,4]
    # Model argmax for verify_input [last=5, d0=3, d1=4]:
    #   pos 0 → 3 (matches d0) ✓
    #   pos 1 → 4 (matches d1) ✓
    #   pos 2 → 77 (bonus)
    model = _MockLanguageModel(argmax_plan=[[3, 4, 77]])
    gen = _make_gen(model, pld_enabled=True, is_hybrid=False)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer([10, 10])]  # B=1, offset=10
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # Sequential verify (PR #172): K+1=3 single-token forwards
    assert len(model.calls) == 3
    # Primary is bonus
    assert int(sampled[0].item()) == 77
    # Extras = both accepted drafts
    assert req.scratch_extra_tokens == [3, 4]
    # KV offset advanced by 3 (no rewind) for row 0
    assert cache[0].offset.tolist()[0] == 13


def test_full_reject_pure_attention():
    """K=2, neither matches → emit primary=correction, no extras, rewind by 2."""
    model = _MockLanguageModel(argmax_plan=[[88, 89, 90]])  # actual ≠ drafts
    gen = _make_gen(model, pld_enabled=True, is_hybrid=False)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer([10, 10])]  # offset=10
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # Primary = correction at position 0 = 88
    assert int(sampled[0].item()) == 88
    assert req.scratch_extra_tokens is None
    # Offset advanced 3, rewound 2 (K - 0 = 2) → net +1 (just the seed)
    assert cache[0].offset.tolist()[0] == 11


def test_partial_accept_pure_attention():
    """K=2, d0 matches, d1 doesn't → emit primary=correction_at_pos_1, extras=[d0]."""
    # drafts [3,4]. Plan: pos0→3 ✓, pos1→55 (≠d1=4), pos2→ignored
    model = _MockLanguageModel(argmax_plan=[[3, 55, 99]])
    gen = _make_gen(model, pld_enabled=True, is_hybrid=False)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer([10, 10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # n_accept=1, primary = argmax at pos 1 = 55
    assert int(sampled[0].item()) == 55
    assert req.scratch_extra_tokens == [3]
    # Offset: +3 (forward) - 1 (K - n_accept = 1) = +2 net
    assert cache[0].offset.tolist()[0] == 12


# ---------------------------------------------------------------------------
# Hybrid partial-accept simplification (correction only)
# ---------------------------------------------------------------------------


def test_partial_accept_hybrid_preserves_drafts_via_replay():
    """Hybrid partial-accept now PRESERVES accepted drafts via per-row replay (C.5).

    Per-row replay forward restores both KV and SSM via solo cache extraction,
    forward, and writeback. Accepted drafts are emitted as extras alongside
    the next-step seed (correction at the rejected position).
    """
    # drafts [3,4]. Plan: pos0→3 ✓ (accept d0), pos1→55 (reject d1), pos2→99
    model = _MockLanguageModel(argmax_plan=[[3, 55, 99]])
    ssm_state = mx.array([[1.0, 2.0]])
    ssm_layer = _FakeSSMLayer(ssm_state)
    gen = _make_gen(model, pld_enabled=True, is_hybrid=True)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer([10]), ssm_layer]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # New C.5 behaviour: primary = correction at pos n_accept=1 = 55
    # extras = accepted drafts = [3]
    assert int(sampled[0].item()) == 55
    assert req.scratch_extra_tokens == [3]
    # Replay counters incremented
    assert gen._pld_replay_attempts == 1


def test_full_reject_hybrid_emits_correction_only():
    """Hybrid full-reject: n_accept=0 → emit correction at pos 0, no extras.

    Per-row replay forwards [seed] to advance SSM by 1 from snapshot, KV via
    writeback. Cache ends at N+1.
    """
    # drafts [3,4]. Plan: pos0→99 (reject d0)
    model = _MockLanguageModel(argmax_plan=[[99, 88, 77]])
    ssm_state = mx.array([[1.0, 2.0]])
    ssm_layer = _FakeSSMLayer(ssm_state)
    gen = _make_gen(model, pld_enabled=True, is_hybrid=True)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer([10]), ssm_layer]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # n_accept=0, primary = correction = pos 0 = 99
    assert int(sampled[0].item()) == 99
    assert req.scratch_extra_tokens is None
    # Replay counter incremented (replay ran with [seed])
    assert gen._pld_replay_attempts == 1


def test_full_accept_hybrid_no_rollback():
    """Hybrid full-accept: cache stays advanced, no SSM restore needed."""
    model = _MockLanguageModel(argmax_plan=[[3, 4, 77]])
    ssm_state = mx.array([[1.0, 2.0]])
    ssm_layer = _FakeSSMLayer(ssm_state)
    gen = _make_gen(model, pld_enabled=True, is_hybrid=True)
    req = _FakeReq(
        last_token=5, num_tokens=1, output_tokens=[],
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    cache = [_FakeKVLayer([10]), ssm_layer]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)

    # Full accept: emit bonus + extras = both drafts
    assert int(sampled[0].item()) == 77
    assert req.scratch_extra_tokens == [3, 4]
    # SSM was advanced by the (mock) forward but mock doesn't actually mutate;
    # the snapshot+restore should not touch it on full accept either.


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_spec_steps_counter_increments():
    model = _MockLanguageModel(argmax_plan=[[3, 4, 77]])
    gen = _make_gen(model, pld_enabled=True)
    req = _FakeReq(
        last_token=5, num_tokens=1,
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    batch = _FakeBatch([req], [_FakeKVLayer([10])], y=mx.array([5]))
    assert gen._spec_batched_steps == 0
    gen._step_speculative(batch, K=2)
    assert gen._spec_batched_steps == 1


def test_acceptance_ema_updates():
    """Full-accept run sets EMA toward 1.0; full-reject toward 0.0."""
    model = _MockLanguageModel(argmax_plan=[[3, 4, 77]])  # both accepted
    gen = _make_gen(model, pld_enabled=True)
    req = _FakeReq(
        last_token=5, num_tokens=1,
        input_ids=mx.array([10, 5, 3, 4, 99, 88, 10]),
    )
    batch = _FakeBatch([req], [_FakeKVLayer([10])], y=mx.array([5]))
    gen._step_speculative(batch, K=2)
    # alpha=0.1; before=0; new = 0.9*0 + 0.1*1.0 = 0.1
    assert abs(gen._spec_batched_acceptance_ema - 0.1) < 1e-6
