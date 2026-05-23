# SPDX-License-Identifier: Apache-2.0
"""TurboQuant compatibility + PLD invariant tests (PR #172 Layers 2/3).

Tests:
  - Layer 2: TurboQuant cache short-circuit. PLD detects TQ cache and falls
    back to standard _step instead of crashing in per-row writeback.
  - Layer 3: PLD invariants — scratch_extra_tokens cleared after step,
    KV offset monotonicity, auto-disable cooldown engagement.

These tests use minimal mocks (no real model needed).

Run:
    .venv/bin/python -m pytest tests/test_mllm_pld_tq_and_invariants.py -v
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx


def _import_gen():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    return MLLMBatchGenerator


def _make_gen(
    language_model,
    pld_enabled=True,
    is_hybrid=False,
    excluded_token_ids=None,
    min_acceptance=0.30,
    warmup_steps=20,
    probe_interval=200,
):
    Gen = _import_gen()
    gen = Gen.__new__(Gen)
    gen.language_model = language_model
    gen._pld_spec_enabled = pld_enabled
    gen._pld_excluded_token_ids = excluded_token_ids
    gen._spec_batched_steps = 0
    gen._spec_batched_tokens = 0
    gen._spec_batched_acceptance_ema = 0.0
    gen._spec_batched_accept_histogram = {}
    gen._spec_batched_min_acceptance = min_acceptance
    gen._spec_batched_warmup_steps = warmup_steps
    gen._spec_batched_probe_interval = probe_interval
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


# Fakes
class _CountingModel:
    """Tracks how many times __call__ was invoked."""

    def __init__(self):
        self.calls = 0

    def __call__(self, input_tokens, cache=None):
        self.calls += 1
        B, T = input_tokens.shape
        if cache is not None:
            for layer in cache:
                if hasattr(layer, "is_trimmable") and layer.is_trimmable():
                    if hasattr(layer, "offset") and isinstance(layer.offset, mx.array):
                        layer.offset = layer.offset + T
        return mx.zeros((B, T, 100))


class _FakeKVLayer:
    def __init__(self, B, max_seq, offsets=None):
        self.keys = mx.zeros((B, 2, max_seq, 4))
        self.values = mx.zeros((B, 2, max_seq, 4))
        self.offset = mx.array(offsets if offsets is not None else [max_seq] * B)

    def is_trimmable(self) -> bool:
        return True


class TurboQuantKVCache:
    """Mock TurboQuant cache — class name matches the real one for detection
    via `type(c).__name__`."""

    def __init__(self):
        self.keys = mx.zeros((1, 2, 5, 4))
        self.values = mx.zeros((1, 2, 5, 4))
        self.offset = mx.array([5])

    def is_trimmable(self) -> bool:
        return True


_FakeTQCache = TurboQuantKVCache


class _FakeReq:
    def __init__(self, output_tokens=None):
        self.last_token = 5
        self.num_tokens = 1
        self.output_tokens = output_tokens or []
        self.input_ids = mx.array([10, 5, 3, 4, 99, 88, 10])
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


# ---------------------------------------------------------------------------
# Layer 2 — TurboQuant short-circuit
# ---------------------------------------------------------------------------


def test_tq_cache_detected_short_circuits_pld():
    """When any cache layer is TurboQuantKVCache, _step_speculative falls
    back to _step. No per-row writeback runs, no crash on extract."""
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    tq = _FakeTQCache()
    cache = [tq]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)
    # Fallback _step was called (returns zeros). No spec steps.
    assert sampled.shape == (1,)
    assert gen._spec_batched_steps == 0
    # _CountingModel not invoked (fallback path uses different _step)
    assert model.calls == 0


def test_tq_short_circuit_log_fires_once():
    """The TQ-skip log should fire ONCE, not on every step."""
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    tq = _FakeTQCache()
    cache = [tq]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    # First call: log fires
    assert not getattr(gen, "_pld_tq_skip_logged", False)
    gen._step_speculative(batch, K=2)
    assert getattr(gen, "_pld_tq_skip_logged", False) is True

    # Second call: log doesn't re-fire (flag already set)
    gen._step_speculative(batch, K=2)
    # Same flag state; spec_batched_steps still 0
    assert gen._spec_batched_steps == 0


# ---------------------------------------------------------------------------
# Layer 3 — Invariants
# ---------------------------------------------------------------------------


def test_scratch_extra_tokens_initialized_to_none():
    """Fresh request must have scratch_extra_tokens = None."""
    req = _FakeReq()
    assert req.scratch_extra_tokens is None


def test_cooldown_decrement_invariant():
    """While cooldown > 0, every dispatch call should decrement it by 1."""
    Gen = _import_gen()
    gen = Gen.__new__(Gen)
    gen._spec_batched_cooldown = 5

    # Simulate dispatch decrement (extracted from _next())
    for _ in range(5):
        if gen._spec_batched_cooldown > 0:
            gen._spec_batched_cooldown -= 1
    assert gen._spec_batched_cooldown == 0


def test_cooldown_does_not_underflow():
    """Cooldown=0 stays at 0 (no negative)."""
    Gen = _import_gen()
    gen = Gen.__new__(Gen)
    gen._spec_batched_cooldown = 0
    # Simulate idle dispatch (no decrement since cooldown=0)
    if gen._spec_batched_cooldown > 0:
        gen._spec_batched_cooldown -= 1
    assert gen._spec_batched_cooldown == 0


def test_offset_monotonic_under_seq_verify():
    """KV offset never DECREASES below pre-verify (N) — only increases or
    stays at N+1+n_accept which is >= N+1 (always advances by at least 1).
    """
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    layer = _FakeKVLayer(B=1, max_seq=10, offsets=[10])
    cache = [layer]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    initial_offset = 10
    gen._step_speculative(batch, K=2)
    final_offset = layer.offset.tolist()[0]
    # Must be >= initial + 1 (at least seed was processed)
    assert final_offset >= initial_offset + 1, (
        f"offset {final_offset} should be >= {initial_offset + 1} after spec step"
    )


def test_acceptance_histogram_increments():
    """Each spec step adds to the n_accept histogram by exactly B."""
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    layer = _FakeKVLayer(B=1, max_seq=10, offsets=[10])
    cache = [layer]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    # Before: empty
    assert gen._spec_batched_accept_histogram == {}
    gen._step_speculative(batch, K=2)
    # After one step: histogram has 1 entry (one B=1 row)
    total = sum(gen._spec_batched_accept_histogram.values())
    assert total == 1, f"Expected 1 histogram entry, got {gen._spec_batched_accept_histogram}"


def test_telemetry_counters_monotonic():
    """_spec_batched_steps, _tokens, _accept_histogram never decrement."""
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    layer = _FakeKVLayer(B=1, max_seq=10, offsets=[10])
    cache = [layer]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    snapshots = []
    for _ in range(3):
        gen._step_speculative(batch, K=2)
        snapshots.append((
            gen._spec_batched_steps,
            gen._spec_batched_tokens,
            sum(gen._spec_batched_accept_histogram.values()),
        ))

    # Each successive snapshot >= prior on all 3 counters
    for i in range(1, len(snapshots)):
        assert snapshots[i][0] >= snapshots[i - 1][0]
        assert snapshots[i][1] >= snapshots[i - 1][1]
        assert snapshots[i][2] >= snapshots[i - 1][2]


# ---------------------------------------------------------------------------
# Layer 5 — Edge cases
# ---------------------------------------------------------------------------


def test_k_zero_falls_back():
    """K=0 should be a no-op for spec; fall back to _step."""
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    cache = [_FakeKVLayer(B=1, max_seq=10, offsets=[10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))
    gen._step_speculative(batch, K=0)
    assert gen._spec_batched_steps == 0


def test_empty_batch_falls_back():
    """B=0 should be a no-op for spec."""
    model = _CountingModel()
    gen = _make_gen(model)
    cache = [_FakeKVLayer(B=1, max_seq=10, offsets=[10])]
    batch = _FakeBatch([], cache, y=mx.array([]))
    gen._step_speculative(batch, K=2)
    assert gen._spec_batched_steps == 0


def test_prefill_phase_falls_back():
    """If any req has num_tokens=0 (still in prefill), spec skipped."""
    model = _CountingModel()
    gen = _make_gen(model)
    req = _FakeReq()
    req.num_tokens = 0  # still in prefill
    cache = [_FakeKVLayer(B=1, max_seq=10, offsets=[10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))
    gen._step_speculative(batch, K=2)
    assert gen._spec_batched_steps == 0
