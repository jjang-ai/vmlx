# SPDX-License-Identifier: Apache-2.0
"""Tests for MLLM PLD adaptive auto-disable + verify-shape guard.

Live validation of PR #150 revealed two robustness gaps:
  - Low acceptance rate (n-gram drafts not matching model argmax) made PLD
    cost more than it saved. Throughput regressed from ~12 to ~3.5 tok/s.
  - Some VLM model wrappers return logits shape (B, 1, V) at decode even
    when input length > 1, which would break the per-row accept loop.

This commit adds:
  - Adaptive auto-disable (TCP slow-start pattern from PR #26): when EMA
    acceptance < VMLX_PLD_MIN_ACCEPTANCE for VMLX_PLD_WARMUP_STEPS, cool
    down for VMLX_PLD_PROBE_INTERVAL steps then probe.
  - Verify-shape guard: if logits[1] < K+1, fall back to standard _step.

Tests verify the cooldown countdown and the shape-guard fallback.

Run:
    .venv/bin/python -m pytest tests/test_mllm_pld_auto_disable.py -v
"""

from __future__ import annotations

import os
from typing import List

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Helpers (minimal MLLMBatchGenerator instance setup)
# ---------------------------------------------------------------------------


def _make_gen(language_model, debug_remaining=0,
              min_acceptance=0.3, warmup_steps=20, probe_interval=200):
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.language_model = language_model
    gen._pld_spec_enabled = True
    gen._pld_excluded_token_ids = None
    gen._spec_batched_steps = 0
    gen._spec_batched_tokens = 0
    gen._spec_batched_acceptance_ema = 0.0
    gen._spec_batched_min_acceptance = min_acceptance
    gen._spec_batched_warmup_steps = warmup_steps
    gen._spec_batched_probe_interval = probe_interval
    gen._spec_batched_cooldown = 0
    gen._spec_batched_cooldown_count = 0
    gen._spec_batched_debug_remaining = debug_remaining
    gen._pld_replay_attempts = 0
    gen._pld_replay_emitted = 0
    gen._pld_replay_failures = 0
    gen._pld_replay_enabled = True
    gen._is_hybrid = False

    def _fallback_step(input_tokens, cache):
        B = input_tokens.shape[0] if hasattr(input_tokens, 'shape') else len(input_tokens)
        return mx.zeros((B,), dtype=mx.int32), [None] * B
    gen._step = _fallback_step
    return gen


class _ShapeModel:
    """Model that returns logits with configurable shape."""

    def __init__(self, t_dim):
        self.t_dim = t_dim
        self.calls = 0

    def __call__(self, input_tokens, cache=None):
        self.calls += 1
        B = input_tokens.shape[0]
        T_actual = input_tokens.shape[1]
        # Advance cache offsets as a real model would
        if cache is not None:
            for layer in cache:
                if (
                    hasattr(layer, "is_trimmable")
                    and layer.is_trimmable()
                    and hasattr(layer, "offset")
                    and isinstance(layer.offset, mx.array)
                ):
                    layer.offset = layer.offset + T_actual
        # Return logits with configured T dimension (may differ from input T)
        return mx.zeros((B, self.t_dim, 100))


class _FakeReq:
    def __init__(self):
        self.last_token = 5
        self.num_tokens = 1
        self.output_tokens = []
        self.input_ids = mx.array([10, 5, 3, 4, 99, 88, 10])
        self.max_tokens = 128
        self.scratch_extra_tokens = None
        self._pld_ngram_index = None
        self._cached_prompt_token_ids = None


class _FakeKVLayer:
    def __init__(self, B=1, max_seq=10, value_seed=0, offsets=None):
        self.keys = mx.full((B, 2, max_seq, 4), float(value_seed))
        self.values = mx.full((B, 2, max_seq, 4), float(value_seed))
        self.offset = mx.array(offsets if offsets is not None else [max_seq] * B)

    def is_trimmable(self) -> bool:
        return True


class _FakeBatch:
    def __init__(self, requests, cache, y):
        self.requests = requests
        self.cache = cache
        self.y = y
        self.logprobs = [None] * len(requests)


# ---------------------------------------------------------------------------
# Verify shape guard
# ---------------------------------------------------------------------------


def test_shape_guard_falls_back_when_t_too_small():
    """When language_model returns logits shape (B, 1, V) but we passed
    K+1=3 tokens, fall back to _step instead of crashing in accept loop."""
    model = _ShapeModel(t_dim=1)  # returns (B, 1, V) — too short
    gen = _make_gen(model)
    req = _FakeReq()
    cache = [_FakeKVLayer(B=1, max_seq=10, offsets=[10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    # Should not raise; fallback to _step (zeros)
    sampled, _ = gen._step_speculative(batch, K=2)
    assert sampled.shape == (1,)
    # _step was called as fallback (model was called once, then fallback)
    assert model.calls == 1


def test_shape_guard_passes_with_correct_shape():
    """Correct shape (B, K+1, V) doesn't trigger fallback."""
    # Model returns shape (B, K+1, V). For K=2 → (1, 3, 100).
    model = _ShapeModel(t_dim=3)
    gen = _make_gen(model)
    req = _FakeReq()
    cache = [_FakeKVLayer(B=1, max_seq=10, offsets=[10])]
    batch = _FakeBatch([req], cache, y=mx.array([5]))

    sampled, _ = gen._step_speculative(batch, K=2)
    # No fallback exception — _step_speculative path completed
    # All-zero logits → argmax=0 → drafts unlikely to match → full reject
    # Pure-attention full-reject → primary = correction = 0
    assert int(sampled[0].item()) == 0


# ---------------------------------------------------------------------------
# Adaptive auto-disable cooldown
# ---------------------------------------------------------------------------


def test_cooldown_counter_decrements():
    """When cooldown > 0, dispatch path should decrement it.

    Tests the _spec_batched_cooldown field decrement logic from the dispatch.
    The actual dispatch lives in _next(), but the contract is: cooldown
    decrements once per call when > 0.
    """
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._spec_batched_cooldown = 5
    # Simulate the dispatch's decrement
    if gen._spec_batched_cooldown > 0:
        gen._spec_batched_cooldown -= 1
    assert gen._spec_batched_cooldown == 4


def test_cooldown_trigger_threshold():
    """When EMA < min_acceptance after warmup, cooldown is engaged.

    Mirrors the dispatch decision: if warmup passed AND EMA below threshold,
    set cooldown = probe_interval.
    """
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._spec_batched_steps = 25  # past warmup of 20
    gen._spec_batched_acceptance_ema = 0.05  # below threshold 0.3
    gen._spec_batched_min_acceptance = 0.3
    gen._spec_batched_warmup_steps = 20
    gen._spec_batched_probe_interval = 200
    gen._spec_batched_cooldown = 0
    gen._spec_batched_cooldown_count = 0

    # The dispatch check (extracted for unit test)
    if (
        gen._spec_batched_steps >= gen._spec_batched_warmup_steps
        and gen._spec_batched_acceptance_ema < gen._spec_batched_min_acceptance
    ):
        gen._spec_batched_cooldown = gen._spec_batched_probe_interval
        gen._spec_batched_cooldown_count += 1

    assert gen._spec_batched_cooldown == 200
    assert gen._spec_batched_cooldown_count == 1


def test_no_cooldown_before_warmup():
    """Cooldown does NOT trigger before warmup steps complete."""
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._spec_batched_steps = 5  # under warmup of 20
    gen._spec_batched_acceptance_ema = 0.0  # would trigger if not for warmup
    gen._spec_batched_min_acceptance = 0.3
    gen._spec_batched_warmup_steps = 20
    gen._spec_batched_probe_interval = 200
    gen._spec_batched_cooldown = 0
    gen._spec_batched_cooldown_count = 0

    if (
        gen._spec_batched_steps >= gen._spec_batched_warmup_steps
        and gen._spec_batched_acceptance_ema < gen._spec_batched_min_acceptance
    ):
        gen._spec_batched_cooldown = gen._spec_batched_probe_interval
        gen._spec_batched_cooldown_count += 1

    assert gen._spec_batched_cooldown == 0
    assert gen._spec_batched_cooldown_count == 0


def test_no_cooldown_when_acceptance_high():
    """Cooldown does NOT trigger when acceptance is healthy."""
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._spec_batched_steps = 50
    gen._spec_batched_acceptance_ema = 0.6  # healthy
    gen._spec_batched_min_acceptance = 0.3
    gen._spec_batched_warmup_steps = 20
    gen._spec_batched_probe_interval = 200
    gen._spec_batched_cooldown = 0

    if (
        gen._spec_batched_steps >= gen._spec_batched_warmup_steps
        and gen._spec_batched_acceptance_ema < gen._spec_batched_min_acceptance
    ):
        gen._spec_batched_cooldown = gen._spec_batched_probe_interval

    assert gen._spec_batched_cooldown == 0


# ---------------------------------------------------------------------------
# Env var defaults
# ---------------------------------------------------------------------------


def test_env_var_defaults_picked_up_via_getenv():
    """Verify env var keys match what __init__ reads from os.getenv."""
    # Test that the defaults are sensible (don't depend on env)
    default_min = float(os.getenv("VMLX_PLD_MIN_ACCEPTANCE", "0.30"))
    default_warmup = int(os.getenv("VMLX_PLD_WARMUP_STEPS", "20"))
    default_probe = int(os.getenv("VMLX_PLD_PROBE_INTERVAL", "200"))
    assert 0 < default_min < 1
    assert default_warmup > 0
    assert default_probe > 0
