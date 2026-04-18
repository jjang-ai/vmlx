# SPDX-License-Identifier: Apache-2.0
"""Tests for `vmlx_engine.utils.ssm_companion_cache.SSMCompanionCache`.

Owned by Agent 3 (SSM / Hybrid / SSM companion cache) per the 2026-04-07 audit.
This file replaces the assertion semantics of the legacy
`test_mllm_scheduler_cache.py::TestHybridSSMStateCache::test_store_and_fetch`
test (ISSUE-A3-001), which incorrectly expected `fetch()` to return the SAME
object that was stored. The new module deep-copies on fetch per session
2026-03-28b root-cause fix; these tests verify the deep-copy independence
property AND the new `is_complete: bool` flag from REQ-A3-001.

Run:
    .venv/bin/python -m pytest tests/test_ssm_companion_cache.py -v
"""

from __future__ import annotations

import pytest
import mlx.core as mx

from vmlx_engine.utils.ssm_companion_cache import (
    HybridSSMStateCache,  # back-compat alias
    SSMCompanionCache,
    SSMCompanionEntry,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _FakeSSMLayer:
    """Minimal stand-in for an SSM cache layer with a `.cache` list of arrays.

    Real layers (BatchMambaCache, MambaCache, ArraysCache) all expose `.cache`
    as a list of mx.array tensors. The tests only need that contract.
    """

    def __init__(self, marker: float, n_arrays: int = 1, shape=(4,)):
        self.cache = [mx.array([marker] * shape[0]) for _ in range(n_arrays)]
        self.lengths = None  # populated by tests when relevant


# ----------------------------------------------------------------------
# Construction + invariants
# ----------------------------------------------------------------------


def test_construction_default():
    cache = SSMCompanionCache()
    assert cache.size == 0
    # Default lowered 50 → 20 to keep worst-case ~4 GB instead of ~10 GB.
    assert cache.max_entries == 20


def test_construction_custom_max_entries():
    cache = SSMCompanionCache(max_entries=7)
    assert cache.max_entries == 7
    assert cache.size == 0


def test_construction_invalid_max_entries():
    with pytest.raises(ValueError):
        SSMCompanionCache(max_entries=0)
    with pytest.raises(ValueError):
        SSMCompanionCache(max_entries=-3)


def test_legacy_alias_resolves_to_new_class():
    """REQ-A3-001 / Option C: HybridSSMStateCache must alias SSMCompanionCache
    so existing call sites in scheduler.py / mllm_batch_generator.py keep
    importing the same name during the migration window."""
    assert HybridSSMStateCache is SSMCompanionCache
    instance = HybridSSMStateCache(max_entries=5)
    assert isinstance(instance, SSMCompanionCache)


# ----------------------------------------------------------------------
# Store + fetch contract
# ----------------------------------------------------------------------


def test_store_and_fetch_returns_tuple_with_default_is_complete_true():
    """REQ-A3-001 storage shape: fetch returns (states, is_complete) tuple."""
    cache = SSMCompanionCache(max_entries=10)
    states = [_FakeSSMLayer(1.0), _FakeSSMLayer(2.0)]
    cache.store([1, 2, 3, 4, 5], 5, states)
    assert cache.size == 1

    entry: SSMCompanionEntry = cache.fetch([1, 2, 3, 4, 5], 5)
    assert entry is not None
    fetched_states, is_complete = entry
    assert is_complete is True
    assert len(fetched_states) == 2


def test_store_with_is_complete_false():
    cache = SSMCompanionCache(max_entries=10)
    cache.store([7, 8, 9], 3, [_FakeSSMLayer(5.0)], is_complete=False)
    entry = cache.fetch([7, 8, 9], 3)
    assert entry is not None
    states, is_complete = entry
    assert is_complete is False


def test_fetch_miss_returns_none():
    cache = SSMCompanionCache(max_entries=10)
    assert cache.fetch([42], 1) is None


def test_fetch_after_clear_returns_none():
    cache = SSMCompanionCache(max_entries=10)
    cache.store([1, 2, 3], 3, [_FakeSSMLayer(1.0)])
    assert cache.fetch([1, 2, 3], 3) is not None
    cache.clear()
    assert cache.size == 0
    assert cache.fetch([1, 2, 3], 3) is None


# ----------------------------------------------------------------------
# Deep-copy independence — the session 2026-03-28b root-cause invariant
# ----------------------------------------------------------------------


def test_deep_copy_independence_replaces_layer_cache_array():
    """Mutating a fetched layer's `.cache[i]` must NOT corrupt the stored entry.

    This is the assertion that ISSUE-A3-001's broken test
    (`assert result is ssm_states`) should have been from the start. Replaces
    the identity check with a real independence check.
    """
    cache = SSMCompanionCache(max_entries=10)
    original = [_FakeSSMLayer(1.0)]
    cache.store([1, 2, 3], 3, original)

    # First fetch and corrupt the returned tensor in-place
    entry1 = cache.fetch([1, 2, 3], 3)
    states1, _ = entry1
    states1[0].cache[0] = mx.array([999.0, 999.0, 999.0, 999.0])

    # Second fetch must still see the ORIGINAL value, not the 999s
    entry2 = cache.fetch([1, 2, 3], 3)
    states2, _ = entry2
    val = states2[0].cache[0].tolist()
    assert val == [1.0, 1.0, 1.0, 1.0], (
        f"deep-copy independence violated: expected [1,1,1,1] got {val}"
    )


def test_deep_copy_returns_different_object_than_stored():
    """Sanity: the returned states list must NOT be the same object as the
    stored list (which is what the legacy test wrongly required to be `is`)."""
    cache = SSMCompanionCache(max_entries=5)
    original = [_FakeSSMLayer(1.0)]
    cache.store([1], 1, original)
    entry = cache.fetch([1], 1)
    states, _ = entry
    assert states is not original
    assert states[0] is not original[0]


def test_deep_copy_handles_multi_layer():
    cache = SSMCompanionCache(max_entries=5)
    original = [_FakeSSMLayer(float(i)) for i in range(8)]
    cache.store([1, 2], 2, original)
    entry = cache.fetch([1, 2], 2)
    states, _ = entry
    assert len(states) == 8
    for i, layer in enumerate(states):
        assert layer.cache[0].tolist() == [float(i)] * 4


def test_deep_copy_handles_layer_with_none_in_cache_list():
    """Some SSM layers have None entries in their .cache list (e.g. the
    BatchMambaCache initialized with [None, None]). Deep-copy must not crash."""
    cache = SSMCompanionCache(max_entries=5)
    layer = _FakeSSMLayer(1.0, n_arrays=2)
    layer.cache[1] = None  # second array is None
    cache.store([1], 1, [layer])
    entry = cache.fetch([1], 1)
    states, _ = entry
    assert states[0].cache[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert states[0].cache[1] is None


# ----------------------------------------------------------------------
# `lengths` attribute materialization (REQ-A3-001 deep-copy contract for the
# 0.31.2 ArraysCache.lengths field)
# ----------------------------------------------------------------------


def test_lengths_attr_carried_through_deep_copy():
    cache = SSMCompanionCache(max_entries=5)
    layer = _FakeSSMLayer(1.0)
    layer.lengths = mx.array([5, 7, 9])
    cache.store([1], 1, [layer])
    entry = cache.fetch([1], 1)
    states, _ = entry
    assert states[0].lengths is not None
    assert states[0].lengths.tolist() == [5, 7, 9]


def test_lengths_independence_after_fetch():
    cache = SSMCompanionCache(max_entries=5)
    layer = _FakeSSMLayer(1.0)
    layer.lengths = mx.array([5, 7])
    cache.store([1], 1, [layer])
    entry1 = cache.fetch([1], 1)
    states1, _ = entry1
    states1[0].lengths = mx.array([99, 99])
    entry2 = cache.fetch([1], 1)
    states2, _ = entry2
    assert states2[0].lengths.tolist() == [5, 7]


def test_lengths_none_is_passed_through():
    cache = SSMCompanionCache(max_entries=5)
    layer = _FakeSSMLayer(1.0)
    layer.lengths = None
    cache.store([1], 1, [layer])
    entry = cache.fetch([1], 1)
    states, _ = entry
    assert states[0].lengths is None


# ----------------------------------------------------------------------
# LRU semantics
# ----------------------------------------------------------------------


def test_lru_eviction_at_max_entries():
    cache = SSMCompanionCache(max_entries=3)
    for i in range(5):
        cache.store([i], 1, [_FakeSSMLayer(float(i))])
    assert cache.size == 3
    # Oldest entries (0, 1) should be evicted; (2, 3, 4) should remain
    assert cache.fetch([0], 1) is None
    assert cache.fetch([1], 1) is None
    assert cache.fetch([2], 1) is not None
    assert cache.fetch([3], 1) is not None
    assert cache.fetch([4], 1) is not None


def test_lru_re_store_updates_position():
    cache = SSMCompanionCache(max_entries=3)
    cache.store([1], 1, [_FakeSSMLayer(1.0)])
    cache.store([2], 1, [_FakeSSMLayer(2.0)])
    cache.store([3], 1, [_FakeSSMLayer(3.0)])
    # Re-store [1] to bump it to most-recent
    cache.store([1], 1, [_FakeSSMLayer(11.0)])
    # Now insert [4] — should evict [2] (least recently used), not [1]
    cache.store([4], 1, [_FakeSSMLayer(4.0)])
    assert cache.fetch([1], 1) is not None  # survived (was just re-stored)
    assert cache.fetch([2], 1) is None  # evicted
    assert cache.fetch([3], 1) is not None
    assert cache.fetch([4], 1) is not None


def test_lru_fetch_updates_position():
    cache = SSMCompanionCache(max_entries=3)
    cache.store([1], 1, [_FakeSSMLayer(1.0)])
    cache.store([2], 1, [_FakeSSMLayer(2.0)])
    cache.store([3], 1, [_FakeSSMLayer(3.0)])
    # Fetch [1] to bump it to most-recent
    cache.fetch([1], 1)
    # Insert [4] — should evict [2]
    cache.store([4], 1, [_FakeSSMLayer(4.0)])
    assert cache.fetch([1], 1) is not None  # survived
    assert cache.fetch([2], 1) is None  # evicted
    assert cache.fetch([3], 1) is not None
    assert cache.fetch([4], 1) is not None


# ----------------------------------------------------------------------
# Key alignment — LLM (N) vs MLLM (N-1)
# ----------------------------------------------------------------------


def test_key_alignment_n_vs_n_minus_1():
    """LLM key uses N=prompt_len, MLLM key uses N-1. Verify both produce
    distinct hash buckets so the two paths can coexist in the same cache
    without collision."""
    cache = SSMCompanionCache(max_entries=10)
    tokens = [1, 2, 3, 4, 5, 6]
    cache.store(tokens, 6, [_FakeSSMLayer(1.0)])  # LLM key (N)
    cache.store(tokens, 5, [_FakeSSMLayer(2.0)])  # MLLM key (N-1)
    assert cache.size == 2

    e_llm = cache.fetch(tokens, 6)
    e_mllm = cache.fetch(tokens, 5)
    assert e_llm is not None
    assert e_mllm is not None
    assert e_llm[0][0].cache[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert e_mllm[0][0].cache[0].tolist() == [2.0, 2.0, 2.0, 2.0]


# ----------------------------------------------------------------------
# Regression — ISSUE-A3-003: BatchGenerator constructor must not crash
# the patched _merge_caches with empty input
# ----------------------------------------------------------------------


def test_issue_a3_003_patched_merge_caches_empty_input():
    """Regression for ISSUE-A3-003 (Phase 4 live discovery 2026-04-08).

    `BatchGenerator.__init__` calls `PromptProcessingBatch.empty(...)`
    which calls `_merge_caches([])`. Before the fix, the patched
    `_patched_merge_caches` did `range(len(caches[0]))` without an
    empty-input guard, raising `IndexError: list index out of range`
    on every BatchGenerator construction against any model.

    The fix adds `if not caches or not caches[0]: return []` at the
    top of the function. This test exercises the empty-input path
    directly so future refactors can't silently re-introduce the
    crash. The unit suite previously missed this because all tests
    construct `BatchMambaCache` directly without going through the
    full BatchGenerator constructor path.
    """
    # Trigger the patch installation
    from vmlx_engine.utils.mamba_cache import ensure_mamba_support
    ensure_mamba_support()

    # Reach into the patched mlx_lm.generate module
    import importlib
    gen_module = importlib.import_module("mlx_lm.generate")
    patched = gen_module._merge_caches

    # Confirm we're testing the patched version, not the upstream original
    assert "patch_mlx_lm_for_mamba" in patched.__qualname__, (
        "_merge_caches is not the vMLX patched version — patch flow broken"
    )

    # The crashing inputs from the original Phase 4 live failure
    assert patched([]) == []
    assert patched([[]]) == []

    # Sanity: a non-empty input still goes through the regular merge path
    # (we don't construct a real KV/SSM cache here — just confirm the early
    # return doesn't swallow valid input). Use an explicit truthy outer list
    # with an empty inner — should still take the early return for inner=[].
    assert patched([[], []]) == []


def test_issue_a3_003_batch_generator_construct_empty():
    """Higher-level regression for ISSUE-A3-003: confirm a BatchGenerator
    can be CONSTRUCTED for any model class without immediately crashing
    on the empty-cache path. Uses a stub model so the test doesn't need
    real weights.
    """
    from vmlx_engine.utils.mamba_cache import ensure_mamba_support
    ensure_mamba_support()

    from mlx_lm.generate import BatchGenerator
    import mlx.nn as nn

    class _StubModel(nn.Module):
        """Minimal stub: needs `layers` so make_prompt_cache can iterate."""

        def __init__(self):
            super().__init__()
            self.layers = []  # zero layers — exercises the truly-empty path

        def __call__(self, x, cache=None):
            return x

    # Before the fix, this raised IndexError inside _patched_merge_caches.
    # After the fix it should construct cleanly.
    bg = BatchGenerator(_StubModel(), max_tokens=4)
    assert bg is not None
    bg.close()


# ----------------------------------------------------------------------
# A3-BUG-001 — model identity in cache key
# ----------------------------------------------------------------------


def test_bug001_different_model_keys_no_collision():
    """Two caches with different model_keys must not see each other's entries.

    Before the fix, _key() hashed only the token list, so two models with
    identical prompts collided silently. After the fix, model_key is mixed
    into the SHA-256 input, producing distinct keys per model.
    """
    layer = _FakeSSMLayer(1.0)
    cache_a = SSMCompanionCache(model_key="model-A|smelt=0|tq=0")
    cache_b = SSMCompanionCache(model_key="model-B|smelt=50|tq=1")

    cache_a.store([1, 2, 3], 3, [layer])
    # Same tokens, different model identity → must miss.
    assert cache_b.fetch([1, 2, 3], 3) is None
    # Same model identity → must hit.
    assert cache_a.fetch([1, 2, 3], 3) is not None


def test_bug001_empty_model_key_legacy_default():
    """Default model_key='' preserves legacy behavior — no collision flag needed."""
    layer = _FakeSSMLayer(2.0)
    c = SSMCompanionCache()  # default model_key=""
    assert c.model_key == ""
    c.store([7, 8, 9], 3, [layer])
    assert c.fetch([7, 8, 9], 3) is not None


def test_bug001_model_key_property_immutable_after_construct():
    """model_key is set at construction and exposed read-only via property."""
    c = SSMCompanionCache(model_key="abc")
    assert c.model_key == "abc"
    # Property has no setter — assigning would fail. We just confirm read.
    assert c._model_key == "abc"


# ----------------------------------------------------------------------
# is_hybrid_ssm_model helper (Agent 1 F2 dependency)
# ----------------------------------------------------------------------


def test_is_hybrid_ssm_cache_detects_mamba_layer():
    """Built prompt_cache containing a MambaCache-derived layer → True."""
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_cache
    from vmlx_engine.utils.mamba_cache import BatchMambaCache

    layer = BatchMambaCache(size=2, left_padding=None)
    assert is_hybrid_ssm_cache([layer]) is True


def test_is_hybrid_ssm_cache_pure_attention_returns_false():
    """A cache list with only KV layers (no Mamba) → False."""
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_cache
    from mlx_lm.models.cache import KVCache

    assert is_hybrid_ssm_cache([KVCache(), KVCache()]) is False


def test_is_hybrid_ssm_cache_empty_or_none_returns_false():
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_cache

    assert is_hybrid_ssm_cache([]) is False
    assert is_hybrid_ssm_cache(None) is False


def test_is_hybrid_ssm_config_detects_hybrid_pattern():
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_config

    cfg = {"hybrid_override_pattern": "M*M*ME"}
    assert is_hybrid_ssm_config(cfg) is True


def test_is_hybrid_ssm_config_detects_known_model_type():
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_config

    assert is_hybrid_ssm_config({"model_type": "nemotron_h"}) is True
    assert is_hybrid_ssm_config({"model_type": "qwen3_next"}) is True
    assert is_hybrid_ssm_config({"model_type": "llama"}) is False


def test_is_hybrid_ssm_config_handles_text_config_nesting():
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_config

    # VLM wrapper with hybrid text config
    cfg = {"text_config": {"hybrid_override_pattern": "M*ME"}}
    assert is_hybrid_ssm_config(cfg) is True


def test_is_hybrid_ssm_model_polymorphic_dispatch():
    from vmlx_engine.utils.ssm_companion_cache import is_hybrid_ssm_model
    from vmlx_engine.utils.mamba_cache import BatchMambaCache

    # list path
    assert is_hybrid_ssm_model([BatchMambaCache(size=2, left_padding=None)]) is True
    # config dict path
    assert is_hybrid_ssm_model({"model_type": "nemotron_h"}) is True
    # neither
    assert is_hybrid_ssm_model({"model_type": "llama"}) is False


# ----------------------------------------------------------------------
# A3-BUG-003 — in-place advance perf semantics (correctness only)
# ----------------------------------------------------------------------


def test_bug003_advance_in_place_semantics_unchanged():
    """In-place -= must produce identical observable behavior to out-of-place.

    We can't easily measure allocation count from Python, but we can verify
    the math is correct after the change.
    """
    from vmlx_engine.utils.mamba_cache import BatchMambaCache

    c = BatchMambaCache(size=2, left_padding=None)
    c.prepare(lengths=[10, 8, 6])
    c.advance(3)
    import mlx.core as _mx
    assert _mx.array_equal(c.lengths, _mx.array([7, 5, 3])).item()
    c.advance(2)
    assert _mx.array_equal(c.lengths, _mx.array([5, 3, 1])).item()


# ----------------------------------------------------------------------
# A3-BUG-004 — unknown kwargs warned, not silently dropped
# ----------------------------------------------------------------------


def test_bug004_unknown_prepare_kwarg_warns(caplog):
    """Unknown kwargs to prepare() must emit a one-time WARNING."""
    import logging
    from vmlx_engine.utils.mamba_cache import (
        BatchMambaCache,
        _seen_unknown_prepare_kwargs,
    )

    # Reset the dedup set so the test is order-independent.
    _seen_unknown_prepare_kwargs.discard("future_param")

    c = BatchMambaCache(size=2, left_padding=None)
    with caplog.at_level(logging.WARNING, logger="vmlx_engine.utils.mamba_cache"):
        c.prepare(lengths=[5], future_param=42)
    assert any("future_param" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------
# Edge-case guards EC-1 / EC-2 / EC-10
# ----------------------------------------------------------------------


def test_ec1_store_empty_prompt_silently_skipped():
    """num_tokens <= 0 must not pollute the cache."""
    layer = _FakeSSMLayer(3.0)
    c = SSMCompanionCache()
    c.store([], 0, [layer])
    c.store([1, 2, 3], 0, [layer])
    c.store([1, 2, 3], -5, [layer])
    assert c.size == 0


def test_ec1_fetch_empty_prompt_returns_none():
    layer = _FakeSSMLayer(4.0)
    c = SSMCompanionCache()
    c.store([1, 2, 3], 3, [layer])
    assert c.fetch([], 0) is None
    assert c.fetch([1, 2, 3], 0) is None


def test_ec10_store_zero_ssm_layers_silently_skipped():
    """Empty ssm_states list must not pollute the cache."""
    c = SSMCompanionCache()
    c.store([1, 2, 3], 3, [])
    assert c.size == 0


def test_ec2_mllm_n_minus_one_single_token_path():
    """MLLM N-1 with N=1 means num_tokens=0 → store skipped, fetch None."""
    layer = _FakeSSMLayer(5.0)
    c = SSMCompanionCache()
    # Single-token MLLM prompt: real callers compute n = 1 - 1 = 0.
    c.store([42], 0, [layer])
    assert c.size == 0
    assert c.fetch([42], 0) is None
