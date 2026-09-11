"""MTP context telemetry must include restored logical positions, never slack."""
from types import SimpleNamespace

import pytest

from vmlx_engine.patches.mlx_lm_mtp import batch_generator as lane


@pytest.mark.parametrize("prefix,suffix", [(0, 652), (15644, 315), (260000, 12)])
def test_restored_prefix_is_included(prefix, suffix):
    batch = SimpleNamespace(tokens=[[1] * suffix], prompt_cache=[
        SimpleNamespace(), SimpleNamespace(offset=prefix + suffix)])
    assert lane._mtp_context_extent(batch) == (prefix + suffix, "logical_cache_offset")


def test_native_wrapped_offset_ignores_allocated_extent():
    class Cache:
        offset = 257
        @property
        def keys(self):
            pytest.fail("allocated KV buffers are not logical context")
    wrapper = SimpleNamespace(caches=[Cache()])
    wrapper.caches.append(wrapper)  # Do not recurse forever on malformed wrappers.
    assert lane._mtp_context_extent(SimpleNamespace(prompt_cache=[wrapper])) == (
        257, "logical_cache_offset")


@pytest.mark.parametrize("offset", [None, -1, "invalid", True])
def test_unknown_cache_uses_labelled_admitted_fallback(offset):
    batch = SimpleNamespace(tokens=[[1, 2, 3]], prompt_cache=[SimpleNamespace(offset=offset)])
    assert lane._mtp_context_extent(batch) == (3, "admitted_tokens_fallback")


def test_missing_extent_is_unknown():
    assert lane._mtp_context_extent(SimpleNamespace()) == (0, "unknown")


def test_real_batched_offsets_and_unrelated_physical_capacity():
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache
    cache = BatchKVCache([0])
    cache.update_and_fetch(mx.zeros((1, 1, 257, 4)), mx.zeros((1, 1, 257, 4)))
    assert cache.keys.shape[2] > 257
    batch = SimpleNamespace(tokens=[[1, 2]], prompt_cache=[cache])
    assert lane._mtp_context_extent(batch) == (257, "logical_cache_offset")
    cache.offset = mx.array([257, 128])
    assert lane._mtp_context_extent(batch) == (2, "admitted_tokens_fallback")


def test_health_preserves_context_value_and_provenance():
    stats = lane._MtpStats(seed_context_tokens=15959,
                           seed_context_source="logical_cache_offset")
    payload = lane._native_mtp_payload("request", stats, "stop")
    assert payload["seed_context_tokens"] == 15959
    assert payload["seed_context_source"] == "logical_cache_offset"


@pytest.mark.parametrize("offset", [1.5, float('inf'), True])
def test_noninteger_scalar_array_is_not_an_offset(offset):
    import mlx.core as mx
    batch = SimpleNamespace(tokens=[[1, 2]], prompt_cache=[
        SimpleNamespace(offset=mx.array([offset]))])
    assert lane._mtp_context_extent(batch) == (2, "admitted_tokens_fallback")
