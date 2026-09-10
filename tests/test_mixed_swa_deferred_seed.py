"""A deferred clean store may resume only its still-owned immutable checkpoint."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vmlx_engine.scheduler import Scheduler


def fixture():
    scheduler = Scheduler.__new__(Scheduler)
    table = SimpleNamespace(num_tokens=13, block_ids=[1])
    request = SimpleNamespace(request_id="r", block_table=table, cached_tokens=13)
    caches = [KVCache(), RotatingKVCache(max_size=8)]
    for cache in caches:
        cache.offset = 13
    scheduler.model = SimpleNamespace(layers=[1, 2])
    scheduler._mixed_attention_cache_model = True
    scheduler._kv_cache_bits = 0
    scheduler.block_aware_cache = SimpleNamespace(
        _request_tables={"r": SimpleNamespace(block_table=table)},
        reconstruct_cache=Mock(return_value=caches),
    )
    return scheduler, request, caches


def test_seed_reconstructs_owned_checkpoint_not_live_decode_state():
    scheduler, request, caches = fixture()
    request.prompt_cache = [SimpleNamespace(offset=99)]
    seed, boundary = scheduler._mixed_swa_deferred_seed(request, list(range(25)))
    assert boundary == 13
    assert len(seed) == 2 and all(c.offset == 13 for c in seed)
    assert all(a is not b for a, b in zip(seed, caches))
    assert request.prompt_cache[0].offset == 99
    scheduler.block_aware_cache.reconstruct_cache.assert_called_once_with(request.block_table)


@pytest.mark.parametrize("failure", ["detached", "wrong_owner", "boundary", "beyond", "offset", "layers", "unknown", "missing", "exception", "other_family", "kv_quant", "turboquant"])
def test_untrusted_seed_declines_to_clean_full_prefill(failure):
    scheduler, request, caches = fixture()
    if failure == "detached": scheduler.block_aware_cache._request_tables.clear()
    if failure == "wrong_owner": scheduler.block_aware_cache._request_tables["r"].block_table = object()
    if failure == "boundary": request.cached_tokens = 12
    if failure == "beyond": request.block_table.num_tokens = request.cached_tokens = 26
    if failure == "offset": caches[0].offset = 14
    if failure == "layers": scheduler.model.layers.append(3)
    if failure == "unknown": caches[0] = SimpleNamespace(offset=13)
    if failure == "missing": scheduler.block_aware_cache.reconstruct_cache.return_value = None
    if failure == "exception": scheduler.block_aware_cache.reconstruct_cache.side_effect = RuntimeError("unreadable block")
    if failure == "other_family": scheduler._mixed_attention_cache_model = False
    if failure == "kv_quant": scheduler._kv_cache_bits = 8
    if failure == "turboquant": scheduler._tq_active = True
    assert scheduler._mixed_swa_deferred_seed(request, list(range(25))) == (None, 0)


def test_resume_can_be_disabled_without_changing_cache_settings(monkeypatch):
    scheduler, request, _ = fixture()
    monkeypatch.setenv("VMLX_DISABLE_MIXED_SWA_CLEAN_RESUME", "1")
    assert scheduler._mixed_swa_deferred_seed(request, list(range(25))) == (None, 0)
    scheduler.block_aware_cache.reconstruct_cache.assert_not_called()


@pytest.mark.parametrize("resume_fails", [False, True])
def test_materialization_resumes_then_falls_back_without_reordering(resume_fails, monkeypatch):
    scheduler, request, caches = fixture()
    request._deferred_prompt_cache = {"family": "Mixed-SWA", "mode": "paged", "key_tokens": list(range(25))}
    scheduler._uses_dsv4_cache = False
    scheduler.disk_cache = None
    scheduler._prefill_for_prompt_only_cache = Mock(side_effect=[None, caches] if resume_fails else [caches])
    scheduler._extract_cache_states = Mock(return_value=[{"state": "owned"}])
    monkeypatch.setattr("vmlx_engine.utils.turboquant_config.turboquant_cache_telemetry", lambda _: {})
    scheduler._materialize_deferred_prompt_cache("r", request)
    calls = scheduler._prefill_for_prompt_only_cache.call_args_list
    assert calls[0].kwargs["base_token_count"] == 13
    assert all(c.offset == 13 for c in calls[0].kwargs["base_cache"])
    assert len(calls) == (2 if resume_fails else 1)
    if resume_fails: assert calls[1].kwargs == {}
    assert request._deferred_prompt_cache is None
    assert request._extracted_cache_key_tokens == list(range(25))
    assert request._extracted_cache == [{"state": "owned"}]
