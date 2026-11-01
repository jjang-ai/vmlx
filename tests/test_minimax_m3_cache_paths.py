# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MiniMax-M3's three-lane MSA cache.

M3 sparse layers are not plain KVCache layers.  They must carry keys, values,
and idx_keys together; dropping idx_keys silently changes Lightning-Indexer
block selection on cache reuse.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import mlx.core as mx


def _m3_cache(seq: int = 8):
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    c = MiniMaxM3SparseCache()
    keys = mx.arange(1 * 4 * seq * 8, dtype=mx.float32).reshape(1, 4, seq, 8)
    values = (keys + 1000).astype(mx.float32)
    idx_keys = mx.arange(1 * 1 * seq * 8, dtype=mx.float32).reshape(1, 1, seq, 8)
    c.state = (keys, values, idx_keys)
    return c


def _assert_m3_cache(c, seq: int) -> None:
    assert type(c).__name__ == "MiniMaxM3SparseCache"
    assert c.offset == seq
    assert c.keys.shape[2] == seq
    assert c.values.shape[2] == seq
    assert c.idx_keys is not None
    assert c.idx_keys.shape[2] == seq
    state = c.state
    assert len(state) == 3
    assert state[2] is not None
    assert state[0].shape[2] == state[1].shape[2] == state[2].shape[2] == seq


def test_m3_block_cache_keys_are_scoped_by_prefill_shape():
    """Same tokens from different M3 prefill shapes must not alias one block.

    MLX can produce slightly different projection values for a token prefix
    when the enclosing prefill matrix has a different sequence length. M3's
    Lightning Indexer amplifies that drift, so token-only block hashes are not
    a sufficient native-state identity.
    """
    from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    class _M3Model:
        config = SimpleNamespace(model_type="minimax_m3_vl")

        def make_cache(self):
            return [MiniMaxM3SparseCache()]

    cache = BlockAwarePrefixCache(
        _M3Model(),
        PagedCacheManager(block_size=64, max_blocks=8),
    )
    shared_block_tokens = list(range(64))
    short_shape = cache._shape_scoped_cache_extra_keys(
        list(range(487)),
        None,
    )
    long_shape = cache._shape_scoped_cache_extra_keys(
        list(range(15912)),
        None,
    )
    same_long_shape = cache._shape_scoped_cache_extra_keys(
        list(reversed(range(15912))),
        None,
    )
    stored_long_shape = cache._shape_scoped_cache_extra_keys(
        list(range(15911)),
        None,
        stored_prompt_boundary=True,
    )

    assert short_shape != long_shape
    assert long_shape == same_long_shape
    assert long_shape == stored_long_shape
    assert compute_block_hash(
        None,
        shared_block_tokens,
        extra_keys=short_shape,
    ) != compute_block_hash(
        None,
        shared_block_tokens,
        extra_keys=long_shape,
    )


def test_m3_prefill_shape_scope_preserves_request_discriminators():
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    class _M3Model:
        config = SimpleNamespace(model_type="minimax_m3_vl")

        def make_cache(self):
            return [MiniMaxM3SparseCache()]

    cache = BlockAwarePrefixCache(
        _M3Model(),
        PagedCacheManager(block_size=64, max_blocks=8),
    )
    request_extra = {"generation_prompt": "assistant-suffix"}
    scoped = cache._shape_scoped_cache_extra_keys(
        [1, 2, 3],
        request_extra,
    )
    stored_scoped = cache._shape_scoped_cache_extra_keys(
        [1, 2],
        request_extra,
        stored_prompt_boundary=True,
    )

    assert scoped["__vmlx_request_extra_keys__"] is request_extra
    assert stored_scoped == scoped
    assert scoped["__vmlx_native_cache_shape__"] == {
        "schema": "minimax_m3_prefill_shape_v1",
        "cache_key_tokens": 3,
    }


def test_qwen4_exp_reused_sparse_container_does_not_inherit_m3_shape_key():
    """Qwen QSA payload compatibility must not partition clean media lookup."""
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    class _Qwen4ExpModel:
        config = SimpleNamespace(model_type="qwen4_exp")

        def make_cache(self):
            return [MiniMaxM3SparseCache()]

    cache = BlockAwarePrefixCache(
        _Qwen4ExpModel(),
        PagedCacheManager(block_size=64, max_blocks=8),
    )
    request_extra = {
        "generation_prompt": "assistant-suffix",
        "mllm_media_0000": "blue-image",
    }

    warm_request = cache._shape_scoped_cache_extra_keys(
        list(range(5445)), request_extra
    )
    clean_boundary = cache._shape_scoped_cache_extra_keys(
        list(range(5440)),
        request_extra,
        stored_prompt_boundary=True,
    )

    assert warm_request is request_extra
    assert clean_boundary is request_extra
    assert "__vmlx_native_cache_shape__" not in warm_request


def test_paged_gen_prompt_strip_truncates_all_minimax_m3_cache_lanes():
    """The paged key and M3's K/V/index state must represent one token count."""
    from vmlx_engine.scheduler import _truncate_minimax_m3_state_dict

    cache = _m3_cache(seq=9)
    state_dict = {
        "class_name": "MiniMaxM3SparseCache",
        "state": cache.state,
        "meta_state": ("9",),
    }

    trimmed = _truncate_minimax_m3_state_dict(state_dict, target=6)

    assert trimmed is not None
    keys, values, idx_keys = trimmed["state"]
    assert keys.shape[2] == values.shape[2] == idx_keys.shape[2] == 6
    assert trimmed["meta_state"] == ("6",)
    assert keys[0, 0, -1, 0].item() == cache.keys[0, 0, 5, 0].item()
    assert idx_keys[0, 0, -1, 0].item() == cache.idx_keys[0, 0, 5, 0].item()


def test_paged_gen_prompt_strip_rejects_minimax_m3_without_index_lane():
    from vmlx_engine.scheduler import _truncate_minimax_m3_state_dict

    cache = _m3_cache(seq=9)
    state_dict = {
        "class_name": "MiniMaxM3SparseCache",
        "state": (cache.keys, cache.values, None),
        "meta_state": ("9",),
    }

    assert _truncate_minimax_m3_state_dict(state_dict, target=6) is None


def test_llm_scheduler_truncation_preserves_minimax_m3_idx_keys():
    from vmlx_engine.scheduler import Scheduler

    result = Scheduler._truncate_cache_to_prompt_length([_m3_cache(seq=8)], prompt_len=6)

    assert result is not None
    _assert_m3_cache(result[0], seq=5)


def test_llm_scheduler_truncation_materializes_minimax_m3_slices(monkeypatch):
    import vmlx_engine.models.minimax_m3.cache as m3_cache
    from vmlx_engine.scheduler import Scheduler

    original = m3_cache.clone_minimax_m3_sparse
    seen_copy_fns = []

    def wrapped(cache, length=None, *, copy_fn=None, require_idx_keys=True):
        seen_copy_fns.append(copy_fn)
        return original(
            cache,
            length,
            copy_fn=copy_fn,
            require_idx_keys=require_idx_keys,
        )

    monkeypatch.setattr(m3_cache, "clone_minimax_m3_sparse", wrapped)

    result = Scheduler._truncate_cache_to_prompt_length([_m3_cache(seq=8)], prompt_len=6)

    assert result is not None
    _assert_m3_cache(result[0], seq=5)
    assert seen_copy_fns
    assert callable(seen_copy_fns[0])


def test_memory_cache_truncation_materializes_minimax_m3_slices(monkeypatch):
    import vmlx_engine.models.minimax_m3.cache as m3_cache
    from vmlx_engine.memory_cache import MemoryAwarePrefixCache

    original = m3_cache.clone_minimax_m3_sparse
    seen_copy_fns = []

    def wrapped(cache, length=None, *, copy_fn=None, require_idx_keys=True):
        seen_copy_fns.append(copy_fn)
        return original(
            cache,
            length,
            copy_fn=copy_fn,
            require_idx_keys=require_idx_keys,
        )

    monkeypatch.setattr(m3_cache, "clone_minimax_m3_sparse", wrapped)

    result = MemoryAwarePrefixCache._truncate_cache([_m3_cache(seq=8)], target_len=5)

    assert result is not None
    _assert_m3_cache(result[0], seq=5)
    assert seen_copy_fns
    assert callable(seen_copy_fns[0])


def test_memory_cache_truncation_materializes_dense_kv_companion_layers(monkeypatch):
    """M3 cache hits include dense KV layers 0-2, not only sparse MSA layers.

    The memory-aware fetch path runs on the API thread and the returned cache is
    consumed on the scheduler worker stream.  Dense KV slices must be
    materialized too; otherwise the sparse MSA layers are isolated but the
    companion dense layers still carry lazy API-thread views into generation.
    """
    import numpy as np
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.memory_cache import MemoryAwarePrefixCache

    dense = KVCache()
    keys = mx.arange(1 * 2 * 8 * 4, dtype=mx.float32).reshape(1, 2, 8, 4)
    dense.state = (keys, keys + 100)

    calls = []
    original_array = np.array

    def wrapped_array(value, *args, **kwargs):
        calls.append(getattr(value, "shape", None))
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(np, "array", wrapped_array)

    result = MemoryAwarePrefixCache._truncate_cache([dense], target_len=5)

    assert result is not None
    assert result[0].offset == 5
    assert result[0].keys.shape[2] == 5
    assert result[0].values.shape[2] == 5
    assert calls, "dense KV fetch clone must materialize through numpy"


def test_mllm_scheduler_truncation_preserves_minimax_m3_idx_keys():
    from vmlx_engine.mllm_scheduler import MLLMScheduler

    scheduler = object.__new__(MLLMScheduler)
    result = scheduler._truncate_hybrid_cache([_m3_cache(seq=8)], prompt_len=6)

    assert result is not None
    _assert_m3_cache(result[0], seq=5)


def test_single_batch_prompt_snapshot_clones_minimax_m3_idx_keys():
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    original = _m3_cache(seq=8)
    snapshot = SingleBatchGenerator._clone_prompt_cache_snapshot([original])

    assert snapshot is not None
    assert snapshot[0] is not original
    _assert_m3_cache(snapshot[0], seq=8)


def test_scheduler_memory_object_store_prefers_minimax_m3_prompt_snapshot(monkeypatch):
    """Memory-aware M3 stores must use the clean prompt-boundary snapshot.

    The default M3 app route has paged cache OFF, so prefix reuse goes through
    the scheduler's object-cache path. That path must not store the live
    post-generation cache when SingleBatchGenerator already supplied a clean
    prompt snapshot.
    """
    from vmlx_engine.request import Request, RequestStatus, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    raw_post_decode = [_m3_cache(seq=10)]
    prompt_snapshot = [_m3_cache(seq=7)]

    request = Request(
        request_id="m3-cache-store-test",
        prompt=[11, 12, 13, 14, 15, 16, 17],
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.status = RequestStatus.RUNNING

    scheduler = object.__new__(Scheduler)
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler._tq_active = False
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = object()
    scheduler._mixed_attention_cache_model = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0

    class _Detok:
        text = ""

        def finalize(self):
            return None

    monkeypatch.setattr(Scheduler, "_get_detokenizer", lambda _self, _rid: _Detok())

    response = SimpleNamespace(
        uid=1,
        token=0,
        finish_reason="stop",
        prompt_cache=raw_post_decode,
        prompt_cache_snapshot=prompt_snapshot,
    )

    _outputs, finished_ids = scheduler._process_batch_responses([response])

    assert request.request_id in finished_ids
    assert getattr(request, "_extracted_cache", None) is prompt_snapshot
    _assert_m3_cache(request._extracted_cache[0], seq=7)


def test_scheduler_m3_cache_hit_store_rederives_clean_prompt_cache(monkeypatch):
    """M3 must not donate cache-hit-derived MSA state back to prefix storage.

    A memory/SSD hit restores keys/values/idx_keys, then the scheduler replays
    the uncached tail.  Live app failures showed the resulting extended state is
    not safe to persist as the next prompt prefix: the following exact hit can
    answer an earlier turn.  Cache-hit M3 stores must therefore re-prefill the
    prompt-boundary key directly and store that clean cache.
    """
    from vmlx_engine.request import Request, RequestStatus, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    raw_post_decode = [_m3_cache(seq=10)]
    tail_replay_snapshot = [_m3_cache(seq=7)]
    clean_rederived = [_m3_cache(seq=6)]
    rederive_calls = []

    request = Request(
        request_id="m3-cache-hit-store-test",
        prompt=[11, 12, 13, 14, 15, 16, 17],
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.cached_tokens = 4
    request.status = RequestStatus.RUNNING

    scheduler = object.__new__(Scheduler)
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler._tq_active = False
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = object()
    scheduler._mixed_attention_cache_model = False
    scheduler._uses_m3_msa_cache = True
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._model_type_for_runtime = "minimax_m3"  # set by Scheduler.__init__ in production
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0

    class _Detok:
        text = ""

        def finalize(self):
            return None

    def _fake_rederive(_self, tokens):
        rederive_calls.append(list(tokens))
        return clean_rederived

    monkeypatch.setattr(Scheduler, "_get_detokenizer", lambda _self, _rid: _Detok())
    monkeypatch.setattr(Scheduler, "_prefill_for_prompt_only_cache", _fake_rederive)

    response = SimpleNamespace(
        uid=1,
        token=0,
        finish_reason="stop",
        prompt_cache=raw_post_decode,
        prompt_cache_snapshot=tail_replay_snapshot,
    )

    _outputs, finished_ids = scheduler._process_batch_responses([response])

    assert request.request_id in finished_ids
    assert rederive_calls == []
    assert request._deferred_prompt_cache == {
        "family": "MiniMax-M3",
        "mode": "object",
        "key_tokens": [11, 12, 13, 14, 15, 16],
    }

    scheduler._materialize_deferred_prompt_cache(request.request_id, request)

    assert rederive_calls == [[11, 12, 13, 14, 15, 16]]
    assert getattr(request, "_extracted_cache", None) is clean_rederived
    assert request._extracted_cache is not raw_post_decode
    assert request._extracted_cache is not tail_replay_snapshot
    assert request._extracted_cache_key_tokens == [11, 12, 13, 14, 15, 16]
    assert request._extracted_cache_from_prompt_snapshot is True
    _assert_m3_cache(request._extracted_cache[0], seq=6)


def test_scheduler_paged_m3_cache_hit_store_rederives_clean_prompt_cache(monkeypatch):
    """Paged M3 must not persist a snapshot extended from reconstructed MSA."""
    from vmlx_engine.request import Request, RequestStatus, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    raw_post_decode = [_m3_cache(seq=10)]
    hit_derived_snapshot = [_m3_cache(seq=7)]
    clean_rederived = [_m3_cache(seq=6)]
    extracted = [{"class_name": "MiniMaxM3SparseCache", "state": clean_rederived[0].state}]
    rederive_calls = []

    request = Request(
        request_id="m3-paged-cache-hit-store-test",
        prompt=[11, 12, 13, 14, 15, 16, 17],
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.cached_tokens = 4
    request.status = RequestStatus.RUNNING

    scheduler = object.__new__(Scheduler)
    scheduler.config = SimpleNamespace(enable_prefix_cache=True)
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler._tq_active = False
    scheduler.block_aware_cache = object()
    scheduler.disk_cache = None
    scheduler._is_hybrid = False
    scheduler._kv_cache_bits = 0
    scheduler._mixed_attention_cache_model = False
    scheduler._uses_m3_msa_cache = True
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._model_type_for_runtime = "minimax_m3"  # set by Scheduler.__init__ in production
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0
    scheduler._dsv4_trace_timing = lambda *_args, **_kwargs: None

    class _Detok:
        text = ""

        def finalize(self):
            return None

    def _fake_rederive(_tokens):
        rederive_calls.append(list(_tokens))
        return clean_rederived

    scheduler._prefill_for_prompt_only_cache = _fake_rederive
    scheduler._extract_cache_states = lambda cache: extracted if cache is clean_rederived else None

    monkeypatch.setattr(Scheduler, "_get_detokenizer", lambda _self, _rid: _Detok())

    response = SimpleNamespace(
        uid=1,
        token=0,
        finish_reason="stop",
        prompt_cache=raw_post_decode,
        prompt_cache_snapshot=hit_derived_snapshot,
    )

    _outputs, finished_ids = Scheduler._process_batch_responses(scheduler, [response])

    assert request.request_id in finished_ids
    assert rederive_calls == []
    assert request._deferred_prompt_cache == {
        "family": "MiniMax-M3",
        "mode": "paged",
        "key_tokens": [11, 12, 13, 14, 15, 16],
    }

    scheduler._materialize_deferred_prompt_cache(request.request_id, request)

    assert rederive_calls == [[11, 12, 13, 14, 15, 16]]
    assert request._extracted_cache is extracted
    assert request._extracted_cache_key_tokens == [11, 12, 13, 14, 15, 16]
    assert request._extracted_cache_from_prompt_snapshot is True


def test_m3_paged_store_avoids_full_numpy_mirror_and_deferred_disk_payloads():
    """Pin the bounded-memory shape used by the live long-prompt gate."""
    import inspect

    from vmlx_engine.prefix_cache import BlockAwarePrefixCache

    source = inspect.getsource(BlockAwarePrefixCache.store_cache)
    m3_mirror_start = source.index("if _is_minimax_m3_cache_class(cls):")
    m3_mirror_end = source.index(
        "state = layer_state.get", m3_mirror_start
    )
    m3_mirror_branch = source[m3_mirror_start:m3_mirror_end]

    assert "np.array(" not in m3_mirror_branch
    assert "np_block = block_kv_data" in source
    # the deferred-payload condition grew native-quantized / cache-list kinds
    assert "has_minimax_m3_cache_data\n                            or has_dsv4_delta_cache_data" in source
    assert "minimax_m3=has_minimax_m3_cache_data" in source


def test_scheduler_memory_aware_m3_store_also_writes_prompt_disk_l2(monkeypatch):
    """M3's default paged-off route must still populate SSD prompt L2.

    The live M3 route uses MemoryAwarePrefixCache because paged cache is forced
    off. DiskCacheManager already round-trips MiniMaxM3SparseCache state, so the
    memory-aware finished-request store must write disk L2 with the full
    generation-prompt-stripped key and an N-1 payload.
    """
    import vmlx_engine.scheduler as scheduler_mod
    from vmlx_engine.request import Request, RequestStatus, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    monkeypatch.setattr(
        scheduler_mod,
        "clear_mlx_memory_cache",
        lambda log=None: None,
    )

    request = Request(
        request_id="m3-memory-disk-store",
        prompt=[10, 11, 12, 90, 91],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.output_token_ids = []
    request.status = RequestStatus.RUNNING
    request._gen_prompt_len = 2
    request._extracted_cache = [_m3_cache(seq=3)]

    memory_stores = []
    disk_stores = []

    class _MemoryCache:
        def store(self, tokens, cache, cache_type="assistant"):
            memory_stores.append((list(tokens), cache, cache_type))
            return True

    class _DiskCache:
        def store(self, tokens, cache, cache_type="assistant"):
            disk_stores.append((list(tokens), cache, cache_type))
            return True

        def flush_pending_writes(self, tokens, cache_extra_keys=None):
            assert list(tokens) == disk_stores[-1][0]
            return True

    scheduler = object.__new__(Scheduler)
    scheduler.running = {request.request_id: request}
    scheduler.requests = {request.request_id: request}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.finished_req_ids = set()
    scheduler.batch_generator = None
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = _MemoryCache()
    scheduler.prefix_cache = None
    scheduler.disk_cache = _DiskCache()
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._pld_pending = {}
    scheduler._pld_ngram_indices = {}
    scheduler._pick_cache_type_for_request = lambda _request: "user"
    scheduler._cleanup_detokenizer = lambda _request_id: None
    scheduler.model = object()

    Scheduler._cleanup_finished(scheduler, {request.request_id})

    assert memory_stores
    assert memory_stores[0][0] == [10, 11]
    _assert_m3_cache(memory_stores[0][1][0], seq=2)
    assert disk_stores
    assert disk_stores[0][0] == [10, 11, 12]
    _assert_m3_cache(disk_stores[0][1][0], seq=2)
    assert disk_stores[0][2] == "user"


def test_disk_cache_fetch_longest_prefix_uses_stored_prompt_lengths(tmp_path, monkeypatch):
    """SSD prompt L2 must be a prefix cache, not exact-prompt-only.

    After an engine restart the in-memory prefix cache is empty, so M3 depends
    on disk L2 finding the longest stored prompt prefix for the current,
    longer, multi-turn prompt.
    """
    import sqlite3

    from vmlx_engine.disk_cache import DiskCacheManager, _hash_tokens

    mgr = DiskCacheManager(str(tmp_path), max_size_gb=0)
    try:
        rows = [
            ([7, 8, 9], "short.safetensors"),
            ([7, 8, 9, 10, 11], "longest.safetensors"),
            ([7, 8, 99, 100], "wrong-branch.safetensors"),
        ]
        conn = sqlite3.connect(mgr._db_path)
        now = 1.0
        try:
            for tokens, file_name in rows:
                conn.execute(
                    "INSERT INTO cache_entries "
                    "(token_hash, file_name, num_tokens, file_size, created_at, "
                    "last_accessed, access_count, metadata, cache_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _hash_tokens(tokens),
                        file_name,
                        len(tokens),
                        1,
                        now,
                        now,
                        1,
                        "{}",
                        "user",
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        fetched = []
        sentinel = [object()]

        def fake_fetch(tokens):
            fetched.append(list(tokens))
            return sentinel

        monkeypatch.setattr(mgr, "fetch", fake_fetch)

        cache, matched_tokens = mgr.fetch_longest_prefix([7, 8, 9, 10, 11, 12, 13])

        assert cache is sentinel
        assert matched_tokens == [7, 8, 9, 10, 11]
        assert fetched == [[7, 8, 9, 10, 11]]
    finally:
        mgr.shutdown()


def test_disk_cache_fetch_longest_prefix_accepts_shared_n_minus_1_payload(
    tmp_path, monkeypatch
):
    """A changed generation sentinel must not hide a reusable N-1 payload.

    Reasoning templates can render a base prompt ending in ``<think>`` and its
    multi-turn replay ending in ``</think>`` at the same boundary. The stored
    cache owns only the tokens before that sentinel, so the full-key hash is
    allowed to differ when the complete N-1 payload prefix still matches.
    """
    import sqlite3

    from vmlx_engine.disk_cache import DiskCacheManager, _hash_tokens
    from vmlx_engine.scheduler import Scheduler

    mgr = DiskCacheManager(str(tmp_path), max_size_gb=0)
    try:
        stored_tokens = [7, 8, 9, 10]  # token 10 is the old generation sentinel
        stored_hash = _hash_tokens(stored_tokens)
        conn = sqlite3.connect(mgr._db_path)
        now = 1.0
        try:
            conn.execute(
                "INSERT INTO cache_entries "
                "(token_hash, file_name, num_tokens, file_size, created_at, "
                "last_accessed, access_count, metadata, cache_type, "
                "payload_prefix_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stored_hash,
                    "n-minus-1.safetensors",
                    len(stored_tokens),
                    1,
                    now,
                    now,
                    1,
                    "{}",
                    "user",
                    _hash_tokens(stored_tokens[:-1]),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        fetched = []
        sentinel = [object()]

        def fake_fetch_indexed(token_hash, current_prefix_tokens):
            fetched.append((token_hash, list(current_prefix_tokens)))
            return sentinel

        monkeypatch.setattr(mgr, "_fetch_indexed_hash", fake_fetch_indexed)

        current = [7, 8, 9, 99, 100, 101]  # token 99 replaces old sentinel 10
        cache, matched_tokens = mgr.fetch_longest_prefix(current)

        assert cache is sentinel
        assert matched_tokens == [7, 8, 9, 99]
        assert fetched == [(stored_hash, [7, 8, 9, 99])]

        tail, cached = Scheduler._disk_prefix_hit_tail_and_cached_tokens(
            fetch_tokens=current,
            matched_tokens=matched_tokens,
            gen_prompt_suffix=[],
        )
        assert cached == 3
        assert tail == [99, 100, 101]
    finally:
        mgr.shutdown()


def test_disk_cache_n_minus_1_index_does_not_accept_earlier_divergence(
    tmp_path, monkeypatch
):
    """The N-1 alias is exact over the payload; earlier divergence is a miss."""
    import sqlite3

    from vmlx_engine.disk_cache import DiskCacheManager, _hash_tokens

    mgr = DiskCacheManager(str(tmp_path), max_size_gb=0)
    try:
        stored_tokens = [7, 8, 9, 10]
        conn = sqlite3.connect(mgr._db_path)
        now = 1.0
        try:
            conn.execute(
                "INSERT INTO cache_entries "
                "(token_hash, file_name, num_tokens, file_size, created_at, "
                "last_accessed, access_count, metadata, cache_type, "
                "payload_prefix_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _hash_tokens(stored_tokens),
                    "wrong-branch.safetensors",
                    len(stored_tokens),
                    1,
                    now,
                    now,
                    1,
                    "{}",
                    "user",
                    _hash_tokens(stored_tokens[:-1]),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(
            mgr,
            "_fetch_indexed_hash",
            lambda *_args, **_kwargs: pytest.fail("divergent payload must not load"),
        )
        cache, matched = mgr.fetch_longest_prefix([7, 80, 9, 99, 100])
        assert cache is None
        assert matched == []
    finally:
        mgr.shutdown()


def test_scheduler_disk_l2_prefix_hit_replays_uncached_tail(monkeypatch):
    """Disk L2 prefix hits must replay P[-1] plus current prompt tail.

    Disk stores an N-1 cache payload under prompt key P.  If current prompt F
    extends P, the scheduler must restore the cache and prefill
    P[-1] + F[len(P):] (+ generation suffix), not just the generation suffix.
    """
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    request = Request(
        request_id="m3-disk-prefix-hit",
        prompt=[10, 11, 12, 13, 14, 90, 91],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request._gen_prompt_len = 2

    class _MemoryCache:
        def fetch(self, tokens):
            return None, list(tokens)

        def store(self, *args, **kwargs):
            return True

    class _DiskCache:
        _last_fetch_tq_native = False
        _last_fetch_cache_type = "user"

        def fetch_longest_prefix(self, tokens):
            assert tokens == [10, 11, 12, 13, 14]
            # Store contract: key P=[10,11,12,13] (4 tokens) but the payload is
            # truncated to N-1=3 tokens (_truncate_cache_to_prompt_length) so the
            # last matched token P[-1]=13 is re-fed on a hit. Mock the payload at
            # seq=3 to mirror the real N-1 disk store.
            return [_m3_cache(seq=3)], [10, 11, 12, 13]

    scheduler = object.__new__(Scheduler)
    scheduler.memory_aware_cache = _MemoryCache()
    scheduler.prefix_cache = None
    scheduler.block_aware_cache = None
    scheduler.disk_cache = _DiskCache()
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._pld_spec_enabled = False
    scheduler._pld_auto_enabled = False
    scheduler._pld_ngram_indices = {}
    scheduler._pld_pending = {}
    scheduler._prefix_hit_tail_and_cached_tokens = Scheduler._prefix_hit_tail_and_cached_tokens
    scheduler.requests = {}
    scheduler.waiting = []

    Scheduler.add_request(scheduler, request)

    assert request.prompt_cache is not None
    _assert_m3_cache(request.prompt_cache[0], seq=3)
    # N-1 contract: matched key P=[10,11,12,13] carries a 3-token payload, so the
    # cache offset is len(P)-1=3 and the scheduler re-feeds P[-1]=13 ahead of the
    # uncached tail F[len(P):]=[14] and the generation suffix [90,91].
    assert request.cached_tokens == 3
    assert request.remaining_tokens == [13, 14, 90, 91]
    assert request._cache_detail == "disk"


def test_scheduler_single_batch_cache_hit_passes_cached_prefix_as_all_tokens(monkeypatch):
    """SingleBatch cache hits must keep the generator's logical context whole.

    On a memory/disk prefix hit, the prompt sent to SingleBatchGenerator is only
    the uncached tail.  The restored cache already represents the prefix, so the
    generator also needs `all_tokens` seeded with that cached prefix.  Otherwise
    TokenBuffer/context bookkeeping starts at the tail and diverges from a fresh
    full-prefill request.
    """
    from collections import deque
    from types import SimpleNamespace

    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    request = Request(
        request_id="m3-single-batch-hit-context",
        prompt=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.prompt_cache = [_m3_cache(seq=3)]
    request.cached_tokens = 3
    request.remaining_tokens = [13, 14]
    request._cache_detail = "memory"

    inserts = []

    class SingleBatchGenerator:
        def insert(self, prompts, **kwargs):
            inserts.append((prompts, kwargs))
            return [123]

    scheduler = object.__new__(Scheduler)
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = SimpleNamespace(max_num_seqs=1)
    scheduler.batch_generator = SingleBatchGenerator()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = None
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._long_repetition_context = False
    scheduler._cache_reuse_budget_fraction = lambda: 0.95
    scheduler._cache_merge_memory_multiplier = lambda _cache: 1.0
    scheduler._record_scheduled_cache_hit = lambda _request: None
    scheduler._release_unusable_paged_hit = lambda _request: None
    scheduler._validate_cache = lambda _cache: True
    scheduler.total_prompt_tokens = 0

    monkeypatch.setattr(Scheduler, "_ensure_batch_generator", lambda _self, _sp: False)

    scheduled = Scheduler._schedule_waiting(scheduler)

    assert scheduled == [request]
    assert inserts
    prompts, kwargs = inserts[0]
    assert prompts == [[13, 14]]
    assert kwargs["caches"] == [[request.prompt_cache[0]]]
    assert kwargs["all_tokens"] == [[10, 11, 12]]


def test_m3_vl_cache_replay_requires_prefix_past_every_media_token():
    from vmlx_engine.scheduler import _m3_vl_cached_prefix_covers_media_tokens

    model = SimpleNamespace(
        args=SimpleNamespace(image_token_index=200025, video_token_index=200026)
    )
    image_request = SimpleNamespace(
        prompt_token_ids=[10, 200025, 200025, 13, 14],
        cached_tokens=2,
        pixel_values=object(),
        pixel_values_videos=None,
    )

    assert not _m3_vl_cached_prefix_covers_media_tokens(model, image_request)
    image_request.cached_tokens = 3
    assert _m3_vl_cached_prefix_covers_media_tokens(model, image_request)

    video_request = SimpleNamespace(
        prompt_token_ids=[10, 200026, 12],
        cached_tokens=1,
        pixel_values=None,
        pixel_values_videos=object(),
    )
    assert not _m3_vl_cached_prefix_covers_media_tokens(model, video_request)
    video_request.cached_tokens = 2
    assert _m3_vl_cached_prefix_covers_media_tokens(model, video_request)


def test_engine_core_media_request_gets_content_side_key_without_hard_bypass():
    from vmlx_engine.engine_core import EngineCore

    captured = []
    core = object.__new__(EngineCore)
    core._terminal_cleanup_complete = asyncio.Event()
    core._terminal_cleanup_complete.set()
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core.config = SimpleNamespace(stream_interval=1)
    core.scheduler = SimpleNamespace(add_request=captured.append)

    asyncio.run(
        core.add_request(
            [10, 200025, 12],
            images=["data:image/png;base64,QUJD"],
            pixel_values=mx.ones((2, 3), dtype=mx.float32),
            image_grid_thw=mx.array([[1, 1, 1]], dtype=mx.int32),
            prompt_token_ids=[10, 200025, 12],
        )
    )

    assert len(captured) == 1
    request = captured[0]
    assert request._cache_extra_keys["mllm_media"]
    assert request._m3_vl_media_cache_context is True
    assert request._bypass_prefix_cache is False


def test_scheduler_single_batch_m3_media_hit_replays_tail_without_pixels(monkeypatch):
    from collections import deque

    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    request = Request(
        request_id="m3-media-hit",
        prompt=[10, 200025, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.prompt_cache = [_m3_cache(seq=3)]
    request.cached_tokens = 3
    request.remaining_tokens = [13, 14]
    request.pixel_values = mx.ones((2, 3), dtype=mx.float32)
    request.image_grid_thw = mx.array([[1, 1, 1]], dtype=mx.int32)
    request._cache_extra_keys = {"mllm_media": "image-a"}

    inserts = []

    class SingleBatchGenerator:
        def insert(self, prompts, **kwargs):
            inserts.append((prompts, kwargs))
            return [123]

    scheduler = object.__new__(Scheduler)
    scheduler.model = SimpleNamespace(args=SimpleNamespace(image_token_index=200025))
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = SimpleNamespace(max_num_seqs=1)
    scheduler.batch_generator = SingleBatchGenerator()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = None
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._long_repetition_context = False
    scheduler._cache_reuse_budget_fraction = lambda: 0.95
    scheduler._cache_merge_memory_multiplier = lambda _cache: 1.0
    scheduler._record_scheduled_cache_hit = lambda _request: None
    scheduler._release_unusable_paged_hit = lambda _request: None
    scheduler._validate_cache = lambda _cache: True
    scheduler.total_prompt_tokens = 0

    monkeypatch.setattr(Scheduler, "_ensure_batch_generator", lambda _self, _sp: False)

    assert Scheduler._schedule_waiting(scheduler) == [request]
    prompts, kwargs = inserts[0]
    assert prompts == [[13, 14]]
    assert kwargs["caches"] == [[request.prompt_cache[0]]]
    assert kwargs["all_tokens"] == [[10, 200025, 12]]
    assert "pixel_values" not in kwargs
    assert "image_grid_thw" not in kwargs


def test_scheduler_single_batch_m3_partial_media_hit_falls_back_to_atomic_prefill(
    monkeypatch,
):
    from collections import deque

    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler

    request = Request(
        request_id="m3-media-partial",
        prompt=[10, 200025, 200025, 13, 14],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.prompt_cache = [_m3_cache(seq=2)]
    request.cached_tokens = 2
    request.remaining_tokens = [200025, 13, 14]
    request.pixel_values = mx.ones((2, 3), dtype=mx.float32)
    request.image_grid_thw = mx.array([[1, 1, 1]], dtype=mx.int32)
    request._cache_extra_keys = {"mllm_media": "image-a"}

    inserts = []
    released = []

    class SingleBatchGenerator:
        def insert(self, prompts, **kwargs):
            inserts.append((prompts, kwargs))
            return [123]

    scheduler = object.__new__(Scheduler)
    scheduler.model = SimpleNamespace(args=SimpleNamespace(image_token_index=200025))
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = SimpleNamespace(max_num_seqs=1)
    scheduler.batch_generator = SingleBatchGenerator()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.stop_tokens = set()
    scheduler.block_aware_cache = None
    scheduler._kv_cache_bits = 0
    scheduler._is_hybrid = False
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
    scheduler._long_repetition_context = False
    scheduler._cache_reuse_budget_fraction = lambda: 0.95
    scheduler._cache_merge_memory_multiplier = lambda _cache: 1.0
    scheduler._record_scheduled_cache_hit = lambda _request: None
    scheduler._release_unusable_paged_hit = released.append
    scheduler._validate_cache = lambda _cache: True
    scheduler.total_prompt_tokens = 0

    monkeypatch.setattr(Scheduler, "_ensure_batch_generator", lambda _self, _sp: False)

    assert Scheduler._schedule_waiting(scheduler) == [request]
    prompts, kwargs = inserts[0]
    assert released == [request]
    assert prompts == [[10, 200025, 200025, 13, 14]]
    assert kwargs["caches"] is None
    assert kwargs["pixel_values"][0] is request.pixel_values
    assert kwargs["image_grid_thw"][0] is request.image_grid_thw
    assert "all_tokens" not in kwargs
    execution = scheduler._last_cache_execution
    assert execution["prompt_tokens"] == 5
    assert execution["attempted_cached_tokens"] == 2
    assert execution["cached_tokens"] == 0
    assert execution["uncached_prompt_tokens"] == 5
    assert execution["prefill_tokens"] == 5
    assert execution["cache_outcome"] == "discarded"
    assert execution["cache_reuse_applied"] is False
    assert execution["fallback_reason"] == "media_placeholders_outside_cached_prefix"


def test_single_batch_m3_prefills_full_prompt_before_sampling():
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    class _FakeM3Model:
        def __init__(self):
            self.calls = []

        def make_cache(self):
            return [MiniMaxM3SparseCache()]

        def __call__(self, tokens, cache=None, **_kwargs):
            self.calls.append(tokens.tolist()[0])
            return mx.zeros((1, tokens.shape[1], 8), dtype=mx.float32)

    model = _FakeM3Model()
    gen = SingleBatchGenerator(
        model,
        max_tokens=1,
        sampler=lambda _logits: mx.array([3], dtype=mx.int32),
        stream=None,
    )

    gen.insert([[11, 12, 13]])
    prompt_responses, generation_responses = gen.next()

    assert len(prompt_responses) == 1
    assert generation_responses == []
    assert model.calls == [[11, 12, 13]]


def test_single_batch_m3_video_tensors_are_forwarded_once_on_prefill():
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    class _FakeM3Model:
        def __init__(self):
            self.calls = []

        def make_cache(self):
            return [MiniMaxM3SparseCache()]

        def __call__(self, tokens, cache=None, **kwargs):
            self.calls.append((tokens.tolist()[0], dict(kwargs)))
            return mx.zeros((1, tokens.shape[1], 8), dtype=mx.float32)

    model = _FakeM3Model()
    gen = SingleBatchGenerator(
        model,
        max_tokens=1,
        sampler=lambda _logits: mx.array([3], dtype=mx.int32),
        prefill_step_size=2,
        stream=None,
    )
    video_values = mx.zeros((4, 3, 4), dtype=mx.bfloat16)
    video_grid = mx.array([[4, 1, 1]], dtype=mx.int32)

    gen.insert(
        [[11, 200026, 13]],
        pixel_values_videos=[video_values],
        video_grid_thw=[video_grid],
    )
    prompt_responses, generation_responses = gen.next()

    assert len(prompt_responses) == 1
    assert generation_responses == []
    assert len(model.calls) == 1
    tokens, kwargs = model.calls[0]
    assert tokens == [11, 200026, 13]
    assert kwargs["pixel_values"] is None
    assert kwargs["image_grid_thw"] is None
    assert kwargs["pixel_values_videos"].shape == video_values.shape
    assert kwargs["video_grid_thw"].tolist() == [[4, 1, 1]]


def test_engine_core_salts_video_media_cache_without_hard_bypass():
    from vmlx_engine.engine_core import EngineCore

    seen = {}

    class _Scheduler:
        def add_request(self, request):
            seen["request"] = request

    core = object.__new__(EngineCore)
    core.config = SimpleNamespace(stream_interval=1)
    core.scheduler = _Scheduler()
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core._terminal_cleanup_complete = asyncio.Event()
    core._terminal_cleanup_complete.set()

    request_id = asyncio.run(
        core.add_request(
            prompt=[1, 200026, 2],
            prompt_token_ids=[1, 200026, 2],
            pixel_values_videos=object(),
            video_grid_thw=object(),
        )
    )

    request = seen["request"]
    assert request.request_id == request_id
    assert request.prompt_token_ids == [1, 200026, 2]
    assert request._cache_extra_keys["mllm_media"]
    assert request._m3_vl_media_cache_context is True
    assert request._bypass_prefix_cache is False


def test_batched_m3_text_route_forwards_raw_media_sources_to_engine_core():
    from vmlx_engine.engine.batched import BatchedEngine

    calls = []

    class _Engine:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text="ok",
                output_token_ids=[1],
                logprobs=None,
                prompt_tokens=3,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finish_reason="stop",
            )

    engine = object.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _Engine()

    asyncio.run(
        engine.generate(
            prompt="describe",
            images=["data:image/png;base64,IMAGE-A"],
            videos=["data:video/mp4;base64,VIDEO-A"],
            _m3vl_prompt_token_ids=[1, 200025, 2],
            _m3vl_pixel_values=object(),
            _m3vl_image_grid_thw=object(),
        )
    )

    assert calls[0]["images"] == ["data:image/png;base64,IMAGE-A"]
    assert calls[0]["videos"] == ["data:video/mp4;base64,VIDEO-A"]


def test_batched_m3_stream_route_forwards_raw_media_sources_to_engine_core():
    from vmlx_engine.engine.batched import BatchedEngine

    calls = []

    class _Engine:
        async def add_request(self, **kwargs):
            calls.append(kwargs)
            return "m3-stream"

        async def stream_outputs(self, _request_id):
            yield SimpleNamespace(
                output_text="ok",
                new_text="ok",
                logprobs=None,
                prompt_tokens=3,
                completion_tokens=1,
                cached_tokens=0,
                cache_detail="",
                finished=True,
                finish_reason="stop",
            )

    engine = object.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _Engine()

    async def _collect():
        return [
            output
            async for output in engine.stream_generate(
                prompt="describe",
                images=["data:image/png;base64,IMAGE-A"],
                videos=["data:video/mp4;base64,VIDEO-A"],
                _m3vl_prompt_token_ids=[1, 200025, 2],
                _m3vl_pixel_values=object(),
                _m3vl_image_grid_thw=object(),
            )
        ]

    outputs = asyncio.run(_collect())

    assert outputs[-1].text == "ok"
    assert calls[0]["images"] == ["data:image/png;base64,IMAGE-A"]
    assert calls[0]["videos"] == ["data:video/mp4;base64,VIDEO-A"]


def test_single_batch_m3_chunks_long_prompt_before_final_sample():
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    class _FakeM3Model:
        def __init__(self):
            self.calls = []

        def make_cache(self):
            return [MiniMaxM3SparseCache()]

        def __call__(self, tokens, cache=None, **_kwargs):
            self.calls.append(tokens.tolist()[0])
            return mx.zeros((1, tokens.shape[1], 8), dtype=mx.float32)

    model = _FakeM3Model()
    gen = SingleBatchGenerator(
        model,
        max_tokens=1,
        sampler=lambda _logits: mx.array([3], dtype=mx.int32),
        prefill_step_size=3,
        stream=None,
    )

    gen.insert([[1, 2, 3, 4, 5, 6, 7, 8]])
    prompt_responses, generation_responses = gen.next()

    assert len(prompt_responses) == 1
    assert generation_responses == []
    assert model.calls == [[1, 2, 3], [4, 5, 6], [7, 8]]


def test_scheduler_uses_minimax_m3_logits_sampler_for_msa_cache(monkeypatch):
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.request import SamplingParams
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    class _FakeM3Model:
        def make_cache(self):
            return [MiniMaxM3SparseCache()]

    scheduler = object.__new__(Scheduler)
    scheduler.model = _FakeM3Model()
    scheduler.config = SchedulerConfig(max_num_seqs=1)
    # _create_batch_generator now sizes the typed prompt snapshot admission
    # limit from the configured RAM and prompt-disk backends. This fixture
    # intentionally bypasses Scheduler.__init__, so model the no-cache case
    # explicitly instead of relying on missing attributes.
    scheduler.memory_aware_cache = None
    scheduler.disk_cache = None
    scheduler.block_aware_cache = None
    scheduler._long_repetition_context = False
    scheduler._uses_m3_msa_cache = True
    monkeypatch.setattr(Scheduler, "_get_stop_tokens", lambda _self: set())

    gen = scheduler._create_batch_generator(
        SamplingParams(max_tokens=8, temperature=1.0, top_p=0.95),
    )

    assert type(gen).__name__ == "SingleBatchGenerator"
    assert getattr(gen.sampler, "_vmlx_accepts_logits", False)
    assert getattr(gen.sampler, "_vmlx_sampler_kind", "") == "minimax_m3_runtime"


def test_live_cache_validator_rejects_m3_without_idx_keys():
    from vmlx_engine.cache_record_validator import validate_live_cache

    cache = _m3_cache(seq=8)
    cache.idx_keys = None

    ok, reason, _ = validate_live_cache([cache], source="test:m3-missing-idx")

    assert not ok
    assert "idx" in reason.lower()


def test_prompt_disk_cache_round_trips_minimax_m3_idx_keys(tmp_path):
    from vmlx_engine.disk_cache import DiskCacheManager

    tokens = [11, 12, 13, 14, 15]
    mgr = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        assert mgr.store(tokens, [_m3_cache(seq=5)])
    finally:
        mgr.shutdown()

    mgr2 = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        restored = mgr2.fetch(tokens)
        assert restored is not None
        _assert_m3_cache(restored[0], seq=5)
    finally:
        mgr2.shutdown()


def test_prompt_disk_cache_rejects_legacy_kv_when_m3_sparse_required(tmp_path):
    from mlx_lm.models.cache import KVCache
    from vmlx_engine.disk_cache import DiskCacheManager

    tokens = [21, 22, 23, 24]
    dense = KVCache()
    keys = mx.arange(1 * 2 * 4 * 4, dtype=mx.float32).reshape(1, 2, 4, 4)
    dense.state = (keys, keys + 100)

    mgr = DiskCacheManager(cache_dir=str(tmp_path), max_size_gb=1.0)
    try:
        assert mgr.store(tokens, [dense])
    finally:
        mgr.shutdown()

    m3_mgr = DiskCacheManager(
        cache_dir=str(tmp_path),
        max_size_gb=1.0,
        required_cache_class="MiniMaxM3SparseCache",
    )
    try:
        assert m3_mgr.fetch(tokens) is None
        assert m3_mgr.misses == 1
    finally:
        m3_mgr.shutdown()


def test_minimax_m3_reasoning_on_off_auto_map_to_template_modes(monkeypatch):
    import vmlx_engine.model_config_registry as registry
    import vmlx_engine.server as server

    class _Registry:
        def lookup(self, _model_key):
            return SimpleNamespace(
                family_name="minimax_m3",
                model_type="minimax_m3_vl",
            )

    monkeypatch.setattr(registry, "get_model_config_registry", lambda: _Registry())

    cases = [
        (False, "disabled"),
        (True, "enabled"),
        (None, "adaptive"),
    ]
    for enable_thinking, expected in cases:
        ct_kwargs = {}
        request = SimpleNamespace(enable_thinking=enable_thinking)

        server._normalize_minimax_m3_thinking_mode(
            ct_kwargs,
            request,
            "MiniMax-M3-test",
        )

        assert ct_kwargs["thinking_mode"] == expected
        assert "enable_thinking" not in ct_kwargs


def test_minimax_m3_vl_preprocess_maps_reasoning_to_thinking_mode(monkeypatch):
    import numpy as np

    from vmlx_engine.models.minimax_m3 import m3_vl_preprocess

    seen_kwargs = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen_kwargs.append(dict(kwargs))
            return "<image> describe"

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(self, *, text, images, return_tensors):
            return {
                "input_ids": np.array([[1, 200025, 2]], dtype=np.int64),
                "pixel_values": np.zeros((1, 1, 1), dtype=np.float32),
                "image_grid_thw": np.array([[1, 1, 1]], dtype=np.int32),
            }

    monkeypatch.setattr(m3_vl_preprocess, "_get_processor", lambda _path: _Processor())
    monkeypatch.setattr(m3_vl_preprocess, "_load_pil_images", lambda _images: [object()])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    cases = [
        (False, "disabled"),
        (True, "enabled"),
        (None, "adaptive"),
    ]

    for enable_thinking, expected in cases:
        m3_vl_preprocess.preprocess_m3_vl_messages(
            "/tmp/m3",
            messages,
            enable_thinking=enable_thinking,
        )
        assert seen_kwargs[-1]["thinking_mode"] == expected
        assert "enable_thinking" not in seen_kwargs[-1]


def test_minimax_m3_vl_preprocess_builds_native_video_tensors(monkeypatch):
    import numpy as np

    from vmlx_engine.models.minimax_m3 import m3_vl_preprocess

    seen = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["template_kwargs"] = dict(kwargs)
            return "<video> describe"

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(self, **kwargs):
            seen["processor_kwargs"] = dict(kwargs)
            return {
                "input_ids": np.array([[1, 200026, 2]], dtype=np.int64),
                "pixel_values_videos": np.zeros((4, 3, 4), dtype=np.float32),
                "video_grid_thw": np.array([[4, 1, 1]], dtype=np.int32),
            }

    monkeypatch.setattr(m3_vl_preprocess, "_get_processor", lambda _path: _Processor())
    def load_video(_videos, *, metadata_out):
        metadata_out.append({"total_num_frames": 8, "fps": 4.0,
                             "frames_indices": [0, 2, 4, 7]})
        return [[object(), object(), object(), object()]]

    monkeypatch.setattr(
        m3_vl_preprocess,
        "_load_pil_videos",
        load_video,
    )

    result = m3_vl_preprocess.preprocess_m3_vl_messages(
        "/tmp/m3",
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_video",
                        "video_url": "data:video/mp4;base64,AA==",
                    },
                    {"type": "input_text", "text": "describe"},
                ],
            }
        ],
        enable_thinking=False,
    )

    ids, pixel_values, image_grid, video_values, video_grid = result
    assert ids == [1, 200026, 2]
    assert pixel_values is None
    assert image_grid is None
    assert video_values.shape == (4, 3, 4)
    assert video_grid.tolist() == [[4, 1, 1]]
    assert seen["messages"][0]["content"][0] == {"type": "video"}
    assert seen["processor_kwargs"]["videos_kwargs"] == {
        "do_resize": True, "video_metadata": [{"total_num_frames": 8,
        "fps": 4.0, "frames_indices": [0, 2, 4, 7]}],
    }
    assert len(seen["processor_kwargs"]["videos"][0]) == 4


def test_minimax_m3_model_args_preserve_video_token_index():
    from vmlx_engine.models.minimax_m3.minimax_m3 import ModelArgs

    args = ModelArgs.from_dict(
        {
            "model_type": "minimax_m3_vl",
            "text_config": {"hidden_size": 32},
            "video_token_index": 4242,
        }
    )

    assert args.video_token_index == 4242


def test_minimax_m3_reasoning_parser_accepts_fallback_think_tags():
    from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

    parser = MiniMaxM3ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "<think>private planning</think>Visible answer."
    )

    assert reasoning == "private planning"
    assert content == "Visible answer."


def test_minimax_m3_reasoning_parser_streams_fallback_think_tags():
    from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

    parser = MiniMaxM3ReasoningParser()

    first = parser.extract_reasoning_streaming(
        "",
        "<think>private planning",
        "<think>private planning",
    )
    second = parser.extract_reasoning_streaming(
        "<think>private planning",
        "<think>private planning</think>Visible answer.",
        "</think>Visible answer.",
    )

    assert first is not None
    assert first.reasoning == "private planning"
    assert first.content is None
    assert second is not None
    assert second.reasoning is None
    assert second.content == "Visible answer."


def test_minimax_m3_reasoning_parser_prompt_opened_stream_is_reasoning():
    from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

    parser = MiniMaxM3ReasoningParser()
    parser.reset_state(think_in_prompt=True)

    first = parser.extract_reasoning_streaming(
        "",
        "The user asks for arithmetic.",
        "The user asks for arithmetic.",
    )
    second = parser.extract_reasoning_streaming(
        "The user asks for arithmetic.",
        "The user asks for arithmetic.</mm:think>41",
        "</mm:think>41",
    )

    assert first is not None
    assert first.reasoning == "The user asks for arithmetic."
    assert first.content is None
    assert second is not None
    assert second.reasoning is None
    assert second.content == "41"


def test_minimax_m3_reasoning_parser_prompt_opened_complete_is_reasoning():
    from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

    parser = MiniMaxM3ReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("The user asks for arithmetic.")

    assert reasoning == "The user asks for arithmetic."
    assert content is None


def test_minimax_m3_reasoning_parser_splits_captured_prompt_opened_raw_output():
    from vmlx_engine.reasoning.minimax_m3_parser import MiniMaxM3ReasoningParser

    parser = MiniMaxM3ReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning(
        "The user asks for arithmetic.</mm:think>17 + 25 = 31\n\n\\boxed{31}"
    )

    assert reasoning == "The user asks for arithmetic."
    assert content == "17 + 25 = 31\n\n\\boxed{31}"
    assert "</mm:think>" not in content


def test_minimax_m3_residual_think_markup_is_stripped_for_display_and_tools():
    import vmlx_engine.server as server

    raw = "private <mm:think>hidden</mm:think> visible"
    implicit = "hidden reasoning</mm:think>visible answer"

    assert server._strip_residual_think_markup_for_display(raw) == "private visible"
    assert server._strip_think_for_tool_parse(raw) == "private visible"
    assert server._strip_residual_think_markup_for_display(implicit) == "visible answer"
    assert server._strip_think_for_tool_parse(implicit) == "visible answer"


def test_single_cache_idx_lane_grows_in_steps_and_keeps_logical_contract():
    """A2: the raw index lane is backed by step-sized buffers; every contract
    surface (state, update_index return, trim, restore, validator) sees the
    logical extent only.  Covers a step boundary from both sides."""
    import mlx.core as mx
    import numpy as np
    from vmlx_engine.models.minimax_m3.cache import (
        MiniMaxM3SparseCache,
        restore_minimax_m3_sparse,
    )

    c = MiniMaxM3SparseCache()
    step = c.idx_step
    total = 0
    expected = []
    for n in (step - 1, 1, 1, 3):  # 255 | 256 | 257 | 260
        kv = mx.zeros((1, 1, n, 2))
        c.update_and_fetch(kv, kv)
        chunk = mx.arange(total, total + n, dtype=mx.float32)[None, None, :, None]
        chunk = mx.broadcast_to(chunk, (1, 1, n, 4))
        out = c.update_index(chunk)
        total += n
        expected.append(np.asarray(chunk))
        assert out.shape[2] == total == c.offset == c._idx_offset
        assert c.idx_keys.shape[2] >= total
        assert c.idx_keys.shape[2] % step == 0
        np.testing.assert_array_equal(np.asarray(out), np.concatenate(expected, axis=2))
    k, v, idx = c.state
    assert idx.shape[2] == total
    # trim mid-step then keep appending: contract holds, no stale rows leak
    assert c.trim(5) == 5
    assert c._idx_offset == c.offset == total - 5
    kv = mx.zeros((1, 1, 2, 2)); c.update_and_fetch(kv, kv)
    tail = mx.full((1, 1, 2, 4), 9.0)
    out = c.update_index(tail)
    assert out.shape[2] == total - 3
    assert float(out[0, 0, -1, 0]) == 9.0 and float(out[0, 0, -3, 0]) == float(total - 6)
    # restore from the persisted tuple is exact-length and equal
    r = restore_minimax_m3_sparse(*c.state)
    assert r.idx_keys.shape[2] == r.offset == total - 3
    np.testing.assert_array_equal(np.asarray(r.idx_keys), np.asarray(c.state[2]))


def test_validator_rejects_under_and_over_appended_index_lane_and_counts_derived():
    """Spare capacity must not hide an index lane whose INITIALIZED extent
    differs from the K/V offset (under-append: rows past _idx_offset are
    unwritten; over-append: rows past offset are stale)."""
    import mlx.core as mx
    from vmlx_engine.cache_record_validator import validate_live_cache
    from vmlx_engine.models.minimax_m3.cache import (
        MiniMaxM3SparseCache,
        clone_minimax_m3_sparse,
        restore_minimax_m3_sparse,
    )

    c = MiniMaxM3SparseCache()
    kv = mx.zeros((1, 1, 5, 2)); c.update_and_fetch(kv, kv)
    c.update_index(mx.zeros((1, 1, 5, 4)))
    ok, reason, nbytes = validate_live_cache([c])
    assert ok, reason
    # under-append: initialized extent 4 < offset 5 with capacity 256
    c._idx_offset = 4
    ok, reason, _ = validate_live_cache([c])
    assert not ok and "initialized index extent 4 != logical offset 5" in reason
    # over-append: initialized extent 7 > offset 5
    c._idx_offset = 7
    ok, reason, _ = validate_live_cache([c])
    assert not ok and "initialized index extent 7" in reason
    c._idx_offset = 5
    # the append itself fails closed on a desynchronised lane
    d = MiniMaxM3SparseCache()
    kv = mx.zeros((1, 1, 3, 2)); d.update_and_fetch(kv, kv)
    d.update_index(mx.zeros((1, 1, 3, 4)))
    kv = mx.zeros((1, 1, 2, 2)); d.update_and_fetch(kv, kv)  # KV advanced by 2
    import pytest as _pytest
    with _pytest.raises(ValueError, match="desynchronised"):
        d.update_index(mx.zeros((1, 1, 1, 4)))  # index advanced by 1 only
    # restore and clone produce exact-length lanes that validate
    r = restore_minimax_m3_sparse(*c.state)
    ok, reason, _ = validate_live_cache([r]); assert ok, reason
    k = clone_minimax_m3_sparse(c, 4)
    ok, reason, _ = validate_live_cache([k]); assert ok, reason
    # derived bytes are reported by the cache and counted by the validator
    class _Derived:
        nbytes = 4096
        def truncate_to_tokens(self, n): pass
    c.derived["probe"] = _Derived()
    assert c.derived_nbytes == 4096
    ok, reason, nbytes_with = validate_live_cache([c])
    assert ok, reason
    assert nbytes_with == nbytes + 4096


def test_validator_accepts_step_backed_live_lane_and_rejects_short_lane():
    import mlx.core as mx
    from vmlx_engine.cache_record_validator import validate_live_cache
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    c = MiniMaxM3SparseCache()
    kv = mx.zeros((1, 1, 7, 2)); c.update_and_fetch(kv, kv)
    c.update_index(mx.zeros((1, 1, 7, 4)))
    assert c.idx_keys.shape[2] == c.idx_step > c.offset
    ok, reason, _ = validate_live_cache([c])
    assert ok, reason
    c.idx_keys = c.idx_keys[..., :3, :]  # lane shorter than the logical offset
    ok, reason, _ = validate_live_cache([c])
    assert not ok and "idx_keys length" in reason
