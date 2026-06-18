# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MiniMax-M3's three-lane MSA cache.

M3 sparse layers are not plain KVCache layers.  They must carry keys, values,
and idx_keys together; dropping idx_keys silently changes Lightning-Indexer
block selection on cache reuse.
"""

from __future__ import annotations

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
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler._tq_active = False
    scheduler.block_aware_cache = None
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
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler._tq_active = False
    scheduler.block_aware_cache = None
    scheduler._mixed_attention_cache_model = False
    scheduler._uses_m3_msa_cache = True
    scheduler._uses_dsv4_cache = False
    scheduler._uses_zaya_cache = False
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
    assert rederive_calls == [[11, 12, 13, 14, 15, 16]]
    assert getattr(request, "_extracted_cache", None) is clean_rederived
    assert request._extracted_cache is not raw_post_decode
    assert request._extracted_cache is not tail_replay_snapshot
    assert request._extracted_cache_key_tokens == [11, 12, 13, 14, 15, 16]
    assert request._extracted_cache_from_prompt_snapshot is True
    _assert_m3_cache(request._extracted_cache[0], seq=6)


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
            return [_m3_cache(seq=4)], [10, 11, 12, 13]

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
    _assert_m3_cache(request.prompt_cache[0], seq=4)
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
