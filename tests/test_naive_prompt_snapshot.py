"""Native Naive N-1 capture, immutable ring/indexer state, and bounded admission."""
import importlib
from types import SimpleNamespace

import pytest
mx = pytest.importorskip('mlx.core')
pytest.importorskip('mlx_lm')


def make_model():
    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime
    register_naive_n05_flash_runtime()
    rt = importlib.import_module('mlx_lm.models.naive_n05_flash')
    mx.random.seed(928)
    model = rt.Model(rt.ModelArgs(
        vocab_size=32, hidden_size=64, num_hidden_layers=2, intermediate_size=96,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, v_head_dim=32,
        swa_num_attention_heads=2, swa_num_key_value_heads=2,
        swa_head_dim=64, swa_v_head_dim=32, index_n_heads=2, index_head_dim=64,
        partial_rotary_factor=.5, index_top_k=4, sliding_window=4,
        hybrid_layer_pattern=[0, 1], moe_layer_freq=[0, 0],
    ))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def leaves(cache):
    for entry in cache:
        if hasattr(entry, 'caches'):
            yield from leaves(entry.caches)
        else:
            yield entry


@pytest.mark.parametrize('generation_suffix', [0, 3])
def test_actual_generator_snapshot_is_n_minus_one_and_detached(monkeypatch, generation_suffix):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import clone_prompt_cache
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    model = make_model()
    tokens = list(range(1, 12))
    generator = sbg.SingleBatchGenerator(model, max_tokens=5, prefill_step_size=4)
    generator.insert([tokens], max_tokens=[5], gen_prompt_lens=[generation_suffix])
    prompt, _ = generator.next()
    snapshot = prompt[0].prompt_cache_snapshot
    assert snapshot is not None
    assert [c.offset for c in leaves(snapshot)] == [10] * 3
    before = [[x * 1 for x in c.state] for c in leaves(snapshot)]
    metadata = [c.meta_state for c in leaves(snapshot)]
    mx.eval(*[x for pair in before for x in pair])
    # Continue real decode, allowing both KV writes and SWA wrap.
    while generator._request is not None:
        generator.next()
    for c, expected, meta in zip(leaves(snapshot), before, metadata):
        assert c.meta_state == meta
        assert all(bool(mx.array_equal(x, y)) for x, y in zip(c.state, expected))
    assert snapshot[0][1].state[0].dtype == mx.float32
    assert snapshot[0][1].state[1].shape[-1] == 0
    resumed = clone_prompt_cache(snapshot)
    out = model(mx.array([[tokens[-1]]]), cache=resumed)
    mx.eval(out)
    assert int(mx.argmax(out[:, -1, :]).item()) == prompt[0].token


def test_generator_hit_at_native_boundary_does_not_replay_prefix(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import clone_prompt_cache
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    model = make_model()
    tokens = list(range(1, 12))
    cold = sbg.SingleBatchGenerator(model, max_tokens=2, prefill_step_size=4)
    cold.insert([tokens], max_tokens=[2], gen_prompt_lens=[3])
    initial, _ = cold.next()
    snapshot = clone_prompt_cache(initial[0].prompt_cache_snapshot)
    cold.remove([initial[0].uid])
    warm = sbg.SingleBatchGenerator(model, max_tokens=2, prefill_step_size=4)
    warm.insert([tokens[10:]], max_tokens=[2], caches=[snapshot],
                all_tokens=[tokens[:10]], gen_prompt_lens=[3])
    resumed, _ = warm.next()
    assert resumed[0].token == initial[0].token
    assert [c.offset for c in leaves(resumed[0].prompt_cache_snapshot)] == [10, 10, 10]
    assert warm._request.context_tokens[:len(tokens)] == tokens
    warm.remove([resumed[0].uid])


@pytest.mark.parametrize('case', ['backend', 'headroom', 'unknown-headroom', 'trailer', 'other-family'])
def test_snapshot_declines_before_copy(monkeypatch, case):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash import cache_snapshot as snap
    model = make_model()
    c = model.make_cache();mx.eval(model(mx.array([[1, 2, 3]]), cache=c))
    generator = sbg.SingleBatchGenerator(model, prompt_snapshot_max_bytes=0 if case == 'backend' else 1 << 30)
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (1 << 30, 1 << 30) if case == 'headroom' else (0, 0) if case == 'unknown-headroom' else (0, 1 << 30))
    copies = []
    def unexpected_copy(cache):
        copies.append(cache)
        raise AssertionError('copy must not be attempted')
    monkeypatch.setattr(snap, 'clone_prompt_cache', unexpected_copy)
    if case == 'other-family':model.model_type='unqualified-family'
    req = SimpleNamespace(cache=c, gen_prompt_len=-1 if case == 'trailer' else 0)
    assert generator._clone_naive_prompt_snapshot(req) is None
    assert copies == []
    if case == 'backend':assert generator.prompt_snapshot_oversize_skips == 1
    if case in ('headroom', 'unknown-headroom'):assert generator.prompt_snapshot_headroom_skips == 1


def test_rejects_mismatched_native_boundaries():
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import snapshot_size
    model = make_model();c=model.make_cache();mx.eval(model(mx.array([[1,2,3]]),cache=c))
    c[0][1].offset -= 1
    with pytest.raises(ValueError, match='unequal boundaries'):
        snapshot_size(c)


def test_rejects_snapshot_bound_to_wrong_token_key():
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import snapshot_size
    model=make_model();c=model.make_cache();mx.eval(model(mx.array([[1,2,3]]),cache=c))
    with pytest.raises(ValueError,match='token key'):
        snapshot_size(c,expected_tokens=2)
    assert snapshot_size(c,expected_tokens=3)>0


def test_scheduler_stores_snapshot_with_full_mixed_swa_key(tmp_path, monkeypatch):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.utils import single_batch_generator as sbg

    class Tokenizer:
        clean_up_tokenization_spaces = False
        def decode(self, tokens):
            return ''.join(chr(65 + int(t)) for t in tokens)

    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    scheduler = Scheduler(make_model(), Tokenizer(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=4, enable_prefix_cache=True,
        use_paged_cache=False, enable_block_disk_cache=True,
        block_disk_cache_dir=str(tmp_path), paged_cache_block_size=4,
        max_cache_blocks=32,
    ))
    assert scheduler._mixed_attention_cache_model
    replays = []
    def unexpected_replay(*args, **kwargs):
        replays.append(args)
        raise AssertionError('a captured native boundary must not be re-prefilled')
    monkeypatch.setattr(scheduler, '_prefill_for_prompt_only_cache', unexpected_replay)
    tokens = list(range(1, 12))
    request = Request(request_id='snapshot-key', prompt=tokens,
                      sampling_params=SamplingParams(max_tokens=2, temperature=0))
    # Same source weights and prefill partitions isolate storage from shape-dependent math.
    reference = scheduler.model.make_cache()
    for chunk in (tokens[:4], tokens[4:8], tokens[8:10]):
        scheduler.model(mx.array([chunk]), cache=reference)
        mx.eval(*[array for leaf in leaves(reference) for array in leaf.state])
    request.prompt_token_ids=tokens
    request.num_prompt_tokens=len(tokens)
    request._gen_prompt_len=3
    try:
        scheduler.add_request(request)
        for _ in range(5):
            result=scheduler.step()
            if any(o.finished for o in result.outputs):break
        else:pytest.fail('request did not finish')
        assert replays == []
        # Terminal storage now retains all consumed output tokens, while the
        # separate N-1 checkpoint still serves an exact prompt repeat below.
        assert request._extracted_cache_key_tokens == tokens + request.output_token_ids[:-1]
        terminal_key = request._extracted_cache_key_tokens
        terminal_table, terminal_tail = scheduler.block_aware_cache.fetch_cache(
            'terminal-refault', terminal_key + [12])
        assert terminal_table is not None and terminal_table.num_tokens == len(terminal_key)
        assert terminal_tail == [12]
        terminal_restored = scheduler.block_aware_cache.reconstruct_cache(terminal_table)
        assert terminal_restored is not None
        terminal_reference = scheduler.model.make_cache()
        for chunk in (tokens[:4], tokens[4:8], tokens[8:10]):
            mx.eval(scheduler.model(mx.array([chunk]), cache=terminal_reference))
        for token in [tokens[-1]] + request.output_token_ids[:-1]:
            mx.eval(scheduler.model(mx.array([[token]]), cache=terminal_reference))
        for actual, expected in zip(leaves(terminal_restored), leaves(terminal_reference)):
            assert actual.meta_state == expected.meta_state
            assert all(bool(mx.array_equal(a, b)) for a, b in zip(actual.state, expected.state))
        a = scheduler.model(mx.array([[12]]), cache=terminal_restored)
        b = scheduler.model(mx.array([[12]]), cache=terminal_reference)
        mx.eval(a, b)
        assert bool(mx.array_equal(a, b))
        disk_reads = []
        disk = scheduler.block_aware_cache.paged_cache._disk_store
        read = disk.read_block_for_reconstruction
        def observed_read(*args, **kwargs):
            disk_reads.append(args[0])
            return read(*args, **kwargs)
        monkeypatch.setattr(disk, 'read_block_for_reconstruction', observed_read)
        table, remainder=scheduler.block_aware_cache.fetch_cache('snapshot-refault', tokens)
        assert table is not None and table.num_tokens == len(tokens)-1
        assert remainder == tokens[-1:]
        restored=scheduler.block_aware_cache.reconstruct_cache(table)
        assert restored is not None
        assert disk_reads, 'numerical comparison must exercise real L2 reads'
        assert [c.offset for c in leaves(restored)] == [10,10,10]
        # Teacher-force enough suffix tokens to wrap SWA after the restored boundary.
        for token in tokens[-1:] + [12, 13, 14, 15, 16]:
            expected = scheduler.model(mx.array([[token]]), cache=reference)
            actual = scheduler.model(mx.array([[token]]), cache=restored)
            mx.eval(expected, actual)
            assert bool(mx.array_equal(expected, actual))
        assert [c.offset for c in leaves(restored)] == [16, 16, 16]
        assert restored[0][1].state[0].dtype == mx.float32
        assert restored[0][1].state[1].shape[-1] == 0
    finally:
        scheduler.shutdown()


def test_prefill_realizes_each_native_chunk_on_owned_stream(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    cache = model.make_cache()
    req = SimpleNamespace(cache=cache, context_tokens=[], logits_processors=[],
                          pixel_values=None, pixel_values_videos=None, dsv41_vl_images=None)
    real_eval = mx.eval
    boundaries = []
    def observed_eval(*values):
        assert mx.default_stream(mx.default_device()) == generator._stream
        boundaries.append([c.offset for c in leaves(cache)])
        return real_eval(*values)
    monkeypatch.setattr(mx, 'eval', observed_eval)
    generator._prefill(list(range(1, 11)), req)
    assert boundaries == [[4, 4, 4], [8, 8, 8], [10, 10, 10]]
    monkeypatch.setattr(mx, 'eval', real_eval)
    reference = model.make_cache()
    with generator._stream_context():
        for chunk in ([1,2,3,4], [5,6,7,8], [9,10]):
            model(mx.array([chunk]), cache=reference)
        real_eval([c.state for c in reference])
    for actual, expected in zip(leaves(cache), leaves(reference)):
        assert actual.meta_state == expected.meta_state
        assert all(bool(mx.array_equal(x, y)) for x, y in zip(actual.state, expected.state))


def test_prefill_materialization_failure_stops_before_next_chunk(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    cache = model.make_cache()
    req = SimpleNamespace(cache=cache, context_tokens=[], logits_processors=[],
                          pixel_values=None, pixel_values_videos=None, dsv41_vl_images=None)
    def fail(*values):
        raise RuntimeError('native cache evaluation failed')
    monkeypatch.setattr(mx, 'eval', fail)
    with pytest.raises(RuntimeError, match='native cache evaluation failed'):
        generator._prefill(list(range(1, 11)), req)
    assert [c.offset for c in leaves(cache)] == [4, 4, 4]
    assert req.context_tokens == []


def test_scheduler_serializes_concurrent_naive_requests(caplog):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig
    from vmlx_engine.request import Request, SamplingParams

    class Tokenizer:
        clean_up_tokenization_spaces = False
        def decode(self, tokens):
            return ''.join(chr(65 + int(t)) for t in tokens)

    scheduler = Scheduler(make_model(), Tokenizer(), SchedulerConfig(
        max_num_seqs=2, prefill_batch_size=2, completion_batch_size=2,
        prefill_step_size=4, enable_prefix_cache=False,
    ))
    try:
        assert (scheduler.config.max_num_seqs,
                scheduler.config.prefill_batch_size,
                scheduler.config.completion_batch_size) == (1, 1, 1)
        expected = {'native-a', 'native-b'}
        for i, request_id in enumerate(sorted(expected)):
            tokens = list(range(1, 12 + i))
            request = Request(request_id=request_id, prompt=tokens,
                              sampling_params=SamplingParams(max_tokens=2, temperature=0))
            request.prompt_token_ids = tokens
            request.num_prompt_tokens = len(tokens)
            scheduler.add_request(request)
        finished = set()
        for _ in range(12):
            for output in scheduler.step().outputs:
                if output.finished:
                    assert output.finish_reason != 'error'
                    finished.add(output.request_id)
            if finished == expected:
                break
        assert finished == expected
        assert not any('recovering with cache clear' in rec.message for rec in caplog.records)
    finally:
        scheduler.shutdown()


def test_native_family_detection_preserves_explicit_config_precedence():
    from vmlx_engine.scheduler import Scheduler

    native = SimpleNamespace(args=SimpleNamespace(model_type="naive_n05_flash"))
    detect = Scheduler._detect_model_type_for_runtime
    assert detect(native) == "naive_n05_flash"
    assert detect(SimpleNamespace(model=native)) == "naive_n05_flash"
    assert detect(SimpleNamespace(model_type="naive_n05_flash")) == "naive_n05_flash"
    assert detect(SimpleNamespace(config={"model_type": "explicit"}, model=native)) == "explicit"
    assert detect(SimpleNamespace(args=SimpleNamespace(model_type="fallback"),
                                  model=SimpleNamespace(config={"model_type": "inner"}))) == "inner"


@pytest.mark.parametrize("bits", [4, 8])
def test_storage_quantization_preserves_native_search_index(bits, tmp_path):
    from mlx_lm.models.cache import KVCache, CacheList, QuantizedKVCache
    from vmlx_engine.scheduler import Scheduler
    from vmlx_engine.models.naive_n05_flash.register import register_naive_n05_flash_runtime
    register_naive_n05_flash_runtime()
    _fp8_round = importlib.import_module("mlx_lm.models.naive_n05_flash")._fp8_round

    mx.random.seed(928)
    attention, indexer = KVCache(), KVCache()
    attention.update_and_fetch(
        mx.random.normal((1, 4, 11, 192)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 11, 128)).astype(mx.bfloat16))
    indexer.update_and_fetch(_fp8_round(mx.random.normal((1, 1, 11, 128))),
                             mx.zeros((1, 1, 11, 0), dtype=mx.float32))
    mx.eval(attention.state, indexer.state)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_cache_bits = bits
    scheduler._kv_cache_group_size = 64
    scheduler._model_type_for_runtime = "naive_n05_flash"
    scheduler.block_aware_cache = object()
    stored = scheduler._quantize_cache_for_storage([CacheList(attention, indexer)])
    assert isinstance(stored[0][0], QuantizedKVCache)
    assert stored[0][1] is indexer
    restored = scheduler._dequantize_cache_for_use(stored)
    mx.eval([c.state for c in restored])
    assert restored[0][1].keys.dtype == mx.float32
    assert restored[0][1].values.shape[-1] == 0
    assert restored[0][1].offset == 11
    assert bool(mx.array_equal(restored[0][1].keys, indexer.keys))

    # Reopen the real SSD store: no in-memory payload can satisfy this read.
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    model = SimpleNamespace(args=SimpleNamespace(num_key_value_heads=4),
                            cache_list_head_counts=((4, 1),))
    tokens = list(range(11))
    writer = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        manager = PagedCacheManager(block_size=4, max_blocks=32,
                                    disk_store=writer, disk_only=True)
        prefix = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
        state = scheduler._extract_cache_states(stored)
        table = prefix.store_cache("quantized-writer", tokens, state)
        assert table is not None
        hashes = [manager.allocated_blocks[i].block_hash for i in table.block_ids]
        assert writer.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
    finally:
        writer.shutdown()
    reader = BlockDiskStore(str(tmp_path), max_size_gb=0)
    try:
        manager = PagedCacheManager(block_size=4, max_blocks=32,
                                    disk_store=reader, disk_only=True)
        prefix = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
        table, suffix = prefix.fetch_cache("quantized-reader", tokens + [99])
        assert table is not None and table.num_tokens == 11 and suffix == [99]
        rebuilt = prefix.reconstruct_cache(table)
        assert rebuilt is not None
        assert isinstance(rebuilt[0][0], QuantizedKVCache)
        disk_restored = scheduler._dequantize_cache_for_use(rebuilt)
        for actual, expected in zip(disk_restored[0].caches, restored[0].caches):
            assert actual.offset == expected.offset == 11
            for a, b in zip(actual.state, expected.state):
                assert a.dtype == b.dtype
                assert bool(mx.array_equal(a, b))
    finally:
        reader.shutdown()


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_naive_storage_status_reports_native_indexer_policy(bits):
    from vmlx_engine.server import _native_cache_status

    scheduler = SimpleNamespace(
        _model_type_for_runtime="naive_n05_flash",
        _mixed_attention_cache_model=True,
        _kv_cache_bits=bits,
        _kv_cache_group_size=64,
    )
    policy = _native_cache_status(scheduler)["storage_quantization"]
    assert policy["enabled"] is bool(bits)
    assert policy["applies_to"] == "full_attention_kv_only"
    assert policy["sliding_window_policy"] == "native_rotating_kv_state"
    assert policy["indexer_policy"] == "preserve_native_fp32_key_only_state"


@pytest.mark.parametrize("bits", [4, 8])
def test_explicit_naive_quantized_scheduler_ssd_continuation(tmp_path, monkeypatch, bits):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.utils import single_batch_generator as sbg

    class Tokenizer:
        clean_up_tokenization_spaces = False
        def decode(self, tokens):
            return "".join(chr(65 + int(t)) for t in tokens)

    monkeypatch.setattr(sbg, "get_effective_metal_working_set_bytes",
                        lambda mx: (0, 64 << 30))
    scheduler = Scheduler(make_model(), Tokenizer(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=4, enable_prefix_cache=True,
        use_paged_cache=False, enable_block_disk_cache=True,
        block_disk_cache_dir=str(tmp_path), paged_cache_block_size=4,
        max_cache_blocks=32, kv_cache_quantization=f"q{bits}",
        kv_cache_quantization_explicit=True, kv_cache_group_size=32,
    ))
    try:
        assert scheduler._kv_cache_bits == bits
        tokens = list(range(1, 12))
        reference = scheduler.model.make_cache()
        for chunk in (tokens[:4], tokens[4:8], tokens[8:10]):
            scheduler.model(mx.array([chunk]), cache=reference)
            mx.eval(*[a for leaf in leaves(reference) for a in leaf.state])
        reference = scheduler._dequantize_cache_for_use(
            scheduler._quantize_cache_for_storage(reference))
        request = Request(request_id="quantized-native", prompt=tokens,
                          sampling_params=SamplingParams(max_tokens=2, temperature=0))
        request.prompt_token_ids = tokens
        request.num_prompt_tokens = len(tokens)
        scheduler.add_request(request)
        for _ in range(5):
            result = scheduler.step()
            if any(o.finished for o in result.outputs):
                break
        else:
            pytest.fail("request did not finish")
        table, suffix = scheduler.block_aware_cache.fetch_cache("quantized-refault", tokens)
        assert table is not None and table.num_tokens == 10 and suffix == tokens[-1:]
        restored = scheduler._dequantize_cache_for_use(
            scheduler.block_aware_cache.reconstruct_cache(table))
        assert restored[0][1].keys.dtype == mx.float32
        for token in tokens[-1:] + [12, 13, 14, 15, 16]:
            expected = scheduler.model(mx.array([[token]]), cache=reference)
            actual = scheduler.model(mx.array([[token]]), cache=restored)
            mx.eval(expected, actual)
            assert bool(mx.array_equal(expected, actual))
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("length_limited", [False, True])
def test_consumed_terminal_key_matches_actual_generator(monkeypatch, length_limited):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import terminal_cache_key
    monkeypatch.setattr(sbg, "get_effective_metal_working_set_bytes", lambda mx: (0, 64 << 30))
    model = make_model()
    tokens = list(range(1, 12))
    generator = sbg.SingleBatchGenerator(
        model, max_tokens=2, prefill_step_size=4,
        sampler=lambda logits: mx.array([31]),
        stop_tokens=[] if length_limited else [31],
    )
    generator.insert([tokens], max_tokens=[2])
    final = None
    while generator._request is not None or final is None:
        prompt, generation = generator.next()
        for response in prompt + generation:
            final = response
        if final is not None and final.finish_reason is not None:
            break
    key = terminal_cache_key(final.prompt_cache, final.all_tokens, tokens)
    assert len(key) == len(tokens) + 1
    assert len(final.all_tokens) - len(key) == int(length_limited)
    assert all(c.offset == len(key) for c in leaves(final.prompt_cache))
    with pytest.raises(ValueError):
        terminal_cache_key(final.prompt_cache, final.all_tokens, [30] + tokens[1:])


def test_terminal_scheduler_ssd_retains_generated_tokens(tmp_path, monkeypatch):
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.utils import single_batch_generator as sbg
    class Tokenizer:
        clean_up_tokenization_spaces = False
        def decode(self, tokens):
            return "".join(chr(65 + int(t)) for t in tokens)
    monkeypatch.setattr(sbg, "get_effective_metal_working_set_bytes", lambda mx: (0, 64 << 30))
    scheduler = Scheduler(make_model(), Tokenizer(), SchedulerConfig(
        max_num_seqs=1, prefill_step_size=4, enable_prefix_cache=True,
        use_paged_cache=False, enable_block_disk_cache=True,
        block_disk_cache_dir=str(tmp_path), paged_cache_block_size=4,
        max_cache_blocks=32,
    ))
    try:
        tokens = list(range(1, 12))
        request = Request(request_id="terminal-native", prompt=tokens,
                          sampling_params=SamplingParams(max_tokens=3, temperature=0))
        request.prompt_token_ids = tokens
        request.num_prompt_tokens = len(tokens)
        scheduler.add_request(request)
        for _ in range(6):
            if any(o.finished for o in scheduler.step().outputs):
                break
        else:
            pytest.fail("request did not finish")
        repeated, remaining = scheduler.block_aware_cache.fetch_cache("prompt-branch", tokens)
        assert repeated is not None and repeated.num_tokens == len(tokens) - 1
        assert remaining == tokens[-1:]
        scheduler.block_aware_cache.release_cache("prompt-branch")
        key = request._extracted_cache_key_tokens
        assert len(key) == len(tokens) + 2
        table, suffix = scheduler.block_aware_cache.fetch_cache("terminal-refault", key + [17])
        assert table is not None and table.num_tokens == len(key)
        assert suffix == [17]
        restored = scheduler.block_aware_cache.reconstruct_cache(table)
        assert all(c.offset == len(key) for c in leaves(restored))
        reference = scheduler.model.make_cache()
        # Match original prompt chunking followed by autoregressive single tokens.
        for chunk in (tokens[:4], tokens[4:8], tokens[8:10], tokens[10:]):
            mx.eval(scheduler.model(mx.array([chunk]), cache=reference))
        for token in key[len(tokens):] + [17, 18, 19]:
            expected = scheduler.model(mx.array([[token]]), cache=reference)
            mx.eval(expected)
            if reference[0][0].offset > len(key):
                actual = scheduler.model(mx.array([[token]]), cache=restored)
                mx.eval(actual)
                assert bool(mx.array_equal(expected, actual))
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("numpy_path", [False, True])
def test_terminal_ring_preserves_physical_state_and_next_logits(numpy_path):
    import numpy as np
    from vmlx_engine.prefix_cache import _rotating_terminal_window
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import clone_prompt_cache
    model = make_model()
    native = model.make_cache()
    for token in range(1, 12):
        mx.eval(model(mx.array([[token]]), cache=native))
    restored = clone_prompt_cache(native)
    ring = restored[1]
    keys, values = ring.state
    dtype = keys.dtype
    if numpy_path:
        keys, values = np.array(keys.astype(mx.float32)), np.array(values.astype(mx.float32))
    k, v, size, keep, offset, idx = _rotating_terminal_window(
        keys, values, ring.meta_state, expected_offset=11,
        concatenate=np.concatenate if numpy_path else mx.concatenate,
        preserve_bounded_ring=True,
    )
    ring.state = [mx.array(k).astype(dtype), mx.array(v).astype(dtype)]
    ring.meta_state = tuple(map(str, (keep, size, offset, idx)))
    assert ring.meta_state == native[1].meta_state
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(ring.state, native[1].state))
    for token in (12, 13, 14):
        expected = model(mx.array([[token]]), cache=native)
        actual = model(mx.array([[token]]), cache=restored)
        mx.eval(expected, actual)
        assert bool(mx.array_equal(expected, actual))

def test_native_mixed_state_size_eviction_and_republication(tmp_path):
    """Real aggregate budget eviction cannot leave a restorable stale chain."""
    from vmlx_engine.scheduler import Scheduler
    from vmlx_engine.block_disk_store import BlockDiskStore
    from vmlx_engine.paged_cache import PagedCacheManager
    from vmlx_engine.prefix_cache import BlockAwarePrefixCache
    from vmlx_engine.global_disk_cache_budget import get_global_disk_cache_budget

    model = make_model()
    cache = model.make_cache()
    tokens = list(range(1, 12))
    for part in (tokens[:4], tokens[4:8], tokens[8:10], tokens[10:]):
        mx.eval(model(mx.array([part]), cache=cache))
    extractor = Scheduler.__new__(Scheduler)
    payload = extractor._extract_cache_states(cache)
    store = BlockDiskStore(str(tmp_path), max_size_gb=0)
    manager = PagedCacheManager(block_size=4, max_blocks=32, disk_store=store, disk_only=True)
    prefix = BlockAwarePrefixCache(model=model, paged_cache_manager=manager)
    pressure = None
    try:
        def publish(request_id):
            table = prefix.store_cache(request_id, tokens, payload)
            assert table is not None
            hashes = [manager.allocated_blocks[i].block_hash for i in table.block_ids]
            assert store.wait_for_blocks(hashes, timeout=5.0) == set(hashes)
            # The next scheduler step releases SSD-only write-behind RAM copies.
            prefix.settle_write_behind(wait=True)
            prefix.release_cache(request_id)

        publish('before-pressure')
        pressure = get_global_disk_cache_budget(store.global_cache_root, 1)
        result = pressure.enforce(force=True)
        assert result.evicted_entries > 0
        table, remaining = prefix.fetch_cache('evicted-read', tokens + [12])
        assert table is None or prefix.reconstruct_cache(table) is None
        prefix.release_cache('evicted-read')
        pressure.close()
        pressure = None
        publish('after-pressure')
        table, remaining = prefix.fetch_cache('recovered-read', tokens + [12])
        assert table is not None and table.num_tokens == len(tokens)
        assert remaining == [12]
        restored = prefix.reconstruct_cache(table)
        assert restored is not None
        for actual, expected in zip(leaves(restored), leaves(cache)):
            assert actual.meta_state == expected.meta_state
            assert all(bool(mx.array_equal(a, b)) for a, b in zip(actual.state, expected.state))
        expected = model(mx.array([[12]]), cache=cache)
        actual = model(mx.array([[12]]), cache=restored)
        mx.eval(expected, actual)
        assert bool(mx.array_equal(expected, actual))
        prefix.release_cache('recovered-read')
    finally:
        if pressure is not None:
            pressure.close()
        store.shutdown()


def test_oversized_chronological_snapshot_compacts_without_changing_continuation():
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import (
        clone_prompt_cache, snapshot_size,
    )
    model = make_model()
    native = model.make_cache()
    mx.eval(model(mx.array([list(range(1, 11))]), cache=native))
    assert native[1].state[0].shape[2] > native[1].max_size
    original = [[a * 1 for a in c.state] for c in leaves(native)]
    metadata = [c.meta_state for c in leaves(native)]
    mx.eval(*[a for state in original for a in state])
    full_bytes = sum(a.nbytes for c in leaves(native) for a in c.state)
    estimate = snapshot_size(native, expected_tokens=10)
    compact = clone_prompt_cache(native)
    assert estimate < full_bytes
    assert estimate == sum(a.nbytes for c in leaves(compact) for a in c.state)
    assert compact[1].state[0].shape[2] == native[1].max_size
    assert compact[1].meta_state == ('0', '4', '10', '4')
    assert compact[0][1].state[0].dtype == mx.float32
    assert compact[0][1].state[1].shape[-1] == 0
    for c, arrays, meta in zip(leaves(native), original, metadata):
        assert c.meta_state == meta
        assert all(bool(mx.array_equal(a, b)) for a, b in zip(c.state, arrays))
    for tokens in ([11], [12, 13]):
        before = [[a * 1 for a in c.state] for c in leaves(native)]
        mx.eval(*[a for state in before for a in state])
        actual = model(mx.array([tokens]), cache=compact)
        mx.eval(actual)
        # Updating the detached snapshot must not write into the live source.
        for c, arrays in zip(leaves(native), before):
            assert all(bool(mx.array_equal(a, b)) for a, b in zip(c.state, arrays))
        expected = model(mx.array([tokens]), cache=native)
        mx.eval(expected)
        assert bool(mx.array_equal(actual, expected))
        for a, b in zip(leaves(compact), leaves(native)):
            assert a.meta_state == b.meta_state
            assert all(bool(mx.array_equal(x, y)) for x, y in zip(a.state, b.state))


@pytest.mark.parametrize('budget_kind', ['backend', 'headroom'])
def test_compact_snapshot_admission_uses_planned_copy_bytes(monkeypatch, budget_kind):
    from vmlx_engine.utils import single_batch_generator as sbg
    from vmlx_engine.utils import memory_limits
    from vmlx_engine.models.naive_n05_flash import cache_snapshot as snap
    model = make_model()
    native = model.make_cache()
    mx.eval(model(mx.array([list(range(1, 11))]), cache=native))
    estimated = snap.snapshot_size(native)
    full_bytes = sum(a.nbytes for c in leaves(native) for a in c.state)
    assert estimated < full_bytes
    between = (estimated + full_bytes) // 2
    monkeypatch.setattr(memory_limits, 'get_metal_ws_guard_threshold', lambda: 100)
    maximum = between if budget_kind == 'headroom' else 1 << 30
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, maximum))
    generator = sbg.SingleBatchGenerator(
        model, prompt_snapshot_max_bytes=between if budget_kind == 'backend' else 1 << 30,
    )
    req = SimpleNamespace(cache=native, gen_prompt_len=0)
    result = generator._clone_naive_prompt_snapshot(req)
    assert result is not None
    assert generator.prompt_snapshot_last_estimated_bytes == estimated
    assert sum(a.nbytes for c in leaves(result) for a in c.state) == estimated
    if budget_kind == 'backend':
        generator.prompt_snapshot_max_bytes = estimated - 1
    else:
        maximum = estimated - 1
    monkeypatch.setattr(snap, 'clone_prompt_cache', lambda cache: pytest.fail('copy before admission'))
    assert generator._clone_naive_prompt_snapshot(req) is None
    assert generator.prompt_snapshot_last_reason == ('backend_budget' if budget_kind == 'backend' else 'metal_headroom')


@pytest.mark.parametrize('case', ['keep', 'nonchronological', 'bounded_ring'])
def test_noneligible_rotating_snapshots_preserve_native_state(case):
    from mlx_lm.models.cache import RotatingKVCache
    from vmlx_engine.models.naive_n05_flash.cache_snapshot import (
        clone_prompt_cache, snapshot_size,
    )
    cache = RotatingKVCache(max_size=4, keep=1 if case == 'keep' else 0)
    if case == 'bounded_ring':
        for i in range(7):
            cache.update_and_fetch(mx.full((1, 2, 1, 8), i), mx.full((1, 2, 1, 4), i))
    else:
        cache.update_and_fetch(mx.ones((1, 2, 8, 8)), mx.ones((1, 2, 8, 4)))
        if case == 'nonchronological':
            # Unsupported oversized insertion state is copied, not reinterpreted.
            cache._idx = 3
    mx.eval(cache.state)
    copied = clone_prompt_cache([cache])[0]
    assert snapshot_size([cache]) == sum(a.nbytes for a in cache.state)
    assert copied.meta_state == cache.meta_state
    assert all(bool(mx.array_equal(a, b)) for a, b in zip(copied.state, cache.state))


def test_naive_prefill_receipt_ends_before_sampler_and_survives_terminal(monkeypatch):
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    monkeypatch.setattr(sbg, 'get_effective_metal_working_set_bytes', lambda mx: (0, 64 << 30))
    clock = [1000.0]
    monkeypatch.setattr(sbg.time, 'perf_counter', lambda: clock[0])
    original_prefill = generator._prefill
    original_clone = generator._clone_naive_prompt_snapshot
    original_eval = generator._eval_on_stream
    original_sample = generator._sample_from_logits
    initial_evals = []
    sampler_receipts = []

    def prefill(tokens, req):
        original_prefill(tokens, req)
        clock[0] += 11.0

    def clone(req):
        result = original_clone(req)
        clock[0] += 7.0
        return result

    def evaluate(*values):
        result = original_eval(*values)
        req = generator._request
        if req.prefill_started is not None and len(values) == 1 and values[0].shape == (1, 32):
            initial_evals.append(req.uid)
            clock[0] += 3.0
        return result

    def sample(logits, req, input_tokens, **kwargs):
        assert req.prefill_started is None
        assert req.prefill_usage is not None
        sampler_receipts.append((req.uid, dict(req.prefill_usage)))
        clock[0] += 100.0  # Must not enter prefill timing, including lookahead.
        return original_sample(logits, req, input_tokens, **kwargs)

    monkeypatch.setattr(generator, '_prefill', prefill)
    monkeypatch.setattr(generator, '_clone_naive_prompt_snapshot', clone)
    monkeypatch.setattr(generator, '_eval_on_stream', evaluate)
    monkeypatch.setattr(generator, '_sample_from_logits', sample)
    generator.insert([list(range(1, 12))], max_tokens=[3])
    first, _ = generator.next()
    receipt = first[0].prefill_usage
    assert receipt == {'tokens': 11, 'seconds': 21.0, 'scope': 'model_prefill_and_prompt_state'}
    snapshot = first[0].prompt_cache_snapshot
    responses = list(first)
    while generator._request is not None:
        prompt, generated = generator.next()
        responses.extend(prompt + generated)
    assert responses[-1].finish_reason is not None
    assert all(r.prefill_usage == receipt for r in responses)
    assert initial_evals == [first[0].uid]

    # Reuse this generator with an N-1 snapshot and a one-token uncached suffix.
    clock[0] += 500.0  # Idle/restore time cannot contaminate the new request.
    generator.insert([[11]], max_tokens=[1], caches=[snapshot], all_tokens=[list(range(1, 11))])
    warm, _ = generator.next()
    assert warm[0].prefill_usage == {'tokens': 1, 'seconds': 10.0, 'scope': 'model_prefill_and_prompt_state'}
    assert warm[0].finish_reason == 'length'
    assert initial_evals == [first[0].uid, warm[0].uid]
    assert sampler_receipts[-1][1] == warm[0].prefill_usage


def test_every_single_batch_family_emits_prefill_receipt(monkeypatch):
    """The chat footer's pp/s needs usage.vmlx_prefill from every family on this
    generator, not only Naive (Spark, MiniMax, Laguna, ... showed no pp/s at all).
    Cost: one materialization of the final prompt logits, never a decode fence."""
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    model.model_type = 'other-family'
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    original_eval = generator._eval_on_stream
    final_logits_evals = []

    def evaluate(*values):
        if len(values) == 1 and values[0].shape == (1, 32):
            final_logits_evals.append(values[0])
        return original_eval(*values)

    monkeypatch.setattr(generator, '_eval_on_stream', evaluate)
    generator.insert([[1, 2, 3]], max_tokens=[3])
    outputs, _ = generator.next()
    receipt = outputs[0].prefill_usage
    assert receipt['tokens'] == 3 and receipt['seconds'] > 0
    assert receipt['scope'] == 'model_prefill_and_prompt_state'
    responses = list(outputs)
    while generator._request is not None:
        prompt, generated = generator.next()
        responses.extend(prompt + generated)
    assert all(r.prefill_usage == receipt for r in responses)
    assert len(final_logits_evals) == 1


def test_full_prompt_prefill_path_emits_prefill_receipt(monkeypatch):
    """M3 MSA / VL prompts prefill in one full forward (_prefill_full_and_sample)."""
    from vmlx_engine.utils import single_batch_generator as sbg
    model = make_model()
    model.model_type = 'minimax_m3'
    generator = sbg.SingleBatchGenerator(model, prefill_step_size=4)
    monkeypatch.setattr(generator, '_cache_uses_m3_msa', lambda cache: True)
    generator.insert([list(range(1, 10))], max_tokens=[2])
    outputs, _ = generator.next()
    receipt = outputs[0].prefill_usage
    assert receipt['tokens'] == 9 and receipt['seconds'] > 0
    assert receipt['scope'] == 'model_prefill_and_prompt_state'
