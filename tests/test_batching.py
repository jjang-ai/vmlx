# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching system.

These tests verify the scheduler, engine, and request handling
for the vLLM-style continuous batching implementation.
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from vmlx_engine.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)
from vmlx_engine.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingPolicy,
)
from vmlx_engine.paged_cache import BlockTable


class TestRequest:
    """Tests for Request class."""

    def test_request_creation(self):
        """Test basic request creation."""
        params = SamplingParams(max_tokens=100, temperature=0.8)
        request = Request(
            request_id="test-1",
            prompt="Hello, world!",
            sampling_params=params,
        )

        assert request.request_id == "test-1"
        assert request.prompt == "Hello, world!"
        assert request.sampling_params.max_tokens == 100
        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

    def test_request_status_transitions(self):
        """Test request status transitions."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

        request.status = RequestStatus.RUNNING
        assert not request.is_finished()

        request.set_finished(RequestStatus.FINISHED_STOPPED)
        assert request.is_finished()
        assert request.get_finish_reason() == "stop"

    def test_request_output_tokens(self):
        """Test appending output tokens."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3

        assert request.num_output_tokens == 0
        assert request.num_tokens == 3

        request.append_output_token(100)
        request.append_output_token(101)

        assert request.num_output_tokens == 2
        assert request.num_tokens == 5
        assert request.output_token_ids == [100, 101]

    def test_request_comparison(self):
        """Test request comparison for priority queue."""
        req1 = Request(
            request_id="req-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=1.0,
        )
        req2 = Request(
            request_id="req-2",
            prompt="World",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=0.5,
        )
        req3 = Request(
            request_id="req-3",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=2.0,
        )

        # Lower priority value = higher priority
        assert req1 < req2
        # Same priority, earlier arrival = higher priority
        assert req1 < req3


class TestPrefixHitTailAccounting:
    """Prefix hits store N-1 KV state, so exact hits still need a kickoff token."""

    def test_exact_hit_with_generation_suffix_refeeds_last_prompt_token(self):
        remaining, cached_tokens = Scheduler._prefix_hit_tail_and_cached_tokens(
            fetch_tokens=[10, 11, 12, 13],
            remaining=[],
            gen_prompt_suffix=[90, 91],
        )

        assert remaining == [13, 90, 91]
        assert cached_tokens == 3

    def test_exact_hit_without_suffix_counts_n_minus_one_cached_tokens(self):
        remaining, cached_tokens = Scheduler._prefix_hit_tail_and_cached_tokens(
            fetch_tokens=[10, 11, 12, 13],
            remaining=[],
            gen_prompt_suffix=[],
        )

        assert remaining == []
        assert cached_tokens == 3


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_params(self):
        """Test default sampling parameters."""
        params = SamplingParams()

        assert params.max_tokens == 256
        assert params.temperature == 0.0
        assert params.top_p == 1.0
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_custom_params(self):
        """Test custom sampling parameters."""
        params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=["END"],
            stop_token_ids=[1, 2],
        )

        assert params.max_tokens == 100
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.stop == ["END"]
        assert params.stop_token_ids == [1, 2]


class TestRequestOutput:
    """Tests for RequestOutput."""

    def test_output_creation(self):
        """Test output creation."""
        output = RequestOutput(
            request_id="test-1",
            new_token_ids=[100, 101],
            new_text="Hello",
            output_token_ids=[100, 101],
            output_text="Hello",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=2,
        )

        assert output.request_id == "test-1"
        assert output.finished
        assert output.finish_reason == "stop"

        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 12


class TestSchedulerLogprobs:
    """Opt-in logprobs must be extracted on the scheduler worker, not faked later."""

    def test_greedy_scheduler_sampler_is_marked_logits_compatible(self):
        import inspect

        source = inspect.getsource(Scheduler._create_batch_generator)
        assert "_vmlx_accepts_logits" in source
        assert "sampling_params.temperature" in source

    def test_generation_batch_skips_logsumexp_for_greedy_no_logprobs(self):
        import inspect
        from vmlx_engine.utils import mamba_cache

        source = inspect.getsource(mamba_cache._patch_generation_step_sync)
        assert "can_sample_logits" in source
        assert "not any(requested_logprobs)" in source
        assert "sample_sampler(logits[e : e + 1])" in source
        assert "mx.logsumexp(logits" in source

    def test_generation_batch_cpu_tolist_detour_is_model_gated(self):
        import inspect
        from vmlx_engine.utils import mamba_cache

        patch_source = inspect.getsource(mamba_cache._patch_generation_step_sync)
        gate_source = inspect.getsource(mamba_cache._needs_cpu_tolist_detour)
        assert "_needs_cpu_tolist_detour" in patch_source
        assert "with mx.stream(mx.cpu)" in patch_source
        assert "deepseek_v4" in gate_source
        assert "VMLINUX_FORCE_CPU_TOLIST_DETOUR" in gate_source

    def test_generation_batch_keeps_async_lookahead_for_normal_models(self):
        import inspect
        from vmlx_engine.utils import mamba_cache

        source = inspect.getsource(mamba_cache._patch_generation_step_sync)
        assert "mx.async_eval(*async_items)" in source
        assert "mx.eval(*current_items)" in source
        assert "if needs_cpu_detour:" in source
        assert "A per-token synchronize here costs MiniMax/Qwen throughput" in source

    def test_format_token_logprobs_includes_sampled_token_and_top_k(self):
        import mlx.core as mx

        from vmlx_engine.scheduler import _format_token_logprobs_for_output

        logprobs = mx.array([-4.0, -0.3, -1.2, -0.7])
        formatted = _format_token_logprobs_for_output(
            token_id=2,
            logprobs=logprobs,
            top_k=2,
        )

        assert formatted["token_id"] == 2
        assert formatted["logprob"] == pytest.approx(-1.2)
        assert formatted["top_logprobs"] == [
            (1, pytest.approx(-0.3)),
            (3, pytest.approx(-0.7)),
        ]

    def test_format_token_logprobs_none_keeps_default_fast_path_empty(self):
        from vmlx_engine.scheduler import _format_token_logprobs_for_output

        assert _format_token_logprobs_for_output(2, None, 5) is None

    def test_format_token_logprobs_rejects_invalid_token_id(self):
        import mlx.core as mx

        from vmlx_engine.scheduler import _format_token_logprobs_for_output

        logprobs = mx.array([-4.0, -0.3, -1.2, -0.7])
        assert _format_token_logprobs_for_output(-1, logprobs, 2) is None


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        """Test default scheduler config."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 1
        assert config.policy == SchedulingPolicy.FCFS
        assert config.prefill_batch_size == 512
        assert config.completion_batch_size == 512

    def test_custom_config(self):
        """Test custom scheduler config."""
        config = SchedulerConfig(
            max_num_seqs=64,
            policy=SchedulingPolicy.PRIORITY,
            prefill_batch_size=4,
            completion_batch_size=16,
        )

        assert config.max_num_seqs == 64
        assert config.policy == SchedulingPolicy.PRIORITY


class TestSchedulerBasic:
    """Basic tests for Scheduler (without real model)."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    def test_scheduler_creation(self, mock_model, mock_tokenizer):
        """Test scheduler creation."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=10),
        )

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()

    def test_legacy_turboquant_live_cache_clamps_to_single_sequence(self, mock_tokenizer):
        """Old bundled TurboQuantKVCache lacks real multi-seq batch methods."""

        class TurboQuantKVCache:
            pass

        class TQModel:
            def __init__(self):
                def _tq_make_cache():
                    return [TurboQuantKVCache()]

                self.make_cache = _tq_make_cache

        cfg = SchedulerConfig(
            max_num_seqs=4,
            prefill_batch_size=4,
            completion_batch_size=4,
        )

        scheduler = Scheduler(model=TQModel(), tokenizer=mock_tokenizer, config=cfg)

        assert scheduler._tq_active
        assert scheduler.config.max_num_seqs == 1
        assert scheduler.config.prefill_batch_size == 1
        assert scheduler.config.completion_batch_size == 1

    def test_batch_api_turboquant_live_cache_preserves_configured_batching(self, mock_tokenizer):
        """New TurboQuantKVCache declares the real batch API and may batch."""

        class TurboQuantKVCache:
            _vmlx_batch_api = "turboquant_kv_v1"

            def extend(self, other):
                return None

            def filter(self, keep):
                return None

            def extract(self, idx):
                return self

            def prepare(self, *args, **kwargs):
                return None

            def finalize(self):
                return None

        class TQModel:
            def __init__(self):
                def _tq_make_cache():
                    return [TurboQuantKVCache()]

                self.make_cache = _tq_make_cache

        cfg = SchedulerConfig(
            max_num_seqs=4,
            prefill_batch_size=4,
            completion_batch_size=4,
        )

        scheduler = Scheduler(model=TQModel(), tokenizer=mock_tokenizer, config=cfg)

        assert scheduler._tq_active
        assert scheduler.config.max_num_seqs == 4
        assert scheduler.config.prefill_batch_size == 4
        assert scheduler.config.completion_batch_size == 4

    def test_batch_api_turboquant_merge_preserves_turboquant_cache(self):
        """mlx-lm merge patch must use TurboQuantKVCache.extend, not BatchKVCache."""
        import importlib

        from vmlx_engine.utils.mamba_cache import ensure_mamba_support

        class TurboQuantKVCache:
            _vmlx_batch_api = "turboquant_kv_v1"

            def __init__(self, name):
                self.name = name
                self.extended = []

            def extend(self, other):
                self.extended.append(other.name)

            def filter(self, keep):
                return None

            def extract(self, idx):
                return self

            def prepare(self, *args, **kwargs):
                return None

            def finalize(self):
                return None

        ensure_mamba_support()
        gen_module = importlib.import_module("mlx_lm.generate")

        merged = gen_module._merge_caches([
            [TurboQuantKVCache("a")],
            [TurboQuantKVCache("b")],
        ])

        assert type(merged[0]).__name__ == "TurboQuantKVCache"
        assert merged[0].name == "a"
        assert merged[0].extended == ["b"]

    def test_add_request(self, mock_model, mock_tokenizer):
        """Test adding requests to scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello world",
            sampling_params=SamplingParams(max_tokens=10),
        )

        scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 1
        assert scheduler.has_requests()
        assert scheduler.get_request("test-1") is not None

    def test_memory_cache_exact_hit_with_generation_suffix_refeeds_last_key_token(
        self, mock_model, mock_tokenizer
    ):
        """Memory L1 payloads are N-1, even when the key is an exact hit."""

        class _MemoryCache:
            def fetch(self, tokens):
                assert tokens == [10, 11, 12]
                return ["cache"], []

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = _MemoryCache()
        scheduler.prefix_cache = None
        scheduler.disk_cache = None

        request = Request("req-memory-exact", "x", SamplingParams())
        request.prompt_token_ids = [10, 11, 12, 90, 91]
        request.num_prompt_tokens = len(request.prompt_token_ids)
        request._gen_prompt_len = 2

        scheduler.add_request(request)

        assert request.prompt_cache == ["cache"]
        assert request.cached_tokens == 2
        assert request.remaining_tokens == [12, 90, 91]

    def test_prompt_disk_l2_exact_hit_with_generation_suffix_refeeds_last_key_token(
        self, mock_model, mock_tokenizer
    ):
        """Prompt L2 stores full lookup keys but N-1 KV payloads."""

        class _DiskCache:
            _last_fetch_tq_native = False
            _last_fetch_cache_type = "assistant"

            def fetch(self, tokens):
                assert tokens == [10, 11, 12]
                return ["disk-cache"]

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = None
        scheduler.memory_aware_cache = None
        scheduler.prefix_cache = None
        scheduler.disk_cache = _DiskCache()

        request = Request("req-disk-exact", "x", SamplingParams())
        request.prompt_token_ids = [10, 11, 12, 90, 91]
        request.num_prompt_tokens = len(request.prompt_token_ids)
        request._gen_prompt_len = 2

        scheduler.add_request(request)

        assert request.prompt_cache == ["disk-cache"]
        assert request.cached_tokens == 2
        assert request.remaining_tokens == [12, 90, 91]

    def test_add_duplicate_request(self, mock_model, mock_tokenizer):
        """Test adding duplicate request raises error."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)
        assert scheduler.get_num_waiting() == 1

        result = scheduler.abort_request("test-1")

        assert result is True
        assert scheduler.get_num_waiting() == 0
        assert "test-1" in scheduler.finished_req_ids

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test aborting non-existent request."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        result = scheduler.abort_request("nonexistent")
        assert result is False

    def test_get_stats(self, mock_model, mock_tokenizer):
        """Test getting scheduler stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        stats = scheduler.get_stats()

        assert "num_waiting" in stats
        assert "num_running" in stats
        assert "num_requests_processed" in stats
        assert stats["num_waiting"] == 0
        assert stats["num_running"] == 0
        assert stats["cache_reuse_skips"] == 0
        assert stats["cache_reuse_skip_tokens"] == 0
        assert stats["cache_reuse_partial_downgrades"] == 0
        assert stats["cache_reuse_partial_tokens"] == 0
        assert stats["last_cache_reuse_skip"] is None
        assert stats["last_cache_reuse_partial"] is None

    def test_memory_pressure_partially_reuses_paged_cache(
        self, mock_model, mock_tokenizer, monkeypatch
    ):
        """Large L2/paged hits should shrink to fit instead of full-prefilling."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._validate_cache = lambda cache: True

        original = BlockTable(
            request_id="req-partial",
            block_ids=[1, 2, 3, 4],
            num_tokens=256,
        )
        trimmed = BlockTable(
            request_id="req-partial",
            block_ids=[1, 2],
            num_tokens=128,
        )

        class _BlockAwareCache:
            block_size = 64

            def __init__(self):
                self.trim_calls = []

            def trim_block_table(self, request_id, target_tokens):
                self.trim_calls.append((request_id, target_tokens))
                return trimmed

            def reconstruct_cache(self, block_table):
                assert block_table is trimmed
                return ["small-cache"]

            def get_stats(self):
                return {}

        cache = _BlockAwareCache()
        scheduler.block_aware_cache = cache
        request = Request(
            request_id="req-partial",
            prompt="x",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = list(range(320))
        request.block_table = original
        request.cached_tokens = 256
        request.remaining_tokens = request.prompt_token_ids[256:]
        request.shared_prefix_blocks = 4

        cache_to_use, tokens_to_process = scheduler._shrink_paged_cache_for_memory(
            request=request,
            cache_to_use=["large-cache"],
            cache_bytes=1_000,
            available_bytes=1_100,
            multiplier=2.0,
            budget_fraction=1.0,
        )

        assert cache_to_use == ["small-cache"]
        assert tokens_to_process == request.prompt_token_ids[128:]
        assert request.block_table is trimmed
        assert request.cached_tokens == 128
        assert request.shared_prefix_blocks == 2
        assert request.prompt_cache == ["small-cache"]
        assert cache.trim_calls == [("req-partial", 128)]

        stats = scheduler.get_stats()
        assert stats["cache_reuse_partial_downgrades"] == 1
        assert stats["cache_reuse_partial_tokens"] == 128
        assert stats["last_cache_reuse_partial"]["original_cached_tokens"] == 256
        assert stats["last_cache_reuse_partial"]["used_cached_tokens"] == 128
        assert stats["last_cache_reuse_partial"]["tail_tokens"] == 192

    def test_memory_pressure_partial_reuse_records_effective_budget_and_final_cache(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._validate_cache = lambda cache: True

        class _LargeCache:
            pass

        class _SmallCache:
            pass

        small_cache = _SmallCache()
        trimmed = BlockTable(
            request_id="req-partial-budget",
            block_ids=[1],
            num_tokens=64,
        )

        class _BlockAwareCache:
            block_size = 64

            def trim_block_table(self, request_id, target_tokens):
                return trimmed

            def reconstruct_cache(self, block_table):
                return small_cache

            def get_stats(self):
                return {}

        scheduler.block_aware_cache = _BlockAwareCache()
        request = Request(
            request_id="req-partial-budget",
            prompt="x",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = list(range(256))
        request.block_table = BlockTable("req-partial-budget", [1, 2, 3, 4], 256)
        request.cached_tokens = 256
        request.remaining_tokens = request.prompt_token_ids[256:]

        cache_to_use, _tokens_to_process = scheduler._shrink_paged_cache_for_memory(
            request=request,
            cache_to_use=_LargeCache(),
            cache_bytes=256 * 1024 * 1024,
            available_bytes=256 * 1024 * 1024,
            multiplier=2.0,
            budget_fraction=0.5,
        )

        assert cache_to_use is small_cache
        partial = scheduler.get_stats()["last_cache_reuse_partial"]
        assert partial["original_needed_mb"] == 512.0
        assert partial["budget_mb"] == 128.0
        assert partial["used_needed_mb"] == 128.0
        assert partial["used_cache_mb"] == 64.0
        assert partial["cache_type"] == "_SmallCache"

    def test_cache_merge_multiplier_uses_actual_cache_format_not_global_q4_flag(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._kv_cache_bits = 4

        class KVCache:
            pass

        class QuantizedKVCache:
            bits = 4

        full_precision_cache = [KVCache()]
        quantized_cache = [QuantizedKVCache()]

        assert scheduler._cache_reuse_cache_format(full_precision_cache) == (
            "full_precision_kv"
        )
        assert scheduler._cache_merge_memory_multiplier(full_precision_cache) == 2.0
        assert scheduler._cache_reuse_cache_format(quantized_cache) == (
            "quantized_kv"
        )
        assert scheduler._cache_merge_memory_multiplier(quantized_cache) == 5.0

    def test_schedule_keeps_dequantized_cache_hit_when_q4_storage_flag_would_overbudget(
        self, mock_model, mock_tokenizer, monkeypatch
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._kv_cache_bits = 4
        scheduler._validate_cache = lambda cache: True

        class KVCache:
            pass

        prompt_cache = [KVCache()]

        class _BatchGenerator:
            def __init__(self):
                self.inserted_caches = None
                self.inserted_tokens = None
                self.stop_tokens = set()

            def insert(self, tokens, max_tokens, caches=None, **_kwargs):
                self.inserted_tokens = tokens
                self.inserted_caches = caches
                return [42]

        sampling = SamplingParams(max_tokens=4)
        request = Request("req-q4-dequant-fit", "x", sampling)
        request.prompt_token_ids = list(range(1000))
        request.num_prompt_tokens = len(request.prompt_token_ids)
        request.prompt_cache = prompt_cache
        request.cached_tokens = 999
        request.remaining_tokens = [999]

        scheduler.batch_generator = _BatchGenerator()
        scheduler._current_sampler_params = (
            sampling.temperature,
            sampling.top_p,
            sampling.min_p,
            sampling.top_k,
            sampling.repetition_penalty,
        )
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request

        import types
        import vmlx_engine.memory_cache as memory_cache
        import psutil

        mb = 1024 * 1024
        monkeypatch.setattr(
            memory_cache,
            "estimate_kv_cache_memory",
            lambda _cache: 300 * mb,
        )
        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: types.SimpleNamespace(available=800 * mb),
        )

        scheduled = scheduler._schedule_waiting()

        assert scheduled == [request]
        assert scheduler.batch_generator.inserted_caches == [prompt_cache]
        assert scheduler.batch_generator.inserted_tokens == [[999]]
        stats = scheduler.get_stats()
        assert stats["cache_reuse_skips"] == 0
        assert stats["cache_hit_tokens"] == 999

    def test_memory_pressure_partial_reuse_preserves_generation_prompt_suffix(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._validate_cache = lambda cache: True

        trimmed = BlockTable("req-gpl", [1], 64)

        class _BlockAwareCache:
            block_size = 64

            def trim_block_table(self, request_id, target_tokens):
                return trimmed

            def reconstruct_cache(self, block_table):
                return ["small-cache"]

        scheduler.block_aware_cache = _BlockAwareCache()
        request = Request("req-gpl", "x", SamplingParams())
        request.prompt_token_ids = list(range(130))
        request.block_table = BlockTable("req-gpl", [1, 2], 120)
        request.cached_tokens = 120
        request.remaining_tokens = request.prompt_token_ids[120:]
        request._gen_prompt_len = 10

        cache_to_use, tokens_to_process = scheduler._shrink_paged_cache_for_memory(
            request=request,
            cache_to_use=["large-cache"],
            cache_bytes=1_000,
            available_bytes=1_200,
            multiplier=2.0,
            budget_fraction=1.0,
        )

        assert cache_to_use == ["small-cache"]
        assert tokens_to_process == request.prompt_token_ids[64:]
        assert request.remaining_tokens == request.prompt_token_ids[64:]

    def test_memory_pressure_reuses_shorter_dsv4_terminal_composite_prefix(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._uses_dsv4_cache = True
        scheduler._validate_cache = lambda cache: True

        trimmed = BlockTable("req-dsv4", [1, 2], 128)

        class _BlockAwareCache:
            block_size = 64

            def __init__(self):
                self.trim_calls = []

            def trim_block_table_to_terminal_state(self, request_id, target_tokens, tag):
                self.trim_calls.append((request_id, target_tokens, tag))
                return trimmed

            def reconstruct_cache(self, block_table):
                assert block_table is trimmed
                return ["dsv4-composite-cache"]

            def get_stats(self):
                return {}

        cache = _BlockAwareCache()
        scheduler.block_aware_cache = cache
        request = Request("req-dsv4", "x", SamplingParams())
        request.prompt_token_ids = list(range(320))
        request.block_table = BlockTable("req-dsv4", [1, 2, 3, 4], 256)
        request.cached_tokens = 256
        request.remaining_tokens = request.prompt_token_ids[256:]

        cache_to_use, tokens_to_process = scheduler._shrink_paged_cache_for_memory(
            request=request,
            cache_to_use=["large-dsv4-cache"],
            cache_bytes=1_000,
            available_bytes=1_100,
            multiplier=2.0,
            budget_fraction=1.0,
        )

        assert cache.trim_calls == [("req-dsv4", 128, "deepseek_v4")]
        assert cache_to_use == ["dsv4-composite-cache"]
        assert tokens_to_process == request.prompt_token_ids[128:]
        assert request.block_table is trimmed
        assert request.cached_tokens == 128
        assert request.shared_prefix_blocks == 2
        assert request.prompt_cache == ["dsv4-composite-cache"]
        assert request._cache_reuse_contract == "deepseek_v4_composite"
        assert scheduler.get_stats()["last_cache_reuse_partial"]["cache_contract"] == (
            "deepseek_v4_composite"
        )

    def test_memory_pressure_refuses_dsv4_partial_without_terminal_composite(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._uses_dsv4_cache = True

        class _BlockAwareCache:
            block_size = 64

            def trim_block_table_to_terminal_state(self, request_id, target_tokens, tag):
                return None

            def trim_block_table(self, request_id, target_tokens):
                raise AssertionError("DSV4 must not use plain KV block trimming")

        scheduler.block_aware_cache = _BlockAwareCache()
        request = Request("req-dsv4-miss", "x", SamplingParams())
        request.prompt_token_ids = list(range(256))
        request.block_table = BlockTable("req-dsv4-miss", [1, 2, 3, 4], 256)
        request.cached_tokens = 256

        cache_to_use, tokens_to_process = scheduler._shrink_paged_cache_for_memory(
            request=request,
            cache_to_use=["unsafe-dsv4-cache"],
            cache_bytes=1_000,
            available_bytes=600,
            multiplier=2.0,
        )

        assert cache_to_use is None
        assert tokens_to_process is None
        assert request._cache_reuse_contract == "deepseek_v4_composite"
        assert request._cache_reuse_partial_unavailable_reason == (
            "no_terminal_path_dependent_checkpoint"
        )

    def test_cache_reuse_skip_records_partial_reason_and_full_prefill_action(
        self, mock_model, mock_tokenizer, caplog
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = Request("req-skip", "x", SamplingParams())
        request.prompt_token_ids = list(range(1000))
        request.cached_tokens = 512
        request.remaining_tokens = list(range(512, 1000))
        request._cache_reuse_contract = "hybrid_ssm"
        request._cache_reuse_partial_unavailable_reason = (
            "no_block_aligned_ssm_checkpoint"
        )

        caplog.set_level("WARNING", logger="vmlx_engine.scheduler")
        scheduler._record_cache_reuse_skip(
            request=request,
            cache_to_use=["kv-cache"],
            cache_bytes=512 * 1024 * 1024,
            available_bytes=256 * 1024 * 1024,
            needed_bytes=1024 * 1024 * 1024,
            merge_budget_bytes=128 * 1024 * 1024,
            multiplier=2.0,
            budget_fraction=0.5,
        )

        stats = scheduler.get_stats()
        assert stats["cache_reuse_skips"] == 1
        assert stats["cache_reuse_skip_tokens"] == 512
        skip = stats["last_cache_reuse_skip"]
        assert skip["action"] == "full_prefill"
        assert skip["cache_contract"] == "hybrid_ssm"
        assert skip["cached_tokens"] == 512
        assert skip["dropped_cached_tokens"] == 512
        assert skip["remaining_tokens"] == 488
        assert skip["full_prefill_tokens"] == 1000
        assert skip["partial_reuse_available"] is False
        assert skip["partial_reuse_unavailable_reason"] == (
            "no_block_aligned_ssm_checkpoint"
        )
        assert skip["cache_format"] == "unknown"
        assert (
            "cached 512 tokens, contract=hybrid_ssm, "
            "format=unknown, partial_reason=no_block_aligned_ssm_checkpoint"
        ) in caplog.text
        assert "full-prefilling 1000 prompt tokens" in caplog.text

    def test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._is_hybrid = True
        scheduler._uses_dsv4_cache = False
        scheduler._uses_zaya_cache = False
        scheduler._validate_cache = lambda cache: True

        trimmed = BlockTable("req-hybrid", [1, 2], 128)

        class _BlockAwareCache:
            block_size = 64

            def trim_block_table(self, request_id, target_tokens):
                assert target_tokens == 128
                return trimmed

            def reconstruct_cache(self, block_table):
                assert block_table is trimmed
                return ["kv-only"]

            def get_stats(self):
                return {}

        class _SSMCache:
            def fetch_longest_prefix(self, token_ids, max_len):
                assert max_len == 128
                return (128, ["ssm"], True)

        scheduler.block_aware_cache = _BlockAwareCache()
        scheduler._ssm_state_cache = _SSMCache()
        scheduler._finalize_hybrid_paged_cache_on_worker = (
            lambda request, reconstructed: ["kv+ssm"]
        )
        request = Request("req-hybrid", "x", SamplingParams())
        request.prompt_token_ids = list(range(320))
        request.block_table = BlockTable("req-hybrid", [1, 2, 3, 4], 256)
        request.cached_tokens = 256
        request.remaining_tokens = request.prompt_token_ids[256:]
        request._hybrid_ssm_fetch_tokens = list(range(300))

        cache_to_use, tokens_to_process = scheduler._shrink_paged_cache_for_memory(
            request=request,
            cache_to_use=["large-hybrid-cache"],
            cache_bytes=1_000,
            available_bytes=1_100,
            multiplier=2.0,
            budget_fraction=1.0,
        )

        assert cache_to_use == ["kv+ssm"]
        assert tokens_to_process == request.prompt_token_ids[128:]
        assert request.cached_tokens == 128
        assert request.shared_prefix_blocks == 2
        assert request.prompt_cache == ["kv+ssm"]
        assert scheduler.get_stats()["last_cache_reuse_partial"]["cache_contract"] == (
            "hybrid_ssm"
        )

    def test_prompt_disk_l2_hit_backfills_paged_cache_for_partial_reuse(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._extract_cache_states = lambda cache: ["state-dict"]

        class _DiskCache:
            _last_fetch_tq_native = True
            _last_fetch_cache_type = "assistant"

            def __init__(self):
                self.fetch_tokens = None

            def fetch(self, tokens):
                self.fetch_tokens = list(tokens)
                return ["disk-cache"]

        class _PagedCache:
            def __init__(self):
                self.released = []
                self.detached = []

            def release_request_refs(self, block_table):
                self.released.append(block_table)

            def detach_request(self, request_id):
                self.detached.append(request_id)

        class _BlockAwareCache:
            def __init__(self):
                self.fetch_calls = []
                self.store_calls = []
                self.paged_cache = _PagedCache()
                self._request_tables = {}

            def fetch_cache(self, request_id, tokens):
                self.fetch_calls.append((request_id, list(tokens)))
                return None, list(tokens)

            def store_cache(self, request_id, tokens, cache_data, cache_type="assistant"):
                self.store_calls.append(
                    (request_id, list(tokens), list(cache_data), cache_type)
                )
                return BlockTable(request_id=request_id, block_ids=[1, 2], num_tokens=len(tokens))

            def get_stats(self):
                return {}

        disk = _DiskCache()
        block_cache = _BlockAwareCache()
        scheduler.disk_cache = disk
        scheduler.block_aware_cache = block_cache

        request = Request("req-disk-l2", "x", SamplingParams())
        request.prompt_token_ids = list(range(257))

        scheduler.add_request(request)

        assert disk.fetch_tokens == request.prompt_token_ids
        assert block_cache.store_calls == [
            ("req-disk-l2", request.prompt_token_ids[:-1], ["state-dict"], "assistant")
        ]
        assert request.prompt_cache is None
        assert request.block_table is not None
        assert request.cached_tokens == 256
        assert request.remaining_tokens == [256]
        assert request._paged_block_table_needs_worker_reconstruct is True
        assert request._cache_detail == "paged+disk+tq"

    def test_paged_quantized_kv_hit_defers_worker_dequant_for_batch_generator(
        self, mock_model, mock_tokenizer
    ):
        """Paged q4/q8 hits must not insert QuantizedKVCache into BatchGenerator.

        Block-aware cache reconstruct runs on the scheduler worker. When stored
        prefix blocks are quantized, the worker must dequantize them before
        BatchGenerator sees the cache; otherwise mlx-lm calls .filter() on a
        QuantizedKVCache at request finish and the cached turn returns an
        engine error.
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler._kv_cache_bits = 4
        scheduler._is_hybrid = False
        scheduler._uses_dsv4_cache = False
        scheduler._uses_zaya_cache = False

        class _PagedCache:
            stats = None

        class _BlockAwareCache:
            paged_cache = _PagedCache()

            def fetch_cache(self, request_id, tokens):
                return BlockTable(request_id=request_id, block_ids=[1], num_tokens=64), [64]

            def get_stats(self):
                return {}

        scheduler.block_aware_cache = _BlockAwareCache()

        request = Request("req-paged-q4", "x", SamplingParams())
        request.prompt_token_ids = list(range(65))

        scheduler.add_request(request)

        assert request._paged_block_table_needs_worker_reconstruct is True
        assert request._prompt_cache_needs_worker_dequant is True
        assert request.cached_tokens == 64
        assert request.remaining_tokens == [64]

    def test_hybrid_ssm_checkpoint_alignment_falls_back_to_exact_aligned_state(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = Request("req-align", "x", SamplingParams())
        request.prompt_token_ids = list(range(256))
        request._hybrid_ssm_fetch_tokens = list(range(256))

        class _SSMCache:
            def fetch_longest_prefix(self, token_ids, max_len):
                if max_len >= 150:
                    return (150, ["ssm@150"], True)
                return None

            def fetch(self, token_ids, num_tokens):
                assert num_tokens == 128
                return (["ssm@128"], True)

        scheduler._ssm_state_cache = _SSMCache()

        hit = scheduler._fetch_block_aligned_ssm_checkpoint(
            request,
            max_len=192,
            block_size=64,
        )

        assert hit == (128, ["ssm@128"])

    def test_hybrid_ssm_checkpoint_alignment_rejects_misaligned_state_without_exact(
        self, mock_model, mock_tokenizer
    ):
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = Request("req-misaligned", "x", SamplingParams())
        request.prompt_token_ids = list(range(256))

        class _SSMCache:
            def fetch_longest_prefix(self, token_ids, max_len):
                if max_len >= 150:
                    return (150, ["ssm@150"], True)
                return None

            def fetch(self, token_ids, num_tokens):
                return None

        scheduler._ssm_state_cache = _SSMCache()

        hit = scheduler._fetch_block_aligned_ssm_checkpoint(
            request,
            max_len=192,
            block_size=64,
        )

        assert hit is None
        assert request._cache_reuse_partial_unavailable_reason == (
            "no_block_aligned_ssm_checkpoint"
        )

    def test_reset(self, mock_model, mock_tokenizer):
        """Test resetting scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Add some requests
        for i in range(5):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Hello {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 5

        scheduler.reset()

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()


class TestSchedulerCacheResilience:
    """Tests for scheduler cache error handling and recovery."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    def test_memory_aware_cache_init(self, mock_model, mock_tokenizer):
        """Memory-aware cache is initialized with the scheduler default budget."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(
                use_memory_aware_cache=True,
                use_paged_cache=False,
            ),
        )
        assert scheduler.memory_aware_cache is not None
        assert scheduler.memory_aware_cache.memory_limit_mb > 0

    def test_cache_miss_no_error(self, mock_model, mock_tokenizer):
        """Cache miss should not cause any errors."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(use_memory_aware_cache=True),
        )
        request = Request(
            request_id="test-1",
            prompt="Hello world test prompt",
            prompt_token_ids=list(range(1000)),
            sampling_params=SamplingParams(max_tokens=10),
        )
        # Should not raise even though there's no cached data
        scheduler.add_request(request)
        assert request.prompt_cache is None
        assert len(request.remaining_tokens) == 1000

    def test_cache_validation_rejects_none_cache(self, mock_model, mock_tokenizer):
        """Validate cache rejects None properly."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert scheduler._validate_cache(None) is False
        assert scheduler._validate_cache([]) is False

    def test_cache_validation_accepts_valid_kv_cache(self, mock_model, mock_tokenizer):
        """Validate cache accepts valid KV cache objects."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        mock_layer = MagicMock()
        mock_layer.keys = MagicMock(nbytes=1000)
        mock_layer.values = MagicMock(nbytes=1000)
        # Make sure keys/values are not callable
        mock_layer.keys.callable = False
        type(mock_layer).keys = property(lambda self: MagicMock(nbytes=1000))
        type(mock_layer).values = property(lambda self: MagicMock(nbytes=1000))

        # A list with a mock layer that has keys and values
        layer = MagicMock()
        layer.keys = MagicMock(spec=[])  # Not callable
        layer.keys.nbytes = 1000
        layer.values = MagicMock(spec=[])
        layer.values.nbytes = 1000
        assert scheduler._validate_cache([layer]) is True

    def test_cache_validation_rejects_none_layers(self, mock_model, mock_tokenizer):
        """Validate cache rejects lists with None layers."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert scheduler._validate_cache([None]) is False

    def test_recover_from_cache_error(self, mock_model, mock_tokenizer):
        """Cache recovery clears all state correctly."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(use_memory_aware_cache=True),
        )
        # Simulate having some state
        scheduler.batch_generator = MagicMock()
        scheduler._current_sampler_params = (0.7, 0.9, 0.0)

        scheduler._recover_from_cache_error()

        assert scheduler.batch_generator is None
        assert scheduler._current_sampler_params is None
        assert len(scheduler.memory_aware_cache) == 0

    def test_reschedule_running_requests(self, mock_model, mock_tokenizer):
        """Rescheduling moves running requests back to waiting."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        request = Request(
            request_id="test-1",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
        )
        request.status = RequestStatus.RUNNING
        scheduler.running["test-1"] = request

        scheduler._reschedule_running_requests()

        assert len(scheduler.running) == 0
        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0].request_id == "test-1"
        assert scheduler.waiting[0].status == RequestStatus.WAITING

    def test_is_cache_corruption_error(self, mock_model, mock_tokenizer):
        """Cache corruption detection works for known patterns."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert scheduler._is_cache_corruption_error(
            TypeError("'NoneType' object is not subscriptable")
        )
        assert scheduler._is_cache_corruption_error(
            RuntimeError("BatchKVCache merge failed")
        )
        assert not scheduler._is_cache_corruption_error(
            ValueError("unrelated error")
        )

    def test_extract_cache_states_partial_success(self, mock_model, mock_tokenizer):
        """Extract cache states returns partial results on failure."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        # Create mix of valid and invalid layer caches
        valid_layer = MagicMock()
        valid_layer.state = (MagicMock(), MagicMock())
        valid_layer.meta_state = ("0",)

        invalid_layer = MagicMock(spec=[])  # No state/meta_state attributes

        result = scheduler._extract_cache_states([valid_layer, invalid_layer])
        # Should return the 1 valid layer, not empty list
        assert len(result) == 1
        assert result[0]["class_name"] is not None

    def test_memory_cache_stats_in_scheduler(self, mock_model, mock_tokenizer):
        """Scheduler exposes memory-aware cache stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(use_memory_aware_cache=True),
        )
        stats = scheduler.get_cache_stats()
        assert stats is not None
        assert "hits" in stats
        assert "current_memory_mb" in stats
        assert "max_memory_mb" in stats

    def test_scheduler_records_successful_cache_hit_tokens_by_detail(
        self, mock_model, mock_tokenizer
    ):
        """Successful scheduled cache hits must be visible in scheduler stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=2),
        )

        class _BatchGenerator:
            stop_tokens = set()

            def insert(self, tokens, max_tokens, caches=None, **kwargs):
                self.last_insert = {
                    "tokens": tokens,
                    "max_tokens": max_tokens,
                    "caches": caches,
                    "kwargs": kwargs,
                }
                return [101]

        request = Request(
            request_id="cache-hit-1",
            prompt="cache hit",
            prompt_token_ids=list(range(20)),
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.num_prompt_tokens = 20
        scheduler.requests[request.request_id] = request
        scheduler.waiting.append(request)
        scheduler.batch_generator = _BatchGenerator()
        params = request.sampling_params
        scheduler._current_sampler_params = (
            params.temperature,
            params.top_p,
            params.min_p,
            params.top_k,
            params.repetition_penalty,
        )
        scheduler._validate_cache = lambda cache: True
        request.prompt_cache = [object()]
        request.cached_tokens = 12
        request.remaining_tokens = request.prompt_token_ids[12:]
        request._cache_detail = "paged+dsv4"

        scheduled = scheduler._schedule_waiting()

        assert [r.request_id for r in scheduled] == ["cache-hit-1"]
        stats = scheduler.get_stats()
        assert stats["cache_hit_requests"] == 1
        assert stats["cache_hit_tokens"] == 12
        assert stats["cache_hit_tokens_by_detail"] == {"paged+dsv4": 12}


# Integration tests require actual MLX model
@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests that require a real model."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small test model."""
        try:
            from mlx_lm import load

            model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")

    def test_scheduler_with_real_model(self, model_and_tokenizer):
        """Test scheduler with real model."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=4,
                prefill_batch_size=2,
                completion_batch_size=4,
            ),
        )

        # Add a request
        request = Request(
            request_id="test-1",
            prompt="What is 2+2?",
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(request)

        # Run a few steps
        outputs = []
        for _ in range(20):
            output = scheduler.step()
            if output.outputs:
                outputs.extend(output.outputs)
            if output.finished_request_ids:
                break

        assert len(outputs) > 0
        # Check we got at least one output
        final_output = outputs[-1]
        assert final_output.request_id == "test-1"

    def test_multiple_concurrent_requests(self, model_and_tokenizer):
        """Test handling multiple concurrent requests."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
                completion_batch_size=8,
            ),
        )

        # Add multiple requests
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
        ]

        for i, prompt in enumerate(prompts):
            request = Request(
                request_id=f"test-{i}",
                prompt=prompt,
                sampling_params=SamplingParams(max_tokens=10),
            )
            scheduler.add_request(request)

        # Run until all complete
        finished = set()
        max_steps = 100
        steps = 0

        while len(finished) < len(prompts) and steps < max_steps:
            output = scheduler.step()
            finished.update(output.finished_request_ids)
            steps += 1

        assert len(finished) == len(prompts), f"Only {len(finished)} requests finished"


@pytest.mark.asyncio
class TestEngineAsync:
    """Async tests for the engine."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return model, tokenizer

    async def test_engine_lifecycle(self, mock_model_and_tokenizer):
        """Test engine start/stop lifecycle."""
        from vmlx_engine.engine import AsyncEngineCore, EngineConfig

        model, tokenizer = mock_model_and_tokenizer

        engine = AsyncEngineCore(model, tokenizer, EngineConfig())

        assert not engine.engine.is_running()

        # Use async context manager
        async with engine:
            assert engine.engine.is_running()
            await asyncio.sleep(0.05)

        assert not engine.engine.is_running()

    async def test_engine_context_manager(self, mock_model_and_tokenizer):
        """Test engine as async context manager."""
        from vmlx_engine.engine import AsyncEngineCore

        model, tokenizer = mock_model_and_tokenizer

        async with AsyncEngineCore(model, tokenizer) as engine:
            assert engine.engine.is_running()

        assert not engine.engine.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
