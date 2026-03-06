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


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_params(self):
        """Test default sampling parameters."""
        params = SamplingParams()

        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
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


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        """Test default scheduler config."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.policy == SchedulingPolicy.FCFS
        assert config.prefill_batch_size == 8
        assert config.completion_batch_size == 32

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
        """Memory-aware cache is initialized with 30% default."""
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
