# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLMScheduler batching stability and concurrency.

These tests verify:
- Thread safety of queue mutations using self._queue_lock
- Graceful recovery from pipeline errors in concurrent generation steps
- Unbounded queue avoidance
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture
def mock_mllm_components():
    mock_model = MagicMock()
    mock_processor = MagicMock()
    from vmlx_engine.mllm_scheduler import MLLMSchedulerConfig
    config = MLLMSchedulerConfig()
    return mock_model, mock_processor, config


class TestMLLMSchedulerStability:
    """Tests to verify stability/locking mechanisms in MLLMScheduler."""

    def test_queue_locks_and_maxsize(self, mock_mllm_components):
        """Test queue sizes and lock existence."""
        model, processor, config = mock_mllm_components
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        
        scheduler = MLLMScheduler(model, processor, config)
        
        # Verify internal lock exists
        assert hasattr(scheduler, "_queue_lock")
        assert isinstance(scheduler._queue_lock, type(threading.RLock()))

        # Add a request async
        req_id = asyncio.run(scheduler.add_request_async(prompt="Testing output queues"))
        
        assert req_id in scheduler.output_queues
        q = scheduler.output_queues[req_id]
        
        # Queue should be bounded to 8192
        assert getattr(q, "_maxsize", getattr(q, "maxsize", None)) == 8192

    @patch("vmlx_engine.mllm_scheduler.logger")
    def test_concurrent_mutations_protected(self, mock_logger, mock_mllm_components):
        """Test that mutations to waiting/running state are fully guarded by lock."""
        model, processor, config = mock_mllm_components
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        
        scheduler = MLLMScheduler(model, processor, config)
        
        # Add tasks from different "threads" simultaneously
        def add_tasks(prefix, count):
            for i in range(count):
                scheduler.add_request(prompt=f"{prefix} {i}")

        t1 = threading.Thread(target=add_tasks, args=("A", 50))
        t2 = threading.Thread(target=add_tasks, args=("B", 50))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Safe addition protected by lock
        assert len(scheduler.waiting) == 100
        assert len(scheduler.requests) == 100

        # Now test that get_stats is safe to read
        stats = scheduler.get_stats()
        assert stats["num_waiting"] == 100

    @patch("vmlx_engine.mllm_scheduler.logger")
    def test_error_recovery_rescheduling(self, mock_logger, mock_mllm_components):
        """Test that generation step errors properly clear caches and reschedule."""
        model, processor, config = mock_mllm_components
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        from vmlx_engine.request import RequestStatus

        scheduler = MLLMScheduler(model, processor, config)
        req_id = scheduler.add_request(prompt="Test failure recovery")

        # Mock schedule waiting
        from vmlx_engine.mllm_scheduler import MLLMRequest
        req = scheduler.requests[req_id]
        
        scheduler.running[req_id] = req
        req.status = RequestStatus.RUNNING
        scheduler.waiting.clear()

        # Mock the BatchGenerator blowing up during next()
        mock_bg = MagicMock()
        mock_bg.next.side_effect = RuntimeError("Simulated GPU Cache Corruption")
        scheduler.batch_generator = mock_bg

        # Run step (this should trap the error, release batchgen, and reschedule)
        output = scheduler.step()

        # Assert correct recovery behavior
        assert scheduler.batch_generator is None  # Batch generator thrown away
        assert len(scheduler.running) == 0       # Not running
        assert len(scheduler.waiting) == 1       # Put back on queue
        assert scheduler.waiting[0].request_id == req_id
        
        assert output.has_work is False
        assert len(output.outputs) == 0

    @patch("vmlx_engine.mllm_scheduler.logger")
    def test_fail_all_requests_clears_state(self, mock_logger, mock_mllm_components):
        """Test that _fail_all_requests properly fails all waiting+running requests under lock."""
        model, processor, config = mock_mllm_components
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        from vmlx_engine.request import RequestStatus

        scheduler = MLLMScheduler(model, processor, config)

        # Add several requests
        req_ids = []
        for i in range(5):
            rid = scheduler.add_request(prompt=f"Test {i}")
            req_ids.append(rid)

        # Move some to running
        for i in range(3):
            req = scheduler.requests[req_ids[i]]
            req.status = RequestStatus.RUNNING
            scheduler.running[req_ids[i]] = req
        # Remove the first 3 from waiting since they're now "running"
        for _ in range(3):
            scheduler.waiting.popleft()

        assert len(scheduler.running) == 3
        assert len(scheduler.waiting) == 2

        # Fail all
        scheduler._fail_all_requests("test error")

        # Everything should be cleaned
        assert len(scheduler.running) == 0
        assert len(scheduler.waiting) == 0
        assert len(scheduler.requests) == 0
        assert scheduler.batch_generator is None

    @patch("vmlx_engine.mllm_scheduler.logger")
    def test_reset_no_deadlock(self, mock_logger, mock_mllm_components):
        """Test that reset() doesn't deadlock despite calling abort_request() which also acquires RLock."""
        model, processor, config = mock_mllm_components
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler(model, processor, config)

        # Add requests
        for i in range(10):
            scheduler.add_request(prompt=f"Test {i}")

        assert len(scheduler.waiting) == 10

        # Reset should not deadlock (abort_request acquires _queue_lock, reset already holds it)
        # This works because _queue_lock is an RLock (reentrant)
        scheduler.reset()

        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 0
        assert len(scheduler.requests) == 0

    @patch("vmlx_engine.mllm_scheduler.logger")
    def test_get_stats_thread_safe(self, mock_logger, mock_mllm_components):
        """Test that get_stats returns a consistent snapshot even under concurrent writes."""
        model, processor, config = mock_mllm_components
        from vmlx_engine.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler(model, processor, config)

        errors = []

        def add_and_abort_requests():
            """Continuously add and abort requests."""
            for i in range(100):
                rid = scheduler.add_request(prompt=f"Task {i}")
                scheduler.abort_request(rid)

        def poll_stats():
            """Continuously read stats."""
            for _ in range(200):
                try:
                    stats = scheduler.get_stats()
                    # num_waiting should always be >= 0
                    assert stats["num_waiting"] >= 0
                    assert stats["num_running"] >= 0
                except Exception as e:
                    errors.append(str(e))

        t1 = threading.Thread(target=add_and_abort_requests)
        t2 = threading.Thread(target=poll_stats)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread-safety errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
