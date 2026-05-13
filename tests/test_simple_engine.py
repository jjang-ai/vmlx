# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleEngine concurrency handling."""

import asyncio
import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestSimpleEngineConcurrency:
    """Test SimpleEngine lock behavior with concurrent requests."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that tracks concurrent calls."""
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            # Simulate some work
            import time

            time.sleep(0.05)
            model._concurrent_count -= 1
            result = MagicMock()
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)
        return model

    @pytest.fixture
    def mock_llm_model(self):
        """Create a mock LLM model.

        SimpleEngine.chat() internally routes through `_model.generate()` for
        text-only (LLM) models (is_mllm_model returns False). So this fixture
        mocks `.generate` with a real string `.text` attribute — a bare
        MagicMock returns MagicMock for unset attributes, and that slips into
        `clean_output_text()` which calls re.sub and crashes on non-strings.
        """
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        # Track concurrent executions
        model._concurrent_count = 0
        model._max_concurrent = 0

        def generate_side_effect(**kwargs):
            model._concurrent_count += 1
            model._max_concurrent = max(model._max_concurrent, model._concurrent_count)
            import time
            time.sleep(0.05)
            model._concurrent_count -= 1
            # Use spec so attribute reads raise AttributeError for anything
            # we didn't explicitly wire — prevents silent MagicMock bleed-in.
            result = MagicMock(spec=["text", "tokens", "finish_reason",
                                     "prompt_tokens", "completion_tokens"])
            result.text = "test response"
            result.tokens = [1, 2, 3]
            result.finish_reason = "stop"
            result.prompt_tokens = 3
            result.completion_tokens = 3
            return result

        model.generate = MagicMock(side_effect=generate_side_effect)
        # Some code paths may probe .chat too — mirror to .generate
        model.chat = model.generate
        return model

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_generate(self, mock_model):
        """Test that the lock prevents concurrent generate calls."""
        from vmlx_engine.engine.simple import SimpleEngine

        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            tasks = [
                engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_chat(self, mock_llm_model):
        """Test that the lock prevents concurrent chat calls."""
        from vmlx_engine.engine.simple import SimpleEngine

        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_llm_model
            engine._loaded = True

            # Launch multiple concurrent chat calls
            tasks = [
                engine.chat(
                    messages=[{"role": "user", "content": f"test {i}"}], max_tokens=10
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # With the lock, max concurrent should be 1
            assert mock_llm_model._max_concurrent == 1, (
                f"Expected max concurrent to be 1, but got {mock_llm_model._max_concurrent}. "
                "The lock is not working correctly."
            )

    @pytest.mark.asyncio
    async def test_lock_serializes_stream_generate(self, mock_model):
        """Test that stream_generate uses the same lock as other methods."""
        from vmlx_engine.engine.simple import SimpleEngine

        def stream_generate_side_effect(**kwargs):
            # Yield a few chunks
            for i in range(3):
                chunk = MagicMock()
                chunk.text = f"chunk{i}"
                chunk.prompt_tokens = 5
                chunk.finished = i == 2
                chunk.finish_reason = "stop" if i == 2 else None
                yield chunk

        mock_model.stream_generate = MagicMock(side_effect=stream_generate_side_effect)

        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Test that stream_generate acquires the lock
            # by checking if it blocks when lock is already held
            lock_acquired = asyncio.Event()
            stream_started = asyncio.Event()

            async def hold_lock():
                async with engine._generation_lock:
                    lock_acquired.set()
                    # Wait until stream tries to start
                    await asyncio.sleep(0.1)

            async def try_stream():
                # Wait for lock to be held
                await lock_acquired.wait()
                stream_started.set()
                # This should block until hold_lock releases
                result = []
                async for chunk in engine.stream_generate(prompt="test", max_tokens=10):
                    result.append(chunk)
                return result

            # Start both tasks
            hold_task = asyncio.create_task(hold_lock())
            stream_task = asyncio.create_task(try_stream())

            # Wait a bit for stream to try to acquire lock
            await asyncio.sleep(0.05)

            # Stream should have started but be blocked on the lock
            assert stream_started.is_set(), "Stream should have attempted to start"

            # Stream task should not be done yet (blocked on lock)
            assert not stream_task.done(), "Stream should be blocked waiting for lock"

            # Let hold_lock finish
            await hold_task

            # Now stream should complete
            result = await stream_task
            assert len(result) == 3, f"Expected 3 chunks, got {len(result)}"

    @pytest.mark.asyncio
    async def test_engine_initialization_creates_lock(self):
        """Test that SimpleEngine creates a lock on initialization."""
        from vmlx_engine.engine.simple import SimpleEngine

        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")

            assert hasattr(engine, "_generation_lock")
            assert isinstance(engine._generation_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_requests_complete_in_order(self, mock_model):
        """Test that concurrent requests complete (may be in any order due to lock)."""
        from vmlx_engine.engine.simple import SimpleEngine

        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            # Launch multiple concurrent generate calls
            results = await asyncio.gather(
                *[
                    engine.generate(prompt=f"test prompt {i}", max_tokens=10)
                    for i in range(3)
                ]
            )

            # All requests should complete
            assert len(results) == 3
            for result in results:
                assert result.text == "test response"

    @pytest.mark.asyncio
    async def test_mllm_stream_chat_generation_error_raises_not_model_text(self):
        """Runtime failures from MLLM streaming must surface as errors, not assistant text."""
        from vmlx_engine.engine.simple import SimpleEngine

        model = MagicMock()
        model.stream_chat.side_effect = RuntimeError(
            "There is no Stream(gpu, 0) in current thread."
        )

        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("test-vlm")
            engine._model = model
            engine._loaded = True

            with pytest.raises(RuntimeError, match="There is no Stream"):
                async for _chunk in engine.stream_chat(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=4,
                ):
                    pass

    def test_simple_engine_model_work_uses_dedicated_executor_not_to_thread(self):
        """SimpleEngine must not use arbitrary default-executor threads for MLX.

        MLX streams are thread-local. Direct/no-continuous-batching mode must
        load and execute on one dedicated model worker, otherwise JANG/JANGTQ,
        VLM, and hybrid kernels can crash with Stream(gpu,N) missing from the
        current thread.
        """
        from vmlx_engine.engine.simple import SimpleEngine

        source = inspect.getsource(SimpleEngine)
        assert "ThreadPoolExecutor(" in source
        assert "thread_name_prefix=\"simple-engine-model\"" in source
        assert "asyncio.to_thread" not in source

    @pytest.mark.asyncio
    async def test_simple_engine_load_generate_and_stream_next_stay_on_one_thread(self):
        """Load, generate, stream iterator creation, and next() share one thread."""
        from types import SimpleNamespace

        from vmlx_engine.engine.simple import SimpleEngine

        thread_events: list[tuple[str, int, str]] = []

        class _Tokenizer:
            def encode(self, text):
                return text.split()

            def apply_chat_template(self, messages, **_kwargs):
                return " ".join(str(m["content"]) for m in messages)

        class _Model:
            tokenizer = _Tokenizer()

            def load(self):
                thread_events.append(
                    ("load", threading.get_ident(), threading.current_thread().name)
                )

            def generate(self, **_kwargs):
                thread_events.append(
                    ("generate", threading.get_ident(), threading.current_thread().name)
                )
                return SimpleNamespace(
                    text="READY",
                    tokens=[1],
                    prompt_tokens=1,
                    completion_tokens=1,
                    finish_reason="stop",
                )

            def stream_generate(self, **_kwargs):
                thread_events.append(
                    (
                        "stream_create",
                        threading.get_ident(),
                        threading.current_thread().name,
                    )
                )

                def _gen():
                    thread_events.append(
                        (
                            "stream_next",
                            threading.get_ident(),
                            threading.current_thread().name,
                        )
                    )
                    yield SimpleNamespace(
                        text="READY",
                        prompt_tokens=1,
                        finished=True,
                        finish_reason="stop",
                    )

                return _gen()

        with (
            patch("vmlx_engine.engine.simple.is_mllm_model", return_value=False),
            patch("vmlx_engine.models.llm.MLXLanguageModel", return_value=_Model()),
        ):
            engine = SimpleEngine("test-model")
            await engine.start()
            await engine.generate("one two")
            chunks = [
                chunk
                async for chunk in engine.stream_generate("one two", max_tokens=4)
            ]
            await engine.stop()

        assert chunks[-1].text == "READY"
        worker_ids = {ident for _event, ident, _name in thread_events}
        assert len(worker_ids) == 1
        assert all("simple-engine-model" in name for _event, _ident, name in thread_events)
