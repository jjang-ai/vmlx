# SPDX-License-Identifier: Apache-2.0
"""
Engine Core for vllm-mlx continuous batching.

This module provides the EngineCore class that coordinates:
- Model loading and management
- Request scheduling via Scheduler
- Async request processing
- Output streaming

The design follows vLLM's engine architecture adapted for MLX.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .request import Request, RequestOutput, SamplingParams
from .scheduler import Scheduler, SchedulerConfig
from .output_collector import RequestOutputCollector, RequestStreamState
from .model_registry import get_registry

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: Optional[SchedulerConfig] = None
    step_interval: float = 0.001  # 1ms between steps
    stream_interval: int = 1  # Tokens to batch before streaming (1=every token)


class EngineCore:
    """
    Core engine for vllm-mlx inference with continuous batching.

    This engine runs the generation loop and manages request lifecycle.
    It provides both sync and async interfaces for request handling.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
        engine_id: Optional[str] = None,
        force_model_ownership: bool = True,
    ):
        """
        Initialize the engine.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Engine configuration
            engine_id: Optional unique ID for this engine (auto-generated if None)
            force_model_ownership: If True (default), forcibly take model ownership
                                   from any existing engine. If False, raises
                                   ModelOwnershipError if model is in use.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False

        # Acquire model ownership
        registry = get_registry()
        registry.acquire(
            model=model,
            engine=self,
            engine_id=self._engine_id,
            force=force_model_ownership,
        )
        self._owns_model = True

        # Create scheduler
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=scheduler_config,
        )

        # Output collectors for low-latency streaming (vLLM pattern)
        self._output_collectors: Dict[str, RequestOutputCollector] = {}
        self._stream_states: Dict[str, RequestStreamState] = {}
        self._finished_events: Dict[str, asyncio.Event] = {}

        # Engine state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._steps_executed = 0

        logger.debug(f"Engine {self._engine_id} initialized")

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())
        logger.info("Engine started")

    async def stop(self) -> None:
        """Stop the engine loop and flush caches."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Flush disk caches before exit
        if hasattr(self, 'scheduler'):
            self.scheduler.shutdown()
        logger.info("Engine stopped")

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    async def _engine_loop(self) -> None:
        """Main engine loop - optimized for minimal overhead."""
        # Cache config values for faster access
        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = stream_interval == 1

        # Ghost request detection: tracks consecutive orphan outputs
        # (outputs with no matching collector). If this exceeds a threshold,
        # those requests are aborted to prevent a permanent spin loop.
        orphan_counts: dict[str, int] = {}
        _ORPHAN_ABORT_THRESHOLD = 10  # abort after 10 consecutive orphan outputs

        # Periodic check for scheduler/collector mismatch (ghost requests
        # in scheduler.running with no output collector). Runs every N steps.
        _GHOST_CHECK_INTERVAL = 50  # check every 50 steps
        _ghost_check_counter = 0

        while self._running:
            try:
                if self.scheduler.has_requests():
                    # Run one generation step
                    output = self.scheduler.step()
                    self._steps_executed += 1

                    # Periodic ghost request detection: find requests in
                    # scheduler.running that have no output collector.
                    # These are "ghost" requests that will spin forever.
                    _ghost_check_counter += 1
                    if _ghost_check_counter >= _GHOST_CHECK_INTERVAL:
                        _ghost_check_counter = 0
                        ghost_ids = [
                            rid for rid in self.scheduler.running
                            if rid not in self._output_collectors
                        ]
                        for rid in ghost_ids:
                            logger.warning(
                                f"Aborting ghost request {rid}: in scheduler.running "
                                f"but no output collector"
                            )
                            self.scheduler.abort_request(rid)

                    # Fast path: distribute outputs to collectors
                    outputs = output.outputs
                    if outputs:
                        collectors = self._output_collectors
                        states = self._stream_states
                        events = self._finished_events

                        for req_output in outputs:
                            rid = req_output.request_id
                            collector = collectors.get(rid)

                            if collector is not None:
                                # Has a consumer — clear orphan tracking
                                orphan_counts.pop(rid, None)
                                # Optimized: skip stream_interval check when interval=1
                                if use_simple_streaming:
                                    collector.put(req_output)
                                else:
                                    state = states.get(rid)
                                    if state and state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished,
                                    ):
                                        collector.put(req_output)
                                        state.mark_sent(req_output.completion_tokens)
                            else:
                                # No collector — ghost/orphan request
                                count = orphan_counts.get(rid, 0) + 1
                                orphan_counts[rid] = count
                                if count >= _ORPHAN_ABORT_THRESHOLD:
                                    logger.warning(
                                        f"Aborting ghost request {rid}: "
                                        f"{count} outputs with no consumer"
                                    )
                                    self.scheduler.abort_request(rid)
                                    orphan_counts.pop(rid, None)

                            if req_output.finished:
                                orphan_counts.pop(rid, None)
                                event = events.get(rid)
                                if event:
                                    event.set()

                        # Always yield after distributing outputs so the
                        # event loop can flush SSE chunks per-token.
                        # Without this, the tight loop aggregates many tokens
                        # before the consumer gets a chance to read them.
                        await asyncio.sleep(0)
                else:
                    # No work, yield control
                    await asyncio.sleep(step_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Engine loop error: {e}", exc_info=True)
                # Signal all active requests as failed so consumers don't hang
                self._fail_active_requests(str(e))
                orphan_counts.clear()
                await asyncio.sleep(0.1)

    def _fail_active_requests(self, error_msg: str) -> None:
        """Signal all active requests as failed so waiting consumers unblock.

        Called when the engine loop catches an unexpected exception. Creates
        a synthetic error output for each tracked request so stream_outputs()
        and generate() callers receive an error instead of hanging forever.

        Also proactively aborts all requests in the scheduler to prevent
        ghost requests from lingering in scheduler.running / BatchGenerator,
        which would cause a permanent spin loop blocking the event loop.
        """
        # Signal consumers first so they can unblock
        for rid, collector in list(self._output_collectors.items()):
            try:
                error_output = RequestOutput(
                    request_id=rid,
                    finished=True,
                    finish_reason="error",
                    new_text=f"\n\n[Engine error: {error_msg}]",
                )
                collector.put(error_output)
            except Exception:
                pass  # Best effort
            event = self._finished_events.get(rid)
            if event:
                event.set()

        # Proactively abort all scheduler requests to prevent ghost entries
        # in scheduler.running and BatchGenerator. Without this, orphaned
        # requests cause a tight spin loop doing GPU work with no consumer.
        try:
            running_ids = list(self.scheduler.running.keys())
            waiting_ids = [r.request_id for r in self.scheduler.waiting]
            for rid in running_ids + waiting_ids:
                self.scheduler.abort_request(rid)
            if running_ids or waiting_ids:
                logger.warning(
                    f"Aborted {len(running_ids)} running + {len(waiting_ids)} "
                    f"waiting requests after engine error"
                )
        except Exception as e:
            logger.error(f"Failed to abort scheduler requests: {e}")

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ) -> str:
        """
        Add a request for processing.

        Args:
            prompt: Input prompt (string or token IDs)
            sampling_params: Generation parameters
            request_id: Optional custom request ID
            images: Optional images for multimodal
            videos: Optional videos for multimodal

        Returns:
            The request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            images=images,
            videos=videos,
        )

        # Setup output collector with stream_interval from config
        self._output_collectors[request_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[request_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[request_id] = asyncio.Event()

        # Add to scheduler
        self.scheduler.add_request(request)

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        # _cleanup_request handles scheduler.abort_request() internally —
        # don't call it here too (was causing double abort)
        self._cleanup_request(request_id)
        return True

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up request tracking.

        Uses abort_request() instead of remove_finished_request() to ensure
        complete cleanup of ALL scheduler state: running dict, BatchGenerator
        UIDs, paged cache tracking, detokenizer, and UID mappings. Without
        this, ghost requests can linger in scheduler.running and the
        BatchGenerator, causing a permanent spin loop that blocks the event
        loop and makes the API unresponsive.
        """
        collector = self._output_collectors.pop(request_id, None)
        if collector:
            # Put a finished sentinel so any consumer blocked in
            # collector.get() unblocks and exits cleanly. Without this,
            # cancel API calls while a consumer is waiting hang forever.
            # Don't call clear() after — let the consumer read the sentinel.
            # The collector is already popped from _output_collectors so the
            # engine loop won't send more outputs to it.
            try:
                collector.put(RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="aborted",
                ))
            except Exception:
                pass
        self._stream_states.pop(request_id, None)
        event = self._finished_events.pop(request_id, None)
        if event:
            event.set()
        self.scheduler.abort_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request with low-latency non-blocking pattern.

        Uses the vLLM pattern: get_nowait() or await get()
        This avoids unnecessary task switches when output is available.

        Args:
            request_id: The request ID
            timeout: Optional timeout in seconds

        Yields:
            RequestOutput objects as tokens are generated
        """
        collector = self._output_collectors.get(request_id)
        if collector is None:
            # Request might not be added yet or already cleaned up
            return

        try:
            while True:
                try:
                    # Non-blocking drain pattern from vLLM
                    # Try get_nowait first to avoid task switch if output ready
                    if timeout:
                        output = collector.get_nowait()
                        if output is None:
                            output = await asyncio.wait_for(
                                collector.get(), timeout=timeout
                            )
                    else:
                        output = collector.get_nowait()
                        if output is None:
                            output = await collector.get()

                    yield output

                    if output.finished:
                        break

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for request {request_id}")
                    break

        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate a complete response (non-streaming).

        This method is optimized to avoid streaming overhead when
        you only need the final result.

        Args:
            prompt: Input prompt
            sampling_params: Generation parameters
            request_id: Optional request ID

        Returns:
            Final RequestOutput with complete text
        """
        request_id = await self.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

        # Wait for completion using event instead of streaming
        # This avoids the waiting_consumer tracking overhead
        event = self._finished_events.get(request_id)
        if event is None:
            raise RuntimeError(f"No event for request {request_id}")

        # Wait for the request to finish
        await event.wait()

        # Get the final output from collector
        collector = self._output_collectors.get(request_id)
        if collector is None:
            raise RuntimeError(f"No collector for request {request_id}")

        # Drain all outputs and get the last one
        final_output = None
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            final_output = output

        # Cleanup
        self._cleanup_request(request_id)

        if final_output is None:
            raise RuntimeError(f"No output for request {request_id}")

        return final_output

    def generate_batch_sync(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """
        Generate responses synchronously for maximum throughput.

        This bypasses the async engine loop entirely, running the scheduler
        directly for optimal batching performance. Use this when you don't
        need streaming and want maximum throughput.

        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters (same for all)

        Returns:
            List of RequestOutput in same order as prompts
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Add all requests to scheduler
        request_ids = []
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            request = Request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            self.scheduler.add_request(request)
            request_ids.append(request_id)

        # Process until all done - direct scheduler access, no async overhead
        results: Dict[str, RequestOutput] = {}
        while self.scheduler.has_requests():
            output = self.scheduler.step()
            for req_output in output.outputs:
                if req_output.finished:
                    results[req_output.request_id] = req_output

        # Cleanup
        for rid in request_ids:
            self.scheduler.remove_finished_request(rid)

        # Return in original order
        return [results[rid] for rid in request_ids]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._output_collectors),
            "stream_interval": self.config.stream_interval,
            **scheduler_stats,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.scheduler.get_cache_stats()

    def _release_model(self) -> None:
        """Release model ownership."""
        if self._owns_model and not self._closed:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

    def close(self) -> None:
        """
        Explicitly close the engine and release resources.

        This should be called when done using the engine, especially
        if you plan to create another engine with the same model.
        """
        if self._closed:
            return

        # Release model ownership BEFORE setting _closed
        # (_release_model checks not self._closed)
        if self._owns_model:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

        self._closed = True

        # Flush disk caches before clearing in-memory state
        self.scheduler.shutdown()

        # Reset scheduler to clear BatchGenerator and all caches
        self.scheduler.deep_reset()

        # Clear output collectors
        for collector in self._output_collectors.values():
            collector.clear()
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()

        logger.debug(f"Engine {self._engine_id} closed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._release_model()
        except Exception:
            # Ignore errors during garbage collection
            pass

    @property
    def engine_id(self) -> str:
        """Get the engine ID."""
        return self._engine_id


class AsyncEngineCore:
    """
    Async context manager wrapper for EngineCore.

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            request_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs(request_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
    ):
        self.engine = EngineCore(model, tokenizer, config)

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.engine.stop()

    def start(self) -> None:
        """Start engine (creates task in current loop)."""
        asyncio.create_task(self.engine.start())

    async def stop(self) -> None:
        """Stop the engine."""
        await self.engine.stop()

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Add a request."""
        return await self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        return await self.engine.abort_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Stream outputs."""
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> RequestOutput:
        """Generate complete response."""
        return await self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            **kwargs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine stats."""
        return self.engine.get_stats()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.engine.get_cache_stats()
