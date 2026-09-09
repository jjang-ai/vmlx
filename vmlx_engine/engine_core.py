# SPDX-License-Identifier: Apache-2.0
"""
Engine Core for vmlx-engine continuous batching.

This module provides the EngineCore class that coordinates:
- Model loading and management
- Request scheduling via Scheduler
- Async request processing
- Output streaming

The design follows vLLM's engine architecture adapted for MLX.
"""

from .persistence_outcome import LEDGER as _PERSIST, format_outcome as _format_persistence_outcome
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .scheduler import Scheduler, SchedulerConfig
from .mllm_batch_generator import _mllm_media_cache_extra_keys
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
    Core engine for vmlx-engine inference with continuous batching.

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

        # Non-terminal deltas reach API/UI consumers immediately.  The engine
        # loop queues the finished output before potentially slow cache
        # persistence, then reopens this gate after paged/TQ/SSM/native cleanup.
        # Public stream consumers hold that finished output at the gate so a
        # completed tool call cannot be executed before its turn is durable;
        # the same gate also prevents the next request from performing prefix
        # lookup against a partially-written cache.
        self._terminal_cleanup_complete = asyncio.Event()
        self._terminal_cleanup_complete.set()

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
        # A terminal output is queued internally before its prefix/TQ/SSM
        # cleanup, but public stream consumers hold it behind the persistence
        # gate. Do not cancel that cleanup when the user immediately
        # stops/restarts the model after seeing the answer. The same event gates
        # terminal visibility, next-turn admission, and shutdown durability.
        if self._task and not self._terminal_cleanup_complete.is_set():
            logger.info("Waiting for terminal cache cleanup before engine stop")
            try:
                await asyncio.wait_for(
                    self._terminal_cleanup_complete.wait(), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Terminal cache cleanup did not finish within 5s; "
                    "continuing engine shutdown"
                )
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
        orphan_first_seen: dict[str, float] = {}
        _ORPHAN_ABORT_THRESHOLD = 10  # abort after 10 consecutive orphan outputs

        # Periodic check for scheduler/collector mismatch (ghost requests
        # in scheduler.running with no output collector). Runs every N steps.
        _GHOST_CHECK_INTERVAL = 50  # check every 50 steps
        _ORPHAN_TIMEOUT_S = 30.0  # abort after 30s with no consumer
        _ghost_check_counter = 0

        # JANGTQ Metal streams are thread-local: step() must run on the
        # SAME thread that loaded the model. BatchedEngine plumbs a
        # single-worker `llm-worker` ThreadPoolExecutor through
        # SchedulerConfig.step_executor → Scheduler._step_executor.
        # When present, dispatch step() onto it (mirrors the MLLM
        # scheduler's _process_loop pattern, mllm_scheduler.py:2371-2374).
        # Without this, DSV4-Flash + every JANGTQ LLM bundle crashes with
        # `RuntimeError: There is no Stream(gpu, N) in current thread.`
        _step_executor = getattr(self.scheduler, "_step_executor", None)
        _async_loop = asyncio.get_running_loop() if _step_executor is not None else None

        while self._running:
            try:
                if self.scheduler.has_requests():
                    # Run one generation step
                    if _step_executor is not None:
                        output = await _async_loop.run_in_executor(
                            _step_executor,
                            lambda: self.scheduler.step(defer_finished_cleanup=True),
                        )
                    else:
                        output = self.scheduler.step(defer_finished_cleanup=True)
                    self._steps_executed += 1
                    finished_ids = set(output.finished_request_ids)
                    if finished_ids:
                        # Close request admission before exposing the terminal
                        # output.  A consumer can submit its next turn as soon
                        # as it receives that output, while cache persistence
                        # is intentionally still pending.
                        self._terminal_cleanup_complete.clear()

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
                                orphan_first_seen.pop(rid, None)
                                # Optimized: skip stream_interval check when interval=1
                                if use_simple_streaming:
                                    collector.put(req_output)
                                else:
                                    state = states.get(rid)
                                    if state and state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished,
                                    ):
                                        # Merge any accumulated text from skipped tokens
                                        pending_text, pending_ids = state.drain_pending()
                                        if pending_text or pending_ids:
                                            req_output.new_text = pending_text + req_output.new_text
                                            req_output.new_token_ids = pending_ids + req_output.new_token_ids
                                        collector.put(req_output)
                                        state.mark_sent(req_output.completion_tokens)
                                    elif state:
                                        # Not sending yet — accumulate for next send
                                        state.accumulate(req_output.new_text, req_output.new_token_ids)
                            else:
                                # No collector — ghost/orphan request
                                count = orphan_counts.get(rid, 0) + 1
                                orphan_counts[rid] = count
                                # Track first-seen time for time-based reaping
                                if rid not in orphan_first_seen:
                                    orphan_first_seen[rid] = time.monotonic()
                                elapsed = time.monotonic() - orphan_first_seen[rid]
                                if count >= _ORPHAN_ABORT_THRESHOLD or elapsed > _ORPHAN_TIMEOUT_S:
                                    logger.warning(
                                        f"Aborting ghost request {rid}: "
                                        f"{count} outputs with no consumer "
                                        f"({elapsed:.1f}s elapsed)"
                                    )
                                    self.scheduler.abort_request(rid)
                                    orphan_counts.pop(rid, None)
                                    orphan_first_seen.pop(rid, None)

                            if req_output.finished:
                                orphan_counts.pop(rid, None)
                                orphan_first_seen.pop(rid, None)
                                event = events.get(rid)
                                if event:
                                    event.set()

                        # Always yield after distributing outputs so the
                        # event loop can flush SSE chunks per-token.
                        # Without this, the tight loop aggregates many tokens
                        # before the consumer gets a chance to read them.
                        await asyncio.sleep(0)

                    # Cache/TQ/SSM persistence for a long prompt can take
                    # seconds. The terminal object is already in the collector;
                    # public stream consumers hold it on
                    # _terminal_cleanup_complete while cleanup runs on the same
                    # model worker. Non-terminal deltas remain low-latency, but
                    # tool completion and the next request cannot race a
                    # partially-written cache.
                    if finished_ids:
                        if not outputs:
                            await asyncio.sleep(0)
                        try:
                            if _step_executor is not None:
                                await _async_loop.run_in_executor(
                                    _step_executor,
                                    self.scheduler._cleanup_finished_after_terminal_dispatch,
                                    finished_ids,
                                )
                            else:
                                self.scheduler._cleanup_finished_after_terminal_dispatch(
                                    finished_ids
                                )
                        finally:
                            self._terminal_cleanup_complete.set()
                else:
                    # Idle: drain AT MOST ONE maintenance task (e.g. deferred
                    # SSM re-derive) per iteration, on the same executor as
                    # step() — Metal streams are thread-local. This branch is
                    # only reached after the previous iteration dispatched all
                    # outputs and finished terminal cleanup, so idle tasks can
                    # never delay a response (vmlx#245). Tasks park themselves
                    # when foreground work arrives; the next iteration takes
                    # the step() branch first.
                    if getattr(self.scheduler, "has_idle_tasks", None) and (
                        self.scheduler.has_idle_tasks()
                    ):
                        if _step_executor is not None:
                            await _async_loop.run_in_executor(
                                _step_executor,
                                self.scheduler.run_one_idle_task,
                            )
                        else:
                            self.scheduler.run_one_idle_task()
                        await asyncio.sleep(0)
                    else:
                        # No work, yield control
                        await asyncio.sleep(step_interval)

            except asyncio.CancelledError:
                self._terminal_cleanup_complete.set()
                self._fail_active_requests("Engine cancelled")
                break
            except Exception as e:
                self._terminal_cleanup_complete.set()
                logger.error(f"Engine loop error: {e}", exc_info=True)
                # Signal all active requests as failed so consumers don't hang
                self._fail_active_requests(str(e))
                orphan_counts.clear()
                orphan_first_seen.clear()
                await asyncio.sleep(0.1)

    def _fail_active_requests(self, error_msg: str) -> None:
        """Signal all active requests as failed so waiting consumers unblock.

        Called when the engine loop catches an unexpected exception. Creates
        a structured error output for each tracked request so stream_outputs()
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
                    new_text="",
                    output_text="",
                    error=f"Engine loop error: {error_msg}",
                    error_code="engine_loop_error",
                    error_source="engine_loop",
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
        gen_prompt_len: int = 0,
        num_messages: int = 1,
        segment_boundaries: Optional[List[Any]] = None,
        bypass_prefix_cache: bool = False,
        encode_add_special_tokens: Optional[bool] = None,
        max_prompt_tokens: Optional[int] = None,
        pixel_values: Optional[Any] = None,
        image_grid_thw: Optional[Any] = None,
        pixel_values_videos: Optional[Any] = None,
        video_grid_thw: Optional[Any] = None,
        prompt_token_ids: Optional[List[int]] = None,
        cache_extra_keys: Optional[Any] = None,
        dsv4_thinking_soft_cap: Optional[int] = None,
        tools_present: bool = False,
    ) -> str:
        """
        Add a request for processing.

        Args:
            prompt: Input prompt (string or token IDs)
            sampling_params: Generation parameters
            request_id: Optional custom request ID
            images: Optional images for multimodal
            videos: Optional videos for multimodal
            gen_prompt_len: Number of generation prompt tokens to strip from
                prefix cache key (prevents cache misses on thinking models)
            num_messages: Number of messages in the conversation (for cache
                skip heuristic — multi-turn conversations keep cache)
            max_prompt_tokens: Optional exact prompt/context cap enforced
                after tokenization and before prefill/cache lookup

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
        # Preserve the exact template capability on the scheduler request.
        # DSV4's reasoning-dropping shadow re-key is useful only for tools-off
        # renders: tools-on continuations keep a different prompt/template
        # prefix and already use the normal extended native chain.
        request._vmlx_tools_present = bool(tools_present)

        # M3 VL: carry preprocessed image/video tensors and input ids whose
        # media placeholders are already expanded by the bundle processor.
        if prompt_token_ids is not None:
            request.prompt_token_ids = list(prompt_token_ids)
            request.num_prompt_tokens = len(request.prompt_token_ids)
        if pixel_values is not None:
            request.pixel_values = pixel_values
        if image_grid_thw is not None:
            request.image_grid_thw = image_grid_thw
        if pixel_values_videos is not None:
            request.pixel_values_videos = pixel_values_videos
        if video_grid_thw is not None:
            request.video_grid_thw = video_grid_thw

        # MiniMax-M3 VL is text-routed, but its image/video placeholders are
        # content-dependent just like the generic MLLM path. Mix the shared
        # byte-derived media fingerprint into paged/block keys so same-shaped
        # different media cannot cross-reuse vision-conditioned MSA state.
        # Scheduler admission below permits only the salted paged/block path;
        # unsalted legacy/memory/prompt-L2 paths remain disabled for media.
        if cache_extra_keys is not None:
            request._cache_extra_keys = dict(cache_extra_keys) if isinstance(
                cache_extra_keys, dict
            ) else {"request": repr(cache_extra_keys)}
        if pixel_values is not None or pixel_values_videos is not None:
            media_extra = _mllm_media_cache_extra_keys(request)
            if media_extra:
                if getattr(request, "_cache_extra_keys", None):
                    merged = dict(request._cache_extra_keys)
                    merged.update(media_extra)
                    request._cache_extra_keys = merged
                else:
                    request._cache_extra_keys = media_extra
            request._m3_vl_media_cache_context = bool(
                getattr(request, "_cache_extra_keys", None)
            )

        # Attach gen_prompt_len for prefix cache key stripping.
        # The scheduler reads this via getattr(request, '_gen_prompt_len', 0)
        # to exclude generation prompt tokens from the cache key hash,
        # which prevents 100% cache misses in multi-turn conversations
        # for thinking models (where gen prompt includes <think>).
        if gen_prompt_len > 0:
            request._gen_prompt_len = gen_prompt_len

        # Attach _has_history for cache skip heuristic.
        # Multi-turn conversations (>2 messages = system+user+assistant+...)
        # should always store cache even for short outputs.
        request._has_history = num_messages > 2

        # F11 (audit 2026-04-08): attach per-segment boundaries from the chat
        # template render so the LLM scheduler can store cache entries with
        # the correct cache_type (system/user/assistant) — drives Agent 1's
        # PrefixCacheManager priority LRU breakthrough.
        if segment_boundaries:
            request._segment_boundaries = list(segment_boundaries)

        # Per-request prefix cache bypass (benchmark isolation).
        # When set, the scheduler skips EVERY prefix cache layer for this
        # request — both lookup and store — including the SSM companion
        # cache on hybrid models. Set by the server gateway layer when the
        # API request carried cache_salt or skip_prefix_cache=true.
        if bypass_prefix_cache:
            request._bypass_prefix_cache = True

        # Chat templates already include every control token they need
        # (BOS, role markers, generation prompt, think delimiters). Some
        # tokenizers, notably ZAYA's GemmaTokenizerFast, append EOS when
        # encode() is called with default add_special_tokens=True; that
        # places an end-of-turn token after the assistant prefix and makes
        # generation stop immediately. Leave None as the legacy raw-completion
        # behavior; templated chat paths pass False explicitly.
        if encode_add_special_tokens is not None:
            request._encode_add_special_tokens = bool(encode_add_special_tokens)

        if max_prompt_tokens is not None and int(max_prompt_tokens) > 0:
            request._max_prompt_tokens = int(max_prompt_tokens)

        # DSV4 answer-reserve soft cap: enforced by DSV4BatchGenerator only
        # while generation is still inside the thinking rail; lifts at </think>
        # so the same pass continues into the reserved answer budget.
        if dsv4_thinking_soft_cap is not None and int(dsv4_thinking_soft_cap) > 0:
            request._dsv4_thinking_soft_cap = int(dsv4_thinking_soft_cap)

        # A prior terminal output may already be visible to the caller while
        # its prefix/TQ/SSM state is still being persisted. Mark this foreground
        # admission BEFORE waiting so idle cache-maintenance tasks yield rather
        # than starting a full Metal prefill in the gap before scheduler lookup.
        # The marker is cleared for success, validation failure, and cancellation.
        begin_admission = getattr(
            self.scheduler, "_begin_foreground_admission", None
        )
        end_admission = getattr(self.scheduler, "_end_foreground_admission", None)
        admission_marked = False
        if callable(begin_admission):
            begin_admission(request_id)
            admission_marked = True

        try:
            # Wait before creating request bookkeeping so a cancelled waiter
            # cannot leak a collector and prefix lookup cannot observe a stale
            # partial cache entry.
            await self._terminal_cleanup_complete.wait()

            # Setup output collector with stream_interval from config
            self._output_collectors[request_id] = RequestOutputCollector(
                aggregate=True
            )
            self._stream_states[request_id] = RequestStreamState(
                stream_interval=self.config.stream_interval
            )
            self._finished_events[request_id] = asyncio.Event()

            # If validation rejects the request (for example, DSV4 long-prefill
            # guard), remove the collector/event state so no ghost bookkeeping
            # remains.
            try:
                self.scheduler.add_request(request)
            except Exception:
                self._cleanup_request(request_id)
                raise
        finally:
            if admission_marked and callable(end_admission):
                end_admission(request_id)

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request. Returns True if the request was found."""
        # Check if request exists before cleanup
        found = request_id in self._output_collectors or request_id in self._finished_events
        # _cleanup_request handles scheduler.abort_request() internally —
        # don't call it here too (was causing double abort)
        self._cleanup_request(request_id)
        return found

    def request_progress(self, request_id: str) -> Optional[int]:
        """Monotonic progress counter for a live request, or None if unknown."""
        return self.scheduler.request_progress(request_id)

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
            #
            # Drain any pending accumulated text from stream_interval > 1
            # so the abort sentinel carries the full output so far.
            state = self._stream_states.get(request_id)
            pending_text = ""
            pending_ids: list = []
            if state:
                pending_text, pending_ids = state.drain_pending()
            try:
                collector.put(RequestOutput(
                    request_id=request_id,
                    new_text=pending_text,
                    new_token_ids=pending_ids,
                    finished=True,
                    finish_reason="aborted",
                ))
            except Exception:
                pass
        self._stream_states.pop(request_id, None)
        event = self._finished_events.pop(request_id, None)
        if event:
            event.set()
        request = self.scheduler.get_request(request_id)
        if request is None or not RequestStatus.is_finished(request.status):
            self.scheduler.abort_request(request_id)

    async def _wait_for_terminal_persistence(self, request_id: str) -> None:
        """Public JSON and streaming results share the same completion fence.

        Completion of this fence means cleanup settled, not that a store
        succeeded. Keep the actual stored/skipped/failed outcome in the log.
        """
        started = time.perf_counter()
        was_pending = not self._terminal_cleanup_complete.is_set()
        await self._terminal_cleanup_complete.wait()
        outcome = _PERSIST.take(request_id)
        logger.info(
            "Terminal durability barrier: request=%s wait_ms=%.3f waited=%s %s",
            request_id,
            (time.perf_counter() - started) * 1000.0,
            "true" if was_pending else "false",
            _format_persistence_outcome(outcome),
        )

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

                    if output.finished:
                        # The scheduler intentionally queues the terminal object
                        # before synchronous prefix/TQ/SSM/native persistence so
                        # its model-thread ownership is not lost.  Do not expose
                        # that object to Chat/Responses/Electron until the
                        # existing cleanup fence is durable: clients commonly
                        # execute a parsed tool as soon as finish_reason or
                        # response.completed arrives.
                        await self._wait_for_terminal_persistence(request_id)

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
        try:
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

            if final_output is None:
                raise RuntimeError(f"No output for request {request_id}")

            # The finished event is internal dispatch, before deferred SSD/KV
            # cleanup. A JSON tool response is just as actionable as a streamed
            # terminal; never publish it or remove its collector ahead of the
            # same fence used by stream_outputs.
            await self._wait_for_terminal_persistence(request_id)
            return final_output
        finally:
            # Always clean up request state to prevent permanent leaks
            self._cleanup_request(request_id)

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

        # Process until all done - direct scheduler access, no async overhead.
        # Same thread-pinning rationale as `_engine_loop`: when an
        # `_step_executor` is present we route step() through it so
        # JANGTQ Metal kernels stay on the loader thread.
        _step_executor = getattr(self.scheduler, "_step_executor", None)
        results: Dict[str, RequestOutput] = {}
        while self.scheduler.has_requests():
            if _step_executor is not None:
                output = _step_executor.submit(self.scheduler.step).result()
            else:
                output = self.scheduler.step()
            for req_output in output.outputs:
                if req_output.finished:
                    results[req_output.request_id] = req_output

        # Cleanup — use abort_request for complete cleanup (BatchGenerator UIDs,
        # paged cache tracking, detokenizer, UID mappings). remove_finished_request
        # only removes from the running dict, leaving ghost entries elsewhere.
        for rid in request_ids:
            self.scheduler.abort_request(rid)

        # Sync path has no engine loop to drain idle maintenance (deferred SSM
        # re-derive). Drain here so companion snapshots aren't stranded in the
        # queue — a stranded snapshot makes every later request fall back to a
        # much shorter companion. Bounded so a forever-parking task can't hang
        # the caller; with no concurrent producer each task completes promptly.
        if getattr(self.scheduler, "has_idle_tasks", None):
            for _ in range(4 * 8):  # 4x SSM_REDERIVE_QUEUE_CAP
                if self.scheduler.has_requests() or (
                    not self.scheduler.has_idle_tasks()
                ):
                    break
                if _step_executor is not None:
                    _step_executor.submit(
                        self.scheduler.run_one_idle_task
                    ).result()
                else:
                    self.scheduler.run_one_idle_task()

        # Return in original order — use .get() to handle requests that were
        # aborted or never produced output (avoids KeyError on missing keys)
        return [results.get(rid, RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="aborted",
        )) for rid in request_ids]

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        uptime = time.time() - self._start_time if self._start_time else 0
        collector_request_ids = sorted(self._output_collectors)

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(collector_request_ids),
            # These are the engine-owned consumers, not prefix-cache request
            # tables.  Keep the IDs path-free and prompt-free so /health can
            # attest which request owns a scheduler lifecycle without
            # confusing cache bookkeeping with HTTP/gateway activity.
            "engine_collector_count": len(collector_request_ids),
            "engine_collector_request_ids": collector_request_ids,
            "terminal_cleanup_pending": (
                not self._terminal_cleanup_complete.is_set()
            ),
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

    def request_progress(self, request_id: str) -> Optional[int]:
        """Monotonic progress counter for a live request, or None if unknown."""
        return self.engine.request_progress(request_id)

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
