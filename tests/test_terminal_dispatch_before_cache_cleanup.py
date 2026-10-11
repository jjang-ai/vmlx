"""Terminals publish before cache cleanup; the next admission is the fence.

VMLX_TERMINAL_PUBLISH=strict restores holding the terminal behind cleanup.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import MethodType, SimpleNamespace

import pytest

from vmlx_engine.engine_core import EngineCore
from vmlx_engine.mllm_scheduler import MLLMScheduler, MLLMSchedulerOutput
from vmlx_engine.persistence_outcome import LEDGER
from vmlx_engine.request import RequestOutput
from vmlx_engine.scheduler import Scheduler, SchedulerOutput


def test_engine_loop_failure_uses_structured_error_not_assistant_text() -> None:
    engine = EngineCore.__new__(EngineCore)
    seen: list[RequestOutput] = []

    class _Collector:
        def put(self, output: RequestOutput) -> None:
            seen.append(output)

    class _Scheduler:
        running = {"req": object()}
        waiting = []

        def abort_request(self, request_id: str) -> None:
            assert request_id == "req"

    done = asyncio.Event()
    engine.scheduler = _Scheduler()
    engine._output_collectors = {"req": _Collector()}
    engine._finished_events = {"req": done}

    engine._fail_active_requests("[full] Negative dimensions not allowed.")

    assert done.is_set()
    assert len(seen) == 1
    output = seen[0]
    assert output.finished is True
    assert output.finish_reason == "error"
    assert output.error == "Engine loop error: [full] Negative dimensions not allowed."
    assert output.error_code == "engine_loop_error"
    assert output.error_source == "engine_loop"
    assert output.new_text == ""
    assert "[Engine error:" not in output.output_text


@pytest.mark.asyncio
async def test_llm_engine_dispatches_terminal_before_deferred_cleanup() -> None:
    order: list[str] = []
    engine = EngineCore.__new__(EngineCore)

    class _Collector:
        def put(self, output: RequestOutput) -> None:
            assert output.finished
            assert not engine._terminal_cleanup_complete.is_set()
            order.append("dispatch")

    class _Scheduler:
        _step_executor = None
        running = {}

        def has_requests(self) -> bool:
            return engine._running

        def step(self, *, defer_finished_cleanup: bool = False) -> SchedulerOutput:
            assert defer_finished_cleanup is True
            return SchedulerOutput(
                outputs=[RequestOutput(request_id="req", finished=True)],
                finished_request_ids={"req"},
            )

        def _cleanup_finished_after_terminal_dispatch(
            self, finished_ids: set[str]
        ) -> None:
            assert finished_ids == {"req"}
            assert not engine._terminal_cleanup_complete.is_set()
            order.append("cleanup")
            engine._running = False

    engine.scheduler = _Scheduler()
    engine.config = SimpleNamespace(step_interval=0.0, stream_interval=1)
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {"req": _Collector()}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._terminal_cleanup_complete = asyncio.Event()
    engine._terminal_cleanup_complete.set()

    await engine._engine_loop()

    assert order == ["dispatch", "cleanup"]
    assert engine._terminal_cleanup_complete.is_set()


def test_llm_deferred_cleanup_clears_allocator_after_cleanup_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final clear must run after text cache-owning cleanup locals are gone."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running = {}
    order: list[str] = []

    def _cleanup(self, finished_ids: set[str]) -> None:
        assert finished_ids == {"req"}
        order.append("cleanup-returned")

    scheduler._cleanup_finished = MethodType(_cleanup, scheduler)
    monkeypatch.setattr(
        "vmlx_engine.scheduler.clear_mlx_memory_cache",
        lambda **_kwargs: order.append("allocator-cleared"),
    )

    scheduler._cleanup_finished_after_terminal_dispatch({"req"})

    assert order == ["cleanup-returned", "allocator-cleared"]


@pytest.mark.asyncio
async def test_mllm_loop_dispatches_terminal_before_worker_cleanup() -> None:
    order: list[str] = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._running = True
    scheduler._step_executor = ThreadPoolExecutor(max_workers=1)
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._terminal_cleanup_complete.set()

    def _has_requests(self) -> bool:
        return self._running

    def _step(self, *, defer_finished_cleanup: bool = False) -> MLLMSchedulerOutput:
        assert defer_finished_cleanup is True
        return MLLMSchedulerOutput(
            outputs=[RequestOutput(request_id="req", finished=True)],
            finished_request_ids={"req"},
        )

    def _dispatch(self, output: MLLMSchedulerOutput) -> None:
        assert output.outputs[0].finished
        assert not self._terminal_cleanup_complete.is_set()
        order.append("dispatch")

    def _cleanup(self, finished_ids: set[str]) -> None:
        assert finished_ids == {"req"}
        assert not self._terminal_cleanup_complete.is_set()
        order.append("cleanup")
        self._running = False

    scheduler.has_requests = MethodType(_has_requests, scheduler)
    scheduler.step = MethodType(_step, scheduler)
    scheduler._dispatch_outputs = MethodType(_dispatch, scheduler)
    scheduler._cleanup_finished_after_terminal_dispatch = MethodType(
        _cleanup, scheduler
    )

    try:
        await scheduler._process_loop()
    finally:
        scheduler._step_executor.shutdown(wait=True)

    assert order == ["dispatch", "cleanup"]
    assert scheduler._terminal_cleanup_complete.is_set()


def test_mllm_deferred_cleanup_clears_allocator_after_cleanup_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final clear must run after cache-owning cleanup locals are gone."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.running = {}
    scheduler._queue_lock = Lock()
    order: list[str] = []

    def _cleanup(self, finished_ids: set[str]) -> None:
        assert finished_ids == {"req"}
        order.append("cleanup-returned")

    scheduler._cleanup_finished = MethodType(_cleanup, scheduler)
    monkeypatch.setattr(
        "vmlx_engine.mllm_scheduler.clear_mlx_memory_cache",
        lambda **_kwargs: order.append("allocator-cleared"),
    )

    scheduler._cleanup_finished_after_terminal_dispatch({"req"})

    assert order == ["cleanup-returned", "allocator-cleared"]


@pytest.mark.asyncio
async def test_llm_request_admission_waits_for_terminal_cache_cleanup() -> None:
    added: list[str] = []
    pending_admissions: set[str] = set()
    engine = EngineCore.__new__(EngineCore)

    class _Scheduler:
        def _begin_foreground_admission(self, request_id: str) -> None:
            pending_admissions.add(request_id)

        def _end_foreground_admission(self, request_id: str) -> None:
            pending_admissions.discard(request_id)

        def add_request(self, request) -> None:
            added.append(request.request_id)

    engine.scheduler = _Scheduler()
    engine.config = SimpleNamespace(stream_interval=1)
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._terminal_cleanup_complete = asyncio.Event()

    pending = asyncio.create_task(
        engine.add_request(prompt="next turn", request_id="next")
    )
    await asyncio.sleep(0)
    assert not pending.done()
    assert added == []
    assert pending_admissions == {"next"}

    engine._terminal_cleanup_complete.set()
    assert await pending == "next"
    assert added == ["next"]
    assert pending_admissions == set()


def _queued_terminal_engine() -> EngineCore:
    engine = EngineCore.__new__(EngineCore)
    engine._terminal_cleanup_complete = asyncio.Event()

    class _Collector:
        def __init__(self) -> None:
            self._items = [RequestOutput(request_id="req", finished=True)]

        def get_nowait(self):
            return self._items.pop(0) if self._items else None

        async def get(self):
            raise AssertionError("terminal object was already queued")

    engine._output_collectors = {"req": _Collector()}
    engine._stream_states = {}
    engine._finished_events = {}
    return engine


@pytest.mark.asyncio
async def test_llm_stream_publishes_terminal_before_cache_cleanup() -> None:
    engine = _queued_terminal_engine()
    aborted: list[str] = []
    engine.scheduler = SimpleNamespace(
        get_request=lambda _request_id: None,
        abort_request=aborted.append,
    )

    stream = engine.stream_outputs("req")
    output = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert output.finished is True
    assert not engine._terminal_cleanup_complete.is_set()
    await stream.aclose()

    # Scheduler bookkeeping for the published request waits for the cleanup
    # still running on the model worker, then runs exactly as before.
    await asyncio.sleep(0)
    assert aborted == []
    engine._terminal_cleanup_complete.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert aborted == ["req"]


@pytest.mark.asyncio
async def test_llm_stream_holds_terminal_until_cache_cleanup_in_strict_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VMLX_TERMINAL_PUBLISH", "strict")
    engine = EngineCore.__new__(EngineCore)
    engine._terminal_cleanup_complete = asyncio.Event()

    class _Collector:
        def __init__(self) -> None:
            self._items = [RequestOutput(request_id="req", finished=True)]

        def get_nowait(self):
            return self._items.pop(0) if self._items else None

        async def get(self):
            raise AssertionError("terminal object was already queued")

    engine._output_collectors = {"req": _Collector()}
    engine._stream_states = {}
    engine._finished_events = {}
    engine.scheduler = SimpleNamespace(
        get_request=lambda _request_id: None,
        abort_request=lambda _request_id: None,
    )

    stream = engine.stream_outputs("req")
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert not pending.done()

    engine._terminal_cleanup_complete.set()
    output = await pending
    assert output.finished is True
    await stream.aclose()


@pytest.mark.asyncio
async def test_mllm_stream_publishes_terminal_before_cache_cleanup() -> None:
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler.output_queues = {"req": asyncio.Queue()}
    scheduler.running = {}
    scheduler.output_queues["req"].put_nowait(
        RequestOutput(request_id="req", finished=True)
    )

    stream = scheduler.stream_outputs("req")
    output = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert output.finished is True
    assert not scheduler._terminal_cleanup_complete.is_set()
    await stream.aclose()


def test_mllm_settled_persistence_records_outcome_after_dispatch() -> None:
    import time as _time

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    stats = SimpleNamespace(last_durability=None)
    scheduler.batch_generator = SimpleNamespace(_stats=stats)
    scheduler._terminal_dispatched_at = _time.perf_counter() - 0.25
    LEDGER.record("settle-a", "stored", "paged 4 blocks", retained_tokens=4281)
    LEDGER.record("settle-b", "skipped", "partial output")

    scheduler._record_terminal_persistence(
        {"settle-a", "settle-b"},
        [
            RequestOutput(request_id="settle-a", finished=True, finish_reason="stop"),
            RequestOutput(request_id="settle-b", finished=True, finish_reason="aborted"),
        ],
    )

    record = stats.last_durability
    assert record["request_id"] == "settle-a"
    assert record["cache_outcome"] == "stored"
    assert record["retained_tokens"] == 4281
    assert record["published_early"] is True
    assert record["wait_ms"] >= 250.0
    assert LEDGER.peek("settle-a") is None
    # An aborted partial publishes no prefix and gets no durability record.
    assert LEDGER.peek("settle-b") is not None
    LEDGER.take("settle-b")


def test_mllm_dispatch_reports_admission_wait_beside_prefill_usage() -> None:
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"req": asyncio.Queue()}
    scheduler._admission_waits_by_request = {"req": 0.4}
    usage = {"tokens": 4281, "seconds": 1.5, "scope": "model_prefill_and_prompt_state"}
    step = MLLMSchedulerOutput(
        outputs=[RequestOutput(request_id="req", prefill_usage=usage)]
    )

    scheduler._dispatch_outputs(step)

    delivered = scheduler.output_queues["req"].get_nowait()
    assert delivered.prefill_usage == {**usage, "admission_wait_seconds": 0.4}
    # The prefill measurement itself is untouched: pp/s never absorbs the wait.
    assert usage == {"tokens": 4281, "seconds": 1.5, "scope": "model_prefill_and_prompt_state"}


@pytest.mark.asyncio
async def test_mllm_stream_holds_terminal_until_cache_cleanup_in_strict_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VMLX_TERMINAL_PUBLISH", "strict")
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler.output_queues = {"req": asyncio.Queue()}
    scheduler.running = {}
    scheduler.output_queues["req"].put_nowait(
        RequestOutput(request_id="req", finished=True)
    )

    stream = scheduler.stream_outputs("req")
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert not pending.done()

    scheduler._terminal_cleanup_complete.set()
    output = await pending
    assert output.finished is True
    await stream.aclose()


def test_llm_pending_admission_parks_idle_tasks_before_scheduler_lookup() -> None:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = []
    scheduler.running = {}
    scheduler.unprocessed_requests = []
    scheduler._pending_aborts = set()

    assert scheduler._foreground_pending() is False
    scheduler._begin_foreground_admission("next")
    assert scheduler._foreground_pending() is True
    scheduler._end_foreground_admission("next")
    assert scheduler._foreground_pending() is False


@pytest.mark.asyncio
async def test_mllm_async_admission_waits_for_terminal_cache_cleanup() -> None:
    added: list[str] = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler.output_queues = {}

    def _add_request(self, **kwargs) -> str:
        added.append(kwargs["prompt"])
        return "next-vl"

    scheduler.add_request = MethodType(_add_request, scheduler)
    pending = asyncio.create_task(scheduler.add_request_async(prompt="next media turn"))
    await asyncio.sleep(0)
    assert not pending.done()
    assert added == []

    scheduler._terminal_cleanup_complete.set()
    assert await pending == "next-vl"
    assert added == ["next media turn"]


@pytest.mark.asyncio
async def test_llm_stop_waits_for_terminal_cache_cleanup_before_cancelling() -> None:
    order: list[str] = []
    engine = EngineCore.__new__(EngineCore)

    class _Scheduler:
        def shutdown(self) -> None:
            order.append("shutdown")

    async def _loop() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            order.append("cancel")
            raise

    engine.scheduler = _Scheduler()
    engine._running = True
    engine._terminal_cleanup_complete = asyncio.Event()
    engine._task = asyncio.create_task(_loop())
    await asyncio.sleep(0)

    stopping = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not stopping.done()
    assert order == []

    engine._terminal_cleanup_complete.set()
    await stopping

    assert order == ["cancel", "shutdown"]
    assert engine._task is None


@pytest.mark.asyncio
async def test_mllm_stop_waits_for_terminal_cache_cleanup_before_cancelling() -> None:
    order: list[str] = []
    scheduler = MLLMScheduler.__new__(MLLMScheduler)

    async def _loop() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            order.append("cancel")
            raise

    scheduler._running = True
    scheduler._terminal_cleanup_complete = asyncio.Event()
    scheduler._processing_task = asyncio.create_task(_loop())
    scheduler.batch_generator = None
    await asyncio.sleep(0)

    stopping = asyncio.create_task(scheduler.stop())
    await asyncio.sleep(0)
    assert not stopping.done()
    assert order == []

    scheduler._terminal_cleanup_complete.set()
    await stopping

    assert order == ["cancel"]
    assert scheduler._running is False


def test_direct_scheduler_steps_retain_synchronous_cleanup_default() -> None:
    llm_source = inspect.getsource(Scheduler.step)
    mllm_source = inspect.getsource(MLLMScheduler.step)

    assert "defer_finished_cleanup: bool = False" in llm_source
    assert "if not defer_finished_cleanup" in llm_source
    assert "defer_finished_cleanup: bool = False" in mllm_source
    assert "if not defer_finished_cleanup" in mllm_source


def test_finished_stream_consumers_do_not_abort_deferred_cleanup() -> None:
    llm_source = inspect.getsource(EngineCore._abort_unfinished)
    mllm_source = inspect.getsource(MLLMScheduler.stream_outputs)

    assert "RequestStatus.is_finished(request.status)" in llm_source
    assert "_abort_unfinished_after_cleanup" in inspect.getsource(
        EngineCore._cleanup_request
    )
    assert "RequestStatus.is_finished(request.status)" in mllm_source


def test_m3_hit_rederive_materializes_in_post_dispatch_cleanup_shape() -> None:
    scheduler = Scheduler.__new__(Scheduler)
    order: list[str] = []

    def _prefill(tokens):
        order.append("rederive")
        assert tokens == [11, 12, 13]
        return ["native-m3-cache"]

    def _extract(cache):
        order.append("extract")
        assert cache == ["native-m3-cache"]
        return [{"class_name": "MiniMaxM3SparseCache", "state": "typed"}]

    scheduler._prefill_for_prompt_only_cache = _prefill
    scheduler._extract_cache_states = _extract
    scheduler._kv_cache_bits = 0
    scheduler.disk_cache = None
    scheduler._is_hybrid = False
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13, 14],
        _deferred_prompt_cache={
            "family": "MiniMax-M3",
            "mode": "paged",
            "key_tokens": [11, 12, 13],
        },
    )

    # Terminal delivery happens in EngineCore before this cleanup helper is
    # called; the descriptor itself must not do any eager work.
    order.append("terminal-dispatch")
    scheduler._materialize_deferred_prompt_cache("m3", request)

    assert order == ["terminal-dispatch", "rederive", "extract"]
    assert request._deferred_prompt_cache is None
    assert request._extracted_cache_key_tokens == [11, 12, 13]
    assert request._extracted_cache_from_prompt_snapshot is True
    assert request._extracted_cache == [
        {"class_name": "MiniMaxM3SparseCache", "state": "typed"}
    ]


def test_m3_response_finalization_only_schedules_clean_rederive() -> None:
    source = inspect.getsource(Scheduler._process_batch_responses)
    paged_marker = '"mode": "paged"'
    object_marker = '"mode": "object"'

    assert source.count("request._deferred_prompt_cache = {") == 6
    assert paged_marker in source
    assert object_marker in source

    # Both MiniMax-M3 hit branches now leave the expensive prefill for
    # _cleanup_finished. Other architecture-specific rederive branches are
    # audited separately and may still call this helper in finalization.
    for marker in (paged_marker, object_marker):
        branch = source[source.index(marker) - 900 : source.index(marker) + 250]
        assert "_prefill_for_prompt_only_cache" not in branch


def test_all_llm_terminal_response_paths_defer_clean_prompt_reprefill() -> None:
    source = inspect.getsource(Scheduler._process_batch_responses)

    assert "_prefill_for_prompt_only_cache" not in source
    for family in ("DSV4", "ZAYA", "Mixed-SWA", "MiniMax-M3"):
        assert f'"family": "{family}"' in source
    assert source.count('"mode": "paged"') == 4
    assert source.count('"mode": "object"') == 2


def test_admission_wait_is_reported_even_without_a_prefill_receipt() -> None:
    from vmlx_engine.utils.terminal_persistence import with_admission_wait

    receipt = with_admission_wait(None, 0.4567891)
    assert receipt == {"scope": "admission_only", "admission_wait_seconds": 0.456789}
    # A real engine receipt keeps its own measurement and gains the wait.
    usage = {"tokens": 10, "seconds": 0.5, "scope": "model_prefill_and_prompt_state"}
    assert with_admission_wait(usage, 0.1) == {**usage, "admission_wait_seconds": 0.1}
    assert with_admission_wait(usage, None) is usage


def test_llm_dispatch_attaches_admission_only_receipt() -> None:
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"req": asyncio.Queue()}
    scheduler._admission_waits_by_request = {"req": 0.25}
    scheduler._dispatch_outputs(MLLMSchedulerOutput(outputs=[RequestOutput(request_id="req")]))
    delivered = scheduler.output_queues["req"].get_nowait()
    assert delivered.prefill_usage == {"scope": "admission_only", "admission_wait_seconds": 0.25}


def test_later_outputs_keep_the_real_receipt_instead_of_admission_only() -> None:
    """Servers keep the latest non-empty prefill_usage: a decode output after
    the real receipt must not replace it with an admission-only one."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"req": asyncio.Queue()}
    scheduler._admission_waits_by_request = {"req": 0.2}
    usage = {"tokens": 64, "seconds": 0.5, "scope": "model_prefill_and_prompt_state"}
    scheduler._dispatch_outputs(MLLMSchedulerOutput(outputs=[RequestOutput(request_id="req")]))
    scheduler._dispatch_outputs(MLLMSchedulerOutput(outputs=[RequestOutput(request_id="req", prefill_usage=usage)]))
    scheduler._dispatch_outputs(MLLMSchedulerOutput(outputs=[RequestOutput(request_id="req")]))
    q = scheduler.output_queues["req"]
    before, real, after = q.get_nowait(), q.get_nowait(), q.get_nowait()
    assert before.prefill_usage == {"scope": "admission_only", "admission_wait_seconds": 0.2}
    assert real.prefill_usage == {**usage, "admission_wait_seconds": 0.2}
    assert after.prefill_usage == real.prefill_usage
