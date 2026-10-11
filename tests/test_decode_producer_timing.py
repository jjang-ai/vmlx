"""Decode telemetry must not measure terminal SSD publication latency."""
from types import SimpleNamespace

import pytest

from vmlx_engine.request import RequestOutput
from vmlx_engine.engine.base import GenerationOutput


def test_request_output_timestamps_production_not_later_delivery(monkeypatch):
    import vmlx_engine.request as requests
    clock = [10.0]
    monkeypatch.setattr(requests.time, "perf_counter", lambda: clock[0])
    output = RequestOutput(request_id="terminal", completion_tokens=9, finished=True)
    clock[0] = 15.0  # simulate waiting for SSD durability before publication
    assert output.generated_at == 10.0


def test_decode_prefers_producer_clock_over_post_fence_observation(monkeypatch):
    import vmlx_engine.server as server
    monkeypatch.setattr(server.time, "perf_counter", lambda: 15.0)
    output = GenerationOutput(text="done", completion_tokens=9, generated_at=10.0)
    assert server._decode_output_timestamp(output) == 10.0


@pytest.mark.parametrize("timestamp", [None, float("nan"), float("inf"), -1, 0, True, "10"])
def test_absent_or_invalid_producer_clock_uses_observation(monkeypatch, timestamp):
    import vmlx_engine.server as server
    monkeypatch.setattr(server.time, "perf_counter", lambda: 15.0)
    assert server._decode_output_timestamp(SimpleNamespace(generated_at=timestamp)) == 15.0


def test_first_batched_output_does_not_invent_unobserved_intervals():
    from vmlx_engine.server import _decode_usage_snapshot
    # First observed output already contains four accepted tokens.
    receipt = _decode_usage_snapshot(
        completion_tokens=12, first_token_count=4,
        first_token_ts=10.0, last_token_ts=12.0,
    )
    assert receipt == {"tokens": 8, "seconds": 2.0, "tokens_per_second": 4.0}
    assert _decode_usage_snapshot(
        completion_tokens=4, first_token_count=4,
        first_token_ts=10.0, last_token_ts=12.0,
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mllm", [False, True])
async def test_batched_adapters_preserve_scheduler_timestamp(mllm):
    from vmlx_engine.engine.batched import BatchedEngine

    class Scheduler:
        async def add_request(self, **kwargs):
            return "timing"

        add_request_async = add_request

        async def stream_outputs(self, request_id):
            yield RequestOutput(
                request_id=request_id, output_text="AB", new_text="AB",
                completion_tokens=12, generated_at=12.0, finished=True,
                prefill_usage={"tokens": 7, "seconds": 0.25, "scope": "model_prefill_and_prompt_state"},
            )

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = mllm
    engine._engine = Scheduler()
    engine._mllm_scheduler = Scheduler() if mllm else None
    outputs = [output async for output in engine.stream_generate("hi")]
    assert len(outputs) == 1
    assert outputs[0].generated_at == 12.0
    assert outputs[0].completion_tokens == 12
    assert outputs[0].prefill_usage["tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["chat", "responses"])
@pytest.mark.parametrize("telemetry", [True, False])
async def test_stream_uses_producer_window_and_preserves_terminal_usage(monkeypatch, caplog, dialect, telemetry):
    import json
    from unittest.mock import AsyncMock
    import vmlx_engine.server as server
    from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest, StreamOptions

    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)

        async def stream_chat(self, **kwargs):
            yield GenerationOutput(
                text="A", new_text="A", prompt_tokens=20, completion_tokens=4,
                generated_at=10.0, finished=False,
                prefill_usage={"tokens": 20, "seconds": 0.5, "scope": "model_prefill_and_prompt_state"},
            )
            # Delivery occurs now, but generation was observed at t=12 before
            # the scheduler's terminal cleanup. Do not timestamp this in server.
            yield GenerationOutput(
                text="AB", new_text="B", prompt_tokens=20, completion_tokens=12,
                generated_at=12.0, finished=True,
            )

    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", "timing-test")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    options = dict(model="timing-test", stream=True, stream_options=StreamOptions(include_usage=True))
    messages = [{"role": "user", "content": "hi"}]
    http = SimpleNamespace(headers={"x-vmlx-stream-usage": "incremental"} if telemetry else {}, is_disconnected=AsyncMock(return_value=False))
    if dialect == "responses":
        request = ResponsesRequest(input="hi", **options)
        iterator = server.stream_responses_api(Engine(), messages, request, fastapi_request=http)
    else:
        request = ChatCompletionRequest(messages=messages, **options)
        iterator = server.stream_chat_completion(Engine(), messages, request, fastapi_request=http)
    with caplog.at_level("INFO"):
        events = [event async for event in iterator]
    assert "(4.0 tok/s decode)" in caplog.text
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payloads.append(json.loads(line[6:]))
    if dialect == "responses":
        terminal = [p for p in payloads if p.get("type") == "response.completed"][-1]
        assert terminal["response"]["usage"]["output_tokens"] == 12
        if telemetry:
            private = [p for p in payloads if p.get("type") == "response.usage"][-1]
            assert private["usage"]["vmlx_prefill"] == {"tokens": 20, "seconds": 0.5, "scope": "model_prefill_and_prompt_state"}
            assert "vmlx_prefill" not in terminal["response"]["usage"]
            assert private["usage"]["vmlx_decode"]["tokens"] == 8
            assert private["usage"]["vmlx_decode"]["seconds"] == 2.0
        else:
            assert not [p for p in payloads if p.get("type") == "response.usage"]
    else:
        assert [p["usage"]["completion_tokens"] for p in payloads if p.get("usage")][-1] == 12
        last_usage = [p["usage"] for p in payloads if p.get("usage")][-1]
        assert ("vmlx_prefill" in last_usage) is telemetry
        assert ("vmlx_decode" in last_usage) is telemetry
        if telemetry:
            assert last_usage["vmlx_prefill"]["seconds"] == 0.5
            assert last_usage["vmlx_decode"] == {"tokens": 8, "seconds": 2.0, "tokens_per_second": 4.0}
        assert sum(event.count("data: [DONE]") for event in events) == 1


@pytest.mark.parametrize('has_receipt', [False, True])
def test_text_scheduler_forwards_prefill_receipt_through_terminal(monkeypatch, has_receipt):
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler
    request = Request(request_id='prefill-clock', prompt=[1, 2], sampling_params=SamplingParams(max_tokens=3))
    request.prompt_token_ids = [1, 2]
    request.num_prompt_tokens = 2
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(enable_prefix_cache=False)
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0

    class Detok:
        text = ''
        def add_token(self, token):
            self.text += 'x'
        def finalize(self):
            pass

    detok = Detok()
    monkeypatch.setattr(Scheduler, '_get_detokenizer', lambda *args: detok)
    monkeypatch.setattr(Scheduler, '_advance_request_state_machine', lambda *args: None)
    monkeypatch.setattr(Scheduler, '_queue_dsv4_shadow_rekey', lambda *args: None)
    receipt = {'tokens': 2, 'seconds': .25, 'scope': 'model_prefill_and_prompt_state'}
    for token, finish in [(3, None), (0, 'stop')]:
        response = SimpleNamespace(uid=1, token=token, finish_reason=finish)
        if has_receipt:
            response.prefill_usage = receipt
        outputs, finished = scheduler._process_batch_responses([response])
        assert outputs[0].prefill_usage == (receipt if has_receipt else None)
        assert outputs[0].finished is (finish is not None)


@pytest.mark.parametrize('generator_receipt', [False, True])
def test_text_scheduler_times_prefill_for_generators_without_a_receipt(monkeypatch, generator_receipt):
    """Stock mlx_lm BatchGenerator (native-MTP text, max_num_seqs>1) and DSV4 emit no
    usage.vmlx_prefill; the scheduler times insert -> first token so the chat footer
    still shows pp/s. A generator's own receipt always wins."""
    import time as _time
    from vmlx_engine.request import Request, SamplingParams
    from vmlx_engine.scheduler import Scheduler
    request = Request(request_id='prefill-fallback', prompt=[1, 2, 3], sampling_params=SamplingParams(max_tokens=3))
    request.prompt_token_ids = [1, 2, 3]
    request.num_prompt_tokens = 3
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(enable_prefix_cache=False)
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.stop_tokens = {0}
    scheduler._pld_spec_enabled = False
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0

    class Detok:
        text = ''
        def add_token(self, token):
            self.text += 'x'
        def finalize(self):
            pass

    detok = Detok()
    monkeypatch.setattr(Scheduler, '_get_detokenizer', lambda *args: detok)
    monkeypatch.setattr(Scheduler, '_advance_request_state_machine', lambda *args: None)
    monkeypatch.setattr(Scheduler, '_queue_dsv4_shadow_rekey', lambda *args: None)
    # Two uncached tokens inserted after a one-token cache restore, 0.5 s before the first token.
    request._vmlx_prefill_insert = (_time.perf_counter() - 0.5, 2)
    own = {'tokens': 2, 'seconds': .25, 'scope': 'model_prefill_and_prompt_state'}
    seen = []
    for token, finish in [(3, None), (4, None), (0, 'stop')]:
        response = SimpleNamespace(uid=1, token=token, finish_reason=finish)
        if generator_receipt:
            response.prefill_usage = own
        outputs, _ = scheduler._process_batch_responses([response])
        seen.append(outputs[0].prefill_usage)
    if generator_receipt:
        assert seen == [own, own, own]
    else:
        receipt = seen[0]
        assert receipt['tokens'] == 2 and receipt['scope'] == 'model_prefill_and_prompt_state'
        assert 0.5 <= receipt['seconds'] < 5.0
        assert receipt['path'] == 'scheduler_first_token'
        assert seen == [receipt, receipt, receipt]
    assert request._vmlx_prefill_insert is None
