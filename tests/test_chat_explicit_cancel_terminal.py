# SPDX-License-Identifier: Apache-2.0
"""T01: explicit scheduler cancellation must not become Chat/Messages success.

Model-free owning regression. Reuse the B01 real queue/abort/stream fixture and
BatchedEngine bridge; only request setup is stubbed. Native nonstream tests call
the actual route function with a request-body shim (not socket/ASGI/live proof).
No model, Metal, cache-publication, family-wide or GPU correctness is claimed.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_mllm_explicit_cancel_terminal import _queue_partial, _scheduler
from vmlx_engine.api.anthropic_adapter import AnthropicStreamAdapter
from vmlx_engine.api.models import ChatCompletionRequest, StreamOptions
from vmlx_engine.engine.batched import BatchedEngine
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser
from vmlx_engine.request import RequestOutput, RequestStatus
from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser


MODEL = "terminal-cancel-fixture"
TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Read a value.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]},
}}]


def _bridge(monkeypatch, *, parts, cancel=True, finish="stop", reasoning=False, tools=False):
    import vmlx_engine.server as server

    class Bridge:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = True
        preserve_native_tool_format = False

        def __init__(self):
            self.calls = 0
            self.abort_calls = []
            self.outputs = []
            self.scheduler = None
            self.request = None
            self.engine = None

        async def abort_request(self, rid):
            self.abort_calls.append(rid)
            return await self.engine.abort_request(rid)

        async def stream_chat(self, **kwargs):
            self.calls += 1
            assert self.calls == 1, "Cancellation must not initiate another generation"
            rid = kwargs["request_id"]
            scheduler, request = _scheduler(monkeypatch, request_id=rid)
            self.scheduler, self.request = scheduler, request
            _queue_partial(scheduler, request, parts)

            async def existing_request(**setup):
                assert setup["request_id"] == rid
                return rid

            scheduler.add_request_async = existing_request
            engine = BatchedEngine.__new__(BatchedEngine)
            engine._loaded = True
            engine._is_mllm = True
            engine._mllm_scheduler = scheduler
            self.engine = engine
            if not cancel:
                request.status = (RequestStatus.FINISHED_LENGTH_CAPPED if finish == "length"
                                  else RequestStatus.FINISHED_STOPPED)
                request.finish_reason = finish
                terminal = RequestOutput(
                    request_id=rid, output_text="".join(parts), finished=True,
                    finish_reason=finish, prompt_tokens=64, completion_tokens=len(parts),
                    cached_tokens=32, cache_detail="block-disk+ssm",
                )
                scheduler.output_queues[rid].put_nowait(terminal)
                scheduler.output_queues[rid].put_nowait(None)
            cancelled = False
            if cancel and not parts:
                assert (await server.cancel_chat_completion(rid))["success"] is True
                cancelled = True
            async for output in engine.stream_generate(
                prompt="fixture", request_id=rid, max_tokens=16, temperature=0,
            ):
                self.outputs.append(output)
                yield output
                if cancel and not cancelled:
                    assert (await server.cancel_chat_completion(rid))["success"] is True
                    cancelled = True

    bridge = Bridge()
    monkeypatch.setattr(server, "_engine", bridge)
    monkeypatch.setattr(server, "get_engine", lambda: bridge)
    monkeypatch.setattr(server, "_default_timeout", 2.0)
    monkeypatch.setattr(server, "_model_name", MODEL)
    monkeypatch.setattr(server, "_served_model_name", MODEL)
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser() if reasoning else None)
    # Production stores a registry name, not a parser instance. An instance
    # misses registry lookup and silently exercises the generic fallback.
    monkeypatch.setattr(server, "_tool_call_parser", "qwen" if tools else None)
    if tools:
        assert server.ToolParserManager.get_tool_parser(server._tool_call_parser) is QwenToolParser
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_mcp_manager", None)
    return bridge


def _rows(frames):
    return [json.loads(line[6:]) for frame in frames for line in frame.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def _choices(rows):
    return [choice for row in rows for choice in row.get("choices", [])]


def _assert_upstream(bridge, count, finish):
    assert bridge.calls == 1
    terminals = [output for output in bridge.outputs if output.finished]
    assert len(terminals) == 1 and bridge.outputs[-1] is terminals[0]
    terminal = terminals[0]
    assert terminal.finish_reason == finish
    assert terminal.prompt_tokens == 64 and terminal.completion_tokens == count
    assert terminal.cached_tokens == 32 and terminal.cache_detail == "block-disk+ssm"
    assert not terminal.error
    if finish == "aborted":
        assert bridge.abort_calls == [bridge.request.request_id]
        assert bridge.request.status == RequestStatus.FINISHED_ABORTED
        assert bridge.request.cancel_event.is_set()
        # Collection has finished: stream_outputs owns these two removals in
        # finally, independently of the deferred worker/tensor cleanup below.
        rid = bridge.request.request_id
        assert rid not in bridge.scheduler.output_queues
        assert rid not in bridge.scheduler._stream_abort_outputs
        bridge.scheduler.batch_generator.remove.assert_not_called()
    else:
        assert bridge.abort_calls == []


async def _chat(bridge, *, reasoning=False, tools=False, required=False, include_usage=True):
    import vmlx_engine.server as server

    request = ChatCompletionRequest(
        model=MODEL, messages=[{"role": "user", "content": "Continue the task."}],
        stream=True, stream_options=StreamOptions(include_usage=include_usage),
        enable_thinking=reasoning, max_tokens=16,
        tools=TOOLS if tools else None, tool_choice="required" if required else None,
    )
    upstream = []

    async def capture():
        async for frame in server.stream_chat_completion(
            bridge, [{"role": "user", "content": "Continue the task."}], request,
            response_id="chatcmpl-t01-cancel", max_tokens=16, temperature=0,
        ):
            upstream.append(frame)
            yield frame

    async def collect():
        return [frame async for frame in server._terminal_finish_guard(
            capture(), required_tool_call=required)]

    frames = await asyncio.wait_for(collect(), timeout=2)
    return upstream, frames


def _cancel_contract(upstream, frames, count, *, include_usage=True):
    for captured in (upstream, frames):
        rows = _rows(captured)
        errors = [row for row in rows if row.get("error")]
        assert len(errors) == 1
        assert errors[0]["error"]["type"] == "invalid_request_error"
        assert errors[0]["error"]["code"] == "request_cancelled"
        assert errors[0]["error"]["message"]
        assert not any(choice.get("finish_reason") for choice in _choices(rows))
        assert sum(frame.strip() == "data: [DONE]" for frame in captured) == 1
        assert captured[-1].strip() == "data: [DONE]"
        usage_rows = [row for row in rows if row.get("usage") and not row.get("error")]
        usage = errors[0]["usage"]
        assert usage["prompt_tokens"] == 64 and usage["completion_tokens"] == count
        assert usage["total_tokens"] == 64 + count
        assert usage["prompt_tokens_details"]["cached_tokens"] == 32
        assert usage["prompt_tokens_details"]["cache_detail"] == "block-disk+ssm"
        if include_usage:
            assert len(usage_rows) == 1 and usage_rows[0]["choices"] == []
            assert usage_rows[0]["usage"] == usage
            assert rows.index(errors[0]) < rows.index(usage_rows[0]) == len(rows) - 1
        else:
            assert usage_rows == []
            assert rows[-1] is errors[0]
    # Guard must not append a synthetic stop/tool-required error to cancellation.
    assert frames == upstream
    adapter = AnthropicStreamAdapter(MODEL, "msg_t01_cancel")
    translated = []
    for frame in frames:
        translated.extend(adapter.process_chunk(frame))
    translated.extend(adapter.finalize())
    assert adapter.finalize() == []
    native = _rows(translated)
    errors = [row for row in native if row.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["type"] == "invalid_request_error"
    assert errors[0]["error"]["code"] == "request_cancelled"
    assert errors[0]["usage"] == {"input_tokens": 32, "output_tokens": count,
                                   "cache_read_input_tokens": 32,
                                   "cache_creation_input_tokens": 0}
    assert not any(row.get("type") in {"message_delta", "message_stop"} for row in native)
    return _rows(frames), native


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_explicit_cancel_chat_guard_and_messages_are_typed_not_success(monkeypatch, partial):
    parts = ["1\n", "2\n"] if partial else []
    bridge = _bridge(monkeypatch, parts=parts)
    upstream, frames = await _chat(bridge)
    rows, native = _cancel_contract(upstream, frames, len(parts))
    _assert_upstream(bridge, len(parts), "aborted")
    expected = "".join(parts)
    assert "".join(choice.get("delta", {}).get("content", "") or "" for choice in _choices(rows)) == expected
    assert "".join(row.get("delta", {}).get("text", "") for row in native) == expected
    assert "".join(output.new_text for output in bridge.outputs) == expected


@pytest.mark.asyncio
async def test_cancel_without_usage_tail_keeps_error_usage_and_partial_content(monkeypatch):
    parts = ["1\n", "2\n"]
    bridge = _bridge(monkeypatch, parts=parts)
    upstream, frames = await _chat(bridge, include_usage=False)
    rows, native = _cancel_contract(upstream, frames, 2, include_usage=False)
    _assert_upstream(bridge, 2, "aborted")
    expected = "".join(parts)
    assert "".join(choice.get("delta", {}).get("content", "") or "" for choice in _choices(rows)) == expected
    assert "".join(row.get("delta", {}).get("text", "") for row in native) == expected
    assert "".join(output.new_text for output in bridge.outputs) == expected


@pytest.mark.asyncio
async def test_reasoning_only_cancel_does_not_promote_thought_or_regenerate(monkeypatch):
    parts = ["<think>Check sum. ", "Still reasoning."]
    bridge = _bridge(monkeypatch, parts=parts, reasoning=True)
    upstream, frames = await _chat(bridge, reasoning=True)
    rows, native = _cancel_contract(upstream, frames, 2)
    _assert_upstream(bridge, 2, "aborted")
    deltas = [choice.get("delta", {}) for choice in _choices(rows)]
    assert not any(delta.get("content") or delta.get("tool_calls") for delta in deltas)
    thought = "".join(delta.get("reasoning_content") or delta.get("reasoning") or "" for delta in deltas)
    assert thought == "Check sum. Still reasoning."
    assert "".join(row.get("delta", {}).get("thinking", "") for row in native) == thought


@pytest.mark.asyncio
async def test_buffered_tool_cancel_skips_final_parse_and_required_tool_promotion(monkeypatch):
    import vmlx_engine.server as server

    parts = ['<tool_call>{"name":"lookup",', '"arguments":{"query":"alpha"}']
    bridge = _bridge(monkeypatch, parts=parts, tools=True)
    parse = Mock(wraps=server._parse_tool_calls_with_parser)
    monkeypatch.setattr(server, "_parse_tool_calls_with_parser", parse)
    upstream, frames = await _chat(bridge, tools=True, required=True)
    rows, native = _cancel_contract(upstream, frames, 2)
    _assert_upstream(bridge, 2, "aborted")
    parse.assert_not_called()
    assert not any(choice.get("delta", {}).get("tool_calls") for choice in _choices(rows))
    assert not any(choice.get("delta", {}).get("content") for choice in _choices(rows))
    assert not any(row.get("content_block", {}).get("type") == "tool_use" for row in native)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,tools,expected", [
    ("stop", False, "end_turn"), ("length", False, "max_tokens"),
    ("stop", True, "tool_use"),
])
async def test_natural_stop_length_and_tool_keep_existing_success_semantics(monkeypatch, finish, tools, expected):
    parts = (['<tool_call>{"name":"lookup",', '"arguments":{"query":"alpha"}}</tool_call>']
             if tools else ["Natural ", "answer."])
    bridge = _bridge(monkeypatch, parts=parts, cancel=False, finish=finish, tools=tools)
    _, frames = await _chat(bridge, tools=tools)
    rows = _rows(frames)
    assert not any(row.get("error") for row in rows)
    reasons = [choice["finish_reason"] for choice in _choices(rows) if choice.get("finish_reason")]
    assert reasons == ["tool_calls" if tools else finish]
    _assert_upstream(bridge, 2, finish)
    adapter = AnthropicStreamAdapter(MODEL)
    native = []
    for frame in frames:
        native.extend(adapter.process_chunk(frame))
    native.extend(adapter.finalize())
    native = _rows(native)
    assert [row["delta"]["stop_reason"] for row in native if row.get("type") == "message_delta"] == [expected]
    assert sum(row.get("type") == "message_stop" for row in native) == 1
    if tools:
        starts = [row["content_block"] for row in native if row.get("content_block", {}).get("type") == "tool_use"]
        assert len(starts) == 1 and starts[0]["name"] == "lookup" and starts[0]["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_native_messages_nonstream_explicit_cancel_returns_nonretryable_400(monkeypatch, partial):
    import vmlx_engine.server as server

    parts = ["1\n", "2\n"] if partial else []
    bridge = _bridge(monkeypatch, parts=parts)

    class RequestBody:
        headers = {}

        async def json(self):
            return {"model": MODEL, "messages": [{"role": "user", "content": "Count."}],
                    "stream": False, "max_tokens": 16, "thinking": {"type": "disabled"}}

        async def is_disconnected(self):
            return False

    response = await asyncio.wait_for(server.create_anthropic_message(RequestBody()), timeout=2)
    assert getattr(response, "status_code", None) == 400
    body = json.loads(response.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "request_cancelled"
    assert body["error"]["message"]
    assert body["usage"] == {"input_tokens": 32, "output_tokens": len(parts),
                             "cache_read_input_tokens": 32,
                             "cache_creation_input_tokens": 0}
    assert "content" not in body and "stop_reason" not in body
    _assert_upstream(bridge, len(parts), "aborted")
