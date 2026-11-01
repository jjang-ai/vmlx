"""Native EOS classification must precede tool extraction, without generating again."""
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.reasoning.qwen3_parser import Qwen3ReasoningParser


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
@pytest.mark.parametrize("text", [
    '<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>',
    '<tool_call>{"name":"read_file","arguments":{"path":"notes.txt"}}</tool_call>',
])
async def test_native_eos_tool_is_executable_not_visible_or_reasoning(monkeypatch, surface, stream, tool_choice, text):
    class Engine:
        tokenizer = SimpleNamespace(has_thinking=True)
        is_mllm = False
        preserve_native_tool_format = True
        calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            return GenerationOutput(text=text, raw_text=text, prompt_tokens=10,
                                    completion_tokens=26, finish_reason="stop", finished=True)

        async def stream_chat(self, **kwargs):
            self.calls += 1
            for i in range(0, len(text), 2):
                end = min(i + 2, len(text))
                yield GenerationOutput(text=text[:end], new_text=text[i:end], prompt_tokens=10,
                                       completion_tokens=26, finished=end == len(text),
                                       finish_reason="stop" if end == len(text) else None)

    engine = Engine()
    server._begin_tool_call_drop_capture()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", "MiniCPM-V-4.7-35B")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", Qwen3ReasoningParser())
    monkeypatch.setattr(server, "_tool_call_parser", "qwen")
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(server, "_engine_prompt_starts_in_reasoning", lambda *a, **k: True)
    monkeypatch.setattr(server, "_template_completes_thinking", lambda *a, **k: False)
    schema = {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}, "required": ["path"]}}
    messages = [{"role": "user", "content": "Read notes.txt"}]
    if surface == "chat":
        request = server.ChatCompletionRequest(model="MiniCPM-V-4.7-35B", messages=messages,
            tools=[{"type": "function", "function": schema}], tool_choice=tool_choice,
            stream=stream, enable_thinking=True)
        if not stream:
            try:
                response = await server.create_chat_completion(request, fastapi_request=None)
            except HTTPException as error:
                assert tool_choice == "none" and error.status_code == 422
                assert error.detail["code"] == "tool_calls_rejected"
                assert engine.calls == 1
                return
            body = response.model_dump() if hasattr(response, "model_dump") else json.loads(response.body)
            message = body["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            assert not message.get("content") and not message.get("reasoning_content") and not message.get("reasoning")
            if tool_choice == "auto":
                assert body["choices"][0]["finish_reason"] == "tool_calls"
        else:
            iterator = server._terminal_finish_guard(server.stream_chat_completion(
                engine, messages, request, fastapi_request=None, max_tokens=128))
    else:
        request = server.ResponsesRequest(model="MiniCPM-V-4.7-35B", input=messages,
            tools=[{"type": "function", **schema}], tool_choice=tool_choice,
            stream=stream, enable_thinking=True)
        if not stream:
            response = await server.create_response(request, fastapi_request=None)
            items = [x.model_dump() for x in response.output]
        else:
            iterator = server.stream_responses_api(engine, messages, request, fastapi_request=None, max_tokens=128)
    if stream:
        chunks = [c async for c in iterator]
        events = [json.loads(line[6:]) for c in chunks for line in c.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        if surface == "chat":
            pieces = [d for e in events for choice in e.get("choices", [])
                      for d in choice.get("delta", {}).get("tool_calls", [])]
            args = "".join(p.get("function", {}).get("arguments", "") for p in pieces)
            name = "".join(p.get("function", {}).get("name", "") for p in pieces)
            calls = [{"function": {"name": name, "arguments": args}}] if pieces else []
            assert not any(c.get("delta", {}).get("content") for e in events for c in e.get("choices", []))
            terminal = [c["finish_reason"] for e in events for c in e.get("choices", []) if c.get("finish_reason")]
            if tool_choice == "auto":
                assert terminal == ["tool_calls"]
        else:
            terminal = [e for e in events if e.get("type") in {"response.completed", "response.failed", "response.incomplete"}]
            assert len(terminal) == 1
            items = terminal[0]["response"]["output"]
            assert not terminal[0]["response"]["output_text"]
    if surface == "responses":
        calls = [{"function": item} for item in items if item.get("type") == "function_call"]
        assert not any(p.get("text") for item in items if item.get("type") == "reasoning"
                       for p in item.get("summary", []))
        assert not any(p.get("text") for item in items if item.get("type") == "message"
                       for p in item.get("content", []))
    assert len(calls) == int(tool_choice == "auto")
    if calls:
        assert calls[0]["function"]["name"] == "read_file"
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": "notes.txt"}
    assert engine.calls == 1
