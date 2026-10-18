"""All HTTP surfaces hide malformed generic tool control and report termination."""
import json
from types import SimpleNamespace

import httpx
import pytest

from vmlx_engine import server
from vmlx_engine.engine.base import GenerationOutput


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("text", [
    "<tool_call>UNSAFE_ORPHAN_SUFFIX",
    '<tool_call>{"name":"unavailable_tool","arguments":{"path":"x"}}</tool_call>',
    '<tool_call>{"name":"file_info","arguments":{}}</tool_call>',
    '<tool_call>{"name":"file_info","arguments":[]}</tool_call>',
])
async def test_generic_control_rejection_http(monkeypatch, api, stream, finish, text):
    class Engine:
        tokenizer = SimpleNamespace(has_thinking=False)
        is_mllm = False
        preserve_native_tool_format = True
        output_text = text
        calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            return GenerationOutput(text=self.output_text, tokens=[],
                                    prompt_tokens=12, completion_tokens=2,
                                    finished=True, finish_reason=finish)

        async def stream_chat(self, **kwargs):
            self.calls += 1
            accumulated = ""
            for i, delta in enumerate([self.output_text[:11], self.output_text[11:]]):
                accumulated += delta
                yield GenerationOutput(text=accumulated, new_text=delta, tokens=[],
                                       prompt_tokens=12, completion_tokens=i+1,
                                       finished=bool(i), finish_reason=finish if i else None)

    engine = Engine()
    monkeypatch.setattr(server, "_engine", engine)
    for field in ("_model_name", "_served_model_name"):
        monkeypatch.setattr(server, field, "generic-rejection-control")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", "auto")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_api_key", None)
    fn = {"name": "file_info", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
    body = {"model": "generic-rejection-control", "stream": stream, "tool_choice": "auto"}
    if api == "chat":
        body.update(messages=[{"role": "user", "content": "Inspect if needed."}],
                    tools=[{"type": "function", "function": fn}], max_tokens=64)
        if stream:
            body["stream_options"] = {"include_usage": True}
    else:
        body.update(input="Inspect if needed.", tools=[{"type": "function", **fn}], max_output_tokens=64)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
        response = await client.post("/v1/" + ("chat/completions" if api == "chat" else "responses"), json=body)
        # No hidden retry, and a subsequent request can use the same engine.
        assert engine.calls == 1
        engine.output_text = "Recovered answer"
        followup = dict(body, stream=False)
        if api == "responses":
            if stream:
                captured = [json.loads(line[5:]) for line in response.text.splitlines()
                            if line.startswith("data:") and line[5:].strip() != "[DONE]"]
                rejected_response = next(e["response"] for e in captured if e.get("type") == "response.incomplete")
            else:
                rejected_response = response.json()
            followup["previous_response_id"] = rejected_response["id"]
            followup["input"] = "Continue using only the declared tools."
        recovered = await client.post("/v1/" + ("chat/completions" if api == "chat" else "responses"), json=followup)
        assert recovered.status_code == 200
        assert engine.calls == 2
        recovered_data = recovered.json()
        if api == "chat":
            assert recovered_data["choices"][0]["message"]["content"] == "Recovered answer"
        else:
            assert recovered_data["output_text"] == "Recovered answer"
        assert "tool_calls_rejected" not in recovered.text
    assert "UNSAFE_ORPHAN_SUFFIX" not in response.text
    assert "<tool_call>" not in response.text
    if not stream:
        data = response.json()
        if api == "chat" and finish == "stop":
            assert response.status_code == 422
            assert data["detail"]["code"] == "tool_calls_rejected"
            assert data["detail"]["recoverable"] is True
            assert data["detail"]["recovery_action"] == "send_followup"
            assert data["detail"]["usage"]["completion_tokens"] == 2
            assert data["detail"]["warnings"]
        elif api == "chat":
            assert response.status_code == 200
            assert not data["choices"][0]["message"]["content"]
            assert data["choices"][0]["finish_reason"] == "length"
            assert data["warnings"]
        else:
            assert response.status_code == 200
            assert data["status"] == "incomplete" and data["output_text"] is None
            assert data["output"] == []
            assert data["incomplete_details"]["reason"] == ("max_output_tokens" if finish == "length" else "tool_calls_rejected")
        return
    assert response.status_code == 200
    events = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:") and line[5:].strip() != "[DONE]"]
    if api == "chat":
        errors = [e["error"]["code"] for e in events if e.get("error")]
        terminals = [c["finish_reason"] for e in events for c in e.get("choices", []) if c.get("finish_reason")]
        assert errors == (["tool_calls_rejected"] if finish == "stop" else [])
        if errors:
            error = next(e["error"] for e in events if e.get("error"))
            assert error["recoverable"] is True
            assert error["recovery_action"] == "send_followup"
        assert terminals == (["length"] if finish == "length" else [])
        assert response.text.rstrip().endswith("data: [DONE]")
        assert [e["usage"]["completion_tokens"] for e in events if e.get("usage")] == [2]
    else:
        terminals = [e for e in events if e.get("type") in {"response.completed", "response.incomplete", "response.failed"}]
        assert len(terminals) == 1 and terminals[0]["type"] == "response.incomplete"
        assert terminals[0]["response"]["incomplete_details"]["reason"] == ("max_output_tokens" if finish == "length" else "tool_calls_rejected")
