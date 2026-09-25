# SPDX-License-Identifier: Apache-2.0
"""Missing Responses chains must fail before template or model execution."""

import asyncio
from collections import OrderedDict
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request


class ReachedTemplateBoundary(Exception):
    pass


@pytest.fixture
def route_probe(monkeypatch):
    from vmlx_engine import server

    captured = []
    monkeypatch.setattr(server, "_responses_history", OrderedDict())
    monkeypatch.setattr(server, "_responses_was_reasoning_only", set())
    monkeypatch.setattr(server, "_resolve_model_name", lambda: "unit-model")
    monkeypatch.setattr(server, "_model_name", "unit-model")
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "get_engine", lambda: SimpleNamespace(is_mllm=False))
    monkeypatch.setattr(server, "_enforce_text_only_override", lambda *a: None)
    monkeypatch.setattr(server, "_m3_vl_response_media_supported", lambda *a: False)

    def capture(messages):
        captured.extend(messages)
        raise ReachedTemplateBoundary

    monkeypatch.setattr(server, "_canonicalize_mimo_v26_tool_history", capture)
    return server, captured


async def call_route(server, **kwargs):
    from vmlx_engine.api.models import ResponsesRequest

    return await server.create_response(
        ResponsesRequest(model="unit-model", input="What did we establish?", **kwargs),
        Request({"type": "http", "headers": []}),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("missing_kind", ["unknown", "evicted", "cleared"])
def test_missing_chain_rejected_before_template(route_probe, monkeypatch, stream, missing_kind):
    server, captured = route_probe
    if missing_kind != "unknown":
        server._responses_store_history("resp_missing", [{"role": "user", "content": "secret"}])
    if missing_kind == "evicted":
        monkeypatch.setattr(server, "_RESPONSES_HISTORY_MAX", 1)
        server._responses_store_history("resp_new", [{"role": "user", "content": "new"}])
    elif missing_kind == "cleared":
        server._responses_history.clear()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(call_route(server, previous_response_id="resp_missing", stream=stream))
    assert exc.value.status_code == 404
    assert "previous_response_id" in exc.value.detail
    assert captured == []


def test_valid_chain_preserves_history(route_probe):
    server, captured = route_probe
    history = [{"role": "user", "content": "Remember cedar"}, {"role": "assistant", "content": "cedar"}]
    server._responses_store_history("resp_valid", history)
    with pytest.raises(ReachedTemplateBoundary):
        asyncio.run(call_route(server, previous_response_id="resp_valid"))
    assert captured == history + [{"role": "user", "content": "What did we establish?"}]


def test_existing_empty_slot_is_not_missing(route_probe):
    server, captured = route_probe
    server._responses_store_history("resp_empty", [])
    with pytest.raises(ReachedTemplateBoundary):
        asyncio.run(call_route(server, previous_response_id="resp_empty"))
    assert captured == [{"role": "user", "content": "What did we establish?"}]


def test_explicit_history_without_chain_id_preserved(route_probe):
    from vmlx_engine.api.models import ResponsesRequest

    server, captured = route_probe
    history = [{"role": "user", "content": "Remember cedar"}, {"role": "assistant", "content": "cedar"}, {"role": "user", "content": "Repeat it"}]
    with pytest.raises(ReachedTemplateBoundary):
        asyncio.run(server.create_response(
            ResponsesRequest(model="unit-model", input=history),
            Request({"type": "http", "headers": []}),
        ))
    assert captured == history


def test_valid_tool_result_chain_preserves_call_adjacency(route_probe):
    from vmlx_engine.api.models import ResponsesRequest

    server, captured = route_probe
    history = [
        {"role": "user", "content": "Look up cedar"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_lookup", "type": "function", "function": {
                "name": "lookup", "arguments": '{"name":"cedar"}'
            }}
        ]},
    ]
    server._responses_store_history("resp_tool", history)
    with pytest.raises(ReachedTemplateBoundary):
        asyncio.run(server.create_response(
            ResponsesRequest(
                model="unit-model", previous_response_id="resp_tool",
                input=[{"type": "function_call_output", "call_id": "call_lookup", "output": "28"}],
            ),
            Request({"type": "http", "headers": []}),
        ))
    assert captured == history + [{"role": "tool", "tool_call_id": "call_lookup", "content": "28"}]
