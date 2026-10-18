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


@pytest.mark.parametrize("family,is_mllm,retained", [
    ("glm5_next", True, True),
    ("glm5_next_text", True, True),
    ("mimo_v2", True, True),
    ("mimo_v2", False, False),
    ("glm5_next", False, False),
    ("qwen3_vl", True, False),
    (None, True, False),
])
@pytest.mark.parametrize("as_string", [True, False])
def test_loaded_qualified_text_chain_preserves_authored_media(
    route_probe, monkeypatch, family, is_mllm, retained, as_string
):
    from copy import deepcopy
    from vmlx_engine.api.models import ResponsesRequest

    server, captured = route_probe
    monkeypatch.setattr(server, "get_engine", lambda: SimpleNamespace(is_mllm=is_mllm))
    monkeypatch.setattr(server, "_current_model_config", lambda: SimpleNamespace(family_name=family))
    monkeypatch.setattr(server, "_loaded_runtime_modalities", lambda: ["text", "image", "video"])
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "Inspect the card."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE"}},
        ]},
        {"role": "assistant", "content": "Card inspected."},
        {"role": "user", "content": [
            {"type": "text", "text": "Inspect the clip."},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,VIDEO"}},
        ]},
        {"role": "assistant", "content": "Clip inspected."},
    ]
    original = deepcopy(history)
    server._responses_store_history("resp_media", history)
    current = "Compare their undisclosed details."
    if not as_string:
        current = [{"role": "user", "content": [{"type": "input_text", "text": current}]}]
    with pytest.raises(ReachedTemplateBoundary):
        asyncio.run(server.create_response(
            ResponsesRequest(model="unit-model", previous_response_id="resp_media", input=current),
            Request({"type": "http", "headers": []}),
        ))
    assert history == original
    assert server._responses_get_history("resp_media") == original
    if retained:
        assert captured[:len(history)] == original
    else:
        assert captured[0]["content"] == "Inspect the card."
        assert captured[2]["content"] == "Inspect the clip."


@pytest.mark.parametrize("stream", [False, True])
def test_loaded_glm_rejects_unsupported_historical_audio_before_template(
    route_probe, monkeypatch, stream
):
    server, captured = route_probe
    monkeypatch.setattr(server, "get_engine", lambda: SimpleNamespace(is_mllm=True))
    monkeypatch.setattr(server, "_current_model_config", lambda: SimpleNamespace(family_name="glm5_next"))
    monkeypatch.setattr(server, "_loaded_runtime_modalities", lambda: ["text", "image", "video"])
    history = [{"role": "user", "content": [
        {"type": "text", "text": "Listen to this recording."},
        {"type": "input_audio", "input_audio": {"data": "OLD_AUDIO", "format": "wav"}},
    ]}, {"role": "assistant", "content": "Recording inspected."}]
    server._responses_store_history("resp_prior_audio_model", history)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(call_route(server, previous_response_id="resp_prior_audio_model", stream=stream))
    assert exc.value.status_code == 400
    assert "unsupported media modality audio" in exc.value.detail
    assert captured == []
    assert server._responses_get_history("resp_prior_audio_model") == history
