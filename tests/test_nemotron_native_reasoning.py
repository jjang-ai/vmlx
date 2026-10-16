"""Native efficient reasoning must remain distinct from legacy inferred Medium."""
import json

import pytest
from fastapi import HTTPException

from vmlx_engine import server
from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest


@pytest.fixture(params=[ChatCompletionRequest, ResponsesRequest])
def make_request(request):
    cls = request.param
    key = "messages" if cls is ChatCompletionRequest else "input"
    def make(**kwargs):
        return cls(model="test", **{key: [{"role": "user", "content": "19 + 23?"}]}, **kwargs)
    return make


@pytest.fixture
def native_bundle(tmp_path, monkeypatch):
    stamp = {"capabilities": {"family": "nemotron_h", "reasoning_modes": {
        "on": {"enable_thinking": True},
        "medium": {"enable_thinking": True, "medium_effort": True},
        "off": {"enable_thinking": False},
    }}}
    (tmp_path / "jang_config.json").write_text(json.dumps(stamp))
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {})
    return tmp_path, stamp


def merged(req):
    return server._merge_ct_kwargs(req.chat_template_kwargs, req.reasoning_effort,
                                   enable_thinking=req.enable_thinking, native_request=req)


@pytest.mark.parametrize("fields", [{}, {"enable_thinking": True}, {"thinking_mode": "on"},
                                    {"thinking_mode": "reasoning"}, {"reasoning": {"type": "enabled"}}])
def test_default_and_on_do_not_invent_efficient_mode(native_bundle, make_request, fields):
    req = make_request(**fields)
    assert req._explicit_reasoning_effort is None
    assert "medium_effort" not in merged(req)
    assert "_explicit_reasoning_effort" not in req.model_dump()


@pytest.mark.parametrize("fields", [
    {"reasoning_effort": "medium"},
    {"reasoning_effort": "medium", "thinking_mode": "reasoning"},
    {"reasoning": {"effort": "medium"}},
    {"thinking_mode": "medium"},
    {"chat_template_kwargs": {"reasoning_effort": "medium"}},
])
def test_explicit_medium_reaches_native_template(native_bundle, make_request, fields):
    req = make_request(**fields)
    assert merged(req)["medium_effort"] is True
    assert merged(req)["enable_thinking"] is True
    # Streaming and gateway request copies must not lose the original intent.
    assert merged(req.model_copy())["medium_effort"] is True


@pytest.mark.parametrize("effort,native_medium", [("low", True), ("minimal", True), ("high", False), ("xhigh", False), ("max", False)])
def test_unimplemented_effort_is_coerced_and_logged_not_refused(native_bundle, make_request, effort, native_medium, caplog):
    # no 400 (an OpenAI client sending "high" by default must still work); nearest native tier; never silent
    with caplog.at_level("WARNING", logger="vmlx_engine.server"):
        kwargs = merged(make_request(reasoning_effort=effort))
    assert kwargs.get("enable_thinking") is True
    assert (kwargs.get("medium_effort") is True) is native_medium
    assert any("no native Nemotron tier" in r.getMessage() for r in caplog.records)


def test_native_kwarg_and_off_keep_precedence(native_bundle, make_request):
    req = make_request(reasoning_effort="medium", chat_template_kwargs={"medium_effort": False})
    assert merged(req)["medium_effort"] is False
    req = make_request(reasoning_effort="medium", enable_thinking=False)
    assert "medium_effort" not in merged(req)
    req = make_request(chat_template_kwargs={"medium_effort": True})
    assert merged(req)["medium_effort"] is True


def test_native_on_overrides_inherited_medium_but_not_caller_kwargs(native_bundle, make_request, monkeypatch):
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {"medium_effort": True})
    assert merged(make_request(reasoning_effort="high"))["medium_effort"] is False
    assert merged(make_request(reasoning_effort="high", chat_template_kwargs={"medium_effort": True}))["medium_effort"] is True


@pytest.mark.parametrize("fields,effective", [
    ({"reasoning_effort": "high"}, "on"),
    ({"reasoning_effort": "medium", "chat_template_kwargs": {"medium_effort": False}}, "on"),
    ({"reasoning_effort": "high", "enable_thinking": False}, "off"),
])
def test_native_effort_diagnostics_describe_actual_template_mode(native_bundle, make_request, fields, effective):
    from vmlx_engine.context_limits import pop_effort_substitution

    path, _ = native_bundle
    req = make_request(**fields)
    ct = merged(req)
    before = dict(ct)
    chat = {"reasoning_effort": req.reasoning_effort, "enable_thinking": req.enable_thinking}
    server._apply_stamped_effort_policy(chat, ct, model_key=str(path), request_id="native-mode-diagnostic", native_request=req)
    diagnostic = pop_effort_substitution("native-mode-diagnostic")
    assert diagnostic is not None
    assert diagnostic["effective_effort"] == effective
    assert ct == before  # diagnostics must not run a second template policy


def test_native_stamp_advertises_only_medium_without_changing_default(native_bundle):
    path, _ = native_bundle
    assert server._stamped_reasoning_effort_contract(str(path)) == (("medium",), None)


def test_other_family_preserves_existing_kwargs(native_bundle, make_request):
    path, stamp = native_bundle
    stamp["capabilities"]["family"] = "qwen4_exp"
    (path / "jang_config.json").write_text(json.dumps(stamp))
    assert "medium_effort" not in merged(make_request(reasoning_effort="medium"))
    assert server._stamped_reasoning_effort_contract(str(path)) == ((), None)


def test_normalized_off_and_explicit_on_precedence(native_bundle, make_request):
    req = make_request(reasoning_effort="medium", chat_template_kwargs={"enable_thinking": "false"})
    assert merged(req)["enable_thinking"] is False
    assert "medium_effort" not in merged(req)
    req = make_request(reasoning_effort="medium", enable_thinking=True,
                       chat_template_kwargs={"enable_thinking": False})
    assert merged(req)["medium_effort"] is True


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/api/chat"])
@pytest.mark.parametrize("explicit", [False, True])
def test_http_native_medium_survives_all_chat_adapters(native_bundle, monkeypatch, endpoint, explicit):
    """Actual HTTP routing, synthetic generation; not a model-output proof."""
    from fastapi.testclient import TestClient
    from tests.test_ollama_reasoning_parity import _run_streaming_ollama_chat

    path, _ = native_bundle
    messages = [{"role": "user", "content": "19 + 23?"}]
    _run_streaming_ollama_chat(monkeypatch, family_name="nemotron_h", model_type="nemotron_h",
                             body={"model": "test", "messages": messages, "stream": True})
    monkeypatch.setattr(server, "_model_path", str(path))
    captured = {}

    async def capture(engine, messages, request, fastapi_request=None, **kwargs):
        captured.update(kwargs)
        yield 'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        yield 'data: [DONE]\n\n'

    monkeypatch.setattr(server, "stream_chat_completion", capture)
    monkeypatch.setattr(server, "stream_responses_api", capture)
    body = {"model": "test", "stream": True}
    if endpoint == "/v1/responses":
        body.update(input=messages, thinking_mode="on")
    elif endpoint == "/v1/messages":
        body.update(messages=messages, max_tokens=32, thinking={"type": "enabled"})
    elif endpoint == "/api/chat":
        body.update(messages=messages, think=True)
    else:
        body.update(messages=messages, thinking_mode="on")
    if explicit:
        body["reasoning_effort"] = "medium"
    client = TestClient(server.app)
    try:
        response = client.post(endpoint, json=body)
    finally:
        client.close()
    assert response.status_code == 200, response.text
    assert captured, response.text
    assert captured.get("chat_template_kwargs", {}).get("medium_effort", False) is explicit
    assert captured["enable_thinking"] is True


def test_inferred_medium_does_not_claim_an_explicit_substitution(native_bundle, make_request):
    from vmlx_engine.context_limits import pop_effort_substitution

    path, _ = native_bundle
    req = make_request(thinking_mode="on")
    ct = merged(req)
    server._apply_stamped_effort_policy({"reasoning_effort": req.reasoning_effort}, ct,
                                      model_key=str(path), request_id="inferred-native-mode", native_request=req)
    assert pop_effort_substitution("inferred-native-mode") is None
    assert "medium_effort" not in ct


@pytest.mark.parametrize("thinking", [None, True, False])
@pytest.mark.parametrize("budget", [64, 1500, 16384])
def test_native_modes_do_not_invent_an_answer_reserve(native_bundle, thinking, budget):
    assert server._dsv4_answer_pass_thinking_cap("nemotron_h", thinking, budget) is None


def test_native_budget_exception_does_not_change_other_family_reserves(native_bundle):
    assert server._dsv4_answer_pass_thinking_cap("deepseek_v4", True, 1500) == 1125


def test_native_mode_is_visible_in_resolved_kwargs_log(native_bundle, caplog):
    with caplog.at_level("INFO", logger="vmlx_engine.server"):
        server._log_resolved_sampling_kwargs(
            "/v1/chat/completions", "test",
            {"chat_template_kwargs": {"medium_effort": False, "enable_thinking": True}},
        )
    assert any("'medium_effort': False" in r.getMessage() for r in caplog.records)
