"""DeepSeek-V4.1 reasoning effort: every endpoint and both render paths hand the bundle encoder a value it accepts.

The bundle encoder (encoding/encoding.py) accepts an int 1..100 or low/high/max and asserts on anything else. The
server maps the wider effort vocabulary through the Swift runtime's ladder (minimal 25, low 50, medium 62, high 75,
xhigh 88, max 100; ints / numeric strings clamp to 1..100); ``none`` selects the chat rail. No weights are loaded:
the encoder here is a stand-in with the bundle's exact effort contract (set DSV41_BUNDLE to a real bundle directory
to also render through its own encoder).
"""
from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmlx_engine.models.deepseek_v4_1 import chat_encoding as CE
from vmlx_engine.utils.chat_template_kwargs import template_kwarg_strip_order

# Stand-in for encoding/encoding.py with the bundle's effort contract (render_reasoning_effort asserts).
_FAKE_ENCODING = textwrap.dedent('''
    REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
    CALLS = []

    def render_reasoning_effort(effort):
        if effort is None:
            effort = "high"
        assert (type(effort) is int and 1 <= effort <= 100) or effort in REASONING_EFFORT_MAPPINGS, effort
        return REASONING_EFFORT_MAPPINGS[effort] if type(effort) is str else effort

    def encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True,
                        reasoning_effort=None, return_multi_modal_data=False):
        budget = render_reasoning_effort(reasoning_effort)
        CALLS.append({"messages": messages, "thinking_mode": thinking_mode, "budget": budget,
                      "effort": reasoning_effort})
        prompt = f"<bos>[{thinking_mode}:{budget}]" + "|".join(str(m.get("content")) for m in messages)
        if return_multi_modal_data:
            return prompt, {"images": []}
        return prompt
''')


@pytest.fixture
def fake_bundle(tmp_path):
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding" / "encoding.py").write_text(_FAKE_ENCODING)
    return tmp_path


# --- the ladder -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("effort,budget", [
    ("minimal", 25), ("low", 50), ("medium", 62), ("high", 75), ("xhigh", 88), ("max", 100),
    (" High ", 75), ("40", 40), ("150", 100), ("-3", 1), ("62.4", 62), (0, 1), (250, 100), (33, 33), (70.6, 71),
])
def test_ladder_matches_swift_runtime(effort, budget):
    assert CE.reasoning_budget(effort) == budget


@pytest.mark.parametrize("effort", [None, "", "banana", True, "none", "off", "disabled", "0"])
def test_unknown_and_off_have_no_budget(effort):
    assert CE.reasoning_budget(effort) is None


@pytest.mark.parametrize("effort", ["none", "NONE", "off", "false", "disabled", "disable", "0"])
def test_thinking_off_spellings(effort):
    assert CE.is_thinking_off_effort(effort)
    assert not CE.is_thinking_off_effort("low")


def test_encoder_effort_drops_unknown_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert CE.encoder_effort("banana") is None
    assert "banana" in caplog.text
    assert CE.encoder_effort("medium") == 62


def test_bundle_encoding_loaded_once(fake_bundle):
    a = CE.load_bundle_encoding(fake_bundle)
    b = CE.load_bundle_encoding(str(fake_bundle))
    assert a is b


# --- the text path: the loader's apply_chat_template shim ----------------------------------------------------------

class _Tok:
    def encode(self, prompt):
        return [ord(c) for c in prompt]


def _shim(bundle):
    from vmlx_engine.loaders.load_jangtq_dsv41 import _attach_chat_encoder

    tok = _Tok()
    _attach_chat_encoder(tok, Path(bundle))
    return tok, CE.load_bundle_encoding(bundle)


@pytest.mark.parametrize("effort,budget", [
    (None, 75), ("medium", 62), ("xhigh", 88), ("minimal", 25), (62, 62), ("max", 100), ("banana", 75),
])
def test_shim_renders_every_effort(fake_bundle, effort, budget):
    tok, enc = _shim(fake_bundle)
    msgs = [{"role": "user", "content": "hi"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=True, reasoning_effort=effort)
    assert prompt.startswith(f"<bos>[thinking:{budget}]")


def test_shim_effort_none_is_chat_rail(fake_bundle):
    tok, enc = _shim(fake_bundle)
    prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], enable_thinking=True, reasoning_effort="none")
    assert prompt.startswith("<bos>[chat:")


def test_shim_keeps_tools_with_out_of_set_effort(fake_bundle):
    tok, enc = _shim(fake_bundle)
    tools = [{"type": "function", "function": {"name": "f"}}]
    tok.apply_chat_template([{"role": "user", "content": "hi"}], enable_thinking=True,
                            reasoning_effort="medium", tools=tools, thinking_budget=4096)
    call = enc.CALLS[-1]
    assert call["budget"] == 62
    assert call["messages"][0] == {"role": "system", "content": "", "tools": tools}


_TOOLS = [{"type": "function", "function": {"name": "get_weather"}}, {"type": "function", "function": {"name": "calc"}}]


@pytest.mark.parametrize("tool_choice,tools,named", [
    ("required", _TOOLS, None),
    ({"type": "function", "function": {"name": "calc"}}, _TOOLS, "calc"),
    ("required", _TOOLS[:1], "get_weather"),  # one tool available: named like the Swift runtime
])
def test_shim_required_tool_choice_adds_latest_reminder(fake_bundle, tool_choice, tools, named):
    tok, enc = _shim(fake_bundle)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "go"}]
    tok.apply_chat_template(msgs, enable_thinking=False, tools=tools, tool_choice=tool_choice)
    sent = enc.CALLS[-1]["messages"]
    # right before the turn being answered, so the prompt still ends with the assistant rail
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "latest_reminder", "user"]
    text = sent[3]["content"]
    assert text.startswith("The active API tool_choice is required for this assistant turn. ")
    assert "<\uff5cDSML\uff5c calls>" in text
    assert text.endswith(f" Use the `{named}` function.") if named else text.endswith("before the tool result.")
    assert msgs[-1] == {"role": "user", "content": "go"}  # the caller's list is not mutated


def test_shim_reminder_precedes_the_latest_tool_results_as_one_run(fake_bundle):
    tok, enc = _shim(fake_bundle)
    call = {"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    msgs = [{"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [dict(call, id="c1"), dict(call, id="c2")]},
            {"role": "tool", "tool_call_id": "c1", "content": "18C"}, {"role": "tool", "tool_call_id": "c2", "content": "9C"}]
    tok.apply_chat_template(msgs, enable_thinking=False, tools=_TOOLS, tool_choice="required")
    # the encoder merges consecutive tool results into one user turn: the reminder goes before the run, not inside it
    assert [m["role"] for m in enc.CALLS[-1]["messages"]] == ["system", "user", "assistant", "latest_reminder",
                                                              "tool", "tool"]


def test_shim_reminder_not_added_twice(fake_bundle):
    tok, enc = _shim(fake_bundle)
    msgs = [{"role": "latest_reminder", "content": "r"}, {"role": "user", "content": "go"}]
    tok.apply_chat_template(msgs, enable_thinking=False, tools=_TOOLS, tool_choice="required")
    assert [m["role"] for m in enc.CALLS[-1]["messages"]] == ["system", "latest_reminder", "user"]


@pytest.mark.parametrize("tool_choice,tools", [("auto", _TOOLS), ("none", _TOOLS), (None, _TOOLS), ("required", None),
                                               ({"type": "function", "function": {"name": " "}}, _TOOLS)])
def test_shim_no_reminder_otherwise(fake_bundle, tool_choice, tools):
    tok, enc = _shim(fake_bundle)
    tok.apply_chat_template([{"role": "user", "content": "hi"}], enable_thinking=False, tools=tools,
                            tool_choice=tool_choice)
    assert all(m["role"] != "latest_reminder" for m in enc.CALLS[-1]["messages"])


def test_shim_tokenize_returns_ids(fake_bundle):
    tok, _ = _shim(fake_bundle)
    ids = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=True)
    assert isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids)


@pytest.mark.skipif(not os.environ.get("DSV41_BUNDLE"), reason="set DSV41_BUNDLE to a V4.1 bundle directory")
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "37", None])
def test_real_bundle_encoder_accepts_every_effort(effort):
    from vmlx_engine.loaders.load_jangtq_dsv41 import _attach_chat_encoder

    tok = _Tok()
    _attach_chat_encoder(tok, Path(os.environ["DSV41_BUNDLE"]))
    tools = [{"type": "function", "function": {"name": "f", "description": "d",
                                               "parameters": {"type": "object", "properties": {}}}}]
    prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], enable_thinking=True,
                                     reasoning_effort=effort, tools=tools)
    budget = CE.reasoning_budget(effort) or 75
    assert f"Reasoning Effort: {budget} " in prompt


@pytest.mark.skipif(not os.environ.get("DSV41_BUNDLE"), reason="set DSV41_BUNDLE to a V4.1 bundle directory")
@pytest.mark.parametrize("thinking,rail", [(False, "<\uff5cAssistant\uff5c></think>"), (True, "<\uff5cAssistant\uff5c><think>")])
def test_real_bundle_required_tool_choice_ends_with_the_assistant_rail(thinking, rail):
    from vmlx_engine.loaders.load_jangtq_dsv41 import _attach_chat_encoder

    tok = _Tok()
    _attach_chat_encoder(tok, Path(os.environ["DSV41_BUNDLE"]))
    call = {"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    for msgs in ([{"role": "user", "content": "Hello there!"}],
                 [{"role": "user", "content": "Paris and Tokyo?"}, {"role": "assistant", "content": "",
                  "tool_calls": [dict(call, id="c1")]}, {"role": "tool", "tool_call_id": "c1", "content": "18C"}]):
        prompt = tok.apply_chat_template(msgs, enable_thinking=thinking, tools=_TOOLS, tool_choice="required")
        assert "<\uff5clatest_reminder\uff5c>The active API tool_choice is required" in prompt
        assert prompt.endswith(rail), prompt[-200:]


# --- the image path ---------------------------------------------------------------------------------------------

def test_vl_preprocess_normalizes_effort(fake_bundle, monkeypatch):
    pytest.importorskip("jang_tools.dsv41.vision")
    from vmlx_engine.models.deepseek_v4_1 import dsv41_vl_preprocess as P

    enc = CE.load_bundle_encoding(fake_bundle)
    seen = {}

    def _encode(messages, thinking_mode, reasoning_effort=None, return_multi_modal_data=False, **_):
        enc.render_reasoning_effort(reasoning_effort)  # the bundle's assertion
        seen.update(thinking_mode=thinking_mode, effort=reasoning_effort)
        return "<bos>p", {"images": []}  # no images -> preprocess returns None after rendering

    monkeypatch.setattr(enc, "encode_messages", _encode)
    model = SimpleNamespace(_rt=SimpleNamespace(vision_args=object()))
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                                         {"type": "text", "text": "what?"}]}]
    assert P.preprocess_dsv41_vl(model, str(fake_bundle), msgs, enable_thinking=True,
                                 reasoning_effort="medium") is None
    assert seen == {"thinking_mode": "thinking", "effort": 62}
    P.preprocess_dsv41_vl(model, str(fake_bundle), msgs, enable_thinking=True, reasoning_effort="none")
    assert seen == {"thinking_mode": "chat", "effort": None}


# --- server: /v1/chat/completions, /v1/responses, /v1/messages ------------------------------------------------------

@pytest.fixture
def dsv41_server(monkeypatch):
    import vmlx_engine.server as server

    monkeypatch.setattr(server, "_is_loaded_dsv41_model", lambda model="": True)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {}, raising=False)
    return server


def _apply(server, request, endpoint="test"):
    gen, ct = {}, {}
    if getattr(request, "reasoning_effort", None) is not None:  # what _forward_reasoning_effort_kwargs leaves
        gen["reasoning_effort"] = request.reasoning_effort
        ct["reasoning_effort"] = request.reasoning_effort
        ct["thinking_budget"] = 8192
    server._apply_dsv41_reasoning_policy(gen, ct, request, "m", endpoint=endpoint)
    return gen, ct


def _chat(**kw):
    from vmlx_engine.api.models import ChatCompletionRequest

    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **kw)


@pytest.mark.parametrize("kw,budget", [
    ({"reasoning_effort": "medium"}, 62),
    ({"reasoning_effort": "xhigh"}, 88),
    ({"reasoning_effort": "minimal"}, 25),
    ({"reasoning_effort": "55"}, 55),
    ({"chat_template_kwargs": {"reasoning_effort": "max"}}, 100),
    ({"reasoning": {"effort": "medium"}}, 62),
])
def test_chat_completions_effort_becomes_budget(dsv41_server, kw, budget):
    gen, ct = _apply(dsv41_server, _chat(**kw))
    assert ct["reasoning_effort"] == budget and gen["reasoning_effort"] == budget
    assert "thinking_budget" not in ct


def test_generic_thinking_mode_keeps_bundle_default(dsv41_server):
    req = _chat(thinking_mode="reasoning")
    assert req.reasoning_effort == "medium"  # the request model's generic inference
    gen, ct = _apply(dsv41_server, req)
    assert "reasoning_effort" not in ct and "reasoning_effort" not in gen
    assert req.enable_thinking is True


@pytest.mark.parametrize("kw", [{"reasoning_effort": "none"}, {"chat_template_kwargs": {"reasoning_effort": "none"}},
                                {"reasoning": {"effort": "none"}}])
def test_effort_none_is_thinking_off(dsv41_server, kw):
    req = _chat(**kw)
    gen, ct = _apply(dsv41_server, req)
    assert req.enable_thinking is False
    assert ct.get("enable_thinking", False) is False and gen.get("enable_thinking", False) is False
    assert "reasoning_effort" not in ct


def test_responses_reasoning_effort(dsv41_server):
    from vmlx_engine.api.models import ResponsesRequest

    req = ResponsesRequest(model="m", input="hi", reasoning={"effort": "xhigh"})
    gen, ct = _apply(dsv41_server, req)
    assert ct["reasoning_effort"] == 88


def test_messages_output_config_effort(dsv41_server):
    from vmlx_engine.api.anthropic_adapter import AnthropicRequest, to_chat_completion

    areq = AnthropicRequest(model="m", max_tokens=64, messages=[{"role": "user", "content": "hi"}],
                            output_config={"effort": "xhigh"})
    chat_req = to_chat_completion(areq)
    gen, ct = _apply(dsv41_server, chat_req)
    assert ct["reasoning_effort"] == 88


def test_thinking_budget_is_logged_and_dropped(dsv41_server, caplog):
    req = _chat(reasoning_effort="high", max_thinking_tokens=512)
    with caplog.at_level(logging.INFO, logger="vmlx_engine.server"):
        gen, ct = _apply(dsv41_server, req, endpoint="/v1/chat/completions")
    assert ct["reasoning_effort"] == 75 and "thinking_budget" not in ct
    msgs = [r.getMessage() for r in caplog.records if "thinking budget" in r.getMessage()]
    assert len(msgs) == 1 and "512" in msgs[0]


def test_other_families_untouched(monkeypatch):
    import vmlx_engine.server as server

    monkeypatch.setattr(server, "_is_loaded_dsv41_model", lambda model="": False)
    gen, ct = {"reasoning_effort": "medium"}, {"reasoning_effort": "medium", "thinking_budget": 8192}
    server._apply_dsv41_reasoning_policy(gen, ct, _chat(reasoning_effort="medium"), "m")
    assert ct == {"reasoning_effort": "medium", "thinking_budget": 8192}


# --- the kwarg-strip retry ------------------------------------------------------------------------------------

def test_strip_order_drops_extras_before_tools():
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True, "thinking": True,
              "reasoning_effort": "medium", "thinking_budget": 8192, "tools": [{"x": 1}]}
    assert template_kwarg_strip_order(kwargs) == [
        "thinking_budget", "reasoning_effort", "tools", "thinking", "enable_thinking"]


def test_batched_render_retry_keeps_tools(monkeypatch):
    """A template that rejects an extra kwarg must still render the tool schema (V4.1 pre-fix: 'medium')."""
    from vmlx_engine.engine.batched import BatchedEngine

    calls = []

    class Renderer:
        chat_template = "{# native #}"

        def apply_chat_template(self, messages, **kw):
            calls.append(dict(kw))
            if kw.get("reasoning_effort") == "medium":
                raise AssertionError("Invalid reasoning effort")
            names = [t["function"]["name"] for t in kw.get("tools") or []]
            return "PROMPT native-tools=" + ",".join(names) + " lookup_weather"

    class FakeModel:
        config = {"model_type": "deepseek_v41"}

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._processor = None
    engine._tokenizer = Renderer()
    engine._model = FakeModel()
    engine._model_name = "dsv41-test"
    engine._model_tool_parser_name = lambda: None
    engine._model_family_name = lambda: None
    tools = [{"type": "function", "function": {"name": "lookup_weather",
                                               "parameters": {"type": "object", "properties": {}}}}]
    out = engine._apply_chat_template(
        [{"role": "user", "content": "hi"}], tools=tools, enable_thinking=True,
        extra_template_kwargs={"reasoning_effort": "medium"},
    )
    assert out.startswith("PROMPT native-tools=lookup_weather")
    assert "tools" in calls[-1] and "reasoning_effort" not in calls[-1]
