"""Visible prose that precedes a DeepSeek-V4/V4.1 DSML tool call must stream.

Live symptom (DSV4.1, Electron Responses wire, ``--stream-interval 8``): the
assistant prose that shared a streamed delta with the first DSML marker never
reached the client ("...waiting" instead of "...waiting for each result.",
"First" instead of "First done. Now the second."). The panel then re-sent the
shortened text as history, so the next tool step also missed the
generation-end prefix-cache entry.

Root causes covered here:

* Responses: the tool-call finalization re-split the reasoning parser's
  CONTENT-only text with the per-request parser. With thinking on, DSV4.1's
  parser is seeded inside an implicit ``<think>`` rail, so close-less content
  was classified as unclosed reasoning and the withheld prose was dropped.
* Chat Completions: an early-stopped (grace-window) DSML turn parsed the RAW
  stream, so the reconciled prose still carried the reasoning rail and either
  failed to match what was streamed (prose dropped) or, with nothing streamed
  yet, leaked reasoning plus ``</think>`` as content.
* Chat Completions: reconciliation compared against a stripped copy of the
  streamed text and re-sent its trailing whitespace ("time,  waiting").

The server is driven exactly like a DSV4.1 launch: registry family
``deepseek_v4_1`` (model_type ``deepseek_v41``), reasoning parser
``deepseek_r1``, tool parser ``deepseek_v41`` (or V4 ``dsml``), fake engine
deltas of several tokens each (stream interval > 1). No model is loaded.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import vmlx_engine.server as server
from vmlx_engine.api.models import ChatCompletionRequest, Message, ResponsesRequest
from vmlx_engine.engine.base import GenerationOutput
from vmlx_engine.reasoning import get_parser
from vmlx_engine.tool_parsers.dsml_tool_parser import DSML_PREFIX
from vmlx_engine.tool_parsers.dsml_v41_tool_parser import DSMLV41ToolParser

D = DSML_PREFIX
EOS = "<｜end▁of▁sentence｜>"
TOOL = {
    "name": "run_command",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}


def _call(parser: str) -> str:
    if parser == "deepseek_v41":
        wrap, inv, prm = " calls", " invoke", " parameter"
    else:
        wrap, inv, prm = "tool_calls", "invoke", "parameter"
    return (
        f"<{D}{wrap}>\n<{D}{inv} name=\"run_command\">\n"
        f"<{D}{prm} name=\"command\" string=\"true\">echo two</{D}{prm}>\n"
        f"</{D}{inv}>\n</{D}{wrap}>"
    )


class _V41BundleEncoder:
    """Production-shaped V4.1 ``encoding/encoding.py`` stand-in.

    Mirrors the bundle contract the parser relies on: the completion must end
    with EOS and ``thinking_mode`` is a required argument. It agrees with the
    strict grammar on the call, so the canonical-agreement path is exercised.
    """

    eos_token = EOS
    calls: list[tuple[str, str]] = []

    @classmethod
    def parse_message_from_completion_text(cls, text, thinking_mode):
        assert text.endswith(cls.eos_token)
        assert f"<{D} calls>" in text
        cls.calls.append((text, thinking_mode))
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "echo two"}',
                    },
                }
            ],
        }


class _Engine:
    calls = 0
    tokenizer = SimpleNamespace(has_thinking=True)

    def __init__(self, deltas):
        self.deltas = deltas

    async def _stream(self):
        type(self).calls += 1
        text = ""
        last = len(self.deltas)
        for i, delta in enumerate(self.deltas, 1):
            text += delta
            yield GenerationOutput(
                text=text,
                new_text=delta,
                tokens=[i],
                prompt_tokens=8,
                completion_tokens=i * 8,
                finished=i == last,
                finish_reason="stop" if i == last else None,
            )

    def stream_chat(self, *, messages, **kwargs):
        return self._stream()


@pytest.fixture(params=["deepseek_v41", "dsml"])
def dsv4_server(request, monkeypatch, tmp_path):
    tool_parser = request.param
    bundle = tmp_path / "dsv41-bundle"
    bundle.mkdir()
    model_type = "deepseek_v41" if tool_parser == "deepseek_v41" else "deepseek_v4"
    (bundle / "config.json").write_text(json.dumps({"model_type": model_type}))
    parser = get_parser("deepseek_r1")()
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", str(bundle))
    monkeypatch.setattr(server, "_model_path", str(bundle))
    monkeypatch.setattr(server, "_reasoning_parser", parser)
    monkeypatch.setattr(server, "_reasoning_parser_id", "deepseek_r1")
    monkeypatch.setattr(server, "_reasoning_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_tool_call_parser", tool_parser)
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
    if tool_parser == "deepseek_v41":
        _V41BundleEncoder.calls = []
        monkeypatch.setattr(
            DSMLV41ToolParser,
            "_canonical_encoding_module",
            lambda self, req: _V41BundleEncoder,
        )
    else:
        # No DSV4 encoder on disk may decide this test (the default lookup
        # searches the machine's model folders).
        import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_enc

        monkeypatch.setattr(
            dsv4_enc, "_load_encoding_dsv4_module", lambda model_path=None: None
        )
    return SimpleNamespace(tool_parser=tool_parser, call=_call(tool_parser))


def _run_chat(deltas, think, tool=TOOL, endpoint_kwargs=None):
    req = ChatCompletionRequest(
        model="dsv41",
        messages=[Message(role="user", content="run")],
        stream=True,
        tools=[{"type": "function", "function": tool}],
        enable_thinking=think,
    )
    endpoint_kwargs = endpoint_kwargs(req) if endpoint_kwargs else {}

    async def go():
        content, reasoning, names, content_deltas = "", "", [], []
        async for line in server.stream_chat_completion(
            _Engine(deltas),
            [m.model_dump(exclude_none=True) for m in req.messages],
            req,
            fastapi_request=None,
            **endpoint_kwargs,
        ):
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_deltas.append(delta["content"])
                    content += delta["content"]
                reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
                for tc in delta.get("tool_calls") or []:
                    name = (tc.get("function") or {}).get("name")
                    if name:
                        names.append(name)
        return content, reasoning, names, content_deltas

    return asyncio.run(go())


def _run_responses(deltas, think, tool=TOOL, endpoint_kwargs=None):
    req = ResponsesRequest(
        model="dsv41",
        input="run",
        stream=True,
        tools=[{"type": "function", **tool}],
        enable_thinking=think,
    )
    endpoint_kwargs = endpoint_kwargs(req) if endpoint_kwargs else {}

    async def go():
        content, reasoning, names, content_deltas = "", "", [], []
        done_texts, completed_texts = [], []
        async for chunk in server.stream_responses_api(
            _Engine(deltas),
            [{"role": "user", "content": "run"}],
            req,
            fastapi_request=None,
            **endpoint_kwargs,
        ):
            for row in chunk.splitlines():
                if not row.startswith("data: "):
                    continue
                event = json.loads(row[len("data: "):])
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    content_deltas.append(event["delta"])
                    content += event["delta"]
                elif kind == "response.output_text.done":
                    done_texts.append(event["text"])
                elif kind in (
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                ):
                    reasoning += event["delta"]
                elif kind == "response.output_item.done" and event["item"].get("type") == "function_call":
                    names.append(event["item"]["name"])
                elif kind == "response.completed":
                    for item in event["response"].get("output", []):
                        if item.get("type") == "message":
                            for part in item.get("content", []):
                                completed_texts.append(part.get("text", ""))
        return content, reasoning, names, content_deltas, done_texts, completed_texts

    return asyncio.run(go())


def _assert_no_markup(text):
    assert D not in text
    assert "</think>" not in text
    assert "echo two" not in text


def _cases(call):
    ramble = [" more"] * (server._STREAM_TOOL_CALL_STOP_GRACE_CHUNKS + 4)
    return {
        # name: (enable_thinking, deltas, expected_content, expected_reasoning)
        "prose_and_marker_in_one_delta": (
            False,
            ["I'll run these four commands sequentially, ",
             "waiting for each result.\n\n" + call[:12], call[12:]],
            "I'll run these four commands sequentially, waiting for each result.",
            "",
        ),
        "prose_split_across_deltas": (
            False,
            ["I'll run these four ", "commands sequentially, ", "waiting",
             " for each result.\n\n" + call[:12], call[12:]],
            "I'll run these four commands sequentially, waiting for each result.",
            "",
        ),
        "tool_only_leading_newlines": (
            False,
            ["\n\n" + call[:12], call[12:]],
            "",
            "",
        ),
        "think_prose_and_marker_in_one_delta": (
            True,
            ["Need to run it.", "</think>First",
             " done. Now the second.\n\n" + call[:12], call[12:]],
            "First done. Now the second.",
            "Need to run it.",
        ),
        "think_prose_split_across_deltas": (
            True,
            ["Need to run it.</think>I'll run these four commands ",
             "sequentially, waiting", " for each result.\n\n" + call[:12], call[12:]],
            "I'll run these four commands sequentially, waiting for each result.",
            "Need to run it.",
        ),
        "think_tool_only_leading_newlines": (
            True,
            ["Need to run it.</think>", "\n\n" + call[:12], call[12:]],
            "",
            "Need to run it.",
        ),
        # The model keeps generating past the closed call: the grace-window
        # early stop parses the raw (reasoning + content) candidate.
        "early_stop_prose_and_marker_in_one_delta": (
            False,
            ["I'll run it one at a time, ", "waiting for it.\n\n" + call[:12], call[12:]] + ramble,
            "I'll run it one at a time, waiting for it.",
            "",
        ),
        "think_early_stop_prose_and_marker_in_one_delta": (
            True,
            ["Need to run it.</think>First",
             " done. Now the second.\n\n" + call[:12], call[12:]] + ramble,
            "First done. Now the second.",
            "Need to run it.",
        ),
        "think_early_stop_tool_only": (
            True,
            ["Need to run it.</think>", "\n\n" + call[:12], call[12:]] + ramble,
            "",
            "Need to run it.",
        ),
    }


CASE_NAMES = list(_cases("x").keys())


@pytest.mark.parametrize("case", CASE_NAMES)
def test_chat_completions_streams_prose_before_dsml_call(dsv4_server, case):
    think, deltas, want_content, want_reasoning = _cases(dsv4_server.call)[case]
    content, reasoning, names, content_deltas = _run_chat(deltas, think)
    assert names == ["run_command"]
    assert content == want_content
    assert reasoning == want_reasoning
    _assert_no_markup(content)
    # A tool-only turn shows no blank content at all.
    if not want_content:
        assert content_deltas == []


@pytest.mark.parametrize("case", CASE_NAMES)
def test_responses_streams_prose_before_dsml_call(dsv4_server, case):
    think, deltas, want_content, want_reasoning = _cases(dsv4_server.call)[case]
    content, reasoning, names, content_deltas, done_texts, completed_texts = (
        _run_responses(deltas, think)
    )
    assert names == ["run_command"]
    assert content == want_content
    assert reasoning == want_reasoning
    _assert_no_markup(content)
    if want_content:
        # The terminal text (what the panel stores as history and re-sends)
        # equals the streamed text.
        assert done_texts == [want_content]
        assert completed_texts == [want_content]
    else:
        assert content_deltas == []
        assert not any(t.strip() for t in done_texts + completed_texts)


def test_v41_bundle_encoder_is_consulted(dsv4_server):
    if dsv4_server.tool_parser != "deepseek_v41":
        pytest.skip("V4.1 bundle encoder only")
    think, deltas, want_content, _ = _cases(dsv4_server.call)["think_prose_split_across_deltas"]
    content, _r, names, *_ = _run_responses(deltas, think)
    assert names == ["run_command"] and content == want_content
    assert _V41BundleEncoder.calls, "canonical V4.1 encoder must agree on the call"


def test_chat_invalid_markup_flush_does_not_double_streamed_whitespace(dsv4_server):
    # Schema-invalid call (unknown tool): no call executes, the safe prose
    # before the marker is flushed once, with the whitespace the stream
    # already carried not repeated.
    bad = dsv4_server.call.replace('name="run_command"', 'name="not_offered"')
    deltas = ["I'll run it one at a time, ", "waiting for it.\n\n" + bad[:12], bad[12:]]
    content, _reasoning, names, _deltas = _run_chat(deltas, False)
    assert names == []
    assert content == "I'll run it one at a time, waiting for it."


@pytest.mark.parametrize(
    "candidate, streamed, expected",
    [
        ("time, waiting", "time, ", "waiting"),
        ("time, waiting", "time,", " waiting"),
        ("a.\n\nb", "a.\n\n", "b"),
        ("a.\n\nb", "a.", "\n\nb"),
        ("a. b", "a.\n\n", "b"),
        ("waiting.", "waiting.\n\n", ""),
        ("full", "", "full"),
        ("full", "\n\n", "full"),
        ("other", "unrelated", ""),
        ("", "x", ""),
    ],
)
def test_unstreamed_visible_suffix(candidate, streamed, expected):
    assert server._unstreamed_visible_suffix(candidate, streamed) == expected


# ── MiniMax-M3 / M3.1 ───────────────────────────────────────────────────────
# Same defect, same root cause: M3's ``thinking_mode=enabled`` seeds the
# minimax_m3 reasoning parser inside its implicit ``<mm:think>`` rail, and the
# Responses finalizer re-split content-only text with it. Live: "I'll run these
# commands one at a time in order." showed as "...one at a time in".

M3_NS = "]<]minimax[>["
M3_TOOL = {
    "name": "run_shell",
    "parameters": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
    },
}
# Native M3.1 call markup as emitted live (namespace token before every
# element, no whitespace before the first one).
M3_CALL = (
    f"{M3_NS}<tool_call>\n{M3_NS}<invoke name=\"run_shell\">{M3_NS}<cmd>echo two"
    f"{M3_NS}</cmd>{M3_NS}</invoke>\n{M3_NS}</tool_call>"
)
M3_PROSE = "I'll run these commands one at a time in order."


@pytest.fixture
def m3_server(monkeypatch, tmp_path):
    bundle = tmp_path / "m31-bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "minimax_m3"}))
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(server, "_model_name", str(bundle))
    monkeypatch.setattr(server, "_model_path", str(bundle))
    monkeypatch.setattr(server, "_reasoning_parser", get_parser("minimax_m3")())
    monkeypatch.setattr(server, "_reasoning_parser_id", "minimax_m3")
    monkeypatch.setattr(server, "_reasoning_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_tool_call_parser", "minimax_m3")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True)

    def endpoint_kwargs(req):
        # The endpoints normalize the public toggle onto M3's template
        # vocabulary before streaming; pass the same chat_template_kwargs.
        ct = {}
        if req.enable_thinking is not None:
            ct["enable_thinking"] = req.enable_thinking
        server._normalize_minimax_m3_thinking_mode(ct, req, str(bundle))
        return {"chat_template_kwargs": ct}

    return endpoint_kwargs


M3_CASES = {
    # name: (enable_thinking -> thinking_mode, deltas, content, reasoning)
    "disabled_prose_and_marker_in_one_delta": (
        False,
        [M3_PROSE[:20], M3_PROSE[20:-7], " order." + M3_CALL[:14], M3_CALL[14:]],
        M3_PROSE,
        "",
    ),
    "disabled_prose_split_across_deltas": (
        False,
        [M3_PROSE[:20], M3_PROSE[20:-7], " order.", M3_CALL[:14], M3_CALL[14:]],
        M3_PROSE,
        "",
    ),
    "disabled_tool_only_leading_newlines": (
        False,
        ["\n\n" + M3_CALL[:14], M3_CALL[14:]],
        "",
        "",
    ),
    "enabled_prose_and_marker_in_one_delta": (
        True,
        ["Plan it.", "</mm:think>" + M3_PROSE[:-7], " order." + M3_CALL[:14], M3_CALL[14:]],
        M3_PROSE,
        "Plan it.",
    ),
    "enabled_prose_split_across_deltas": (
        True,
        ["Plan it.</mm:think>" + M3_PROSE[:20], M3_PROSE[20:-7], " order.",
         M3_CALL[:14], M3_CALL[14:]],
        M3_PROSE,
        "Plan it.",
    ),
    "enabled_tool_only_leading_newlines": (
        True,
        ["Plan it.</mm:think>", "\n\n" + M3_CALL[:14], M3_CALL[14:]],
        "",
        "Plan it.",
    ),
    "adaptive_thinks_then_prose_and_marker_in_one_delta": (
        None,
        ["<mm:think>Plan it.</mm:think>" + M3_PROSE[:-7], " order." + M3_CALL[:14], M3_CALL[14:]],
        M3_PROSE,
        "Plan it.",
    ),
    "adaptive_direct_prose_and_marker_in_one_delta": (
        None,
        [M3_PROSE[:20], M3_PROSE[20:-7], " order." + M3_CALL[:14], M3_CALL[14:]],
        M3_PROSE,
        "",
    ),
}


def _assert_no_m3_markup(text):
    assert "minimax[>[" not in text
    assert "<tool_call>" not in text and "<invoke" not in text
    assert "mm:think" not in text
    assert "echo two" not in text


@pytest.mark.parametrize("case", list(M3_CASES))
def test_m3_chat_completions_streams_prose_before_tool_call(m3_server, case):
    think, deltas, want_content, want_reasoning = M3_CASES[case]
    content, reasoning, names, content_deltas = _run_chat(
        deltas, think, tool=M3_TOOL, endpoint_kwargs=m3_server
    )
    assert names == ["run_shell"]
    assert content == want_content
    assert reasoning == want_reasoning
    _assert_no_m3_markup(content)
    if not want_content:
        assert content_deltas == []


@pytest.mark.parametrize("case", list(M3_CASES))
def test_m3_responses_streams_prose_before_tool_call(m3_server, case):
    think, deltas, want_content, want_reasoning = M3_CASES[case]
    content, reasoning, names, content_deltas, done_texts, completed_texts = (
        _run_responses(deltas, think, tool=M3_TOOL, endpoint_kwargs=m3_server)
    )
    assert names == ["run_shell"]
    assert content == want_content
    assert reasoning == want_reasoning
    _assert_no_m3_markup(content)
    if want_content:
        assert done_texts == [want_content]
        assert completed_texts == [want_content]
    else:
        assert content_deltas == []
        assert not any(t.strip() for t in done_texts + completed_texts)


@pytest.mark.parametrize("wire", ["chat", "responses"])
def test_invalid_call_in_unclosed_reasoning_stays_private(dsv4_server, wire):
    bad = dsv4_server.call.replace('name="run_command"', 'name="not_offered"')
    private = "I should privately consider the command before invoking it.\n"
    deltas = [private, bad]
    _Engine.calls = 0
    run = _run_chat if wire == "chat" else _run_responses
    content, reasoning, names, *_ = run(deltas, True)
    assert names == []
    assert content == ""
    assert private.strip() in reasoning
    assert _Engine.calls == 1, "Rejected native calls must not generate a replacement answer"


@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize("terminal_chunk", [False, True])
def test_m3_native_tool_turn_does_not_publish_model_generated_result(m3_server, wire, terminal_chunk):
    tail = "The command already returned alpha without execution."
    deltas = [M3_PROSE + M3_CALL + tail] if terminal_chunk else [M3_PROSE, M3_CALL] + [tail] * 8
    run = _run_chat if wire == "chat" else _run_responses
    content, reasoning, names, *_ = run(
        deltas, False, tool=M3_TOOL, endpoint_kwargs=m3_server
    )
    assert names == ["run_shell"]
    assert content == M3_PROSE
    assert reasoning == ""
