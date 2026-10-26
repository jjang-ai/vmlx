"""The contentless-assistant drop must exist on every dialect's request prep.

cf627b3cd fixed chat + /v1/responses; the Anthropic and streaming-Ollama
preps build their own message lists and were instances 3 and 4 of the
fix-one-of-N class. All four now share _drop_contentless_assistant_turns.
"""

from vmlx_engine.server import _drop_contentless_assistant_turns


def test_drop_removes_assistant_turns_with_no_content_and_no_tool_calls():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant"},
        {"role": "user", "content": "again"},
    ]

    assert _drop_contentless_assistant_turns(messages) == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "again"},
    ]


def test_drop_keeps_tool_call_only_and_normal_turns():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "assistant", "content": "fine"},
        {"role": "tool", "content": ""},
    ]

    assert _drop_contentless_assistant_turns(messages) == messages


def test_every_dialect_request_prep_calls_the_shared_drop():
    import inspect
    import vmlx_engine.server as server

    src = inspect.getsource(server)
    # def + chat + responses + anthropic + ollama-stream
    assert src.count("_drop_contentless_assistant_turns(") >= 5


def test_server_default_thinking_off_degrades_to_auto_on_no_instruct_families(monkeypatch):
    """A server-resolved enable_thinking=false default must not make the
    engine 400 its own configuration on families without an instruct mode
    (lfm2/step37 rejected every text request while a global default was set);
    it degrades to Auto. A CALLER-requested false still 400s."""
    import vmlx_engine.server as server

    class _MC:
        family_name = "lfm2"
        model_type = "lfm2"
        supports_instruct_mode = False
        supports_thinking = True
        reasoning_parser = "qwen3"
        think_in_template = False
        architecture_hints = {}

    class _Registry:
        def lookup(self, key):
            return _MC()

    monkeypatch.setattr(
        "vmlx_engine.model_config_registry.get_model_config_registry",
        lambda: _Registry(),
    )
    monkeypatch.setattr(server, "_default_enable_thinking", False)

    resolved = server._resolve_enable_thinking(
        None, {}, False, "lfm2-test", engine=None
    )
    assert resolved is not False, "server default false must degrade, not force off"

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        server._resolve_enable_thinking(False, {}, False, "lfm2-test", engine=None)


def test_responses_chain_replay_drops_the_stored_reasoning_only_turn():
    """dialect F2: the store deliberately keeps a reasoning-only assistant turn
    (content "", reasoning_content set) so the chain warning and introspection
    work — but replaying it through previous_response_id handed strict
    templates the exact contentless turn every dialect drops from client
    input, reintroducing the fixed 500. The getter is the replay boundary."""
    import vmlx_engine.server as server

    stored = [
        {"role": "user", "content": "think about it"},
        {"role": "assistant", "content": "", "reasoning_content": "private chain"},
        {"role": "user", "content": "follow-up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]
    server._responses_store_history("resp_test_f2", stored, reasoning_only=True)
    try:
        replayed = server._responses_get_history("resp_test_f2")
        roles_contents = [
            (m.get("role"), bool(m.get("content")), bool(m.get("tool_calls")))
            for m in replayed
        ]
        assert ("assistant", False, False) not in roles_contents, replayed
        assert any(r == "assistant" and tc for r, _, tc in roles_contents), (
            "tool-call-only assistant turns are valid and must survive replay"
        )
        with server._responses_history_lock:
            in_store = server._responses_history["resp_test_f2"]
        assert any(
            m.get("role") == "assistant" and m.get("reasoning_content")
            and not m.get("content") and not m.get("tool_calls")
            for m in in_store
        ), "the STORE must keep the reasoning-only turn for the chain warning"
    finally:
        with server._responses_history_lock:
            server._responses_history.pop("resp_test_f2", None)
            server._responses_was_reasoning_only.discard("resp_test_f2")


class _Renderer:
    """A chat renderer that refuses (Mistral-style) or renders an empty assistant turn."""

    def __init__(self, refuses: bool, template: str = "t"):
        self.refuses = refuses
        self.chat_template = template
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        out = []
        for m in messages:
            if m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls"):
                if self.refuses:
                    raise ValueError("Assistant message must have a string or a list of chunks in content")
            out.append(f"<{m['role']}>{m.get('content', '')}")
        return "".join(out) + "<assistant>"


class _Engine:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self._processor = processor
        self._is_mllm = processor is not None
        self.model = None


def _with_engine(monkeypatch, engine):
    import vmlx_engine.server as server

    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_CONTENTLESS_TURN_PROBE", {}, raising=False)
    return server


HISTORY = [
    {"role": "user", "content": "find the logs"},
    {"role": "assistant"},  # a reasoning-only `length` reply, content dropped by model_dump(exclude_none)
    {"role": "user", "content": "run one more command"},
]


def test_template_that_refuses_the_empty_turn_keeps_the_drop(monkeypatch):
    server = _with_engine(monkeypatch, _Engine(_Renderer(refuses=True)))
    assert server._drop_contentless_assistant_turns(HISTORY) == [HISTORY[0], HISTORY[2]]


def test_template_that_renders_the_empty_turn_keeps_it_so_the_prompt_extends_the_cached_one(monkeypatch):
    tok = _Renderer(refuses=False)
    server = _with_engine(monkeypatch, _Engine(tok))
    kept = server._drop_contentless_assistant_turns(HISTORY)
    assert kept == [HISTORY[0], {"role": "assistant", "content": ""}, HISTORY[2]]
    assert HISTORY[1] == {"role": "assistant"}, "the caller's message must not be mutated"
    # The kept history renders as the previous prompt plus the new turns: the prefix cache can extend it.
    previous = tok.apply_chat_template(HISTORY[:1])[: -len("<assistant>")]
    assert tok.apply_chat_template(kept).startswith(previous + "<assistant>")
    calls = tok.calls
    server._drop_contentless_assistant_turns(HISTORY)
    assert tok.calls == calls, "the probe is cached per template"


def test_multimodal_engine_drops_when_its_processor_refuses(monkeypatch):
    server = _with_engine(monkeypatch, _Engine(_Renderer(refuses=False), processor=_Renderer(refuses=True)))
    assert server._drop_contentless_assistant_turns(HISTORY) == [HISTORY[0], HISTORY[2]]


def test_no_engine_keeps_the_drop(monkeypatch):
    server = _with_engine(monkeypatch, None)
    assert server._drop_contentless_assistant_turns(HISTORY) == [HISTORY[0], HISTORY[2]]


def test_chat_path_extraction_keeps_the_turn_for_the_template_aware_drop_to_decide(monkeypatch):
    """/v1/chat/completions (and streaming Ollama) run extract_multimodal_content first; it used to drop every
    empty assistant turn itself, so the template-aware drop never saw one (live DeepSeek-V4.1 A/B, 2026-10-26)."""
    from vmlx_engine.api.utils import extract_multimodal_content

    request = [
        {"role": "user", "content": "find the logs"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "run one more command"},
    ]
    extracted, _, _ = extract_multimodal_content(request)
    assert [m["role"] for m in extracted] == ["user", "assistant", "user"]

    server = _with_engine(monkeypatch, _Engine(_Renderer(refuses=False)))
    assert [m["role"] for m in server._drop_contentless_assistant_turns(extracted)] == ["user", "assistant", "user"]
    server = _with_engine(monkeypatch, _Engine(_Renderer(refuses=True)))
    assert [m["role"] for m in server._drop_contentless_assistant_turns(extracted)] == ["user", "user"]


def test_thinking_off_strip_keeps_a_reasoning_only_turn_the_template_renders(monkeypatch):
    server = _with_engine(monkeypatch, _Engine(_Renderer(refuses=False)))
    monkeypatch.setattr(server, "_template_keeps_prior_reasoning", lambda request_thinking: False)
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "reasoning_content": "unfinished"},
        {"role": "user", "content": "again"},
    ]
    assert server._strip_prior_reasoning_for_thinking_off(history)[1] == {"role": "assistant", "content": ""}
    server = _with_engine(monkeypatch, _Engine(_Renderer(refuses=True)))
    monkeypatch.setattr(server, "_template_keeps_prior_reasoning", lambda request_thinking: False)
    assert [m["role"] for m in server._strip_prior_reasoning_for_thinking_off(history)] == ["user", "user"]
