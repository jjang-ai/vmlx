"""Transport behaviour of engine diagnostics on the gateway dialects, driven by the
RAW chunks retained from the 2026-09-06 schema-lane receipts (Qwen3.8 4S, tool
`set_mode`, warn and enforce modes). These are behavioural checks of the adapters,
not source-string assertions: what does an Ollama or Anthropic client actually
receive when the OpenAI lane emits a `warnings` chunk or a terminal `error`?"""
import json

from vmlx_engine.api.anthropic_adapter import AnthropicStreamAdapter, to_anthropic_response
from vmlx_engine.api.ollama_adapter import (
    openai_chat_chunk_to_ollama_ndjson,
    openai_chat_response_to_ollama,
)

# retained: schema-lanes/after-e2f366f6/lanes-warn-chat-stream.raw (warn mode: call delivered + diagnostic)
WARN_CHUNK = ('data: {"id": "chatcmpl-041b6b0c", "object": "chat.completion.chunk", "created": 1788686689, '
              '"model": "jangq-ai/Qwen3.8-Flash-Next-JANG_4S", "choices": [], "warnings": ["The tool call to '
              "'set_mode' was delivered, but its arguments violate the tool's schema: level: 9 is greater than the "
              "maximum of 5; mode: 'hyperspeed' is not one of ['eco', 'balanced', 'turbo'].\"]}")
# retained: lanes-enforce-chat-stream.raw (enforce mode: call dropped + diagnostic + honest terminal error)
ENFORCE_WARN_CHUNK = ('data: {"id": "chatcmpl-9378d1c9", "object": "chat.completion.chunk", "created": 1788686728, '
                      '"model": "jangq-ai/Qwen3.8-Flash-Next-JANG_4S", "choices": [], "warnings": ["A tool call to '
                      "'set_mode' was dropped because its arguments violate the tool's schema: level: 9 is greater "
                      "than the maximum of 5; mode: 'hyperspeed' is not one of ['eco', 'balanced', 'turbo'].\"]}")
ENFORCE_ERROR_CHUNK = ('data: {"id": "chatcmpl-9378d1c9", "object": "chat.completion.chunk", "error": {"message": '
                       '"The model produced reasoning_content but no visible answer and no tool call. This turn is '
                       'incomplete; retry with a larger output budget or adjust the prompt/reasoning settings.", '
                       '"type": "invalid_response_error", "code": "reasoning_only_no_content"}}')
DIAG_WARN = "arguments violate the tool's schema"
# retained shape of the warn-mode JSON reply (lanes-warn-chat-json.raw): call delivered + warnings field
WARN_JSON = {
    "id": "chatcmpl-warn", "object": "chat.completion", "created": 1788686690, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "set_mode", "arguments": "{\"mode\": \"hyperspeed\", \"level\": 9}"}}]},
        "finish_reason": "tool_calls"}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    "warnings": ["The tool call to 'set_mode' was delivered, but its arguments violate the tool's schema: level: 9 is greater than the maximum of 5."],
}


def test_ollama_stream_turns_the_warning_into_visible_content_and_the_error_into_an_error_line():
    warn = json.loads(openai_chat_chunk_to_ollama_ndjson(WARN_CHUNK, "m"))
    assert warn["message"]["role"] == "assistant" and DIAG_WARN in warn["message"]["content"]
    assert "[vMLX notice]" in warn["message"]["content"] and warn["done"] is False
    dropped = json.loads(openai_chat_chunk_to_ollama_ndjson(ENFORCE_WARN_CHUNK, "m"))
    assert "was dropped because" in dropped["message"]["content"]
    err = json.loads(openai_chat_chunk_to_ollama_ndjson(ENFORCE_ERROR_CHUNK, "m"))
    assert "error" in err and "reasoning_only_no_content" in json.dumps(err) or "no visible answer" in json.dumps(err)


def test_ollama_json_reply_carries_the_call_but_drops_the_warning():
    out = openai_chat_response_to_ollama(WARN_JSON, "m")
    msg = out["message"]
    assert msg["tool_calls"] and msg["tool_calls"][0]["function"]["name"] == "set_mode"
    # documented gap: the Ollama JSON shape has no warnings field and the adapter does not inject prose
    assert "warnings" not in out and DIAG_WARN not in json.dumps(out)


def test_anthropic_stream_turns_the_warning_into_assistant_text_and_the_error_into_an_error_event():
    adapter = AnthropicStreamAdapter("m", "msg_test")
    events = adapter.process_chunk(WARN_CHUNK)
    joined = "".join(events)
    assert "text_delta" in joined and DIAG_WARN in joined and "[vMLX notice]" in joined
    adapter2 = AnthropicStreamAdapter("m", "msg_test2")
    events2 = adapter2.process_chunk(ENFORCE_WARN_CHUNK) + adapter2.process_chunk(ENFORCE_ERROR_CHUNK)
    joined2 = "".join(events2)
    assert "was dropped because" in joined2
    assert 'event: error' in joined2 and "invalid_response_error" in joined2


def test_anthropic_json_reply_carries_the_tool_use_but_drops_the_warning():
    out = to_anthropic_response(WARN_JSON, "m", "msg_json")
    kinds = [b.get("type") for b in out.get("content", [])]
    assert "tool_use" in kinds
    # documented gap: the Anthropic JSON shape has no warnings field and the conversion does not inject prose
    assert "warnings" not in out and DIAG_WARN not in json.dumps(out)
