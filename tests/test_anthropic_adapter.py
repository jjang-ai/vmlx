# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive tests for the Anthropic Messages API adapter.

Tests cover:
- Request conversion (Anthropic → Chat Completions)
- Response conversion (Chat Completions → Anthropic)
- Streaming adapter (SSE event translation)
- Tool calling round-trips
- Thinking/reasoning blocks
- System prompt handling
- Image/multimodal content
- Edge cases and error handling
- Compatibility with prefix cache, continuous batching, reasoning parsers
"""

import json
import pytest

from vmlx_engine.api.anthropic_adapter import (
    AnthropicRequest,
    AnthropicStreamAdapter,
    to_anthropic_response,
    to_chat_completion,
)


# ─── Request Conversion Tests ────────────────────────────────────────


class TestRequestConversion:
    """Tests for Anthropic → Chat Completions request conversion."""

    def test_basic_text_message(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.model == "test-model"
        assert len(chat_req.messages) == 1
        assert chat_req.messages[0].role == "user"
        assert chat_req.messages[0].content == "Hello"
        assert chat_req.max_tokens == 100

    def test_system_prompt_string(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            system="You are a helpful assistant.",
        )
        chat_req = to_chat_completion(req)
        assert len(chat_req.messages) == 2
        assert chat_req.messages[0].role == "system"
        assert chat_req.messages[0].content == "You are a helpful assistant."
        assert chat_req.messages[1].role == "user"

    def test_system_prompt_blocks(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            system=[
                {"type": "text", "text": "System line 1"},
                {"type": "text", "text": "System line 2"},
            ],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.messages[0].role == "system"
        assert "System line 1" in chat_req.messages[0].content
        assert "System line 2" in chat_req.messages[0].content

    def test_temperature_and_top_p_passthrough(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.temperature == 0.7
        assert chat_req.top_p == 0.9
        assert chat_req.top_k == 40

    def test_stop_sequences(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            stop_sequences=["STOP", "END"],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stop == ["STOP", "END"]

    def test_stream_options_on_streaming(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stream is True
        assert chat_req.stream_options is not None
        assert chat_req.stream_options.include_usage is True

    def test_stream_options_absent_non_streaming(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stream is False
        assert chat_req.stream_options is None


# ─── Tool Conversion Tests ───────────────────────────────────────────


class TestToolConversion:
    """Tests for Anthropic tool definition and tool call conversion."""

    def test_tool_definitions_converted(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }],
        )
        chat_req = to_chat_completion(req)
        assert len(chat_req.tools) == 1
        assert chat_req.tools[0].type == "function"
        assert chat_req.tools[0].function["name"] == "get_weather"
        assert chat_req.tools[0].function["description"] == "Get weather for a city"

    def test_tool_choice_any_becomes_required(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice={"type": "any"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.tool_choice == "required"

    def test_tool_choice_auto(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice={"type": "auto"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.tool_choice == "auto"

    def test_tool_choice_none(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice={"type": "none"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.tool_choice == "none"

    def test_tool_choice_specific_tool(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.tool_choice == {"type": "function", "function": {"name": "get_weather"}}

    def test_assistant_tool_use_blocks(self):
        """Anthropic assistant messages with tool_use blocks → OpenAI tool_calls."""
        req = AnthropicRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "call_123", "name": "get_weather",
                     "input": {"city": "Paris"}},
                ]},
            ],
        )
        chat_req = to_chat_completion(req)
        assistant_msg = chat_req.messages[1]
        assert assistant_msg.role == "assistant"
        assert "Let me check." in assistant_msg.content
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(assistant_msg.tool_calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_user_tool_result_blocks(self):
        """Anthropic user messages with tool_result blocks → OpenAI tool response messages."""
        req = AnthropicRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_123",
                     "content": "Sunny, 22°C"},
                ]},
            ],
        )
        chat_req = to_chat_completion(req)
        assert len(chat_req.messages) == 1
        assert chat_req.messages[0].role == "tool"
        assert chat_req.messages[0].content == "Sunny, 22°C"
        assert chat_req.messages[0].tool_call_id == "call_123"


# ─── Thinking / Reasoning Tests ──────────────────────────────────────


class TestThinkingConversion:
    """Tests for thinking/reasoning block conversion."""

    def test_thinking_enabled(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Think carefully."}],
            thinking={"type": "enabled", "budget_tokens": 1000},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.enable_thinking is True

    def test_thinking_disabled(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            thinking={"type": "disabled"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.enable_thinking is False

    def test_thinking_not_specified(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.enable_thinking is None


# ─── Image / Multimodal Tests ────────────────────────────────────────


class TestMultimodalConversion:
    """Tests for image content conversion."""

    def test_base64_image(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo=",
                }},
            ]}],
        )
        chat_req = to_chat_completion(req)
        content = chat_req.messages[0].content
        assert isinstance(content, list)
        assert len(content) == 2
        # Content parts may be dicts or Pydantic models — check via dict access or attribute
        first = content[0] if isinstance(content[0], dict) else content[0].model_dump()
        second = content[1] if isinstance(content[1], dict) else content[1].model_dump()
        assert first["type"] == "text"
        assert second["type"] == "image_url"

    def test_url_image(self):
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "url",
                    "url": "https://example.com/image.jpg",
                }},
                {"type": "text", "text": "Describe this."},
            ]}],
        )
        chat_req = to_chat_completion(req)
        content = chat_req.messages[0].content
        assert isinstance(content, list)


# ─── Response Conversion Tests ───────────────────────────────────────


class TestResponseConversion:
    """Tests for Chat Completions → Anthropic response conversion."""

    def test_basic_text_response(self):
        chat_resp = {
            "choices": [{
                "message": {"content": "Hello there!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = to_anthropic_response(chat_resp, "test-model")
        assert resp["role"] == "assistant"
        assert resp["stop_reason"] == "end_turn"
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][0]["text"] == "Hello there!"
        assert resp["usage"]["input_tokens"] == 10
        assert resp["usage"]["output_tokens"] == 5

    def test_tool_call_response(self):
        chat_resp = {
            "choices": [{
                "message": {
                    "content": "I'll check the weather.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        resp = to_anthropic_response(chat_resp, "test-model")
        assert resp["stop_reason"] == "tool_use"
        # Should have text + tool_use blocks
        types = [c["type"] for c in resp["content"]]
        assert "text" in types
        assert "tool_use" in types
        tool_block = [c for c in resp["content"] if c["type"] == "tool_use"][0]
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "London"}

    def test_reasoning_response(self):
        chat_resp = {
            "choices": [{
                "message": {
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        resp = to_anthropic_response(chat_resp, "test-model")
        types = [c["type"] for c in resp["content"]]
        assert "thinking" in types
        assert "text" in types
        thinking_block = [c for c in resp["content"] if c["type"] == "thinking"][0]
        assert "step by step" in thinking_block["thinking"]

    def test_length_stop_reason(self):
        chat_resp = {
            "choices": [{"message": {"content": "..."}, "finish_reason": "length"}],
            "usage": {},
        }
        resp = to_anthropic_response(chat_resp, "test-model")
        assert resp["stop_reason"] == "max_tokens"

    def test_empty_response(self):
        chat_resp = {"choices": [], "usage": {}}
        resp = to_anthropic_response(chat_resp, "test-model")
        assert resp["content"] == [{"type": "text", "text": ""}]


# ─── Streaming Adapter Tests ─────────────────────────────────────────


class TestStreamingAdapter:
    """Tests for the AnthropicStreamAdapter SSE translation."""

    def test_first_chunk_emits_message_start(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        chunk = 'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}'
        events = adapter.process_chunk(chunk)
        # First event should be message_start
        assert any("message_start" in e for e in events)

    def test_text_content_streaming(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        events.extend(adapter.process_chunk('data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}'))
        events.extend(adapter.process_chunk('data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}'))
        events.extend(adapter.process_chunk('data: {"choices":[{"delta":{"content":" world"},"index":0}]}'))

        text_deltas = [e for e in events if "text_delta" in e]
        assert len(text_deltas) == 2
        assert '"Hello"' in text_deltas[0]
        assert '" world"' in text_deltas[1]

    def test_reasoning_content_streaming(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        events.extend(adapter.process_chunk('data: {"choices":[{"delta":{"reasoning_content":"Think..."},"index":0}]}'))
        events.extend(adapter.process_chunk('data: {"choices":[{"delta":{"content":"Answer."},"index":0}]}'))

        # Should have thinking_delta then content_block_stop then text_delta
        thinking_events = [e for e in events if "thinking_delta" in e]
        assert len(thinking_events) == 1
        assert "Think..." in thinking_events[0]

    def test_tool_call_streaming(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"index":0}]}'
        ))
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":"}}]},"index":0}]}'
        ))

        tool_events = [e for e in events if "tool_use" in e or "input_json_delta" in e]
        assert len(tool_events) >= 1

    def test_finalize_emits_message_stop(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        adapter.process_chunk('data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}')
        events = adapter.finalize()
        assert any("message_stop" in e for e in events)
        assert any("message_delta" in e for e in events)

    def test_finalize_reports_stop_reason_tool_use(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        adapter.process_chunk('data: {"choices":[{"delta":{"content":"text"},"index":0}]}')
        # Open a tool block
        adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"fn","arguments":""}}]},"index":0}]}'
        )
        events = adapter.finalize()
        delta_events = [e for e in events if "message_delta" in e]
        assert any('"tool_use"' in e for e in delta_events)

    def test_usage_tracking(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        adapter.process_chunk('data: {"choices":[{"delta":{"content":"x"},"index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5}}')
        events = adapter.finalize()
        delta_events = [e for e in events if "message_delta" in e]
        # Usage should include output_tokens
        assert any('"output_tokens": 5' in e or '"output_tokens":5' in e for e in delta_events)

    def test_done_marker_ignored(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        events = adapter.process_chunk("data: [DONE]")
        assert events == []

    def test_non_data_lines_ignored(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        events = adapter.process_chunk(": keep-alive")
        assert events == []
        events = adapter.process_chunk("")
        assert events == []

    def test_malformed_json_ignored(self):
        adapter = AnthropicStreamAdapter(model="test-model")
        events = adapter.process_chunk("data: {not valid json}")
        assert events == []


# ─── Integration Compatibility Tests ─────────────────────────────────


class TestAnthropicIntegrationCompat:
    """Tests verifying the adapter works with engine pipeline features."""

    def test_converted_request_has_model_field(self):
        """Prefix cache keys off the model field — must be present."""
        req = AnthropicRequest(
            model="mlx-community/Qwen3-8B-4bit",
            messages=[{"role": "user", "content": "Hello"}],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.model == "mlx-community/Qwen3-8B-4bit"

    def test_converted_request_stream_options_for_batching(self):
        """Continuous batching needs stream_options.include_usage for token counting."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chat_req = to_chat_completion(req)
        assert chat_req.stream_options is not None
        assert chat_req.stream_options.include_usage is True

    def test_enable_thinking_passes_through_for_reasoning_parser(self):
        """Reasoning parsers (deepseek_r1, openai_gptoss) need enable_thinking."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Think carefully."}],
            thinking={"type": "enabled"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.enable_thinking is True

    def test_messages_serializable_for_cache_key(self):
        """Prefix cache hashes messages — they must be JSON-serializable."""
        req = AnthropicRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ],
        )
        chat_req = to_chat_completion(req)
        # All messages must be serializable (for prefix cache token hashing)
        for msg in chat_req.messages:
            serialized = msg.model_dump(exclude_none=True)
            json.dumps(serialized)  # Should not raise

    def test_tool_definitions_compatible_with_tool_parser(self):
        """Tool parsers (hermes, qwen, glm47) need standard OpenAI tool format."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Search for cats"}],
            tools=[{
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }],
        )
        chat_req = to_chat_completion(req)
        tool = chat_req.tools[0]
        # Must match OpenAI function calling format
        assert tool.type == "function"
        assert "name" in tool.function
        assert "parameters" in tool.function
        assert tool.function["parameters"]["type"] == "object"

    def test_multi_turn_with_tool_roundtrip(self):
        """Full tool calling round-trip: user → assistant(tool_use) → user(tool_result) → assistant."""
        req = AnthropicRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "call_xyz", "name": "get_weather",
                     "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_xyz",
                     "content": "Sunny, 20°C"},
                ]},
            ],
        )
        chat_req = to_chat_completion(req)
        assert len(chat_req.messages) == 3
        # Message 0: user
        assert chat_req.messages[0].role == "user"
        # Message 1: assistant with tool_calls
        assert chat_req.messages[1].role == "assistant"
        assert chat_req.messages[1].tool_calls is not None
        # Message 2: tool response
        assert chat_req.messages[2].role == "tool"
        assert chat_req.messages[2].tool_call_id == "call_xyz"

    def test_max_tokens_always_set(self):
        """Engine uses max_tokens for generation — must have a default."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.max_tokens == 4096  # Anthropic default

    def test_empty_messages_rejected_by_validation(self):
        """Edge case: empty message list is rejected by Pydantic (min_length=1)."""
        import pydantic
        req = AnthropicRequest(
            model="test-model",
            messages=[],
        )
        with pytest.raises(pydantic.ValidationError):
            to_chat_completion(req)


# ─── Task 1.1: Request Validation Edge Cases ────────────────────────────


class TestRequestValidationEdgeCases:
    """Edge cases for Anthropic → Chat Completions request conversion."""

    def test_all_optional_fields_set_simultaneously(self):
        """Request with every optional field populated at once."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=200,
            temperature=0.5,
            top_p=0.8,
            top_k=50,
            stop_sequences=["STOP"],
            stream=True,
            tools=[{
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }],
            tool_choice={"type": "auto"},
            thinking={"type": "enabled", "budget_tokens": 500},
            metadata={"user_id": "u123"},
        )
        chat_req = to_chat_completion(req)
        assert chat_req.temperature == 0.5
        assert chat_req.top_p == 0.8
        assert chat_req.top_k == 50
        assert chat_req.stop == ["STOP"]
        assert chat_req.stream is True
        assert chat_req.stream_options is not None
        assert len(chat_req.tools) == 1
        assert chat_req.tool_choice == "auto"
        assert chat_req.enable_thinking is True
        assert chat_req.max_tokens == 200

    def test_deeply_nested_tool_result_content(self):
        """tool_result with a list of content blocks (not a plain string)."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_abc", "content": [
                    {"type": "text", "text": "Line one"},
                    {"type": "text", "text": "Line two"},
                ]},
            ]}],
        )
        chat_req = to_chat_completion(req)
        assert len(chat_req.messages) == 1
        assert chat_req.messages[0].role == "tool"
        assert "Line one" in chat_req.messages[0].content
        assert "Line two" in chat_req.messages[0].content

    def test_mixed_text_and_tool_result_in_user_message(self):
        """User message containing both text blocks and tool_result blocks."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Here are the results:"},
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "Result data"},
            ]}],
        )
        chat_req = to_chat_completion(req)
        # Should produce a user message (text) + tool message (result)
        assert len(chat_req.messages) == 2
        assert chat_req.messages[0].role == "user"
        assert "Here are the results:" in chat_req.messages[0].content
        assert chat_req.messages[1].role == "tool"
        assert chat_req.messages[1].content == "Result data"
        assert chat_req.messages[1].tool_call_id == "call_1"

    def test_assistant_with_only_thinking_blocks(self):
        """Assistant message with only thinking blocks and no text."""
        req = AnthropicRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "Reason about this."},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Deep thought here..."},
                ]},
            ],
        )
        chat_req = to_chat_completion(req)
        assistant_msg = chat_req.messages[1]
        assert assistant_msg.role == "assistant"
        # Thinking blocks are skipped in conversion; content may be None or empty
        assert assistant_msg.tool_calls is None

    def test_multiple_tool_use_blocks_in_assistant(self):
        """Assistant message with multiple tool_use blocks at once."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "assistant", "content": [
                {"type": "text", "text": "I'll use two tools."},
                {"type": "tool_use", "id": "call_a", "name": "search",
                 "input": {"q": "cats"}},
                {"type": "tool_use", "id": "call_b", "name": "weather",
                 "input": {"city": "NYC"}},
            ]}],
        )
        chat_req = to_chat_completion(req)
        msg = chat_req.messages[0]
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0]["id"] == "call_a"
        assert msg.tool_calls[0]["function"]["name"] == "search"
        assert msg.tool_calls[1]["id"] == "call_b"
        assert msg.tool_calls[1]["function"]["name"] == "weather"

    def test_tool_result_with_empty_content_string(self):
        """tool_result block where content is an empty string."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_99", "content": ""},
            ]}],
        )
        chat_req = to_chat_completion(req)
        assert chat_req.messages[0].role == "tool"
        assert chat_req.messages[0].content == ""
        assert chat_req.messages[0].tool_call_id == "call_99"

    def test_system_prompt_empty_string(self):
        """System prompt as empty string should not crash."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            system="",
        )
        chat_req = to_chat_completion(req)
        # Empty string is falsy, so no system message should be prepended
        assert chat_req.messages[0].role == "user"

    def test_system_prompt_empty_list(self):
        """System prompt as empty list should not crash."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            system=[],
        )
        chat_req = to_chat_completion(req)
        # Empty list is falsy, so no system message should be prepended
        assert chat_req.messages[0].role == "user"


# ─── Task 1.2: Tool Parser Compatibility ────────────────────────────────


class TestToolParserCompatibility:
    """Tests that converted tool definitions match what hermes/qwen/glm parsers expect."""

    def test_tool_definitions_have_parameters_key(self):
        """Converted tools must use 'parameters' (not 'input_schema') for OpenAI format."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Use tools"}],
            tools=[{
                "name": "calculator",
                "description": "Do math",
                "input_schema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
            }],
        )
        chat_req = to_chat_completion(req)
        func = chat_req.tools[0].function
        assert "parameters" in func, "Tool must have 'parameters' key for OpenAI compat"
        assert "input_schema" not in func, "input_schema must NOT leak into OpenAI format"
        assert func["parameters"]["type"] == "object"
        assert "expr" in func["parameters"]["properties"]

    def test_tool_call_ids_survive_conversion(self):
        """Tool call IDs must round-trip through Anthropic → OpenAI conversion."""
        req = AnthropicRequest(
            model="test-model",
            messages=[
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_exact_id_123", "name": "fn",
                     "input": {"x": 1}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_exact_id_123",
                     "content": "OK"},
                ]},
            ],
        )
        chat_req = to_chat_completion(req)
        # The tool_use ID in assistant message
        assert chat_req.messages[0].tool_calls[0]["id"] == "toolu_exact_id_123"
        # The tool_result references the same ID
        assert chat_req.messages[1].tool_call_id == "toolu_exact_id_123"

    def test_multiple_tools_same_name_different_inputs(self):
        """Edge case: two tool definitions with the same name but different schemas."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[
                {"name": "fetch", "description": "Fetch v1",
                 "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}}},
                {"name": "fetch", "description": "Fetch v2",
                 "input_schema": {"type": "object", "properties": {"uri": {"type": "string"}}}},
            ],
        )
        chat_req = to_chat_completion(req)
        assert len(chat_req.tools) == 2
        assert chat_req.tools[0].function["name"] == "fetch"
        assert chat_req.tools[1].function["name"] == "fetch"
        # The schemas should be distinct
        assert "url" in chat_req.tools[0].function["parameters"]["properties"]
        assert "uri" in chat_req.tools[1].function["parameters"]["properties"]

    def test_tool_arguments_empty_object(self):
        """Tool call with empty {} arguments must serialize correctly."""
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_empty", "name": "noop", "input": {}},
            ]}],
        )
        chat_req = to_chat_completion(req)
        tc = chat_req.messages[0].tool_calls[0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_tool_arguments_nested_json(self):
        """Tool call with deeply nested JSON arguments."""
        nested_input = {
            "config": {
                "database": {"host": "localhost", "port": 5432},
                "options": {"retry": True, "timeout": 30},
            },
            "tags": ["a", "b"],
        }
        req = AnthropicRequest(
            model="test-model",
            messages=[{"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_nested", "name": "deploy",
                 "input": nested_input},
            ]}],
        )
        chat_req = to_chat_completion(req)
        tc = chat_req.messages[0].tool_calls[0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == nested_input
        assert parsed["config"]["database"]["port"] == 5432
        assert parsed["tags"] == ["a", "b"]


# ─── Task 1.3: Streaming Adapter State Machine ──────────────────────────


class TestStreamingStateMachine:
    """Tests for state transitions in AnthropicStreamAdapter."""

    def test_thinking_to_text_transition(self):
        """thinking -> text must close thinking block before opening text block."""
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"reasoning_content":"Hmm..."},"index":0}]}'
        ))
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"content":"Answer."},"index":0}]}'
        ))
        # Extract event types in order by parsing each SSE event's data
        event_types = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("data: "):
                    try:
                        d = json.loads(line[6:])
                        event_types.append(d.get("type", ""))
                    except json.JSONDecodeError:
                        pass
        # Sequence must be: message_start, content_block_start (thinking),
        # content_block_delta (thinking), content_block_stop, content_block_start (text),
        # content_block_delta (text)
        stop_idx = event_types.index("content_block_stop")
        # Find the text content_block_start AFTER the stop
        remaining = event_types[stop_idx + 1:]
        assert "content_block_start" in remaining, "text block must start after thinking block stops"
        # Verify both content types present
        all_text = "\n".join(events)
        assert "thinking_delta" in all_text
        assert "text_delta" in all_text

    def test_text_to_tool_use_transition(self):
        """text -> tool_use must close text block before opening tool block."""
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"content":"I will call a tool."},"index":0}]}'
        ))
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_t1","function":{"name":"fn","arguments":""}}]},"index":0}]}'
        ))
        all_text = "\n".join(events)
        # Text block must be stopped before tool_use block starts
        text_delta_pos = all_text.find("text_delta")
        # Find the content_block_stop after the text content
        block_stop_after_text = all_text.find("content_block_stop", text_delta_pos)
        tool_start_pos = all_text.find("tool_use", block_stop_after_text)
        assert block_stop_after_text != -1, "text block must be closed"
        assert tool_start_pos != -1, "tool_use block must be opened"
        assert block_stop_after_text < tool_start_pos

    def test_multiple_sequential_tool_calls(self):
        """Multiple tool calls: each new tool closes the previous one."""
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        # First tool
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"fn1","arguments":"{\\"a\\":1}"}}]},"index":0}]}'
        ))
        # Second tool (should close first)
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"fn2","arguments":"{\\"b\\":2}"}}]},"index":0}]}'
        ))
        all_text = "\n".join(events)
        # Count tool_use starts — should be 2
        tool_starts = all_text.count('"type": "tool_use"') + all_text.count('"type":"tool_use"')
        assert tool_starts >= 2, f"Expected 2 tool_use starts, found {tool_starts}"
        # The first tool block should be stopped before the second opens
        first_tool_pos = all_text.find("call_1")
        stop_between = all_text.find("content_block_stop", first_tool_pos)
        second_tool_pos = all_text.find("call_2")
        assert stop_between < second_tool_pos, "First tool block must close before second opens"

    def test_interleaved_thinking_text_tool_all_closed(self):
        """thinking -> text -> tool_use: finalize must close whatever is still open."""
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        # Thinking
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]}'
        ))
        # Text (closes thinking)
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"content":"Here is my answer."},"index":0}]}'
        ))
        # Tool (closes text)
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_f","function":{"name":"fn","arguments":""}}]},"index":0}]}'
        ))
        # Finalize (closes tool)
        events.extend(adapter.finalize())
        # Parse event types from SSE data lines (each event is "event: ...\ndata: ...\n\n")
        data_types = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("data: "):
                    try:
                        d = json.loads(line[6:])
                        data_types.append(d.get("type", ""))
                    except json.JSONDecodeError:
                        pass
        # Count content_block_start and content_block_stop from parsed data
        starts = data_types.count("content_block_start")
        stops = data_types.count("content_block_stop")
        assert starts == stops, f"Unbalanced blocks: {starts} starts vs {stops} stops"
        # There should be exactly 3 content blocks (thinking, text, tool)
        assert starts == 3, f"Expected 3 content blocks, got {starts}"
        # Finalize must emit message_stop
        assert "message_stop" in data_types

    def test_usage_only_chunk_no_crash(self):
        """A chunk with usage data but no choices should not crash."""
        adapter = AnthropicStreamAdapter(model="test-model")
        # First, send a normal chunk to initialize
        events = adapter.process_chunk(
            'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}'
        )
        assert len(events) > 0  # Should have message_start + content
        # Now send a usage-only chunk (no choices)
        events = adapter.process_chunk(
            'data: {"usage":{"prompt_tokens":42,"completion_tokens":7}}'
        )
        # Should not crash; may return empty list or usage events
        assert isinstance(events, list)
        # Finalize should pick up the usage
        final = adapter.finalize()
        delta_events = [e for e in final if "message_delta" in e]
        assert any("7" in e for e in delta_events), "Output tokens should reflect usage-only chunk"

    def test_multiple_reasoning_chunks_accumulated(self):
        """Multiple reasoning_content deltas should each produce a thinking_delta event."""
        adapter = AnthropicStreamAdapter(model="test-model")
        events = []
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"reasoning_content":"Step 1. "},"index":0}]}'
        ))
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"reasoning_content":"Step 2. "},"index":0}]}'
        ))
        events.extend(adapter.process_chunk(
            'data: {"choices":[{"delta":{"reasoning_content":"Step 3."},"index":0}]}'
        ))
        thinking_deltas = [e for e in events if "thinking_delta" in e]
        assert len(thinking_deltas) == 3, f"Expected 3 thinking_delta events, got {len(thinking_deltas)}"
        assert "Step 1." in thinking_deltas[0]
        assert "Step 2." in thinking_deltas[1]
        assert "Step 3." in thinking_deltas[2]
        # Only one content_block_start for thinking (block opened once, not reopened)
        block_starts = [e for e in events if "content_block_start" in e and "thinking" in e]
        assert len(block_starts) == 1, "Thinking block should be opened exactly once"
