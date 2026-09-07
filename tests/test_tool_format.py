"""Tests for tool format conversion, tool_choice handling, and related edge cases.

Covers gaps identified in the comprehensive test audit:
- ResponsesToolDefinition.to_chat_completions_format() conversion
- convert_tools_for_template() roundtrip
- tool_choice suppression and filtering
- response_format.strict validation
- max_tokens fallback chain
"""

from pathlib import Path

import pytest

# ─── ResponsesToolDefinition Conversion ──────────────────────────────────────


class TestResponsesToolDefinitionConversion:
    """Test ResponsesToolDefinition.to_chat_completions_format()."""

    def test_basic_conversion(self):
        from vmlx_engine.api.models import ResponsesToolDefinition

        td = ResponsesToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        result = td.to_chat_completions_format()
        assert result == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }

    def test_with_strict_true(self):
        from vmlx_engine.api.models import ResponsesToolDefinition

        td = ResponsesToolDefinition(name="fn", strict=True)
        result = td.to_chat_completions_format()
        assert result["function"]["strict"] is True

    def test_with_strict_false(self):
        from vmlx_engine.api.models import ResponsesToolDefinition

        td = ResponsesToolDefinition(name="fn", strict=False)
        result = td.to_chat_completions_format()
        assert result["function"]["strict"] is False

    def test_no_optional_fields(self):
        from vmlx_engine.api.models import ResponsesToolDefinition

        td = ResponsesToolDefinition(name="ping")
        result = td.to_chat_completions_format()
        assert result["function"]["name"] == "ping"
        assert "description" not in result["function"]
        assert "parameters" not in result["function"]
        assert "strict" not in result["function"]

    def test_type_is_always_function(self):
        from vmlx_engine.api.models import ResponsesToolDefinition

        td = ResponsesToolDefinition(name="test")
        result = td.to_chat_completions_format()
        assert result["type"] == "function"

    def test_roundtrip_through_tool_definition(self):
        """Converted ResponsesToolDefinition should be valid ToolDefinition input."""
        from vmlx_engine.api.models import ResponsesToolDefinition, ToolDefinition

        flat = ResponsesToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        nested = flat.to_chat_completions_format()
        td = ToolDefinition(**nested)
        assert td.type == "function"
        assert td.function["name"] == "search"
        assert td.function["description"] == "Search the web"


# ─── convert_tools_for_template ──────────────────────────────────────────────


class TestConvertToolsForTemplate:
    """Test convert_tools_for_template() with various inputs."""

    def test_none_returns_none(self):
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        assert convert_tools_for_template(None) is None

    def test_empty_list_returns_none(self):
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        assert convert_tools_for_template([]) is None

    def test_chat_completions_format_dict(self):
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = convert_tools_for_template(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"

    def test_pydantic_tool_definition(self):
        from vmlx_engine.api.models import ToolDefinition
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        tools = [
            ToolDefinition(
                function={"name": "read_file", "description": "Read a file", "parameters": {}}
            )
        ]
        result = convert_tools_for_template(tools)
        assert result is not None
        assert result[0]["function"]["name"] == "read_file"

    def test_flat_responses_format(self):
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        tools = [
            {"type": "function", "name": "search", "description": "Search", "parameters": {}}
        ]
        result = convert_tools_for_template(tools)
        assert result is not None
        assert result[0]["function"]["name"] == "search"

    def test_missing_function_key_skipped(self):
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        tools = [{"type": "web_search"}]
        result = convert_tools_for_template(tools)
        assert result is None  # Non-function tools are skipped

    def test_multiple_tools(self):
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        tools = [
            {
                "type": "function",
                "function": {"name": "tool_a", "description": "A", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "tool_b", "description": "B", "parameters": {}},
            },
        ]
        result = convert_tools_for_template(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"

    def test_default_parameters_for_missing_key(self):
        """When parameters is missing, default should be an empty object schema."""
        from vmlx_engine.api.tool_calling import convert_tools_for_template

        tools = [
            {"type": "function", "function": {"name": "ping", "description": "Ping"}}
        ]
        result = convert_tools_for_template(tools)
        assert result[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


# ─── tool_choice Handling ────────────────────────────────────────────────────


class TestToolChoiceSuppression:
    """Test tool_choice behavior for both Chat Completions and Responses API."""

    def test_tool_choice_none_is_falsy(self):
        """tool_choice='none' should suppress all tools."""
        _tool_choice = "none"
        _suppress_tools = _tool_choice == "none"
        assert _suppress_tools is True

    def test_tool_choice_auto_allows_tools(self):
        _tool_choice = "auto"
        _suppress_tools = _tool_choice == "none"
        assert _suppress_tools is False

    def test_tool_choice_none_type_allows_tools(self):
        """None (not the string 'none') should allow tools."""
        _tool_choice = None
        _suppress_tools = _tool_choice == "none"
        assert _suppress_tools is False

    def test_tool_choice_required_allows_tools(self):
        _tool_choice = "required"
        _suppress_tools = _tool_choice == "none"
        assert _suppress_tools is False

    def test_tool_choice_dict_filter(self):
        """tool_choice as a specific tool dict should filter to that tool."""
        from vmlx_engine.api.models import ToolDefinition

        _tool_choice = {"type": "function", "function": {"name": "search"}}
        target_name = _tool_choice.get("function", {}).get("name") or _tool_choice.get(
            "name"
        )
        assert target_name == "search"

        tools = [
            ToolDefinition(function={"name": "search", "description": "Search"}),
            ToolDefinition(function={"name": "read_file", "description": "Read"}),
        ]
        filtered = [t for t in tools if t.function.get("name") == target_name]
        assert len(filtered) == 1
        assert filtered[0].function["name"] == "search"

    def test_tool_choice_dict_flat_name_format(self):
        """Some clients send tool_choice as {"name": "tool_name"} directly."""
        _tool_choice = {"name": "search"}
        target_name = _tool_choice.get("function", {}).get("name") or _tool_choice.get(
            "name"
        )
        assert target_name == "search"

    def test_tool_choice_dict_no_match_fails_closed(self):
        """A missing specific tool must never authorize the remaining catalog."""
        from fastapi import HTTPException

        from vmlx_engine.api.models import ToolDefinition
        from vmlx_engine.server import _suppress_tool_parsing_when_no_tools

        target_name = "nonexistent"
        tools = [
            ToolDefinition(function={"name": "search", "description": "Search"}),
        ]
        filtered = [t for t in tools if t.function.get("name") == target_name]
        assert filtered == []
        with pytest.raises(HTTPException, match="tool_choice requires"):
            _suppress_tool_parsing_when_no_tools(
                filtered,
                {"type": "function", "function": {"name": target_name}},
                "Chat Completions",
            )

    def test_specific_tool_choice_dict_counts_as_required_for_enforcement(self):
        from vmlx_engine.server import _is_required_tool_choice

        assert _is_required_tool_choice("required") is True
        assert (
            _is_required_tool_choice(
                {"type": "function", "function": {"name": "record_fact"}}
            )
            is True
        )
        assert _is_required_tool_choice({"name": "record_fact"}) is True
        assert _is_required_tool_choice("auto") is False
        assert _is_required_tool_choice(None) is False


# ─── Responses API tool_choice ───────────────────────────────────────────────


class TestResponsesApiToolChoice:
    """Verify the Responses API path now handles tool_choice."""

    def test_responses_request_has_tool_choice(self):
        from vmlx_engine.api.models import ResponsesRequest

        req = ResponsesRequest(model="test", input="hello", tool_choice="none")
        assert req.tool_choice == "none"

    def test_responses_request_tool_choice_default_none(self):
        from vmlx_engine.api.models import ResponsesRequest

        req = ResponsesRequest(model="test", input="hello")
        assert req.tool_choice is None

    def test_responses_request_tool_choice_dict(self):
        from vmlx_engine.api.models import ResponsesRequest

        req = ResponsesRequest(
            model="test",
            input="hello",
            tool_choice={"type": "function", "function": {"name": "search"}},
        )
        assert isinstance(req.tool_choice, dict)


# ─── response_format.strict ──────────────────────────────────────────────────


class TestStrictResponseFormat:
    """Test strict=True enforcement in response_format."""

    def test_strict_true_valid_json_passes(self):
        from vmlx_engine.api.models import ResponseFormat, ResponseFormatJsonSchema
        from vmlx_engine.api.tool_calling import parse_json_output

        text = '{"name": "Alice"}'
        schema = ResponseFormatJsonSchema(
            name="person",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            strict=True,
        )
        rf = ResponseFormat(type="json_schema", json_schema=schema)
        _, parsed, is_valid, error = parse_json_output(text, rf)
        assert is_valid is True
        assert error is None or error == ""

    def test_strict_true_invalid_json_fails(self):
        from vmlx_engine.api.models import ResponseFormat, ResponseFormatJsonSchema
        from vmlx_engine.api.tool_calling import parse_json_output

        text = '{"name": 999}'
        schema = ResponseFormatJsonSchema(
            name="person",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            strict=True,
        )
        rf = ResponseFormat(type="json_schema", json_schema=schema)
        _, parsed, is_valid, error = parse_json_output(text, rf)
        assert is_valid is False

    def test_strict_false_is_default(self):
        from vmlx_engine.api.models import ResponseFormatJsonSchema

        schema = ResponseFormatJsonSchema(name="test", schema={"type": "object"})
        assert schema.strict is False

    def test_strict_flag_extraction_from_dict(self):
        """Verify strict flag can be read from raw dict format."""
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": True, "schema": {"type": "object"}},
        }
        strict = response_format.get("json_schema", {}).get("strict", False)
        assert strict is True

        response_format_no_strict = {
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "object"}},
        }
        strict2 = response_format_no_strict.get("json_schema", {}).get("strict", False)
        assert strict2 is False


# ─── max_tokens Fallback Chain ───────────────────────────────────────────────


class TestMaxTokensFallback:
    """Test the max_tokens fallback chain used throughout server.py."""

    def test_none_uses_default(self):
        """request.max_tokens=None should use server default."""
        _default_max_tokens = 32768
        request_max_tokens = None
        effective = request_max_tokens or _default_max_tokens
        assert effective == 32768

    def test_explicit_value_used(self):
        """Explicit max_tokens should be used as-is."""
        _default_max_tokens = 32768
        request_max_tokens = 1024
        effective = request_max_tokens or _default_max_tokens
        assert effective == 1024

    def test_zero_uses_default_falsy_trap(self):
        """max_tokens=0 is falsy, so it falls back to default.
        This documents current behavior — 0 means "use default", not "generate 0 tokens"."""
        _default_max_tokens = 32768
        request_max_tokens = 0
        effective = request_max_tokens or _default_max_tokens
        assert effective == 32768

    def test_responses_api_max_output_tokens_field(self):
        from vmlx_engine.api.models import ResponsesRequest

        req = ResponsesRequest(model="test", input="hello", max_output_tokens=256)
        assert req.max_output_tokens == 256

    def test_chat_completion_max_tokens_none_default(self):
        from vmlx_engine.api.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="t", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.max_tokens is None

    def test_sampling_params_default(self):
        from vmlx_engine.request import SamplingParams

        sp = SamplingParams()
        assert sp.max_tokens == 256


# ─── Model Config Registry Flags ────────────────────────────────────────────


class TestModelConfigRegistryFlags:
    """Test specific flags on model configs."""

    def _find_by_model_type(self, registry, model_type):
        for config in registry._configs:
            if model_type in config.model_types:
                return config
        return None

    def test_glm4_moe_flash_reasoning_parser(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "glm4_moe")
        assert config is not None
        assert config.reasoning_parser == "openai_gptoss"

    def test_glm4_moe_flash_think_in_template_false(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "glm4_moe")
        assert config is not None
        assert config.think_in_template is False

    def test_qwen3_think_in_template_true(self):
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        config = self._find_by_model_type(registry, "qwen3")
        assert config is not None
        assert config.think_in_template is True

    def test_preserve_native_tool_format_is_bool(self):
        """All configs with preserve_native_tool_format set should have a bool value."""
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        for config in registry._configs:
            if config.preserve_native_tool_format is not None:
                assert isinstance(config.preserve_native_tool_format, bool), (
                    f"{config.family_name}: preserve_native_tool_format should be bool"
                )

    def test_is_mllm_flag_on_vl_models(self):
        """Dedicated VL model configs should have is_mllm=True.
        Shared model_types (qwen3_5) use config.json vision_config instead."""
        from vmlx_engine.model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        # Only dedicated VL model_types
        config = self._find_by_model_type(registry, "qwen3_vl")
        if config is not None:
            assert config.is_mllm is True
        # Shared model_types must NOT have is_mllm=True
        config = self._find_by_model_type(registry, "qwen3_5")
        if config is not None:
            assert config.is_mllm is False


# ─── Audio Model Defaults ───────────────────────────────────────────────────


class TestAudioModelDefaults:
    """Test audio model request defaults."""

    def test_speech_request_defaults(self):
        from vmlx_engine.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(input="test")
        assert req.model == "kokoro"
        assert req.voice == "af_heart"
        assert req.speed == 1.0

    def test_speech_request_speed_range(self):
        from vmlx_engine.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(input="test", speed=0.25)
        assert req.speed == 0.25
        req2 = AudioSpeechRequest(input="test", speed=4.0)
        assert req2.speed == 4.0


# ─── Responses Input Conversion ──────────────────────────────────────────────


class TestResponsesInputConversion:
    """Test _responses_input_to_messages() conversion."""

    def test_string_input(self):
        from vmlx_engine.server import _responses_input_to_messages

        messages = _responses_input_to_messages("hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_string_input_with_instructions(self):
        from vmlx_engine.server import _responses_input_to_messages

        messages = _responses_input_to_messages("hello", instructions="Be helpful")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"

    def test_list_input_user_message(self):
        from vmlx_engine.server import _responses_input_to_messages

        input_data = [{"role": "user", "content": "What is 2+2?"}]
        messages = _responses_input_to_messages(input_data)
        assert len(messages) == 1
        assert messages[0]["content"] == "What is 2+2?"

    def test_function_call_output(self):
        from vmlx_engine.server import _responses_input_to_messages

        input_data = [
            {"role": "user", "content": "test"},
            {
                "type": "function_call",
                "name": "search",
                "call_id": "call_abc",
                "arguments": '{"q": "test"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "result text",
            },
        ]
        messages = _responses_input_to_messages(input_data)
        # Should have: user message, assistant with tool_call, tool result
        assert any(m["role"] == "user" for m in messages)
        assert any(m["role"] == "assistant" for m in messages)
        assert any(m["role"] == "tool" for m in messages)

    def test_zaya_vl_tool_history_uses_text_parts_not_tool_role(self):
        from vmlx_engine.server import (
            _coerce_zaya_vl_tool_history_for_template,
            _responses_input_to_messages,
        )

        messages = _responses_input_to_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "call_audit",
                    "name": "run_command",
                    "arguments": '{"command":"echo ok"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_audit",
                    "output": '{"stdout":"ok"}',
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Continue."}],
                },
            ],
            preserve_multimodal=True,
        )

        coerced = _coerce_zaya_vl_tool_history_for_template(messages)

        assert all(m["role"] != "tool" for m in coerced)
        assert len(coerced) == 1
        assert coerced[0]["role"] == "user"
        assert isinstance(coerced[0]["content"], list)
        assert "tool_calls" not in coerced[0]
        merged_text = coerced[0]["content"][0]["text"]
        assert "tool-call history" in merged_text
        assert "Previous assistant tool call:" in merged_text
        assert "<zyphra_tool_call>" in merged_text
        assert "<function=run_command>" in merged_text
        assert "Tool response: " in merged_text
        assert "<zyphra_tool_response>" not in merged_text
        assert '{"stdout":"ok"}' in merged_text
        assert "Continue." in merged_text

    def test_zaya_vl_tool_history_summarizes_stored_json_result(self):
        from vmlx_engine.server import _coerce_zaya_vl_tool_history_for_template

        coerced = _coerce_zaya_vl_tool_history_for_template(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_smoke_record_fact",
                    "name": "record_fact",
                    "content": '{"ok":true,"stored":"blue-cat"}',
                },
            ]
        )

        text = coerced[0]["content"][0]["text"]
        assert text.startswith("Tool response: ")
        assert "<zyphra_tool_response>" not in text
        assert "STORED blue-cat" in text
        assert '{"ok":true,"stored":"blue-cat"}' not in text

    def test_server_wires_zaya_vl_tool_history_coercion_into_chat_and_responses_paths(self):
        source = Path("vmlx_engine/server.py").read_text()
        assert "_should_coerce_zaya_vl_tool_history(request.model)" in source
        assert source.count("_coerce_zaya_vl_tool_history_for_template(messages)") >= 2

    def test_function_call_assistant_has_template_safe_empty_content(self):
        """Responses function_call history must render on strict templates.

        Mistral 4/Pixtral-style templates allow assistant turns that only carry
        tool_calls, but later evaluate `message['content'] | length`. OpenAI
        `content: null` therefore crashes with len(None). The internal chat
        history should keep the tool_calls and use content="".
        """
        from vmlx_engine.server import _responses_input_to_messages

        messages = _responses_input_to_messages([
            {
                "type": "function_call",
                "name": "list_directory",
                "call_id": "call_audit",
                "arguments": '{"path": "."}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_audit",
                "output": "README.md",
            },
        ])

        assistant = next(m for m in messages if m["role"] == "assistant")
        assert assistant["content"] == ""
        assert assistant["tool_calls"][0]["id"] == "call_audit"

    def test_leading_function_call_history_gets_user_anchor(self):
        """Function-call history without the original user turn still needs
        valid chat alternation for native templates such as DSV4 DSML."""
        from vmlx_engine.server import _responses_input_to_messages

        messages = _responses_input_to_messages([
            {
                "type": "function_call",
                "name": "list_directory",
                "call_id": "call_audit",
                "arguments": '{"path": "."}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_audit",
                "output": "README.md",
            },
            {
                "type": "message",
                "role": "user",
                "content": "Name one listed file.",
            },
        ])

        assert messages[0]["role"] == "user"
        assert "tool-call history" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["id"] == "call_audit"
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "user"

    def test_historical_tool_schema_can_be_rebuilt_from_function_call(self):
        from vmlx_engine.server import (
            _responses_input_to_messages,
            _synthesize_tools_from_message_tool_calls,
        )

        messages = _responses_input_to_messages([
            {
                "type": "function_call",
                "name": "list_directory",
                "call_id": "call_audit",
                "arguments": '{"path": "."}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_audit",
                "output": "README.md",
            },
        ])

        tools = _synthesize_tools_from_message_tool_calls(messages)

        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": (
                        "Previously available tool from conversation history."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

    def test_multimodal_preserved_for_mllm(self):
        from vmlx_engine.server import _responses_input_to_messages

        input_data = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                ],
            }
        ]
        messages = _responses_input_to_messages(input_data, preserve_multimodal=True)
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 2

    def test_multimodal_text_only_for_llm(self):
        from vmlx_engine.server import _responses_input_to_messages

        input_data = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                ],
            }
        ]
        messages = _responses_input_to_messages(input_data, preserve_multimodal=False)
        # For LLM, content should be extracted to text only
        assert isinstance(messages[0]["content"], str)


class TestFallbackToolPromptFormat:
    def test_dsv4_fallback_injects_dsml_not_generic_xml(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = "<｜begin▁of▁sentence｜><｜User｜>use tool<｜Assistant｜>"
        messages = [{"role": "user", "content": "use list_directory"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {"type": "object"},
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True},
        )

        assert "<｜DSML｜invoke" in rendered
        assert "<tool_call>" not in rendered
        assert "list_directory" in rendered

    def test_dsv4_fallback_injects_dsml_even_when_tool_name_present(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def __init__(self):
                self.last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<｜begin▁of▁sentence｜><｜User｜>use tool<｜Assistant｜>\n"
            "Tool: list_directory\nparameters: path\n"
            '<｜DSML｜invoke name="$TOOL_NAME">'
        )
        messages = [{"role": "user", "content": "use list_directory"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True},
        )

        assert "<｜DSML｜invoke" in rendered
        assert 'name="list_directory"' in rendered
        assert "- path (string, required)" in rendered
        assert "VALUE HERE" not in rendered
        assert "<tool_call>" not in rendered

    def test_minimax_fallback_injects_concrete_native_required_tool_call(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(str(m.get("content", "")) for m in messages)

        prompt = (
            "]~b]system\nTools available: record_fact\n"
            "]~b]user\nUse record_fact.\n]~b]ai\n<think>\n"
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "Use the record_fact tool exactly once. Its value argument "
                    "must be the literal string \"blue-cat\"."
                ),
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "description": "Record one exact fact.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            FakeTokenizer(),
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tool_choice": "required",
            },
            tool_parser_id="minimax",
        )

        assert "<minimax:tool_call>" in rendered
        assert '<invoke name="record_fact">' in rendered
        assert '<parameter name="value">blue-cat</parameter>' in rendered
        assert "</invoke>" in rendered
        assert "</minimax:tool_call>" in rendered
        assert "<function=record_fact>" not in rendered

    def test_minimax_fallback_preserves_slash_in_explicit_file_info_path(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        prompt = (
            "]~b]system\nTools available: file_info\n"
            "]~b]user\nUse file_info.\n]~b]ai\n<think>\n"
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "Call the built-in file_info tool exactly once with path "
                    "panel/package.json. After the tool result, reply exactly DONE."
                ),
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "description": "Return information for a filesystem path.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True},
            tool_parser_id="minimax",
        )

        assert "<minimax:tool_call>" in rendered
        assert '<invoke name="file_info">' in rendered
        assert (
            '<parameter name="path">panel/package.json</parameter>' in rendered
        )

    def test_dsv4_fallback_ignores_historical_dsml_when_checking_examples(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<｜begin▁of▁sentence｜><｜User｜>list files<｜Assistant｜>\n"
            '<｜DSML｜invoke name="list_directory">\n'
            '  <｜DSML｜parameter name="path" string="true">.</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "<｜User｜>{\"entries\":[\"README.md\"]}<｜Assistant｜>"
        )
        messages = [{"role": "user", "content": "use list_directory"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True},
        )

        assert tokenizer.calls >= 1
        assert 'name="list_directory"' in rendered
        assert "- path (string, required)" in rendered
        assert "VALUE HERE" not in rendered
        assert "Call them using DSML format" in rendered

    def test_qwen_native_template_keeps_its_tool_scaffold_for_auto_tools(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|im_start|>system\n# Tools\n<tools>\n"
            '{"type":"function","function":{"name":"list_directory","description":"List a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}\n'
            "</tools>\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<|im_end|>\n<|im_start|>user\nUse list_directory<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = [{"role": "user", "content": "Use list_directory for path '.'"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        )

        # The real Qwen template owns the JSON schema, native XML shape,
        # tool-result framing, and the optional pre-tool reasoning contract.
        # Replacing it with a second synthetic system message made Bonsai
        # deliberate over two competing contracts before one simple call.
        assert rendered == prompt
        assert tokenizer.last_kwargs is None

    def test_qwen_required_tool_choice_fallback_injects_hard_first_call_contract(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|im_start|>system\n# Tools\n<tools>\n"
            '{"type":"function","function":{"name":"grep_repo"}}\n'
            "</tools>\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<|im_end|>\n<|im_start|>user\nUse grep_repo<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = [{"role": "user", "content": "Use grep_repo for pattern cache."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "grep_repo",
                    "description": "Search source",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": tools,
                "tool_choice": "required",
            },
        )

        assert "tool_choice=required" in rendered
        assert "emit exactly one native tool call before any prose" in rendered
        assert "first assistant output for this turn must be one of the native tool calls" in rendered
        assert "Current turn API contract: tool_choice=required" in rendered
        assert "Historical tool results do not satisfy this current-turn requirement" in rendered
        assert "Do not answer in prose before the tool call" in rendered
        assert "<function=grep_repo>" in rendered
        assert "<parameter=pattern>" in rendered
        assert "tools" not in tokenizer.last_kwargs

    def test_step3p5_fallback_not_triggered_when_native_examples_present(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|system|>\n# Tools\n<tools>\n"
            '{"type":"function","function":{"name":"list_directory"}}\n'
            "</tools>\n"
            "<tool_call>\n<function=list_directory>\n"
            "<parameter=path>\n.\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<|assistant|>\n"
        )
        messages = [{"role": "user", "content": "Use list_directory"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="step3p5",
        )

        assert rendered == prompt
        assert tokenizer.calls == 0

    def test_step3p5_auto_without_tool_request_keeps_native_schema_only(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|im_start|>system\n<tools>\n"
            '{"type":"function","function":{"name":"list_directory","description":"List a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}\n'
            "</tools>\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>value_1</parameter>\n"
            "</function>\n</tool_call>\n"
            "<|im_end|>\n<|im_start|>user\nSay hello.\n"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        messages = [{"role": "user", "content": "Say hello."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="step3p5",
        )

        assert rendered == prompt
        assert tokenizer.calls == 0

    def test_step3p5_native_tool_result_continuation_keeps_prefix_stable(self):
        """A native Step transcript is sufficient for the answer continuation.

        Step's template shares Qwen's ChatML markers, but its explicit
        ``step3p5`` parser owns a separate tool grammar. Classifying it as Qwen
        injected a new early system instruction only after a tool result,
        rewriting the SSD prefix before the tool schema.
        """
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|im_start|>system\n<tools>\n"
            '{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}\n'
            "</tools>\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>value_1</parameter>\n"
            "</function>\n</tool_call>\n<|im_end|>\n"
            "<|im_start|>user\nInspect the image.<|im_end|>\n"
            "<|im_start|>assistant\n<think>checked</think>\n"
            "<tool_call>\n<function=read_file>\n<parameter=path>README.md"
            "</parameter>\n</function>\n</tool_call><|im_end|>\n"
            "<|im_start|>tool_response\n<tool_response>contents"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        messages = [
            {"role": "user", "content": "Inspect the image."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "contents"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="step3p5",
        )

        assert rendered == prompt
        assert "Native tool-result continuation" not in rendered
        assert tokenizer.calls == 0

    def test_step3p5_explicit_tool_keeps_valid_native_schema_prefix(self):
        """Naming a tool must not swap Step's opening system scaffold."""
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|im_start|>system\n<tools>\n"
            '{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}\n'
            "</tools>\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>value_1</parameter>\n"
            "</function>\n</tool_call>\n<|im_end|>\n"
            "<|im_start|>user\nCall read_file with path README.md."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        messages = [
            {"role": "user", "content": "Call read_file with path README.md."}
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="step3p5",
        )

        assert rendered == prompt
        assert tokenizer.calls == 0

    def test_step3p5_fallback_injects_native_xml_tool_example(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|system|>\n"
            "# Tools\nThe template dropped the native schema.\n"
            "<tool_call>\n<function=example_function_name>\n</function>\n</tool_call>\n"
            "<|assistant|>\n"
        )
        messages = [{"role": "user", "content": "Use list_directory"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="step3p5",
        )

        assert "<tool_call>" in rendered
        assert "<function=list_directory>" in rendered
        assert "<parameter=path>" in rendered
        assert "<tool_call>{\"name\"" not in rendered

    def test_zaya_fallback_injects_concrete_native_tool_example(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|system|>\n# Tools\n<tools>\n"
            '{"type":"function","function":{"name":"list_directory"}}\n'
            "</tools>\n"
            "<zyphra_tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</zyphra_tool_call>\n"
            "<|user|>\nUse list_directory\n<|assistant|>\n"
        )
        messages = [{"role": "user", "content": "Use list_directory for path '.'"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        )

        assert "<zyphra_tool_call>" in rendered
        assert "<function=list_directory>" in rendered
        assert "<parameter=path>" in rendered
        assert "<tool_call>" not in rendered
        assert "fake directory listing" in rendered
        assert "tools" not in tokenizer.last_kwargs

    def test_zaya_fallback_examples_do_not_teach_placeholder_path_values(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "<|user|>\nUse list_directory\n<|assistant|>\n",
            [{"role": "user", "content": "Use list_directory for path '.'"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "VALUE HERE" not in rendered
        assert "<parameter=path>\n.\n</parameter>" in rendered

    def test_zaya_fallback_examples_do_not_teach_literal_example_for_request_values(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "description": "Record one exact fact for a smoke test.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The exact value to record.",
                            }
                        },
                        "required": ["value"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: Use record_fact with value blue-cat\nassistant:",
            [{"role": "user", "content": "Use record_fact with value blue-cat"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=record_fact>" in rendered
        assert "value (string, required): The exact value to record." not in rendered
        assert "Fill fields from the user's request exactly." in rendered
        assert "put only `blue-cat` in `value`" in rendered
        assert "<parameter=value>\nblue-cat\n</parameter>" in rendered
        assert "<parameter=value>\nexample\n</parameter>" not in rendered
        assert "<parameter=value>\nREQUEST_VALUE\n</parameter>" not in rendered

    def test_zaya_fallback_extracts_value_argument_literal_string(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ]
        request = (
            'Use the record_fact tool exactly once. Its value argument must be '
            'the literal string "blue-cat"; preserve every character.'
        )

        rendered = check_and_inject_fallback_tools(
            "user: " + request + "\nassistant:",
            [{"role": "user", "content": request}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<parameter=value>\nblue-cat\n</parameter>" in rendered
        assert "<parameter=value>\nargument\n</parameter>" not in rendered

    def test_zaya_fallback_does_not_teach_unavailable_list_directory_tool(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_fact",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: Use record_fact with value blue-cat\nassistant:",
            [{"role": "user", "content": "Use record_fact with value blue-cat"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "list the current directory" not in rendered
        assert "set path to" not in rendered
        assert "list_directory" not in rendered

    def test_zaya_fallback_injects_native_example_for_each_tool(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|system|>\n# Tools\n<tools>\n"
            '{"type":"function","function":{"name":"list_directory"}}\n'
            '{"type":"function","function":{"name":"write_file"}}\n'
            "</tools>\n"
            "<zyphra_tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</zyphra_tool_call>\n"
            "<|user|>\nUse list_directory then write_file\n<|assistant|>\n"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "Use list_directory then write_file"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=list_directory>" in rendered
        assert "<function=write_file>" in rendered
        assert "<parameter=path>" in rendered
        assert "content (string, required)" not in rendered
        assert "write_file fields: path, content" in rendered
        assert "<parameter=content>" not in rendered
        assert rendered.count("<zyphra_tool_call>") >= 2

    def test_zaya_fallback_scopes_examples_to_requested_tool_name(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                rendered = []
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict)
                        )
                    rendered.append(content)
                return "\n".join(rendered)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ]

        rendered = check_and_inject_fallback_tools(
            "user: Use run_command with command pwd\nassistant:",
            [{"role": "user", "content": "Use run_command with command pwd"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "run_command fields: command" in rendered
        assert "<function=run_command>" in rendered
        assert "write_file fields" not in rendered
        assert "<function=write_file>" not in rendered
        assert "create_directory fields" not in rendered
        assert "<function=create_directory>" not in rendered

    def test_zaya_run_command_prompt_binds_exact_live_command(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: create file\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to create a file named "
                        "real_ui_tool_probe_1.txt in the configured working directory. "
                        "Write the text REAL_UI_LIVE_TOOL_ONE into that file."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "For this request, run_command.command must be exactly:" in rendered
        assert (
            "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt"
            in rendered
        )
        assert "Do not use REAL_UI_LIVE_TOOL_ONE itself as a shell command" in rendered
        assert "<function=run_command>" in rendered
        assert "<parameter=command>" in rendered

    def test_zaya_run_command_prompt_rejects_copying_first_probe_to_second(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: create second file\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to read "
                        "real_ui_tool_probe_1.txt and create "
                        "real_ui_tool_probe_2.txt in the same working directory. "
                        "Write REAL_UI_LIVE_TOOL_TWO into the second file."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "For this request, run_command.command must be exactly:" in rendered
        assert (
            "cat real_ui_tool_probe_1.txt >/dev/null && "
            "printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt"
        ) in rendered
        assert (
            "Do not copy real_ui_tool_probe_1.txt into real_ui_tool_probe_2.txt"
            in rendered
        )
        assert "Do not use REAL_UI_LIVE_TOOL_TWO itself as a shell command" in rendered

    def test_zaya_file_tool_prompt_binds_exact_path_and_content_from_request(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: write file\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the write_file tool exactly once with path "
                        "real_ui_tool_probe_1.txt and content "
                        "REAL_UI_LIVE_TOOL_ONE."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=write_file>" in rendered
        assert "<parameter=path>\nreal_ui_tool_probe_1.txt\n</parameter>" in rendered
        assert "<parameter=content>\nREAL_UI_LIVE_TOOL_ONE\n</parameter>" in rendered
        assert "<parameter=path>\n.\n</parameter>" not in rendered
        assert "list the current directory" not in rendered

    def test_zaya_fallback_preserves_native_tool_scaffold_with_concrete_examples(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                system = ""
                body = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system = str(msg.get("content", ""))
                    else:
                        body.append(str(msg.get("content", "")))
                prompt = "<|im_start|>system\n" + system
                if kwargs.get("tools"):
                    prompt += (
                        "\n\n# Tools\n<tools>\n"
                        "<function><name>write_file</name></function>\n"
                        "</tools>\n"
                        "<IMPORTANT>native zaya tool rules</IMPORTANT>\n"
                        "<zyphra_tool_call>\n"
                        "<function=example_function_name>\n"
                        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                        "</function>\n</zyphra_tool_call>"
                    )
                return prompt + "<|im_end|>\n<|im_start|>user\n" + "\n".join(body)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]
        tokenizer = FakeTokenizer()

        rendered = check_and_inject_fallback_tools(
            "<|im_start|>system\n# Tools\n<tools>\n"
            "<function><name>write_file</name></function>\n"
            "</tools>\n<zyphra_tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</zyphra_tool_call><|im_end|>\n"
            "<|im_start|>user\nUse write_file\n<|im_start|>assistant\n",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the write_file tool exactly once with path "
                        "real_ui_tool_probe_1.txt and content "
                        "REAL_UI_LIVE_TOOL_ONE."
                    ),
                }
            ],
            tools,
            tokenizer,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": tools,
                "enable_thinking": False,
            },
            tool_parser_id="zaya_xml",
        )

        assert tokenizer.last_kwargs["tools"] == tools
        assert "<tools>" in rendered
        assert "<IMPORTANT>native zaya tool rules</IMPORTANT>" in rendered
        assert "<function=write_file>" in rendered
        assert "<parameter=path>\nreal_ui_tool_probe_1.txt\n</parameter>" in rendered
        assert "<parameter=content>\nREAL_UI_LIVE_TOOL_ONE\n</parameter>" in rendered

    def test_zaya_vl_processor_template_receives_concrete_tool_fallback(self):
        from vmlx_engine.engine.batched import BatchedEngine

        class FakeProcessor:
            def apply_chat_template(self, messages, **kwargs):
                system = ""
                body = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system = str(msg.get("content", ""))
                    else:
                        body.append(str(msg.get("content", "")))
                prompt = "<|im_start|>system\n" + system
                if kwargs.get("tools"):
                    prompt += (
                        "\n\n# Tools\n<tools>\n"
                        "<function><name>write_file</name></function>\n"
                        "</tools>\n"
                        "<IMPORTANT>native zaya vl tool rules</IMPORTANT>\n"
                        "<zyphra_tool_call>\n"
                        "<function=example_function_name>\n"
                        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                        "</function>\n</zyphra_tool_call>"
                    )
                return prompt + "<|im_end|>\n<|im_start|>user\n" + "\n".join(body)

        class FakeModel:
            config = {"model_type": "zaya1_vl"}

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = FakeProcessor()
        engine._model = FakeModel()
        engine._model_name = "ZAYA1-VL-8B-JANGTQ4"
        engine._model_tool_parser_name = lambda: "zaya_xml"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]

        rendered = engine._apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Use the write_file tool exactly once with path "
                        "real_ui_tool_probe_1.txt and content "
                        "REAL_UI_LIVE_TOOL_ONE."
                    ),
                }
            ],
            tools=tools,
            enable_thinking=False,
        )

        assert "<tools>" in rendered
        assert "<IMPORTANT>native zaya vl tool rules</IMPORTANT>" in rendered
        assert "<function=write_file>" in rendered
        assert "<parameter=path>\nreal_ui_tool_probe_1.txt\n</parameter>" in rendered
        assert "<parameter=content>\nREAL_UI_LIVE_TOOL_ONE\n</parameter>" in rendered

    def test_zaya_vl_processor_retries_when_template_keeps_only_generic_tool_scaffold(self):
        from vmlx_engine.engine.batched import BatchedEngine

        class GenericOnlyProcessor:
            def __init__(self):
                self.calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                body = []
                for msg in messages:
                    if msg.get("role") == "system":
                        # ZAYA-VL can silently ignore system messages; keep only
                        # the native generic scaffold from the template.
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        body.append(
                            "\n".join(
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict)
                            )
                        )
                    else:
                        body.append(str(content))
                return (
                    "<tools>\n"
                    "<function><name>run_command</name></function>\n"
                    "</tools>\n"
                    "<zyphra_tool_call>\n"
                    "<function=example_function_name>\n"
                    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                    "</function>\n"
                    "</zyphra_tool_call>\n"
                    + "\n".join(body)
                )

        class FakeModel:
            config = {"model_type": "zaya1_vl"}

        processor = GenericOnlyProcessor()
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = processor
        engine._model = FakeModel()
        engine._model_name = "ZAYA1-VL-8B-JANGTQ4"
        engine._model_tool_parser_name = lambda: "zaya_xml"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = engine._apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to create a file "
                        "named real_ui_tool_probe_1.txt. Write the text "
                        "REAL_UI_LIVE_TOOL_ONE into that file."
                    ),
                }
            ],
            tools=tools,
            enable_thinking=False,
        )

        assert processor.calls >= 2
        assert "<function=run_command>" in rendered
        assert "<parameter=command>" in rendered
        assert "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt" in rendered

    def test_zaya_fallback_accepts_flat_responses_function_tool_shape(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        rendered = check_and_inject_fallback_tools(
            "<tools>\n"
            "<function><name>run_command</name></function>\n"
            "</tools>\n"
            "<zyphra_tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n"
            "</zyphra_tool_call>\n"
            "user: Use run_command\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to create a file "
                        "named real_ui_tool_probe_1.txt. Write the text "
                        "REAL_UI_LIVE_TOOL_ONE into that file."
                    ),
                }
            ],
            [
                {
                    "type": "function",
                    "name": "run_command",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True},
            tool_parser_id="zaya_xml",
        )

        assert "<function=run_command>" in rendered
        assert "<parameter=command>" in rendered
        assert "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt" in rendered

    def test_zaya_vl_processor_without_native_scaffold_gets_full_tool_contract(self):
        from vmlx_engine.engine.batched import BatchedEngine

        class ScaffoldlessZayaVlProcessor:
            def apply_chat_template(self, messages, **kwargs):
                rendered = []
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    if isinstance(content, list):
                        text = "\n".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict)
                        )
                    else:
                        text = str(content or "")
                    rendered.append(f"<|im_start|>{role}\n{text}<|im_end|>")
                if kwargs.get("add_generation_prompt"):
                    rendered.append("<|im_start|>assistant\n")
                return "\n".join(rendered)

        class FakeModel:
            config = {"model_type": "zaya1_vl"}

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._processor = ScaffoldlessZayaVlProcessor()
        engine._model = FakeModel()
        engine._model_name = "ZAYA1-VL-8B-JANGTQ4"
        engine._model_tool_parser_name = lambda: "zaya_xml"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Target path"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]

        rendered = engine._apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Use the write_file tool exactly once with path "
                        "real_ui_tool_probe_1.txt and content "
                        "REAL_UI_LIVE_TOOL_ONE."
                    ),
                }
            ],
            tools=tools,
            enable_thinking=False,
        )

        assert "# Tools" in rendered
        assert "<tools>" in rendered
        assert "<function>" in rendered
        assert "<name>write_file</name>" in rendered
        assert "<IMPORTANT>" in rendered
        assert "Function calls MUST follow the specified format" in rendered
        assert "<function=write_file>" in rendered
        assert "<parameter=path>\nreal_ui_tool_probe_1.txt\n</parameter>" in rendered
        assert "<parameter=content>\nREAL_UI_LIVE_TOOL_ONE\n</parameter>" in rendered

    def test_zaya_fallback_skips_concrete_examples_when_request_names_no_tool(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "rerendered <function=write_file>"

        prompt = (
            "<|im_start|>system\n# Tools\n<tools>\n"
            "<function><name>write_file</name></function>\n"
            "</tools>\n"
            "<zyphra_tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</zyphra_tool_call><|im_end|>\n"
            "<|im_start|>user\nWhat is the dominant color of the attached image?"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "What is the dominant color of the attached image?"}],
            tools,
            FakeTokenizer(),
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": tools,
                "enable_thinking": False,
            },
            tool_parser_id="zaya_xml",
        )

        assert FakeTokenizer.calls == 0
        assert rendered == prompt

    def test_zaya_fallback_ignores_stale_prior_tool_requests_for_media_turn(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                return "rerendered <function=run_command>"

        prompt = (
            "<|im_start|>system\n# Tools\n<tools>\n"
            "<function><name>run_command</name></function>\n"
            "</tools>\n"
            "<zyphra_tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</zyphra_tool_call><|im_end|>\n"
            "<|im_start|>user\nWhat is the dominant color of the attached image?"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            [
                {
                    "role": "user",
                    "content": "Use the run_command tool exactly once to create a file.",
                },
                {"role": "assistant", "content": "done"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the dominant color of the attached image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                },
            ],
            tools,
            FakeTokenizer(),
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": tools,
                "enable_thinking": False,
            },
            tool_parser_id="zaya_xml",
        )

        assert FakeTokenizer.calls == 0
        assert rendered == prompt

    def test_zaya_read_file_prompt_binds_path_from_for_phrase(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: read file\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the read_file tool exactly once for "
                        "real_ui_tool_probe_1.txt."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=read_file>" in rendered
        assert "<parameter=path>\nreal_ui_tool_probe_1.txt\n</parameter>" in rendered
        assert "<parameter=path>\n.\n</parameter>" not in rendered
        assert "list the current directory" not in rendered

    def test_zaya_fallback_scopes_examples_to_multiple_requested_tool_names(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                rendered = []
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        rendered.append(
                            "\n".join(
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict)
                            )
                        )
                    else:
                        rendered.append(str(content))
                return "\n".join(rendered)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
        ]

        rendered = check_and_inject_fallback_tools(
            "user: Use run_command then write_file\nassistant:",
            [{"role": "user", "content": "Use run_command then write_file"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=run_command>" in rendered
        assert "<function=write_file>" in rendered
        assert "<function=create_directory>" not in rendered

    def test_zaya_run_command_example_derives_create_file_command_from_request(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: create file\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to create a file named "
                        "real_ui_tool_probe_1.txt in the configured working directory. "
                        "Write the text REAL_UI_LIVE_TOOL_ONE into that file."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=run_command>" in rendered
        assert "<parameter=command>" in rendered
        assert "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt" in rendered

    def test_zaya_run_command_example_derives_read_then_create_command_from_request(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: read then create\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to read "
                        "real_ui_tool_probe_1.txt and create real_ui_tool_probe_2.txt "
                        "in the same working directory. Write REAL_UI_LIVE_TOOL_TWO "
                        "into the second file."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=run_command>" in rendered
        assert "<parameter=command>" in rendered
        assert (
            "cat real_ui_tool_probe_1.txt >/dev/null && "
            "printf %s REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt"
        ) in rendered

    def test_zaya_fallback_uses_compact_examples_not_verbose_schema_prose(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m.get("content", "") for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file to disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "1-based output path.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Text to write.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: Use write_file with path out.txt and content ok\nassistant:",
            [{"role": "user", "content": "Use write_file with path out.txt and content ok"}],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<zyphra_tool_call>" in rendered
        assert "<function=write_file>" in rendered
        assert "Tool: write_file" not in rendered
        assert "description:" not in rendered
        assert "parameters:" not in rendered
        assert "(string, required)" not in rendered
        assert "1-based output path" not in rendered

    def test_zaya_fallback_ignores_historical_tool_calls_when_checking_examples(self):
        """Prior assistant tool-call history is not a concrete instruction exemplar.

        Live ZAYA chained-tool repro: after list_directory then write_file,
        the rendered prompt contained ``<function=list_directory>`` and
        ``<function=write_file>`` in assistant history. The fallback detector
        incorrectly counted those history blocks as native examples, stopped
        injecting the stronger ZAYA tool instructions, and the model repeated
        write_file instead of ending with final content.
        """
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None
            calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<bos><|im_start|>system\n"
            "# Tools\n\n<tools>\n"
            "<function>\n<name>list_directory</name>\n</function>\n"
            "<function>\n<name>write_file</name>\n</function>\n"
            "</tools>\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
            "<zyphra_tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "</function>\n</zyphra_tool_call><|im_end|>\n"
            "<|im_start|>user\nUse list_directory then write_file.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n</think>\n\n"
            "<zyphra_tool_call>\n<function=list_directory>\n"
            "<parameter=path>\n.\n</parameter>\n</function>\n</zyphra_tool_call>\n"
            "<|im_end|>\n<|im_start|>user\n"
            "<zyphra_tool_response>{\"entries\":[\"README.md\"]}</zyphra_tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n"
            "<zyphra_tool_call>\n<function=write_file>\n"
            "<parameter=path>\nvmlx_chained_tool_probe.txt\n</parameter>\n"
            "<parameter=content>\nok\n</parameter>\n"
            "</function>\n</zyphra_tool_call>\n<|im_end|>\n"
            "<|im_start|>user\n"
            "<zyphra_tool_response>{\"path\":\"vmlx_chained_tool_probe.txt\",\"bytes\":2}</zyphra_tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        messages = [{"role": "user", "content": "Use list_directory then write_file."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert tokenizer.calls >= 1
        assert "fake directory listing" in rendered
        assert "<function=list_directory>" in rendered
        assert "<function=write_file>" in rendered

    def test_zaya_parser_id_forces_native_tool_example_for_plain_template(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = "user: Use list_directory for path '.'\nassistant: "
        messages = [{"role": "user", "content": "Use list_directory for path '.'"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<zyphra_tool_call>" in rendered
        assert "<function=list_directory>" in rendered
        assert "<parameter=path>" in rendered
        assert "<tool_call>" not in rendered
        assert "tools" not in tokenizer.last_kwargs

    def test_lfm2_parser_id_forces_python_call_list_tool_example(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = "user: Use run_command to create a file\nassistant: "
        messages = [
            {
                "role": "user",
                "content": (
                    "Use run_command exactly once to create a file named "
                    "real_ui_tool_probe_1.txt. Write the text "
                    "REAL_UI_LIVE_TOOL_ONE into that file."
                ),
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ]

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="lfm2",
        )

        assert "<|tool_call_start|>" in rendered
        assert "run_command(command=" in rendered
        assert "For this request, run_command.command must be exactly:" in rendered
        assert (
            "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt"
            in rendered
        )
        assert "real_ui_tool_probe_1.txt" in rendered
        assert "read_file(" not in rendered
        assert "write_file(" not in rendered
        assert "<|tool_call_end|>" in rendered
        assert "<tool_call>" not in rendered
        # Liquid's native template renders the actual JSON schema from this
        # kwarg. Keep only the explicitly requested schema alongside the
        # concrete Python-call-list example.
        assert tokenizer.last_kwargs["tools"] == [tools[0]]

    def test_lfm2_run_command_prompt_warns_against_bare_payload_command(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(str(m.get("content", "")) for m in messages)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "user: create file\nassistant:",
            [
                {
                    "role": "user",
                    "content": (
                        "Use the run_command tool exactly once to create a file named "
                        "real_ui_tool_probe_1.txt in the configured working directory. "
                        "Write the text REAL_UI_LIVE_TOOL_ONE into that file."
                    ),
                }
            ],
            tools,
            FakeTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="lfm2",
        )

        assert "Do not use REAL_UI_LIVE_TOOL_ONE itself as a shell command" in rendered

    def test_zaya_fallback_survives_templates_that_ignore_system_messages(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class ZayaLikeTokenizer:
            call_count = 0
            last_messages = None

            def apply_chat_template(self, messages, **kwargs):
                self.call_count += 1
                self.last_messages = messages
                rendered = []
                for msg in messages:
                    if msg.get("role") == "user":
                        rendered.append(f"user: {msg.get('content', '')}")
                    elif msg.get("role") == "assistant":
                        rendered.append(f"assistant: {msg.get('content', '')}")
                    elif msg.get("role") == "tool":
                        rendered.append(f"tool: {msg.get('content', '')}")
                rendered.append("assistant: ")
                return "\n".join(rendered)

        prompt = "user: Use list_directory for path '.'\nassistant: "
        messages = [{"role": "user", "content": "Use list_directory for path '.'"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        tokenizer = ZayaLikeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            messages,
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert tokenizer.call_count == 2
        assert tokenizer.last_messages[0]["role"] == "user"
        assert "<zyphra_tool_call>" in rendered
        assert "<function=list_directory>" in rendered
        assert "<parameter=path>" in rendered
        assert "Use list_directory for path '.'" in rendered

    def test_zaya_fallback_uses_list_text_content_for_vl_templates(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class ZayaVlLikeTokenizer:
            call_count = 0
            last_messages = None

            def apply_chat_template(self, messages, **kwargs):
                self.call_count += 1
                self.last_messages = messages
                rendered = []
                for msg in messages:
                    content = msg.get("content")
                    if not isinstance(content, list):
                        continue
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            rendered.append(
                                f"<|im_start|>{msg.get('role')}\n{item.get('text', '')}<|im_end|>"
                            )
                if kwargs.get("add_generation_prompt"):
                    rendered.append("<|im_start|>assistant\n")
                return "\n".join(rendered)

        tokenizer = ZayaVlLikeTokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            "<|im_start|>user\nUse run_command<|im_end|>\n<|im_start|>assistant\n",
            [{"role": "user", "content": "Use run_command with command echo ok"}],
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert tokenizer.call_count == 3
        assert tokenizer.last_messages[0]["role"] == "user"
        assert isinstance(tokenizer.last_messages[0]["content"], list)
        assert tokenizer.last_messages[0]["content"][0]["type"] == "text"
        assert "<|im_start|>user" in rendered
        assert "<zyphra_tool_call>" in rendered
        assert "<function=run_command>" in rendered
        assert rendered.startswith("<|im_start|>user")

    def test_zaya_fallback_skips_when_no_tool_is_explicitly_requested(self):
        from unittest.mock import MagicMock

        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        tokenizer = MagicMock()
        prompt = (
            "<tools>{\"name\":\"list_directory\"}</tools>\n"
            "<zyphra_tool_call>\n"
            "<function=list_directory>\n"
            "<parameter=path>VALUE HERE</parameter>\n"
            "</function>\n"
            "</zyphra_tool_call>"
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ]

        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": "Show me what is in the folder"}],
            tools,
            tokenizer,
            {"tokenize": False, "tools": tools},
        )

        assert rendered == prompt
        tokenizer.apply_chat_template.assert_not_called()

    def test_zaya_explicit_tool_refreshes_unbound_broad_catalog_example(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                rendered = "\n".join(str(m.get("content", "")) for m in messages)
                template_tools = kwargs.get("tools") or []
                if template_tools:
                    rendered += "\n<tools>\n" + "\n".join(
                        f"<function><name>{tool['function']['name']}</name></function>"
                        for tool in template_tools
                    ) + "\n</tools>"
                return rendered

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "description": "Inspect one file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_applescript",
                    "description": "Run AppleScript.",
                    "parameters": {
                        "type": "object",
                        "properties": {"script": {"type": "string"}},
                        "required": ["script"],
                    },
                },
            },
        ]
        prompt = (
            "<tools><function><name>file_info</name></function>"
            "<function><name>run_applescript</name></function></tools>\n"
            "<zyphra_tool_call><function=file_info><parameter=path>VALUE_HERE"
            "</parameter></function></zyphra_tool_call>\n"
            "<zyphra_tool_call><function=run_applescript><parameter=script>"
            "choose folder</parameter></function></zyphra_tool_call>"
        )
        request = (
            "Call the built-in file_info tool exactly once with path "
            "panel/package.json."
        )

        tokenizer = FakeTokenizer()
        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": request}],
            tools,
            tokenizer,
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="zaya_xml",
        )

        assert "<function=file_info>" in rendered
        assert "<parameter=path>\npanel/package.json\n</parameter>" in rendered
        assert "<function=run_applescript>" not in rendered
        assert [
            tool["function"]["name"] for tool in tokenizer.last_kwargs["tools"]
        ] == ["file_info"]

    def test_dsml_parser_rejects_schema_gated_malformed_old_dsv4_tool_call(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<｜DSML｜tool_call_type type="list_directory","" '
            '"attributes":".} 100%}\n\n```json\n<｜DSML｜tool_name type="false"}'
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        result = DSMLToolParser(None).extract_tool_calls(text, request=req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content is None

    def test_dsml_parser_rejects_partial_canonical_invoke(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<｜DSML｜invoke name="list_directory">\n'
            '  <｜DSML｜parameter name="path" string="true">.</｜DSML｜parameter>\n'
            "</"
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        result = DSMLToolParser(None).extract_tool_calls(text, request=req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content is None

    def test_dsml_parser_rejects_dsv4_live_degraded_dsml_params(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<｜DSML｜tool_calls>\n'
            '<｜DSML｜invoke name="list_directory">\n'
            '<｜DSML｜parameter name="path">.</｜DSML｜parameter>\n'
            '</｜DSML｜inv>\n'
            '<｜DSML｜invoke name="write_file">\n'
            '<｜DSML｜parameter name="path">x.txt</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="content">ok</｜DSML｜parameter>\n'
            '</｜DSML｜inv>\n'
            '</｜DSML｜tool_calls>'
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                },
            ]
        }

        result = DSMLToolParser(None).extract_tool_calls(text, request=req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content is None

    def test_dsml_parser_rejects_canonical_attr_residue_without_repair(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<｜DSML｜tool_ctools>\n'
            '<｜DSML｜inv>\n'
            '<｜DSML｜name>write_file</｜DSML｜>\n'
            '<｜DSML｜parameter name="path" string="true">landing-p/proof.html</｜DSML｜>\n'
            '<｜DSML｜parameter name="content" string="true"><html><body>dsv4-default-cache-tool-ok</body></html></｜DSML｜>\n'
            '</｜DSML｜inv>\n'
            '</｜DSML｜tool_calls>'
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ]
        }

        result = DSMLToolParser(None).extract_tool_calls(text, request=req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content is None

    def test_dsml_parser_rejects_schema_placeholder_as_tool_argument(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<｜DSML｜tool_calls>\n'
            '<｜DSML｜invoke name="run_command">\n'
            '<｜DSML｜parameter name="command" string="true"> string='
            '</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n'
            '</｜DSML｜tool_calls>'
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ]
        }
        parser = DSMLToolParser(None)

        result = parser.extract_tool_calls(text, request=req)
        streaming = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=req,
        )
        parser._stream_stop_request = req

        assert not result.tools_called
        assert result.tool_calls == []
        assert streaming is None
        assert parser.stream_tool_calls_complete(text) is False

    def test_dsml_parser_honors_production_two_arg_encoder_and_restores_eos(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest
        from vmlx_engine.loaders import dsv4_chat_encoder
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        calls = []

        class ProductionShapeEncoder:
            eos_token = "<DSV4_EOS>"

            @staticmethod
            def parse_message_from_completion_text(text, thinking_mode):
                calls.append((text, thinking_mode))
                assert text.endswith(ProductionShapeEncoder.eos_token)
                if thinking_mode == "thinking":
                    assert "</think>" in text
                else:
                    assert "</think>" not in text
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need the working directory.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"command": "pwd"}',
                            },
                        }
                    ],
                }

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            lambda model_path=None: ProductionShapeEncoder,
        )
        req = {
            "model_path": "/models/dsv4",
            "enable_thinking": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
        }

        canonical_fragment = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="run_command">\n'
            '<｜DSML｜parameter name="command" string="true">'
            "pwd</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        parser = DSMLToolParser(None)
        result = parser._try_encoding_dsv4_parse(canonical_fragment, request=req)

        assert result is not None
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "run_command"
        assert result.tool_calls[0]["arguments"] == '{"command": "pwd"}'
        assert calls == [
            (
                "\n\n" + canonical_fragment + "<DSV4_EOS>",
                "chat",
            )
        ]

        thinking_fragment = (
            "Need the working directory.</think>\n\n" + canonical_fragment
        )
        thinking_result = parser._try_encoding_dsv4_parse(
            thinking_fragment,
            request=req,
        )
        assert thinking_result is not None
        assert calls[-1] == (
            thinking_fragment + "<DSV4_EOS>",
            "thinking",
        )

        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        monkeypatch.setattr(server, "_model_path", "/models/dsv4")
        api_request = ResponsesRequest(
            model="dsv4",
            input="Call run_command with pwd.",
            enable_thinking=True,
            tools=[
                {
                    "type": "function",
                    "name": "run_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
        )

        cleaned, api_calls = server._parse_tool_calls_with_parser(
            canonical_fragment,
            api_request,
        )

        assert cleaned == ""
        assert api_calls
        assert api_calls[0].function.name == "run_command"
        assert api_calls[0].function.arguments == '{"command": "pwd"}'
        # Public extraction first validates the strict completed grammar, then
        # requires the bundle encoder to agree on the same call.
        assert len(calls) == 3
        assert calls[-1] == (
            "\n\n" + canonical_fragment + "<DSV4_EOS>",
            "chat",
        )

    def test_dsml_parser_does_not_hide_internal_canonical_type_error(
        self, monkeypatch
    ):

        from vmlx_engine.loaders import dsv4_chat_encoder
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        calls = []

        class BrokenEncoder:
            eos_token = "<DSV4_EOS>"

            @staticmethod
            def parse_message_from_completion_text(text, thinking_mode):
                calls.append((text, thinking_mode))
                raise TypeError("internal canonical parser defect")

        monkeypatch.setattr(
            dsv4_chat_encoder,
            "_load_encoding_dsv4_module",
            lambda model_path=None: BrokenEncoder,
        )

        with pytest.raises(TypeError, match="internal canonical parser defect"):
            DSMLToolParser(None)._try_encoding_dsv4_parse(
                "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
                request={"enable_thinking": True},
            )

        assert len(calls) == 1
        assert calls[0][1] == "chat"

    def test_dsml_parser_rejects_partial_invoke_with_malformed_value_attr(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<｜DSML｜invoke name="list_directory">\n'
            '  <｜DSML｜parameter name="path" string value"."></'
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        result = DSMLToolParser(None).extract_tool_calls(text, request=req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content is None

    def test_dsml_parser_rejects_htmlish_invoke_degradation(self):
        from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser

        text = (
            '<invoke_list_directory><br />\n'
            '<param name="path".">.</br />\n'
            '</inv'
        )
        req = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        result = DSMLToolParser(None).extract_tool_calls(text, request=req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content is None

    def test_generic_parser_handles_laguna_arg_key_value_tool_call(self):
        from vmlx_engine.api.tool_calling import parse_tool_calls

        text = (
            "<tool_call>list_directory\n"
            "<arg_key>path</arg_key>\n"
            "<arg_value>.</arg_value>\n"
            "</tool_call>"
        )

        cleaned, calls = parse_tool_calls(text)

        assert cleaned == ""
        assert calls
        assert calls[0].function.name == "list_directory"
        assert '"path": "."' in calls[0].function.arguments

    def test_generic_parser_handles_dsv4_use_json_tool_call(self):
        """Live DSV4 JANGTQ2 can emit a compact <use_tool JSON> call.

        The server filters parsed calls to request.tools, so the generic parser
        can safely recognize this syntax without turning arbitrary prose into a
        callable function.
        """
        from vmlx_engine.api.tool_calling import parse_tool_calls

        text = '<use_list_directory\n\n{\n  "path": "."  \n}'

        cleaned, calls = parse_tool_calls(text)

        assert cleaned == ""
        assert calls
        assert calls[0].function.name == "list_directory"
        assert '"path": "."' in calls[0].function.arguments

    def test_server_tool_parser_filters_unavailable_tool_names(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", None)
        req = ResponsesRequest(
            model="m",
            input="x",
            tools=[
                {
                    "type": "function",
                    "name": "list_directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        )

        cleaned, calls = server._parse_tool_calls_with_parser(
            '<tool_call>{"name":"README.md","arguments":{}}</tool_call>',
            req,
        )

        assert calls is None
        assert "README.md" in cleaned

    def test_server_cleans_suppressed_zaya_tool_markup_without_calling_tool(
        self, monkeypatch
    ):
        """tool_choice=none must strip native tool markup, not emit tool calls."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "zaya_xml")
        req = ResponsesRequest(model="m", input="x", tool_choice="none")

        cleaned = server._clean_suppressed_tool_markup_for_display(
            "Before.\n"
            "<zyphra_tool_call>\n"
            "<function=list_directory>\n"
            "<parameter=path>\n.\n</parameter>\n"
            "</function>\n"
            "</zyphra_tool_call>\n"
            "After.",
            req,
        )

        assert cleaned == "Before.\nAfter."
        assert "<zyphra_tool_call>" not in cleaned
        assert "<function=" not in cleaned

    def test_server_streaming_suppressed_zaya_tool_markup_buffers_until_clean(
        self, monkeypatch
    ):
        """Streaming tool_choice=none must not emit partial native tool tags."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "zaya_xml")
        req = ResponsesRequest(model="m", input="x", tool_choice="none")

        first = server._suppressed_tool_display_delta(
            "Before.\n<zyphra_tool_call>\n",
            "",
            req,
        )
        assert first == "Before."

        second = server._suppressed_tool_display_delta(
            "Before.\n"
            "<zyphra_tool_call>\n"
            "<function=list_directory>\n"
            "<parameter=path>\n.\n</parameter>\n"
            "</function>\n"
            "</zyphra_tool_call>\n"
            "After.",
            "Before.",
            req,
        )
        assert second == "\nAfter."

        hidden = server._suppressed_tool_display_delta(
            "<zyphra_tool_call>\n<function=list_directory>",
            "",
            req,
        )
        assert hidden is None

    def test_server_streaming_suppressed_malformed_qwen_prefix_recovers_answer(
        self, monkeypatch
    ):
        """A rejected Qwen call prefix must not leak or batch a later answer."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "qwen")
        req = ResponsesRequest(model="m", input="x", tool_choice="none")

        prefix = "<tool_call>\n<function=file_info>\n<"
        assert server._clean_suppressed_tool_markup_for_display(prefix, req) == ""
        assert server._suppressed_tool_display_delta(prefix, "", req) is None

        malformed_then_answer = (
            prefix
            + "\n\n\nThe reported file size is 5.2 KB.\n"
            + "Q35-API-TOOL-DONE"
        )
        cleaned = server._clean_suppressed_tool_markup_for_display(
            malformed_then_answer,
            req,
        )
        assert cleaned == (
            "The reported file size is 5.2 KB.\nQ35-API-TOOL-DONE"
        )
        assert "<tool_call" not in cleaned
        assert "<function=" not in cleaned
        assert server._suppressed_tool_display_delta(
            malformed_then_answer,
            "",
            req,
        ) == cleaned

    def test_server_suppressed_cleanup_leaves_ordinary_angle_text_unchanged(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "qwen")
        req = ResponsesRequest(model="m", input="x", tool_choice="none")
        ordinary = "The value 3 < 5 is ordinary visible prose."
        assert server._clean_suppressed_tool_markup_for_display(ordinary, req) == ordinary

    def test_residue_strip_only_applies_on_identified_markup_paths(self):
        """Plain prose must bypass the residue collapse; markup holes still tidy."""
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        with_residue = "Before.\n\n<｜DSML｜tool\n\nAfter."
        cleaned = server._strip_tool_markup_residue_for_display(with_residue)
        assert "DSML" not in cleaned
        assert cleaned == "Before.\n\nAfter."

        # The delta path must NOT route ordinary prose through that collapse.
        req = ResponsesRequest(model="m", input="x", tool_choice="none")
        prose = (
            "The cache stores data.\n\nA cache is fast.\n\n"
            "\\[\nx = 1\n\\]\n\nCaching stores data."
        )
        assert server._suppressed_tool_display_delta(prose, "", req) == prose

    def test_server_suppressed_stream_preserves_paragraph_breaks_no_tools(
        self, monkeypatch
    ):
        """DSV4 no-tools stream: suppressed-display deltas must keep \\n\\n intact.

        A server started with --tool-call-parser dsml suppresses tool parsing on
        plain no-tools turns; every content delta is routed through
        _suppressed_tool_display_delta. The recompute-and-subtract cursor must
        deliver deltas whose concatenation is byte-identical to the raw prose --
        paragraph separators and KaTeX display blocks included.
        """
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", "dsml")
        req = ResponsesRequest(model="m", input="x", tool_choice="none")

        full = (
            "The cache stores data.\n\nA cache is fast.\n\n"
            "\\[\nx = 1\n\\]\n\nCaching stores data."
        )
        # DSV4's tokenizer glues paragraph breaks onto the preceding token
        # (e.g. '.\n\n', '\\]\n\n'), so boundaries land mid-break.
        raw_deltas = [
            "The cache stores data.",
            "\n\n",
            "A cache is fast.\n\n",
            "\\[\n",
            "x = 1\n",
            "\\]\n\n",
            "Caching stores data.",
        ]
        assert "".join(raw_deltas) == full

        accumulated = ""
        streamed = ""
        emitted: list[str] = []
        for raw in raw_deltas:
            accumulated += raw
            delta = server._suppressed_tool_display_delta(accumulated, streamed, req)
            if delta:
                streamed += delta
                emitted.append(delta)

        joined = "".join(emitted)
        assert joined == full
        assert joined.count("\n\n") == 3

    def test_server_suppressed_delta_never_reemits_nonmonotonic_accumulator(
        self, monkeypatch
    ):
        """A cleanup rewrite cannot become a cumulative replacement SSE delta."""
        import vmlx_engine.server as server

        monkeypatch.setattr(
            server,
            "_clean_suppressed_tool_markup_for_display",
            lambda *_args, **_kwargs: "Corrected prefix plus visible suffix",
        )
        assert server._suppressed_tool_display_delta(
            "raw accumulated text",
            "Already emitted prefix",
            None,
        ) is None

    def test_server_repairs_schema_gated_tool_instruction_echo(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", None)
        req = ResponsesRequest(
            model="m",
            input="x",
            tools=[
                {
                    "type": "function",
                    "name": "list_directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        cleaned, calls = server._parse_tool_calls_with_parser(
            "<Use the list_directory tool for path '.' and do not answer in prose.",
            req,
        )

        assert cleaned == ""
        assert calls
        assert calls[0].function.name == "list_directory"
        assert '"path": "."' in calls[0].function.arguments

    def test_server_repairs_required_single_tool_bare_json_arguments(
        self, monkeypatch
    ):
        import vmlx_engine.server as server
        from vmlx_engine.server import ChatCompletionRequest, Message

        monkeypatch.setattr(server, "_tool_call_parser", None)
        req = ChatCompletionRequest(
            model="lfm2.5-8b-a1b-mxfp8",
            messages=[
                Message(
                    role="user",
                    content=(
                        "Call function record_fact with exactly these JSON "
                        'arguments and no other value: {"value":"blue-cat"}.'
                    ),
                )
            ],
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "record_fact",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    },
                }
            ],
        )

        cleaned, calls = server._parse_tool_calls_with_parser(
            '{"value":"blue-cat"}',
            req,
        )

        assert cleaned == ""
        assert calls
        assert calls[0].function.name == "record_fact"
        assert calls[0].function.arguments == '{"value": "blue-cat"}'

    def test_server_repairs_dsv4_partial_tool_intent_from_request_args(self, monkeypatch):
        import vmlx_engine.server as server
        from vmlx_engine.api.models import ResponsesRequest

        monkeypatch.setattr(server, "_tool_call_parser", None)
        req = ResponsesRequest(
            model="m",
            input="Use the list_directory tool for path '.' and do not answer in prose.",
            tools=[
                {
                    "type": "function",
                    "name": "list_directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )

        cleaned, calls = server._parse_tool_calls_with_parser(
            "I'll use the tool to inspect the current directory.\n\n<｜DSML｜tool_c",
            req,
        )

        assert cleaned == ""
        assert calls
        assert calls[0].function.name == "list_directory"
        assert '"path": "."' in calls[0].function.arguments

    def _write_fake_dsv4_encoder(self, model_path):
        enc_dir = model_path / "encoding"
        enc_dir.mkdir(parents=True, exist_ok=True)
        (enc_dir / "encoding_dsv4.py").write_text(
            """
import json

def encode_messages(messages, **kwargs):
    out = []
    for msg in messages:
        if msg.get("role") == "user":
            out.append(f'<｜User｜>{msg.get("content", "")}')
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                for k, v in args.items():
                    out.append(f'<parameter name="{k}">{v}</parameter>')
        if msg.get("role") == "tool":
            out.append(f'<tool_result>{msg.get("content", "")}</tool_result>')
    if messages and messages[-1].get("role") in ("user", "developer"):
        out.append("<｜Assistant｜>")
        out.append("<think>" if kwargs.get("thinking_mode") == "thinking" else "</think>")
    return "\\n".join(out)

def parse_message_from_completion_text(raw_text, **kwargs):
    return {"role": "assistant", "content": raw_text, "tool_calls": []}
""",
            encoding="utf-8",
        )

    def test_dsv4_encoder_keeps_function_arguments_as_dsml_params(self, tmp_path):
        import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_chat_encoder

        model_path = tmp_path / "DeepSeek-V4-Flash-JANGTQ"
        self._write_fake_dsv4_encoder(model_path)
        dsv4_chat_encoder._encoding_cache.clear()

        prompt = dsv4_chat_encoder.apply_chat_template(
            [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "List."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_audit",
                            "type": "function",
                            "function": {
                                "name": "list_directory",
                                "arguments": {"path": "."},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_audit", "content": "README.md"},
                {"role": "user", "content": "What file appeared?"},
            ],
            enable_thinking=False,
            model_path=str(model_path),
        )

        assert 'parameter name="path"' in prompt
        assert 'parameter name="arguments"' not in prompt
        assert "<tool_result>README.md</tool_result>" in prompt

    def test_dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail(self, tmp_path):
        import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_chat_encoder

        model_path = tmp_path / "DeepSeek-V4-Flash-JANGTQ"
        self._write_fake_dsv4_encoder(model_path)
        dsv4_chat_encoder._encoding_cache.clear()
        snippet = (
            "const scene = new THREE.Scene();\n"
            "const renderer = new THREE.WebGLRenderer();\n"
            "const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);\n"
            "const mesh = new THREE.Mesh(new THREE.BoxGeometry(), "
            "new THREE.MeshBasicMaterial());"
        )

        prompt = dsv4_chat_encoder.apply_chat_template(
            [{"role": "user", "content": f"Output exactly this:\n{snippet}"}],
            enable_thinking=False,
            model_path=str(model_path),
        )

        assert "THREE.Scene" in prompt
        assert "THREE.WebGLRenderer" in prompt
        assert "THREE.PerspectiveCamera" in prompt
        assert "THREE.Mesh" in prompt
        assert "THREE.BoxGeometry" in prompt
        assert "THREE.MeshBasicMaterial" in prompt
        assert "WebWebGLRenderer" not in prompt
        assert prompt.endswith("<｜Assistant｜>\n</think>")

    def test_dsv4_encoder_auto_discovers_home_models_encoding(self, tmp_path, monkeypatch):
        import vmlx_engine.loaders.dsv4_chat_encoder as dsv4_chat_encoder

        model_path = tmp_path / "models" / "Sources" / "DeepSeek-V4-Flash"
        self._write_fake_dsv4_encoder(model_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("DSV4_ENCODING_DIR", raising=False)
        monkeypatch.delenv("VMLX_MODELS_DIR", raising=False)
        monkeypatch.delenv("VLLM_MODELS_DIR", raising=False)
        monkeypatch.delenv("VMLINUX_MODELS_DIR", raising=False)
        dsv4_chat_encoder._encoding_cache.clear()

        prompt = dsv4_chat_encoder.apply_chat_template(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "list_directory",
                                "arguments": {"path": "."},
                            },
                        }
                    ],
                }
            ],
            enable_thinking=False,
        )

        assert 'parameter name="path"' in prompt


# ─── ToolDefinition Validation ───────────────────────────────────────────────


class TestToolDefinitionValidation:
    """Test ToolDefinition Pydantic model edge cases."""

    def test_function_is_dict(self):
        from vmlx_engine.api.models import ToolDefinition

        td = ToolDefinition(function={"name": "test"})
        assert isinstance(td.function, dict)
        assert td.function.get("name") == "test"

    def test_type_defaults_to_function(self):
        from vmlx_engine.api.models import ToolDefinition

        td = ToolDefinition(function={"name": "test"})
        assert td.type == "function"

    def test_get_on_function_dict(self):
        """.function is a dict, so .get() is valid (not attribute access)."""
        from vmlx_engine.api.models import ToolDefinition

        td = ToolDefinition(function={"name": "test", "description": "Test tool"})
        assert td.function.get("name") == "test"
        assert td.function.get("description") == "Test tool"
        assert td.function.get("nonexistent") is None


class TestXMLFunctionToolParser:
    """MiMo-style generic XML function-call parser."""

    def test_extracts_xml_function_tool_call(self):
        from vmlx_engine.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser("xml_function")()
        result = parser.extract_tool_calls(
            "ok\n"
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>MiMo V2 cache</parameter>\n"
            "<parameter=limit>3</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        assert result.tools_called is True
        assert result.content == "ok"
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["arguments"] == '{"query": "MiMo V2 cache", "limit": 3}'

    def test_extracts_mimo_nested_invoke_xml_tool_call(self):
        """MiMo JANG_2L can emit tool_name/arguments XML instead of function=."""
        from vmlx_engine.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser("xml_function")()
        result = parser.extract_tool_calls(
            "<tool_call>\n"
            "<function_call>\n"
            "<invoke>\n"
            "<tool_name>record_fact</tool_name>\n"
            "<arguments>\n"
            "<value>blue-cat</value>\n"
            "</arguments>\n"
            "</invoke>\n"
            "</tool_call>"
        )

        assert result.tools_called is True
        assert result.content is None
        assert result.tool_calls[0]["name"] == "record_fact"
        assert result.tool_calls[0]["arguments"] == '{"value": "blue-cat"}'

    def test_streaming_emits_tool_call_only_after_close(self):
        from vmlx_engine.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser("xml_function")()
        current = (
            "<tool_call><function=fetch>"
            "<parameter=url>https://example.test</parameter>"
            "</function></tool_call>"
        )

        assert parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>") is None
        delta = parser.extract_tool_calls_streaming("", current, "</tool_call>")

        assert delta is not None
        assert delta["tool_calls"][0]["type"] == "function"
        assert delta["tool_calls"][0]["function"]["name"] == "fetch"


class TestXMLFunctionDoubledWrapperRecovery:
    """Qwen3.6-35B at temp 0 emits a doubled `<function=function>` wrapper with
    miskeyed `<function=KEY>V</parameter>` parameters (observed verbatim in the
    2026-08-15 smoke). The bogus parse suppressed the markup and left visibly
    empty turns; with the request's tool names the real call is unambiguous."""

    RAW = (
        "<tool_call>\n<function=function>\n<function=record_fact>\n"
        "<function=value>\nblue-cat\n</parameter>\n</function>\n</tool_call>"
    )
    REQUEST = {
        "tools": [
            {"type": "function", "function": {"name": "record_fact", "parameters": {}}}
        ]
    }

    def _parser(self):
        from vmlx_engine.tool_parsers.xml_function_tool_parser import (
            XMLFunctionToolParser,
        )

        return XMLFunctionToolParser.__new__(XMLFunctionToolParser)

    def test_recovers_the_real_call_and_miskeyed_parameter(self):
        import json

        info = self._parser().extract_tool_calls(self.RAW, request=self.REQUEST)

        assert info.tools_called is True
        assert len(info.tool_calls) == 1
        assert info.tool_calls[0]["name"] == "record_fact"
        assert json.loads(info.tool_calls[0]["arguments"]) == {"value": "blue-cat"}
        assert not info.content

    def test_wellformed_calls_do_not_take_the_recovery_path(self):
        import json

        good = (
            "<tool_call>\n<function=record_fact>\n<parameter=value>\nblue-cat\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        info = self._parser().extract_tool_calls(good, request=self.REQUEST)

        assert info.tools_called is True
        assert info.tool_calls[0]["name"] == "record_fact"
        assert json.loads(info.tool_calls[0]["arguments"]) == {"value": "blue-cat"}

    def test_without_request_names_the_bogus_wrapper_is_not_invented(self):
        info = self._parser().extract_tool_calls(self.RAW, request=None)

        # No names to disambiguate: parser keeps its old behavior rather than
        # guessing (the engine's required-mode check then reports it).
        assert all(c["name"] == "function" for c in info.tool_calls)


class TestQwenDoubledWrapperRecovery:
    """The SAME doubled-wrapper raw arrives on the qwen parser route: the live
    Qwen3.6-35B-A3B-MXFP8-CRACK-MTP bundle is jang-stamped tool_parser=qwen, so
    the recovery that first landed on the xml_function route was inert in the
    2026-08-15 batch9 smoke (tool_choice=required 400 with the markup verbatim
    in raw_preview). Recovery is shared on ToolParser; these pin the qwen
    route and both protocol tool shapes."""

    RAW = (
        "<tool_call>\n<function=function>\n<function=record_fact>\n"
        "<function=value>\nblue-cat\n</parameter>\n</function>\n</tool_call>"
    )
    REQUEST_CHAT = {
        "tools": [
            {"type": "function", "function": {"name": "record_fact", "parameters": {}}}
        ]
    }
    REQUEST_RESPONSES = {
        "tools": [{"type": "function", "name": "record_fact", "parameters": {}}]
    }

    def _parser(self):
        from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser

        return QwenToolParser.__new__(QwenToolParser)

    def test_recovers_the_real_call_on_the_qwen_route(self):
        import json

        info = self._parser().extract_tool_calls(self.RAW, request=self.REQUEST_CHAT)

        assert info.tools_called is True
        assert len(info.tool_calls) == 1
        assert info.tool_calls[0]["name"] == "record_fact"
        assert json.loads(info.tool_calls[0]["arguments"]) == {"value": "blue-cat"}
        assert not info.content

    def test_responses_shape_tools_also_disambiguate(self):
        import json

        info = self._parser().extract_tool_calls(
            self.RAW, request=self.REQUEST_RESPONSES
        )

        assert info.tools_called is True
        assert info.tool_calls[0]["name"] == "record_fact"
        assert json.loads(info.tool_calls[0]["arguments"]) == {"value": "blue-cat"}

    def test_wellformed_function_blocks_do_not_take_the_recovery_path(self):
        import json

        good = (
            "<tool_call>\n<function=record_fact>\n<parameter=value>\nblue-cat\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        info = self._parser().extract_tool_calls(good, request=self.REQUEST_CHAT)

        assert info.tools_called is True
        assert info.tool_calls[0]["name"] == "record_fact"
        assert json.loads(info.tool_calls[0]["arguments"]) == {"value": "blue-cat"}

    def test_without_request_names_no_call_is_invented(self):
        info = self._parser().extract_tool_calls(self.RAW, request=None)

        assert all(c.get("name") != "record_fact" for c in info.tool_calls)


class TestOllamaToolIdentityF5:
    """dialect F5: Ollama replays tool results as role:"tool" with tool_name
    and no tool_call_id. The Message model silently discarded the field, the
    non-native flatten rendered an anonymous "[Tool Result ()]", and replayed
    dict-form arguments rendered as a Python repr instead of the JSON the
    model originally emitted."""

    def test_message_model_keeps_tool_name(self):
        from vmlx_engine.api.models import Message

        assert Message(role="tool", tool_name="read_file", content="A").tool_name == "read_file"

    def test_non_native_flatten_labels_by_tool_name_and_renders_json_args(self):
        from vmlx_engine.api.utils import extract_multimodal_content

        msgs = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "x", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": {"path": "a.txt"}}}]},
            {"role": "tool", "tool_name": "read_file", "content": "A"},
        ]
        processed, _, _ = extract_multimodal_content(msgs, preserve_native_format=False)
        tool_texts = [p["content"] for p in processed if p.get("type") == "function_call_output"]
        assert tool_texts == ["[Tool Result (read_file)]: A"]
        call_text = next(p["content"] for p in processed if "[Calling tool:" in str(p.get("content")))
        assert 'read_file({"path": "a.txt"})' in call_text
        assert "{'path'" not in call_text

    def test_native_branch_forwards_name_only_when_present(self):
        from vmlx_engine.api.utils import extract_multimodal_content

        named, _, _ = extract_multimodal_content(
            [{"role": "tool", "tool_name": "read_file", "content": "A"}],
            preserve_native_format=True,
        )
        assert named[0].get("name") == "read_file"
        plain, _, _ = extract_multimodal_content(
            [{"role": "tool", "tool_call_id": "c1", "content": "B"}],
            preserve_native_format=True,
        )
        assert "name" not in plain[0], "existing OpenAI flows must keep identical bytes"


class TestMiskeyedParameterWithProperName:
    """The live Qwen3.6-35B variant on the Responses stream path: the function
    NAME is correct but a parameter opens `<function=KEY>` and closes
    `</parameter>`. The block parse then succeeded with EMPTY arguments and
    the required-args validator dropped the call (observed: run_command
    missing 'command'). Both parser routes recover the miskeyed argument."""

    RAW = (
        "<tool_call>\n<function=run_command>\n<function=command>\n"
        "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    REQUEST = {
        "tools": [
            {"type": "function", "function": {
                "name": "run_command",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}},
                               "required": ["command"]}}}
        ]
    }

    def _parsers(self):
        from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser
        from vmlx_engine.tool_parsers.xml_function_tool_parser import (
            XMLFunctionToolParser,
        )

        return [QwenToolParser.__new__(QwenToolParser),
                XMLFunctionToolParser.__new__(XMLFunctionToolParser)]

    def test_miskeyed_parameter_recovered_on_both_routes(self):
        import json

        for parser in self._parsers():
            info = parser.extract_tool_calls(self.RAW, request=self.REQUEST)
            assert info.tools_called is True, type(parser).__name__
            call = info.tool_calls[0]
            assert call["name"] == "run_command", type(parser).__name__
            args = json.loads(call["arguments"])
            assert "command" in args and "real_ui_tool_probe_1.txt" in args["command"], (
                type(parser).__name__, args)

    def test_wellformed_parameters_still_take_the_strict_path(self):
        import json

        good = ("<tool_call>\n<function=run_command>\n<parameter=command>\nls\n"
                "</parameter>\n</function>\n</tool_call>")
        for parser in self._parsers():
            info = parser.extract_tool_calls(good, request=self.REQUEST)
            assert json.loads(info.tool_calls[0]["arguments"]) == {"command": "ls"}, (
                type(parser).__name__)


class TestSplitKeyParameterVariant:
    """Third live variant, captured verbatim by the raw-suffix log on the
    Responses stream: a bare <parameter> opener with the KEY floated inside
    as `command>` on its own line. The bare-args fallback previously misread
    the literal <parameter> tag as an argument named "parameter"."""

    RAW = (
        "<tool_call>\n<function=run_command>\n<parameter>\ncommand>\n"
        "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    REQUEST = {
        "tools": [
            {"type": "function", "function": {
                "name": "run_command",
                "parameters": {"type": "object",
                               "properties": {"command": {"type": "string"}},
                               "required": ["command"]}}}
        ]
    }

    def _parsers(self):
        from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser
        from vmlx_engine.tool_parsers.xml_function_tool_parser import (
            XMLFunctionToolParser,
        )

        return [QwenToolParser.__new__(QwenToolParser),
                XMLFunctionToolParser.__new__(XMLFunctionToolParser)]

    def test_split_key_parameter_recovered_on_both_routes(self):
        import json

        for parser in self._parsers():
            info = parser.extract_tool_calls(self.RAW, request=self.REQUEST)
            call = info.tool_calls[0]
            args = json.loads(call["arguments"])
            assert call["name"] == "run_command", type(parser).__name__
            assert args.get("command", "").startswith("printf %s REAL_UI_LIVE_TOOL_ONE"), (
                type(parser).__name__, args)
            assert "parameter" not in args, (
                "the literal <parameter> tag must not become an argument name")

    def test_degenerate_empty_opener_is_not_promoted(self):
        degen = "<tool_call>\n<>\n</function>\n</tool_call>"
        for parser in self._parsers():
            info = parser.extract_tool_calls(degen, request=self.REQUEST)
            assert all(
                c.get("name") != "run_command" for c in (info.tool_calls or [])
            ), type(parser).__name__


class TestTruncatedNameAndNamelessVariants:
    """Fourth and fifth live degraded shapes, captured verbatim by the
    catalog-size bisect on the Responses stream (the degeneracy is
    prompt-content-sensitive, not monotonic in catalog size):
    (A) `<function=_command>` — the name truncated, parameters perfect;
    (B) the function opener missing entirely, block starts at a well-formed
    parameter. Both repairs are schema-gated: A renames only when the parsed
    name is a strict suffix of exactly one advertised tool whose schema
    accepts the parsed args; B promotes only when the block starts directly
    at the parameter tag AND the key set matches exactly one advertised
    tool. Explicitly-named unadvertised tools are never reassigned."""

    TOOLS = {"tools": [
        {"type": "function", "function": {"name": "run_command",
         "parameters": {"type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]}}},
        {"type": "function", "function": {"name": "read_file",
         "parameters": {"type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]}}},
        {"type": "function", "function": {"name": "write_file",
         "parameters": {"type": "object",
                        "properties": {"path": {"type": "string"},
                                       "content": {"type": "string"}},
                        "required": ["path"]}}},
    ]}

    def _parser(self):
        from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser

        return QwenToolParser.__new__(QwenToolParser)

    def test_truncated_name_repaired_when_suffix_unique_and_schema_valid(self):
        import json

        raw = ("<tool_call>\n<function=_command>\n<parameter=command>\n"
               "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n"
               "</parameter>\n</function>\n</tool_call>")
        info = self._parser().extract_tool_calls(raw, request=self.TOOLS)
        call = info.tool_calls[0]
        assert call["name"] == "run_command"
        assert json.loads(call["arguments"])["command"].startswith("printf")

    def test_nameless_block_promoted_when_schema_uniquely_matches(self):
        import json

        raw = ("<tool_call>\n<parameter=command>\n"
               "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt\n"
               "</parameter>\n</function>\n</tool_call>")
        info = self._parser().extract_tool_calls(raw, request=self.TOOLS)
        assert info.tools_called is True
        call = info.tool_calls[0]
        assert call["name"] == "run_command"
        assert "command" in json.loads(call["arguments"])

    def test_ambiguous_params_and_ambiguous_suffix_are_not_promoted(self):
        raw_ambiguous_params = (
            "<tool_call>\n<parameter=path>\n/tmp/x\n</parameter>\n"
            "</function>\n</tool_call>")
        info = self._parser().extract_tool_calls(
            raw_ambiguous_params, request=self.TOOLS)
        assert all(c.get("name") not in ("read_file", "write_file")
                   for c in (info.tool_calls or []))

        two_suffix = {"tools": [
            {"type": "function", "function": {"name": "run_command",
             "parameters": {"type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"]}}},
            {"type": "function", "function": {"name": "spawn_command",
             "parameters": {"type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"]}}},
        ]}
        raw_truncated = ("<tool_call>\n<function=_command>\n<parameter=command>\n"
                         "ls\n</parameter>\n</function>\n</tool_call>")
        info = self._parser().extract_tool_calls(raw_truncated, request=two_suffix)
        assert all(c.get("name") not in ("run_command", "spawn_command")
                   for c in (info.tool_calls or []))
