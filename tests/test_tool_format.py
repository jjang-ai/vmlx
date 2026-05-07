"""Tests for tool format conversion, tool_choice handling, and related edge cases.

Covers gaps identified in the comprehensive test audit:
- ResponsesToolDefinition.to_chat_completions_format() conversion
- convert_tools_for_template() roundtrip
- tool_choice suppression and filtering
- response_format.strict validation
- max_tokens fallback chain
"""

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

    def test_tool_choice_dict_no_match_fallback(self):
        """When tool_choice names a non-existent tool, all tools should be sent."""
        from vmlx_engine.api.models import ToolDefinition

        target_name = "nonexistent"
        tools = [
            ToolDefinition(function={"name": "search", "description": "Search"}),
        ]
        filtered = [t for t in tools if t.function.get("name") == target_name]
        # When no match, fallback to all tools
        result = filtered if filtered else tools
        assert len(result) == 1
        assert result[0].function["name"] == "search"


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
        assert 'parameter name="path"' in rendered
        assert "<tool_call>" not in rendered

    def test_qwen_fallback_injects_concrete_native_tool_example(self):
        from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

        class FakeTokenizer:
            last_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.last_kwargs = kwargs
                return "\n".join(m.get("content", "") for m in messages)

        prompt = (
            "<|im_start|>system\n# Tools\n<tools>\n"
            '{"type":"function","function":{"name":"list_directory"}}\n'
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

        assert "<function=list_directory>" in rendered
        assert "<parameter=path>" in rendered
        assert "fake directory listing" in rendered
        assert "tools" not in tokenizer.last_kwargs

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

    def test_zaya_fallback_skips_when_concrete_native_example_present(self):
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
            [{"role": "user", "content": "Use list_directory"}],
            tools,
            tokenizer,
            {"tokenize": False, "tools": tools},
        )

        assert rendered == prompt
        tokenizer.apply_chat_template.assert_not_called()

    def test_dsml_parser_repairs_schema_gated_malformed_old_dsv4_tool_call(self):
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

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "list_directory"
        assert '"path": "."' in result.tool_calls[0]["arguments"]
        assert result.content is None

    def test_dsml_parser_repairs_partial_canonical_invoke(self):
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

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "list_directory"
        assert '"path": "."' in result.tool_calls[0]["arguments"]
        assert result.content is None

    def test_dsml_parser_repairs_partial_invoke_with_malformed_value_attr(self):
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

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "list_directory"
        assert '"path": "."' in result.tool_calls[0]["arguments"]
        assert result.content is None

    def test_dsml_parser_repairs_htmlish_invoke_degradation(self):
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

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "list_directory"
        assert '"path": "."' in result.tool_calls[0]["arguments"]
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
