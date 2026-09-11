# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for tool call parsers."""

import json

import pytest

from vmlx_engine.tool_parsers import (
    AutoToolParser,
    DeepSeekToolParser,
    FunctionaryToolParser,
    Gemma3ToolParser,
    Gemma4ToolParser,
    Glm47ToolParser,
    GraniteToolParser,
    HermesToolParser,
    HunyuanToolParser,
    KimiToolParser,
    LlamaToolParser,
    MiniMaxToolParser,
    MistralToolParser,
    NemotronToolParser,
    QwenToolParser,
    ToolParserManager,
    XMLFunctionToolParser,
    xLAMToolParser,
    ZayaToolParser,
)


class TestToolParserManager:
    """Test the ToolParserManager registry."""

    def test_list_registered(self):
        """Test that all expected parsers are registered."""
        parsers = ToolParserManager.list_registered()
        expected = [
            "auto",
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
            "minimax",
            "gemma3",
            "gemma4",
            "glm47",
            "hunyuan",
            "xml_function",
        ]
        for p in expected:
            assert p in parsers, f"Parser '{p}' not found"

    def test_get_tool_parser_by_name(self):
        """Test getting parsers by name."""
        test_cases = [
            ("mistral", MistralToolParser),
            ("qwen", QwenToolParser),
            ("qwen3", QwenToolParser),
            ("llama", LlamaToolParser),
            ("llama3", LlamaToolParser),
            ("llama4", LlamaToolParser),
            ("auto", AutoToolParser),
            ("deepseek", DeepSeekToolParser),
            ("deepseek_v3", DeepSeekToolParser),
            ("deepseek_r1", DeepSeekToolParser),
            ("kimi", KimiToolParser),
            ("kimi_k2", KimiToolParser),
            ("moonshot", KimiToolParser),
            ("granite", GraniteToolParser),
            ("granite3", GraniteToolParser),
            ("nemotron", NemotronToolParser),
            ("nemotron3", NemotronToolParser),
            ("xlam", xLAMToolParser),
            ("functionary", FunctionaryToolParser),
            ("meetkai", FunctionaryToolParser),
            ("hermes", HermesToolParser),
            ("nous", HermesToolParser),
            ("minimax", MiniMaxToolParser),
            ("minimax_m2", MiniMaxToolParser),
            ("gemma3", Gemma3ToolParser),
            ("gemma3n", Gemma3ToolParser),
            ("gemma4", Gemma4ToolParser),
            ("glm47", Glm47ToolParser),
            ("glm4", Glm47ToolParser),
            ("hunyuan", HunyuanToolParser),
            ("hy_v3", HunyuanToolParser),
            ("tencent", HunyuanToolParser),
            ("xml_function", XMLFunctionToolParser),
            ("mimo_xml_function", XMLFunctionToolParser),
        ]
        for name, expected_cls in test_cases:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert parser_cls == expected_cls, f"Parser '{name}' returned wrong class"

    def test_get_unknown_parser_raises(self):
        """Test that unknown parser raises KeyError."""
        with pytest.raises(KeyError):
            ToolParserManager.get_tool_parser("unknown_parser")

    def test_parser_instantiation(self):
        """Test that all parsers can be instantiated without tokenizer."""
        for name in [
            "auto",
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
            "minimax",
            "gemma3",
            "gemma4",
            "glm47",
            "hunyuan",
            "xml_function",
        ]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            parser = parser_cls()  # Should not raise
            assert parser is not None


class TestMistralToolParser:
    """Test the Mistral tool parser."""

    @pytest.fixture
    def parser(self):
        return MistralToolParser()

    def test_old_format_single(self, parser):
        """Test parsing old Mistral format with single tool call."""
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"

    def test_old_format_multiple(self, parser):
        """Test parsing old Mistral format with multiple tool calls."""
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_time", "arguments": {"timezone": "UTC"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_new_format(self, parser):
        """Test parsing new Mistral format."""
        text = '[TOOL_CALLS]get_weather{"city": "London"}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_new_format_invalid_json_rejected(self, parser):
        """New format with invalid JSON args should be silently rejected."""
        text = '[TOOL_CALLS]get_weather{invalid json here}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_new_format_valid_after_invalid(self, parser):
        """Valid tool call should be parsed even if preceded by invalid one."""
        # Two tool calls separated by [TOOL_CALLS] — first invalid, second valid
        text = '[TOOL_CALLS] [{"name": "good_tool", "arguments": {"x": 1}}]'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "good_tool"

    def test_no_tool_call(self, parser):
        """Test that regular text is not parsed as tool call."""
        text = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text

    def test_content_with_tool_call(self, parser):
        """Test content before tool call is preserved."""
        text = 'Let me check the weather for you.[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me check the weather for you."


class TestQwenToolParser:
    """Test the Qwen tool parser."""

    @pytest.fixture
    def parser(self):
        return QwenToolParser()

    def test_xml_format(self, parser):
        """Test parsing Qwen XML format."""
        text = '<tool_call>{"name": "calculate", "arguments": {"x": 1, "y": 2}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculate"

    def test_xml_string_arguments_use_matching_single_param_schema(self, parser):
        """Qwen3-Coder may emit <command> XML as the argument payload."""
        text = (
            '<tool_call>{"name": "bash", "arguments": '
            '"<command>\\necho \\"Tools are working correctly!\\"\\n</command>"}'
            "</tool_call>"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"command": 'echo "Tools are working correctly!"'}

    def test_plain_tool_name_then_argument_uses_single_tool_schema(self, parser):
        """Qwen3.6 can emit a bare tool name followed by the command text."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls('bash\necho "Tool test successful!"', request=request)

        assert result.tools_called
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "bash"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"command": 'echo "Tool test successful!"'}

    def test_markdown_call_header_uses_single_required_schema(self, parser):
        """Qwen3.6 JANGTQ may emit a markdown-looking call header.

        Observed live on Qwen3.6-35B-A3B-JANGTQ-CRACK: the model wrote
        "# Calling file_info" and "# path=panel/package.json", then continued
        into a fake "# Tool result" block because the parser did not stop the
        stream. Only the schema-valid header+argument pair is the call; the
        fake result must not become content or arguments.
        """
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        text = (
            "# Calling file_info\n"
            "# path=panel/package.json\n\n"
            "# Tool result:\n"
            '# {"path": "panel/package.json", "size": "2.4 KB"}'
        )

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "file_info"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_markdown_call_header_uses_responses_tool_schema(self, parser):
        """Responses tools are top-level function specs, not nested chat specs."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(
            "# Calling file_info\n# path=panel/package.json",
            request=request,
        )

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "file_info"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_markdown_call_requires_matching_schema(self, parser):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(
            "# Calling file_info\n# command=panel/package.json",
            request=request,
        )
        no_schema = parser.extract_tool_calls(
            "# Calling file_info\n# path=panel/package.json"
        )

        assert not result.tools_called
        assert not no_schema.tools_called

    def test_markdown_call_streaming_uses_request_schema(self, parser):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        current = "# Calling file_info\n# path=panel/package.json"

        result = parser.extract_tool_calls_streaming(
            previous_text="# Calling file_info\n",
            current_text=current,
            delta_text="# path=panel/package.json",
            request=request,
        )

        assert result is not None
        assert "tool_calls" in result
        call = result["tool_calls"][0]
        assert call["function"]["name"] == "file_info"
        assert json.loads(call["function"]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_labeled_line_call_uses_schema_after_prompt_marker(self, parser):
        """Qwen3.6 JANGTQ may prefix the tool block with the benchmark marker
        and then write a plain tool-name/parameter pair. The parser may accept
        only the schema-valid pair, not the marker or later answer text.
        """
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        text = (
            "[Q36-JT-UI-TOOL2]\n"
            "file_info\n"
            "path: panel/package.json\n"
            "[Q36-JT-UI-TOOL2-DONE SIZE=3"
        )

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "file_info"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_orphan_function_opener_inside_tool_wrapper_uses_request_schema(self, parser):
        """A closed wrapper with a missing function opener remains executable.

        This exact fenced shape was emitted by the live Qwen3.6 JANGTQ
        Electron full-tool-catalog turn. The advertised tool name and parameter
        schema make it unambiguous; the orphan closing tag and fence are not
        visible assistant content.
        """
        request = {
            "tools": [
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
        text = (
            "```text\n"
            "<tool_call>\n"
            "file_info\n"
            "<parameter=path>\n"
            "panel/package.json\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "```"
        )

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "file_info"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_orphan_function_opener_rejects_unadvertised_tool_or_parameter(self, parser):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        unadvertised = (
            "<tool_call>delete_file<parameter=path>x</parameter>"
            "</function></tool_call>"
        )
        wrong_parameter = (
            "<tool_call>file_info<parameter=command>x</parameter>"
            "</function></tool_call>"
        )

        assert not parser.extract_tool_calls(unadvertised, request=request).tools_called
        assert not parser.extract_tool_calls(wrong_parameter, request=request).tools_called

    def test_empty_parameter_named_for_tool_recovers_missing_function_opener(self, parser):
        """The full UI catalog can make Qwen spell the function as an empty param."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        }
        text = (
            "```text\n<tool_call>\n"
            "<parameter=file_info>\n</parameter>\n"
            "<parameter=path>\npanel/package.json\n</parameter>\n"
            "</function>\n</tool_call>\n```"
        )

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "file_info"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_empty_parameter_does_not_invent_unadvertised_tool(self, parser):
        request = {
            "tools": [
                {
                    "type": "function",
                    "name": "file_info",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        }
        text = (
            "<tool_call><parameter=delete_file></parameter>"
            "<parameter=path>x</parameter></function></tool_call>"
        )

        assert not parser.extract_tool_calls(text, request=request).tools_called

    def test_orphan_function_opener_streaming_completes_only_when_schema_valid(self, parser):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        current = (
            "<tool_call>\nfile_info\n<parameter=path>panel/package.json</parameter>\n"
            "</function>\n</tool_call>"
        )

        result = parser.extract_tool_calls_streaming(
            previous_text=current[:-12],
            current_text=current,
            delta_text="</tool_call>",
            request=request,
        )

        assert result is not None
        call = result["tool_calls"][0]
        assert call["function"]["name"] == "file_info"
        assert json.loads(call["function"]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_labeled_line_call_stream_stop_truncates_after_argument(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        buffered = (
            "[Q36-JT-UI-TOOL2]\n"
            "file_info\n"
            "path: panel/package.json\n"
            "[Q36-JT-UI-TOOL2-DONE SIZE=3"
        )

        assert parser.stream_tool_calls_complete(buffered) is True
        assert parser.stream_tool_call_stop_truncate(buffered) == (
            "[Q36-JT-UI-TOOL2]\nfile_info\npath: panel/package.json"
        )

    def test_exact_once_stream_stop_truncates_after_markdown_call(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        call = "# Calling file_info\n# path=panel/package.json"
        buffered = call + "\n\n# Tool result:\n# fake\nQ36-JT-UI-TOOL1-DONE"

        assert parser.stream_tool_calls_complete(buffered) is True
        assert parser.stream_tool_call_stop_truncate(buffered) == call

    def test_bracket_format(self, parser):
        """Test parsing Qwen bracket format (Qwen3 style)."""
        text = '[Calling tool: add({"a": 5, "b": 3})]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add"

    def test_multiple_xml_calls(self, parser):
        """Test multiple XML tool calls."""
        text = '<tool_call>{"name": "func1", "arguments": {}}</tool_call><tool_call>{"name": "func2", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_parameter_attribute_variant(self, parser):
        """Qwen3.6 sometimes emits the off-format attribute variant
        <function name="..."><parameter name="...">v</parameter></function>
        instead of the canonical <function=...><parameter=...> form. Observed
        live on Qwen3.6-27B-MXFP4: the equals-only regex missed it, the call was
        dropped for a missing required arg, and the raw XML leaked into content.
        The tolerant regex must parse both forms.
        """
        text = (
            '<tool_call><function name="get_weather">'
            '<parameter name="city">Tokyo</parameter>'
            "</function></tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Tokyo"
        # the raw parameter XML must NOT leak into content
        assert result.content is None or "<parameter" not in result.content

    def test_mixed_dialect_equals_function_attribute_parameter(self, parser):
        """Exact live-observed Flash-Next JANG_1L failure shape (temp 0.7,
        tool_choice=required, 2026-08-28): canonical <function=name> mixed
        with attribute <parameter name="...">, multiline value, trailing
        newlines. Under the stale hermes stamp this failed closed with
        tool_calls_required; the qwen parser must accept the mix.
        """
        text = (
            "<tool_call>\n<function=get_weather>\n"
            '<parameter name="city">Paris\n</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Paris"
        assert result.content is None or "<parameter" not in result.content

    def test_hermes_json_body_parses_under_qwen_parser(self, parser):
        """The qwen parser must remain a superset of hermes for the
        <tool_call>{json}</tool_call> body: Flash-Next emits the JSON body on
        most sampled turns, so switching the family off the stale hermes
        stamp must not break those."""
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Tokyo"

    def test_canonical_equals_still_parses(self, parser):
        """Regression guard: canonical equals form still parses after the
        attribute-variant tolerance was added."""
        text = (
            "<tool_call><function=get_weather>"
            "<parameter=city>Paris</parameter>"
            "</function></tool_call>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Paris"

    def test_exact_once_stream_stop_finds_first_schema_valid_call(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        call = (
            "<tool_call>\n<function=file_info>\n"
            "<parameter=path>panel/package.json</parameter>\n"
            "</function>\n</tool_call>"
        )
        buffered = "reasoning before\n" + call + "\npost-call reasoning " * 20

        assert parser.STREAM_STOPS_AFTER_COMPLETE_CALL is False
        assert parser.stream_tool_calls_complete(buffered) is True
        assert parser.stream_tool_call_stop_truncate(buffered) == (
            "reasoning before\n" + call
        )

    def test_exact_once_stream_stop_rejects_missing_required_argument(self, parser):
        parser._stream_stop_request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }
        malformed = (
            "<tool_call><function=file_info>"
            "<parameter>path</parameter>panel/package.json</parameter>"
            "</function></tool_call>"
        )

        assert parser.stream_tool_calls_complete(malformed) is False
        assert parser.stream_tool_call_stop_truncate(malformed) == malformed

    def test_parameter_colon_variant(self, parser):
        """Qwen3.6-27B-JANG_4M under reasoning-on + multi-parameter tools emits a
        COLON name separator: <parameter:city>Tokyo</parameter>. Observed live
        2026-07-08 — the equals/attribute/`>` regex missed it, so the params
        dropped, the call arrived with empty required args, the server dropped it
        "missing required argument", and the whole <tool_call><function=...>
        <parameter:...> XML leaked into visible content. The tolerant regex must
        parse the colon form and keep every parameter.
        """
        text = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter:city>Tokyo</parameter>\n"
            "<parameter:unit>celsius</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Tokyo"
        assert args["unit"] == "celsius"
        assert result.content is None or "<parameter" not in result.content

    def test_orphan_scaffolding_stripped_from_content(self, parser):
        """Holo3-35B-A3B-mxfp4 (2026-07-08) emits MALFORMED tool XML: a premature
        </function> after the first param, then an orphan <parameter=unit> outside
        any function, extra </function> tags, and a hallucinated <result>{...}</result>
        fake tool output. The real call must still parse, but none of the orphan
        scaffolding may leak into visible content.
        """
        text = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nBerlin\n</parameter>\n</function>\n"
            "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</function>\n"
            '<result>\n{\n  "temperature": 15,\n  "condition": "Partly cloudy"\n}\n</result>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Berlin"
        # no orphan tool scaffolding may leak into content
        leaked = result.content or ""
        for marker in ("<parameter", "</function>", "<result", "<tool_call", "<argument", "<value"):
            assert marker not in leaked, f"{marker!r} leaked into content: {leaked!r}"

    def test_orphan_strip_preserves_legitimate_prose(self, parser):
        """The residual strip must only remove tool scaffolding, never a model's
        legitimate trailing prose in a tool-call turn."""
        text = (
            "<tool_call><function=get_weather>"
            "<parameter=city>Paris</parameter>"
            "</function></tool_call>"
            "I'll check the weather in Paris for you now."
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Paris"
        assert result.content is not None
        assert "check the weather in Paris" in result.content

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I can help you with that question."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestLlamaToolParser:
    """Test the Llama tool parser."""

    @pytest.fixture
    def parser(self):
        return LlamaToolParser()

    def test_function_format(self, parser):
        """Test parsing Llama function format."""
        text = '<function=multiply>{"x": 3, "y": 4}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "multiply"

    def test_multiple_functions(self, parser):
        """Test parsing multiple function calls."""
        text = '<function=add>{"a": 1}</function><function=multiply>{"x": 3}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[1]["name"] == "multiply"

    def test_content_with_function(self, parser):
        """Test content before function call."""
        text = 'Computing result<function=calc>{"n": 5}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Computing result"


class TestHermesToolParser:
    """Test the Hermes tool parser."""

    @pytest.fixture
    def parser(self):
        return HermesToolParser()

    def test_tool_call_format(self, parser):
        """Test parsing Hermes format."""
        text = (
            '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_with_reasoning(self, parser):
        """Test with reasoning block."""
        text = '<tool_call_reasoning>I need to search for this</tool_call_reasoning><tool_call>{"name": "search", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert "Reasoning" in (result.content or "")


class TestDeepSeekToolParser:
    """Test the DeepSeek tool parser."""

    @pytest.fixture
    def parser(self):
        return DeepSeekToolParser()

    def test_deepseek_format(self, parser):
        """Test parsing DeepSeek V3 format."""
        text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_multiple_calls(self, parser):
        """Test multiple DeepSeek tool calls."""
        text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>func1
```json
{"a": 1}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>func2
```json
{"b": 2}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_content_before_tools(self, parser):
        """Test content before tool calls is preserved."""
        text = """Let me help you with that.<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me help you with that."

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Here is my response without any tool calls."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestKimiToolParser:
    """Test the Kimi tool parser."""

    @pytest.fixture
    def parser(self):
        return KimiToolParser()

    def test_kimi_format(self, parser):
        """Test parsing Kimi K2 format."""
        text = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|>
<|tool_calls_section_end|>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_simple_function_name(self, parser):
        """Test with simple function name (no functions. prefix)."""
        text = (
            "<|tool_call_begin|>search:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I'll answer your question directly."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestLfm2ToolParser:
    """Test the Liquid LFM2 Python-call-list tool parser."""

    def test_lfm2_python_call_list(self):
        from vmlx_engine.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser("lfm2")()
        text = (
            "I will check that."
            "<|tool_call_start|>[get_weather(city='Paris', units='celsius'), "
            "search(query='local MLX cache')]<|tool_call_end|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "I will check that."
        assert [call["name"] for call in result.tool_calls] == [
            "get_weather",
            "search",
        ]
        assert result.tool_calls[0]["arguments"] == (
            '{"city": "Paris", "units": "celsius"}'
        )
        assert result.tool_calls[1]["arguments"] == '{"query": "local MLX cache"}'

    def test_lfm2_strips_reasoning_before_tools(self):
        from vmlx_engine.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser("liquid")()
        result = parser.extract_tool_calls(
            "<think>use tool</think><|tool_call_start|>[calc(x=2)]<|tool_call_end|>"
        )

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "calc"
        assert result.tool_calls[0]["arguments"] == '{"x": 2}'

    def test_lfm2_strips_malformed_control_markers_from_visible_content(self):
        from vmlx_engine.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser("lfm2")()
        result = parser.extract_tool_calls(
            "<|tool_call_start|>\n\nFile created successfully. REAL_UI_LIVE_TOOL_ONE"
        )

        assert not result.tools_called
        assert result.content == "File created successfully. REAL_UI_LIVE_TOOL_ONE"
        assert "<|tool_call_start|>" not in result.content


class TestGraniteToolParser:
    """Test the Granite tool parser."""

    @pytest.fixture
    def parser(self):
        return GraniteToolParser()

    def test_granite_30_format(self, parser):
        """Test parsing Granite 3.0 format."""
        text = (
            '<|tool_call|>[{"name": "calculate", "arguments": {"expression": "2+2"}}]'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculate"

    def test_granite_31_format(self, parser):
        """Test parsing Granite 3.1 format."""
        text = '<tool_call>[{"name": "search", "arguments": {"query": "test"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_multiple_calls(self, parser):
        """Test multiple tool calls."""
        text = '<|tool_call|>[{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "The answer is 42."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestNemotronToolParser:
    """Test the Nemotron tool parser."""

    @pytest.fixture
    def parser(self):
        return NemotronToolParser()

    def test_parameter_format(self, parser):
        """Test parsing Nemotron parameter format."""
        text = "<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=units>celsius</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"
        assert args["units"] == "celsius"

    def test_function_without_opening_tool_call_marker(self, parser):
        """Nemotron can emit template body without the leading <tool_call>.

        Live Responses auto-tool-choice produced:
        <function=list_directory>... </function></tool_call>
        The parser should still convert it to a structured call instead of
        leaking raw tool markup into output_text.
        """
        text = (
            "<function=list_directory>\n"
            "<parameter=path>\n.\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "list_directory"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["path"] == "."
        assert result.content is None

    def test_json_format(self, parser):
        """Test parsing Nemotron with JSON arguments."""
        text = '<tool_call><function=calculate>{"expression": "2*3"}</function></tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_multiple_calls(self, parser):
        """Test multiple Nemotron tool calls."""
        text = "<tool_call><function=func1><parameter=a>1</parameter></function></tool_call><tool_call><function=func2><parameter=b>2</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Here is the information you requested."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called

class TestZayaToolParser:
    """Test ZAYA/Zyphra XML tool-call parser."""

    @pytest.fixture
    def parser(self):
        return ZayaToolParser()

    def test_nested_parameter_start_repairs_missing_parameter_close(self, parser):
        """Live ZAYA-VL can start the next parameter before closing the first."""
        text = (
            "<zyphra_tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>real_ui_tool_probe_1.txt\n"
            "<parameter=content>REAL_UI_LIVE_TOOL_ONE</parameter>\n"
            "</function>\n"
            "</zyphra_tool_call>"
        )

        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "write_file"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {
            "path": "real_ui_tool_probe_1.txt",
            "content": "REAL_UI_LIVE_TOOL_ONE",
        }

    def test_double_wrapped_arguments_are_unwrapped(self, parser):
        """Zaya JANG_4M can emit the whole args object as a single parameter.

        Live-found (Codex Electron QA 2026-07-13): the model produced
        `<parameter=path>{"path": ..., "offset": 1, "limit": 5}</parameter>`,
        which json.loads collapses to `{"path": {"path": ..., ...}}`; the tool
        then rejected `path` as an object and the model looped the same
        malformed call until the UI guard interrupted it. The parser now
        unwraps the demonstrably double-wrapped object.
        """
        text = (
            "<zyphra_tool_call>\n"
            "<function=read_file>\n"
            '<parameter=path>{"path": "/repo/README.md", "offset": 1, "limit": 5}</parameter>\n'
            "</function>\n"
            "</zyphra_tool_call>"
        )

        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "read_file"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"path": "/repo/README.md", "offset": 1, "limit": 5}

    def test_object_valued_argument_is_not_unwrapped(self, parser):
        """A legitimately object-valued single arg must be left intact.

        The unwrap only fires when the sole key re-appears inside its own dict
        value, so a real object argument (whose key is not a member of the
        object) passes through unchanged.
        """
        text = (
            "<zyphra_tool_call>\n"
            "<function=configure>\n"
            '<parameter=settings>{"width": 80, "height": 24}</parameter>\n'
            "</function>\n"
            "</zyphra_tool_call>"
        )

        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"settings": {"width": 80, "height": 24}}


class TestXLAMToolParser:
    """Test the xLAM tool parser."""

    @pytest.fixture
    def parser(self):
        return xLAMToolParser()

    def test_json_array(self, parser):
        """Test parsing JSON array format."""
        text = '[{"name": "search", "arguments": {"query": "AI"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_code_block(self, parser):
        """Test parsing markdown code block."""
        text = '```json\n[{"name": "calculate", "arguments": {"x": 5}}]\n```'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_after_think(self, parser):
        """Test parsing after </think> tag."""
        text = (
            '<think>Let me search for this</think>[{"name": "search", "arguments": {}}]'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_tool_calls_tag(self, parser):
        """Test [TOOL_CALLS] tag format."""
        text = '[TOOL_CALLS][{"name": "func", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I don't need to use any tools for this."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestFunctionaryToolParser:
    """Test the Functionary tool parser."""

    @pytest.fixture
    def parser(self):
        return FunctionaryToolParser()

    def test_recipient_format(self, parser):
        """Test parsing Functionary v3 recipient format."""
        text = '<|from|>assistant\n<|recipient|>get_weather\n<|content|>{"city": "NYC"}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_function_format(self, parser):
        """Test parsing function format."""
        text = '<function=search>{"query": "test"}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_json_array(self, parser):
        """Test parsing JSON array."""
        text = '[{"name": "func1", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Let me explain that to you."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestAutoToolParser:
    """Test the auto-detecting tool parser."""

    @pytest.fixture
    def parser(self):
        return AutoToolParser()

    def test_detects_mistral(self, parser):
        """Test auto detection of Mistral format."""
        text = '[TOOL_CALLS] [{"name": "search", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_detects_qwen_xml(self, parser):
        """Test auto detection of Qwen XML format."""
        text = '<tool_call>{"name": "calculate", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_detects_qwen_xml_string_arguments_with_schema(self, parser):
        """Auto parser should not leak Qwen <command> XML as raw arguments."""
        text = (
            '<tool_call>{"name": "bash", "arguments": '
            '"<command>\\necho \\"Tools are working correctly!\\"\\n</command>"}'
            "</tool_call>"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "command": 'echo "Tools are working correctly!"'
        }

    def test_detects_qwen_bracket(self, parser):
        """Test auto detection of Qwen bracket format."""
        text = '[Calling tool: add({"a": 1})]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "add"

    def test_detects_llama(self, parser):
        """Test auto detection of Llama format."""
        text = '<function=multiply>{"x": 2}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "multiply"

    def test_detects_nemotron(self, parser):
        """Test auto detection of Nemotron format."""
        text = "<tool_call><function=search><parameter=q>test</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_detects_raw_json(self, parser):
        """Test auto detection of raw JSON format."""
        text = '{"name": "test_func", "arguments": {"key": "value"}}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "test_func"

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "This is just a regular response."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestMiniMaxToolParser:
    """Test the MiniMax tool parser."""

    @pytest.fixture
    def parser(self):
        return MiniMaxToolParser()

    def test_single_tool_call(self, parser):
        """Test parsing a single MiniMax tool call."""
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_direct_schema_child_uses_advertised_responses_argument(self, parser):
        """M2.7 may name the advertised argument directly inside invoke."""
        text = """<minimax:tool_call>
<invoke name="cache_contract_unused">
<value>CACHE-HIERARCHY-27dceffd73d18486b68655370c10f59e-A</value>
</invoke>
</minimax:tool_call>"""
        request = {
            "tools": [
                {
                    "type": "function",
                    "name": "cache_contract_unused",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(text, request=request)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "cache_contract_unused"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "value": "CACHE-HIERARCHY-27dceffd73d18486b68655370c10f59e-A"
        }

    def test_direct_schema_child_streaming_keeps_request_schema(self, parser):
        text = """<minimax:tool_call>
<invoke name="file_info">
<path>panel/package.json</path>
</invoke>
</minimax:tool_call>"""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls_streaming(
            previous_text=text[: -len("</minimax:tool_call>")],
            current_text=text,
            delta_text="</minimax:tool_call>",
            request=request,
        )

        assert result is not None
        call = result["tool_calls"][0]
        assert call["function"]["name"] == "file_info"
        assert json.loads(call["function"]["arguments"]) == {
            "path": "panel/package.json"
        }

    @pytest.mark.parametrize(
        "parser_request",
        [
            None,
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "file_info",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ]
            },
        ],
    )
    def test_direct_child_without_matching_schema_does_not_invent_raw_argument(
        self, parser, parser_request
    ):
        text = """<minimax:tool_call>
<invoke name="cache_contract_unused">
<value>do-not-promote-without-matching-schema</value>
</invoke>
</minimax:tool_call>"""

        result = parser.extract_tool_calls(text, request=parser_request)

        assert not result.tools_called
        assert result.tool_calls == []
        assert not result.content

    def test_multiple_invocations(self, parser):
        """Test multiple <invoke> blocks within a single tool_call."""
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Paris</parameter>
</invoke>
<invoke name="get_time">
<parameter name="timezone">CET</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_content_before_tool_call(self, parser):
        """Test content text before tool call is preserved."""
        text = """Let me check the weather for you.
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Tokyo</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me check the weather for you."
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_orphan_outer_close_with_complete_invoke_is_native_call(self, parser):
        """M2.7 may consume the opening namespace token while decoding."""
        text = '''<invoke name="file_info">
<parameter name="path">panel/package.json</parameter>
</invoke>
</minimax:tool_call>'''
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "file_info"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "path": "panel/package.json"
        }

    def test_visible_invoke_without_orphan_outer_close_remains_content(self, parser):
        """Do not promote arbitrary XML examples without the MiniMax close."""
        text = '<invoke name="file_info"><parameter name="path">x</parameter></invoke>'
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text

    def test_think_tags_with_tool_call(self, parser):
        """Test <think> tags are stripped before parsing."""
        text = """<think>I need to get the weather data.</think>
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">London</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        # Think tags should be stripped, so no residual think content
        assert "<think>" not in (result.content or "")

    def test_no_tool_call(self, parser):
        """Test regular text returns no tool calls."""
        text = "Hello! How can I help you today?"
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text

    def test_array_parameter(self, parser):
        """Test parameter value that is a JSON array."""
        text = """<minimax:tool_call>
<invoke name="create_list">
<parameter name="items">["apple", "banana", "cherry"]</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["items"] == ["apple", "banana", "cherry"]

    def test_quoted_function_name(self, parser):
        """Test function name with quotes."""
        text = """<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">MiniMax M2.5</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search_web"

    def test_quoted_name_with_quotes(self, parser):
        """Test function name wrapped in double quotes."""
        text = '''<minimax:tool_call>
<invoke name="calculate">
<parameter name="expression">2+2</parameter>
</invoke>
</minimax:tool_call>'''
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["expression"] == "2+2"

    def test_unclosed_minimax_block_with_complete_parameter(self, parser):
        """Live MiniMax can stop before closing the outer tool_call block."""
        text = '''<minimax:tool_call>
<invoke name="record_fact">
<parameter name="value">blue-cat</parameter>'''
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "record_fact"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

    def test_minimax_block_missing_invoke_close_with_complete_parameter(self, parser):
        """A complete outer block may still be missing </invoke>."""
        text = '''<minimax:tool_call>
<invoke name="record_fact">
<parameter name="value">blue-cat</parameter>
</minimax:tool_call>'''
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "record_fact"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

    def test_minimax_fragment_without_parameter_value_is_not_fabricated(self, parser):
        """Do not fake a tool call when only a tag prefix is present."""
        text = '''<minimax:tool_call>
<invoke name="record_fact">
<parameter name="'''
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert not result.content

    def test_empty_invoke(self, parser):
        """Test invoke with no parameters."""
        text = """<minimax:tool_call>
<invoke name="get_current_time">
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_current_time"
        assert result.tool_calls[0]["arguments"] == "{}"

    def test_streaming_no_tool_call(self, parser):
        """Test streaming with no tool call markers."""
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello world",
            delta_text="Hello world",
        )
        assert result == {"content": "Hello world"}

    def test_streaming_tool_call_complete(self, parser):
        """Test streaming when tool call block completes."""
        full_text = """<minimax:tool_call>
<invoke name="test">
<parameter name="a">1</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls_streaming(
            previous_text="<minimax:tool_call>\n<invoke name=\"test\">\n<parameter name=\"a\">1</parameter>\n</invoke>\n",
            current_text=full_text,
            delta_text="</minimax:tool_call>",
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "test"

    def test_streaming_accumulating(self, parser):
        """Test streaming while accumulating tool call content."""
        result = parser.extract_tool_calls_streaming(
            previous_text="<minimax:tool_call>",
            current_text="<minimax:tool_call>\n<invoke",
            delta_text="\n<invoke",
        )
        assert result is None  # Still accumulating

    def test_tool_call_id_uniqueness(self, parser):
        """Test that each tool call gets a unique ID."""
        text = """<minimax:tool_call>
<invoke name="func1">
<parameter name="a">1</parameter>
</invoke>
<invoke name="func2">
<parameter name="b">2</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        ids = [tc["id"] for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"

    def test_numeric_parameter(self, parser):
        """Test parameter values that are numbers."""
        text = """<minimax:tool_call>
<invoke name="calculate">
<parameter name="x">42</parameter>
<parameter name="y">3.14</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["x"] == 42
        assert args["y"] == 3.14

    def test_boolean_parameter(self, parser):
        """Test parameter values that are booleans."""
        text = """<minimax:tool_call>
<invoke name="set_option">
<parameter name="enabled">true</parameter>
<parameter name="verbose">false</parameter>
</invoke>
</minimax:tool_call>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["enabled"] is True
        assert args["verbose"] is False


    def test_detects_minimax(self, parser):
        """Test auto detection of MiniMax format."""
        text = '''<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Paris</parameter>
</invoke>
</minimax:tool_call>'''
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test with empty input."""
        parsers = [
            MistralToolParser(),
            QwenToolParser(),
            LlamaToolParser(),
            DeepSeekToolParser(),
            AutoToolParser(),
        ]
        for parser in parsers:
            result = parser.extract_tool_calls("")
            assert not result.tools_called

    def test_malformed_json(self):
        """Test with malformed JSON."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "func", "arguments": {invalid json}]'
        result = parser.extract_tool_calls(text)
        # Should not crash, may or may not parse

    def test_nested_arguments(self):
        """Test with deeply nested arguments."""
        parser = AutoToolParser()
        args = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        text = f'{{"name": "complex", "arguments": {json.dumps(args)}}}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        parsed_args = json.loads(result.tool_calls[0]["arguments"])
        assert parsed_args["level1"]["level2"]["level3"] == [1, 2, 3]

    def test_unicode_in_arguments(self):
        """Test with unicode characters in arguments."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "translate", "arguments": {"text": "日本語"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語"

    def test_special_characters_in_name(self):
        """Test function names with special characters."""
        parser = LlamaToolParser()
        text = '<function=get_user_info>{"user_id": 123}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_user_info"

    def test_tool_call_id_uniqueness(self):
        """Test that each tool call gets a unique ID."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        ids = [tc["id"] for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"


class TestStreamingParsing:
    """Test streaming tool call parsing."""

    def test_mistral_streaming(self):
        """Test Mistral streaming parsing."""
        parser = MistralToolParser()

        # Simulate streaming
        result1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me",
            delta_text="Let me",
        )
        assert result1 == {"content": "Let me"}

        result2 = parser.extract_tool_calls_streaming(
            previous_text="Let me",
            current_text="Let me[TOOL_CALLS]",
            delta_text="[TOOL_CALLS]",
        )
        # Should start tool call parsing

    def test_auto_streaming(self):
        """Test auto parser streaming."""
        parser = AutoToolParser()

        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello world",
            delta_text="Hello world",
        )
        assert result == {"content": "Hello world"}


class TestThinkTagStripping:
    """Test <think> tag stripping in tool parsers (Issue #26)."""

    def test_strip_think_tags_utility(self):
        """Test the strip_think_tags static method."""
        from vmlx_engine.tool_parsers.abstract_tool_parser import ToolParser

        # Basic stripping
        text = "<think>Let me analyze this</think>The answer is 42"
        assert ToolParser.strip_think_tags(text) == "The answer is 42"

        # Multi-line thinking
        text = "<think>Step 1\nStep 2\nStep 3</think>Result"
        assert ToolParser.strip_think_tags(text) == "Result"

        # No think tags
        text = "Just regular text"
        assert ToolParser.strip_think_tags(text) == "Just regular text"

        # Empty think tags
        text = "<think></think>Content"
        assert ToolParser.strip_think_tags(text) == "Content"

    def test_hermes_with_think_tags(self):
        """Test Hermes parser strips think tags before parsing tool calls."""
        parser = HermesToolParser()

        # Model output with think tags AND tool call (Ring-Mini-Linear-2.0 style)
        output = """<think>Let me search for that information.</think>
<tool_call>{"name": "search", "arguments": {"query": "weather"}}</tool_call>"""

        result = parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_qwen_with_think_tags(self):
        """Test Qwen parser strips think tags before parsing tool calls."""
        parser = QwenToolParser()

        # Model output with think tags AND tool call
        output = """<think>I need to get the weather data.</think>
[Calling tool: get_weather({"city": "Tokyo"})]"""

        result = parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_think_tags_with_no_tool_call(self):
        """Test that think tags are stripped even when no tool call is present."""
        parser = HermesToolParser()

        output = "<think>Let me think about this</think>The answer is 42."
        result = parser.extract_tool_calls(output)

        assert result.tools_called is False
        assert result.content == "The answer is 42."


def test_non_string_tool_name_is_not_a_tool_call():
    """A JSON tool name that is not a string must not produce a tool call.

    `if name:` is a truthiness check, so `{"name": 12345}` passed it. The call
    was appended with an int name, response validation then nulled it, and the
    turn emitted a tool call whose function name was None: a phantom call that
    consumes a tool iteration and dispatches nothing. Rejecting and keeping the
    text as content is the safe behaviour.

    Sites that slice the name out of a regex match are already strings and are
    deliberately not covered here.
    """
    from vmlx_engine.tool_parsers.abstract_tool_parser import (
        ToolParserManager,
        is_valid_tool_name,
    )

    assert is_valid_tool_name("read_file")
    assert not is_valid_tool_name(12345)
    assert not is_valid_tool_name(None)
    assert not is_valid_tool_name("")
    assert not is_valid_tool_name("   ")

    malformed = '<tool_call>{"name": 12345, "arguments": {}}</tool_call>'
    valid = '<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'

    for parser_name in ("hermes", "qwen", "qwen3", "nous"):
        parser = ToolParserManager.get_tool_parser(parser_name)(None)

        rejected = parser.extract_tool_calls(malformed, None)
        assert not rejected.tools_called, (
            f"{parser_name} accepted a non-string tool name"
        )
        assert not rejected.tool_calls

        accepted = parser.extract_tool_calls(valid, None)
        assert accepted.tools_called, f"{parser_name} rejected a valid tool call"
