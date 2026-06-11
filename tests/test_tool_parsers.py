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

    def test_streaming_xml_empty_required_args_fail_closed(self, parser):
        """Streaming parse must keep request schema validation."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ]
        }
        text = (
            "Checking what's in /tmp.\n"
            '<tool_call>{"name": "exec_command", "arguments": {}}</tool_call>'
        )

        result = parser.extract_tool_calls_streaming(
            "",
            text,
            "</tool_call>",
            request=request,
        )

        assert result is None

    def test_streaming_xml_required_args_preserved(self, parser):
        """Valid required arguments should still stream as a tool call."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ]
        }
        text = (
            "Checking what's in /tmp.\n"
            '<tool_call>{"name": "exec_command", '
            '"arguments": {"cmd": "ls /tmp"}}</tool_call>'
        )

        result = parser.extract_tool_calls_streaming(
            "",
            text,
            "</tool_call>",
            request=request,
        )

        assert result is not None
        assert result["tool_calls"][0]["function"]["name"] == "exec_command"
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args == {"cmd": "ls /tmp"}

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

    def test_plain_tool_name_then_argument_preserves_spacing_and_newlines(
        self, parser
    ):
        """Schema-gated Qwen plain-line fallback must not rewrite strings."""
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

        result = parser.extract_tool_calls(
            "  bash  \n  printf '<日本語>' && pwd  \nnext  ",
            request=request,
        )

        assert result.tools_called
        assert result.content is None
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"command": "  printf '<日本語>' && pwd  \nnext  "}

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

    def test_xml_empty_arguments_with_required_schema_fails_closed(self, parser):
        text = '<tool_call>{"name": "exec_command", "arguments": {}}</tool_call>'
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(text, request=request)

        assert not result.tools_called
        assert result.tool_calls == []

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

    def test_empty_function_with_required_schema_fails_closed(self, parser):
        text = "<tool_call><function=exec_command></function></tool_call>"
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                }
            ]
        }

        result = parser.extract_tool_calls(text, request=request)

        assert not result.tools_called
        assert result.tool_calls == []

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
        assert "<minimax:tool_call>" in result.content

    def test_bare_invoke_parameter_block(self, parser):
        """Live MiniMax can emit a complete invoke without the outer wrapper."""
        text = '''<invoke name="record_fact">
<parameter name="value">blue-cat</parameter>
</invoke>'''
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0]["name"] == "record_fact"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"value": "blue-cat"}

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
