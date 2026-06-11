# SPDX-License-Identifier: Apache-2.0
"""Required-argument fail-closed coverage across tool parser families.

These tests pin the parser-level contract for the empty-arguments class of
agentic-loop failures: if a model emits a native tool envelope for a schema that
requires ``value`` but omits that field, the parser must not return an
executable ``{}`` argument payload. The server can then retry or fail closed for
``tool_choice=required`` instead of handing clients a malformed call.
"""

import json

import pytest

from vmlx_engine.tool_parsers.deepseek_tool_parser import DeepSeekToolParser
from vmlx_engine.tool_parsers.auto_tool_parser import AutoToolParser
from vmlx_engine.tool_parsers.dsml_tool_parser import DSMLToolParser
from vmlx_engine.tool_parsers.functionary_tool_parser import FunctionaryToolParser
from vmlx_engine.tool_parsers.gemma3_tool_parser import Gemma3ToolParser
from vmlx_engine.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
from vmlx_engine.tool_parsers.glm47_tool_parser import Glm47ToolParser
from vmlx_engine.tool_parsers.granite_tool_parser import GraniteToolParser
from vmlx_engine.tool_parsers.hermes_tool_parser import HermesToolParser
from vmlx_engine.tool_parsers.hunyuan_tool_parser import HunyuanToolParser
from vmlx_engine.tool_parsers.kimi_tool_parser import KimiToolParser
from vmlx_engine.tool_parsers.lfm2_tool_parser import Lfm2ToolParser
from vmlx_engine.tool_parsers.llama_tool_parser import LlamaToolParser
from vmlx_engine.tool_parsers.minimax_tool_parser import MiniMaxToolParser
from vmlx_engine.tool_parsers.mistral_tool_parser import MistralToolParser
from vmlx_engine.tool_parsers.qwen_tool_parser import QwenToolParser
from vmlx_engine.tool_parsers.step3p5_tool_parser import Step3p5ToolParser
from vmlx_engine.tool_parsers.xlam_tool_parser import xLAMToolParser
from vmlx_engine.tool_parsers.zaya_tool_parser import ZayaToolParser


RECORD_FACT_REQUEST = {
    "tools": [
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
}


@pytest.mark.parametrize(
    ("parser_cls", "missing_text", "valid_text"),
    [
        (
            QwenToolParser,
            '[Calling tool: record_fact({})]',
            '[Calling tool: record_fact({"value":"blue-cat"})]',
        ),
        (
            MistralToolParser,
            '[TOOL_CALLS] [{"name":"record_fact","arguments":{}}]',
            '[TOOL_CALLS] [{"name":"record_fact","arguments":{"value":"blue-cat"}}]',
        ),
        (
            HermesToolParser,
            '<tool_call>{"name":"record_fact","arguments":{}}</tool_call>',
            (
                '<tool_call>{"name":"record_fact","arguments":'
                '{"value":"blue-cat"}}</tool_call>'
            ),
        ),
        (
            FunctionaryToolParser,
            "<function=record_fact>{}</function>",
            '<function=record_fact>{"value":"blue-cat"}</function>',
        ),
        (
            LlamaToolParser,
            "<function=record_fact>{}</function>",
            '<function=record_fact>{"value":"blue-cat"}</function>',
        ),
        (
            GraniteToolParser,
            '<|tool_call|>[{"name":"record_fact","arguments":{}}]',
            (
                '<|tool_call|>[{"name":"record_fact","arguments":'
                '{"value":"blue-cat"}}]'
            ),
        ),
        (
            xLAMToolParser,
            '[{"name":"record_fact","arguments":{}}]',
            '[{"name":"record_fact","arguments":{"value":"blue-cat"}}]',
        ),
        (
            Lfm2ToolParser,
            "<|tool_call_start|>[record_fact()]<|tool_call_end|>",
            "<|tool_call_start|>[record_fact(value='blue-cat')]<|tool_call_end|>",
        ),
        (
            KimiToolParser,
            (
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>record_fact:0"
                "<|tool_call_argument_begin|>{}"
                "<|tool_call_end|><|tool_calls_section_end|>"
            ),
            (
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>record_fact:0"
                '<|tool_call_argument_begin|>{"value":"blue-cat"}'
                "<|tool_call_end|><|tool_calls_section_end|>"
            ),
        ),
        (
            HunyuanToolParser,
            (
                "<tool_calls><tool_call>record_fact<tool_sep>"
                "</tool_call></tool_calls>"
            ),
            (
                "<tool_calls><tool_call>record_fact<tool_sep>"
                "<arg_key>value</arg_key><arg_value>blue-cat</arg_value>"
                "</tool_call></tool_calls>"
            ),
        ),
        (
            ZayaToolParser,
            (
                "<zyphra_tool_call><function=record_fact>"
                "</function></zyphra_tool_call>"
            ),
            (
                "<zyphra_tool_call><function=record_fact>"
                "<parameter=value>blue-cat</parameter>"
                "</function></zyphra_tool_call>"
            ),
        ),
        (
            Gemma4ToolParser,
            "<|tool_call>call:record_fact{}<tool_call|>",
            '<|tool_call>call:record_fact{value:<|"|>blue-cat<|"|>}<tool_call|>',
        ),
        (
            Gemma3ToolParser,
            "```tool_code\nrecord_fact()\n```",
            '```tool_code\nrecord_fact(value="blue-cat")\n```',
        ),
        (
            DeepSeekToolParser,
            (
                "<｜tool▁calls▁begin｜>\n"
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>record_fact\n"
                "```json\n{}\n```<｜tool▁call▁end｜>\n"
                "<｜tool▁calls▁end｜>"
            ),
            (
                "<｜tool▁calls▁begin｜>\n"
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>record_fact\n"
                '```json\n{"value":"blue-cat"}\n```<｜tool▁call▁end｜>\n'
                "<｜tool▁calls▁end｜>"
            ),
        ),
        (
            MiniMaxToolParser,
            (
                '<minimax:tool_call><invoke name="record_fact">'
                "</invoke></minimax:tool_call>"
            ),
            (
                '<minimax:tool_call><invoke name="record_fact">'
                '<parameter name="value">blue-cat</parameter>'
                "</invoke></minimax:tool_call>"
            ),
        ),
        (
            DSMLToolParser,
            (
                '<｜DSML｜invoke name="record_fact">'
                "</｜DSML｜invoke>"
            ),
            (
                '<｜DSML｜invoke name="record_fact">'
                '<｜DSML｜parameter name="value" string="true">'
                "blue-cat</｜DSML｜parameter>"
                "</｜DSML｜invoke>"
            ),
        ),
        (
            Glm47ToolParser,
            "<tool_call>record_fact\n</tool_call>",
            (
                "<tool_call>record_fact\n"
                "<arg_key>value</arg_key><arg_value>blue-cat</arg_value>"
                "</tool_call>"
            ),
        ),
        (
            Step3p5ToolParser,
            "<tool_call><function=record_fact></function></tool_call>",
            (
                "<tool_call><function=record_fact>"
                "<parameter=value>blue-cat</parameter>"
                "</function></tool_call>"
            ),
        ),
        (
            AutoToolParser,
            "<tool_call><function=record_fact></function></tool_call>",
            (
                "<tool_call><function=record_fact>"
                "<parameter=value>blue-cat</parameter>"
                "</function></tool_call>"
            ),
        ),
    ],
)
def test_required_arguments_missing_fail_closed_but_valid_calls_survive(
    parser_cls, missing_text, valid_text
):
    parser = parser_cls(tokenizer=None)

    missing = parser.extract_tool_calls(missing_text, request=RECORD_FACT_REQUEST)
    assert missing.tools_called is False
    assert missing.tool_calls == []

    valid = parser.extract_tool_calls(valid_text, request=RECORD_FACT_REQUEST)
    assert valid.tools_called is True
    assert len(valid.tool_calls) == 1
    assert valid.tool_calls[0]["name"] == "record_fact"
    assert json.loads(valid.tool_calls[0]["arguments"]) == {"value": "blue-cat"}
