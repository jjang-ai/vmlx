# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from vmlx_engine.tool_parsers import ToolParserManager
from vmlx_engine.model_config_registry import get_model_config_registry


def call(value, name="code"):
    return f'<function name="write"><param name="{name}">{value}</param></function>'


def parse(text, properties=None, flat=False):
    fn = {"name": "write", "parameters": {"type": "object", "properties": properties or {}}}
    tool = {"type": "function", **fn} if flat else {"type": "function", "function": fn}
    return ToolParserManager.get_tool_parser("minicpm5")(None).extract_tool_calls(text, {"tools": [tool]})


@pytest.mark.parametrize("value", ["007", "3.10", "true", "null", "  indent\n", "", "中文"])
def test_unknown_schema_is_string(value):
    result = parse(call(value))
    assert json.loads(result.tool_calls[0]["arguments"])["code"] == value


@pytest.mark.parametrize("flat", [False, True])
def test_cdata_preserves_code_and_control_literals(flat):
    value = '  if a < b:\r\n    print("<think>x</think></param></function>&")\n'
    result = parse(call(f"<![CDATA[{value}]]>"), {"code": {"type": "string"}}, flat)
    assert json.loads(result.tool_calls[0]["arguments"])["code"] == value


@pytest.mark.parametrize("value,kind,want", [
    ("7", "integer", 7), ("false", "boolean", False), ("wrong", "boolean", "wrong"),
    ("null", ["string", "null"], None), ("007", ["string", "null"], "007"),
    ('{"a":1}', "object", {"a": 1}), ("[1,2]", "array", [1, 2]),
    ("1.5", "integer", "1.5"), ("NaN", "number", "NaN"),
])
def test_types_come_from_schema(value, kind, want):
    result = parse(call(value), {"code": {"type": kind}})
    assert json.loads(result.tool_calls[0]["arguments"])["code"] == want


def test_reasoning_is_never_executable_and_calls_keep_order():
    result = parse("<think>Discuss " + call("bad") + "</think>before " + call("a") + call("b") + " after")
    assert [json.loads(x["arguments"])["code"] for x in result.tool_calls] == ["a", "b"]
    assert result.content == "before  after"
    assert len({x["id"] for x in result.tool_calls}) == 2


@pytest.mark.parametrize("text", [
    '<function name="write"><param name="code">cut',
    '<function name="write"><param name="code">a</param><param name="code">b</param></function>',
    '<function name="write"><param name="code">&evil;</param></function>',
    '<think>' + call("not executable"),
])
def test_malformed_or_reasoning_calls_not_repaired(text):
    assert not parse(text).tools_called


@pytest.mark.parametrize("dialect,parser,thinking", [("minicpm5_xml_function", "minicpm5", True), ("unrelated", "llama", False)])
def test_explicit_dialect_overrides_only_coarse_stamp(tmp_path, dialect, parser, thinking):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "capabilities": {"family": "llama", "tool_parser": "llama", "supports_thinking": False},
        "tool_calling": {"dialect": dialect},
    }))
    config = get_model_config_registry().lookup(str(tmp_path))
    assert config.tool_parser == parser
    assert config.supports_thinking is thinking
    assert config.think_in_template is False
    assert "default_enable_thinking" not in config.architecture_hints
