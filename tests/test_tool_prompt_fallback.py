# SPDX-License-Identifier: Apache-2.0
"""Tool prompt fallback contracts.

These tests pin the native fallback examples that are injected when a model's
chat template drops tool schemas. The examples must not invent fake parameters:
models copy those examples, and a fake arg on a zero-argument tool corrupts
mixed built-in/MCP tool calls on DSV4.
"""

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools


class DSV4LikeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        rendered = []
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                rendered.append(content)
            elif role == "user":
                rendered.append(f"<｜User｜>{content}")
            elif role == "assistant":
                rendered.append(f"<｜Assistant｜>{content}")
        rendered.append("<｜Assistant｜>")
        return "\n".join(rendered)


def test_dsv4_fallback_does_not_invent_arg1_for_zero_arg_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "smoke__echo",
                "description": "Return the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]
    prompt = "<｜User｜>Use the tools.<｜Assistant｜>"

    injected = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": "Use the tools."}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert '<｜DSML｜invoke name="get_current_datetime">' in injected
    assert '<｜DSML｜parameter name="arg1"' not in injected
    datetime_block = injected.split('<｜DSML｜invoke name="get_current_datetime">', 1)[1]
    datetime_block = datetime_block.split("</｜DSML｜invoke>", 1)[0]
    assert "<｜DSML｜parameter" not in datetime_block
    assert '<｜DSML｜invoke name="smoke__echo">' in injected
    assert '<｜DSML｜parameter name="text" string="true">VALUE HERE</｜DSML｜parameter>' in injected


def test_dsv4_schema_only_prompt_gets_concrete_per_tool_examples():
    """DSV4's bundled encoder renders JSON schemas plus generic DSML syntax.

    Live DSV4 JANGTQ-K mixed a zero-argument built-in tool with a one-argument
    MCP tool under that schema-only prompt. The fallback must require concrete
    examples for each actual tool name, not accept the generic DSML block.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "smoke__echo",
                "description": "Return the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]
    schema_only = """
<｜begin▁of▁sentence｜>system
<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
{"name": "get_current_datetime", "parameters": {"type": "object", "properties": {}, "required": []}}
{"name": "smoke__echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}
<｜User｜>Use the tools.
<｜Assistant｜>
""".strip()

    injected = check_and_inject_fallback_tools(
        schema_only,
        [{"role": "user", "content": "Use the tools."}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert injected != schema_only
    assert '<｜DSML｜invoke name="get_current_datetime">' in injected
    assert '<｜DSML｜invoke name="smoke__echo">' in injected
