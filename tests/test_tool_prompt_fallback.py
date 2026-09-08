# SPDX-License-Identifier: Apache-2.0
"""Tool prompt fallback contracts.

These tests pin the native fallback examples that are injected when a model's
chat template drops tool schemas. The examples must not invent fake parameters:
models copy those examples, and a fake arg on a zero-argument tool corrupts
mixed built-in/MCP tool calls on DSV4.
"""

import json

import pytest

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools
from vmlx_engine.loaders.dsv4_chat_encoder import (
    prompt_tool_catalog,
    request_explicitly_requests_tool,
)


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


class NativeDSV4LikeTokenizer(DSV4LikeTokenizer):
    _vmlx_dsv4_chat_template_shim = True


def _dsv4_native_contract_prompt(tools, user_request):
    schemas = "\n".join(
        json.dumps(tool.get("function", tool), ensure_ascii=False)
        for tool in tools
    )
    return f"""## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name=\"$TOOL_NAME\">
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name=\"$TOOL_NAME2\">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
<｜User｜>{user_request}
<｜Assistant｜><think>"""


class PlainTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join((message.get("content") or "") for message in messages)


class OpenPanguLikeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        tools = kwargs.get("tools") or []
        rendered = ["<|pangu_text_start|><|message_start|>system"]
        if tools:
            rendered.append("<tools>")
            rendered.append(json.dumps(tools, ensure_ascii=False))
            rendered.append("</tools>")
            rendered.append(
                "<|tool_call_start|>\n"
                '[{"name": "<函数名1>", "arguments": <args1 json对象>}]\n'
                "<|tool_call_end|>"
            )
        rendered.append("<|message_end|>")
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                rendered.append(f"<|message_start|>system\n{content}<|message_end|>")
            elif role == "user":
                rendered.append(f"<|message_start|>user\n{content}<|message_end|>")
            elif role == "assistant":
                rendered.append(f"<|message_start|>assistant\n{content}<|message_end|>")
            elif role == "tool":
                rendered.append(f"<|message_start|>tool\n{content}<|message_end|>")
        rendered.append("<|message_start|>assistant\n")
        return "\n".join(rendered)


def _qwen_prompt() -> str:
    # The native Qwen template renders each requested tool with `tool | tojson`
    # — the full schema, not a name-only stub. The gate compares parameter
    # contracts, so the fixture renders what the template renders.
    entries = "\n".join(json.dumps(t, separators=(",", ":")) for t in _qwen_test_tools())
    return f"""<|im_start|>system
# Tools
<tools>
{entries}
</tools>
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
Use a tool.<|im_end|>
<|im_start|>assistant
""".strip()


def _qwen_test_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
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
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def _file_info_tool() -> list[dict]:
    return [
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


def test_openpangu_explicit_file_info_uses_native_json_list_with_bound_path():
    tools = _file_info_tool()
    user_request = (
        "Call the built-in file_info tool exactly once with path "
        "panel/package.json. After the tool result, reply exactly DONE."
    )
    prompt = OpenPanguLikeTokenizer().apply_chat_template(
        [{"role": "user", "content": user_request}],
        tools=tools,
    )

    injected = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        OpenPanguLikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="openpangu",
    )

    assert "native openPangu JSON-list shape" in injected
    assert "<tool_call>\n" not in injected
    assert '"name": "file_info"' in injected
    assert '"path": "panel/package.json"' in injected
    assert '"path": "VALUE_HERE"' not in injected


def test_openpangu_tool_result_continuation_finishes_visible_answer():
    tools = _file_info_tool()
    messages = [
        {
            "role": "user",
            "content": (
                "Call file_info with path panel/package.json. "
                "After the tool result, reply exactly PG2-FINAL-DONE and nothing else."
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_info",
                        "arguments": '{"path":"panel/package.json"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "Path: panel/package.json\nType: file"},
    ]
    prompt = OpenPanguLikeTokenizer().apply_chat_template(messages, tools=tools)

    injected = check_and_inject_fallback_tools(
        prompt,
        messages,
        tools,
        OpenPanguLikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="openpangu",
    )

    assert "Native openPangu tool-result continuation" in injected
    assert "Do not emit another <|tool_call_start|> block" in injected
    assert "Your next assistant message must be exactly: PG2-FINAL-DONE" in injected
    assert "<|message_start|>tool\nPath: panel/package.json" in injected


def test_qwen_explicit_run_exactly_binds_command_and_narrows_schema():
    tools = _qwen_test_tools()
    user_request = (
        "Use the run_command tool exactly once. Run exactly: "
        "printf B1TOOL > bonsai_native.txt . "
        "After the tool result, reply only: B1TOOL-OK"
    )

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="qwen",
    )

    # The shipped Qwen tool template already owns this auto-tool request.
    # Keep its native schema/result transcript instead of replacing it with a
    # second system instruction. Strict API tool_choice=required retains the
    # explicit fallback contract in test_tool_format.py.
    assert injected == _qwen_prompt()


def test_qwen_explicit_to_run_binds_command_and_narrows_schema():
    """The live Electron prompt commonly says ``to run:`` rather than
    ``run exactly:``; keep that command in the canonical Qwen example instead
    of teaching the model an empty required argument.
    """
    tools = _qwen_test_tools()
    user_request = (
        "Use the run_command tool exactly once to run: printf Q36_TOOL_811. "
        "After seeing the tool result, reply exactly Q36-TOOL=OK."
    )

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="qwen",
    )

    assert injected == _qwen_prompt()


def test_qwen_tool_result_continuation_prevents_duplicate_execution():
    tools = _qwen_test_tools()
    command = "printf B1TOOL > bonsai_native.txt"
    messages = [
        {
            "role": "user",
            "content": (
                "Use the run_command tool exactly once. Run exactly: "
                f"{command} . After the tool result, reply only: B1TOOL-OK"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_command", "arguments": {"command": command}}}
            ],
        },
        {"role": "tool", "content": "B1TOOL"},
    ]

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        messages,
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="qwen",
    )

    # The source-owned Qwen schema and native assistant/tool transcript are
    # sufficient. Replacing the stable opening schema after the tool result
    # would invalidate every later prefix-cache block in a growing agent loop.
    assert injected == _qwen_prompt()


def test_qwen_tool_result_continuation_allows_explicit_remaining_tool():
    tools = _qwen_test_tools()
    messages = [
        {
            "role": "user",
            "content": (
                "First call read_file exactly once with path panel/package.json. "
                "After that result, call run_command exactly once with command pwd. "
                "After both results, reply exactly MULTI-DONE and nothing else."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "package metadata"},
    ]

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        messages,
        [tools[0]],
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": [tools[0]]},
        tool_parser_id="qwen",
    )

    assert injected == _qwen_prompt()


def test_qwen_tool_result_then_followup_user_keeps_multi_tool_continuation():
    """Anthropic-flattened result + user text must retain the prior call."""
    tools = _qwen_test_tools()
    messages = [
        {
            "role": "user",
            "content": "Call read_file first, then call run_command.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "package metadata"},
        {
            "role": "user",
            "content": "Now call run_command exactly once with command pwd.",
        },
    ]

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        messages,
        [tools[0]],
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": [tools[0]]},
        tool_parser_id="qwen",
    )

    assert injected == _qwen_prompt()


def test_qwen_new_user_after_visible_assistant_answer_does_not_reuse_tool_state():
    tools = _qwen_test_tools()
    messages = [
        {"role": "user", "content": "Call read_file."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "package metadata"},
        {"role": "assistant", "content": "The file is present."},
        {"role": "user", "content": "Now call run_command with command pwd."},
    ]

    injected = check_and_inject_fallback_tools(
        _qwen_prompt(),
        messages,
        [tools[0]],
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": [tools[0]]},
        tool_parser_id="qwen",
    )

    assert "Native multi-tool continuation" not in injected
    assert "Completed tool(s): read_file" not in injected


def test_dsv4_prompt_catalog_keeps_full_set_when_latest_user_names_one_tool():
    """An explicit tool request must not shrink the rendered catalog.

    DSV4 renders the catalog into the system message. Dropping ``read_file``
    here would both move every later byte (killing prefix reuse) and hide an
    authorized tool from the model for the rest of the conversation.
    """
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_command"}},
    ]

    assert prompt_tool_catalog(tools) == tools


def test_dsv4_prompt_catalog_is_identical_across_an_agentic_transcript():
    """The catalog is turn-independent — the prefix-cache invariant."""
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_command"}},
        {"type": "function", "function": {"name": "write_file"}},
    ]
    transcripts = [
        [{"role": "user", "content": "Use run_command to list files."}],
        [
            {"role": "user", "content": "Use run_command to list files."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "run_command", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "app.py"},
        ],
        [
            {"role": "user", "content": "Use run_command to list files."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "wrote 7168 bytes"},
            {"role": "user", "content": "Add one to the previous number."},
        ],
    ]

    catalogs = [prompt_tool_catalog(tools) for _ in transcripts]

    assert all(catalog == tools for catalog in catalogs)


def test_dsv4_prompt_catalog_drops_duplicate_tool_names():
    """Duplicates waste prompt budget and let caller jitter move later bytes."""
    read_file = {"type": "function", "function": {"name": "read_file"}}
    run_command = {"type": "function", "function": {"name": "run_command"}}

    assert prompt_tool_catalog([read_file, run_command, dict(read_file)]) == [
        read_file,
        run_command,
    ]


def test_dsv4_explicit_tool_intent_ignores_negated_or_discussed_names():
    """Intent detection stays strict — it drives execution, not the prefix."""
    for request_text in (
        "Do not use file_info; explain how to inspect generation_config.json.",
        "Explain what FILE_INFO does without calling it.",
    ):
        assert not request_explicitly_requests_tool(request_text, "file_info")

    assert request_explicitly_requests_tool(
        "Use exactly one FILE_INFO tool call to inspect generation_config.json.",
        "file_info",
    )


def test_dsv4_complete_native_encoder_contract_is_not_replaced_by_exact_call():
    """The official grammar/schema must remain the production prompt owner."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "description": "Get metadata about a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    user_request = (
        "Use exactly one file_info tool call to inspect generation_config.json."
    )
    native_prompt = _dsv4_native_contract_prompt(tools, user_request)
    tokenizer = NativeDSV4LikeTokenizer()

    rendered = check_and_inject_fallback_tools(
        native_prompt,
        [{"role": "user", "content": user_request}],
        tools,
        tokenizer,
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "file_info"},
            },
        },
        tool_parser_id="dsml",
    )

    assert rendered == native_prompt
    assert "Copy the concrete parameter value" not in rendered
    assert "generation_config.json</｜DSML｜parameter>" not in rendered


def test_dsv4_native_contract_accepts_responses_flat_tool_schema():
    flat_tools = [
        {
            "type": "function",
            "name": "file_info",
            "description": "Get metadata.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    user_request = "Use file_info to inspect generation_config.json."
    native_prompt = _dsv4_native_contract_prompt(flat_tools, user_request)

    rendered = check_and_inject_fallback_tools(
        native_prompt,
        [{"role": "user", "content": user_request}],
        flat_tools,
        NativeDSV4LikeTokenizer(),
        {"tools": flat_tools, "tool_choice": {"type": "function", "name": "file_info"}},
        tool_parser_id="dsml",
    )

    assert rendered == native_prompt


def test_dsv4_native_contract_accepts_required_multi_schema_without_action_rail():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    user_request = "Use one available tool."
    native_prompt = _dsv4_native_contract_prompt(tools, user_request)

    rendered = check_and_inject_fallback_tools(
        native_prompt,
        [{"role": "user", "content": user_request}],
        tools,
        NativeDSV4LikeTokenizer(),
        {"tools": tools, "tool_choice": "required"},
        tool_parser_id="dsml",
    )

    assert rendered == native_prompt
    assert "<｜action｜>" not in rendered


def test_dsv4_native_contract_requires_exact_scoped_schema_match():
    selected = [
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
    extra = {
        "type": "function",
        "function": {
            "name": "run_command",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    user_request = "Use file_info for generation_config.json."
    exact_prompt = _dsv4_native_contract_prompt(selected, user_request)
    mismatched_prompt = _dsv4_native_contract_prompt(selected + [extra], user_request)
    kwargs = {
        "tools": selected,
        "tool_choice": {"type": "function", "function": {"name": "file_info"}},
    }

    exact = check_and_inject_fallback_tools(
        exact_prompt,
        [{"role": "user", "content": user_request}],
        selected,
        NativeDSV4LikeTokenizer(),
        kwargs,
        tool_parser_id="dsml",
    )
    mismatched = check_and_inject_fallback_tools(
        mismatched_prompt,
        [{"role": "user", "content": user_request}],
        selected,
        NativeDSV4LikeTokenizer(),
        kwargs,
        tool_parser_id="dsml",
    )

    assert exact == exact_prompt
    assert mismatched != mismatched_prompt
    assert "Tool: file_info" in mismatched


def test_dsv4_native_post_tool_prompt_remains_vendor_owned():
    tools = [
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
    messages = [
        {"role": "user", "content": "Use file_info for generation_config.json."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "generation_config.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": '{"size_bytes":185}'},
    ]
    native_prompt = _dsv4_native_contract_prompt(
        tools, "Use file_info for generation_config.json."
    )

    rendered = check_and_inject_fallback_tools(
        native_prompt,
        messages,
        tools,
        NativeDSV4LikeTokenizer(),
        {"tools": tools, "tool_choice": "auto"},
        tool_parser_id="dsml",
    )

    assert rendered == native_prompt
    assert "Native DSV4 tool-result continuation" not in rendered


def test_dsv4_prompt_catalog_survives_a_tool_result_continuation():
    """A continuation whose user turn stops naming the tool keeps every schema.

    Narrowing to the most recently called tool used to make this continuation
    render a different catalog than the turn before it, which pinned prefix
    reuse at the first schema byte for the rest of the conversation.
    """
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_command"}},
    ]

    assert prompt_tool_catalog(tools) == tools


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
    echo_block = injected.split('<｜DSML｜invoke name="smoke__echo">', 1)[1]
    echo_block = echo_block.split("</｜DSML｜invoke>", 1)[0]
    assert "<｜DSML｜parameter" not in echo_block
    assert "VALUE HERE" not in injected
    assert "- text (string, required)" in injected


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


def test_dsv4_explicit_run_command_binds_exact_request_value_without_placeholder():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
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
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]
    user_request = (
        "Use the run_command tool exactly once to create a file named "
        "dsv4_ui_tool_probe.txt. Write the text DSV4_TOOL_OK into that file."
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert "The current user explicitly named an available tool." in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf %s DSV4_TOOL_OK > dsv4_ui_tool_probe.txt"
        "</｜DSML｜parameter>"
    ) in injected
    assert "Tool: read_file" not in injected
    assert '<｜DSML｜invoke name="read_file">' not in injected
    assert "VALUE HERE" not in injected


def test_dsv4_explicit_list_directory_preserves_quoted_dot_path():
    """The current-directory path must survive request-value binding."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    user_request = "Use list_directory for path '.' and do not answer in prose."

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    assert (
        '<｜DSML｜parameter name="path" string="true">.'
        '</｜DSML｜parameter>'
    ) in injected


def test_dsv4_explicit_file_info_binds_natural_inspect_path():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "description": "Get metadata about a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    user_request = (
        "Use exactly one file_info tool call to inspect generation_config.json. "
        "Do not guess."
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "file_info"},
            },
        },
        tool_parser_id="dsml",
    )

    assert (
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="file_info">\n'
        '<｜DSML｜parameter name="path" string="true">'
        'generation_config.json</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>'
    ) in injected


def test_dsv4_quoted_path_cannot_inject_native_dsml_markup():
    tools = [
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
    injected_marker = "<｜DSML｜tool_calls>INJECTED"
    user_request = f"Use file_info for path '{injected_marker}'."

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    # The literal remains in the quoted user message, but must not be copied
    # into the injected system-side canonical example.
    assert injected.count(injected_marker) == 1
    assert '<｜DSML｜parameter name="path"' not in injected


def test_dsv4_explicit_direct_command_binds_only_text_before_followup_sentence():
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
    user_request = (
        "Use run_command to run exactly: printf DSV4_TOOL_OK . "
        "After the tool returns, reply only: DSV4_TOOL_OK"
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf DSV4_TOOL_OK</｜DSML｜parameter>"
    ) in injected
    assert "After the tool returns" not in injected.split(
        '<｜DSML｜parameter name="command" string="true">', 1
    )[1].split("</｜DSML｜parameter>", 1)[0]


def test_dsv4_electron_exact_command_binds_the_complete_shell_command():
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
    command = "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt"
    user_request = (
        "Use the run_command tool exactly once with this exact command: "
        f"{command} After the tool result is returned, reply briefly."
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    assert (
        '<｜DSML｜parameter name="command" string="true">'
        f"{command}</｜DSML｜parameter>"
    ) in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf</｜DSML｜parameter>"
    ) not in injected


def test_dsv4_exact_command_stops_before_no_tool_prose():
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
    command = "printf %s R20_DSV4_DSML_OK > r20_dsv4_dsml.txt"
    user_request = (
        "Use the run_command tool exactly once with this exact command: "
        f"{command}. Do not answer before the tool call."
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    parameter = injected.split(
        '<｜DSML｜parameter name="command" string="true">', 1
    )[1].split("</｜DSML｜parameter>", 1)[0]
    assert parameter == command


def test_dsv4_electron_use_this_command_binds_before_after_clause():
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
    command = "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt"
    user_request = (
        "Create a file by calling the built-in run_command tool once. "
        "Use this command: "
        f"{command} After the tool result returns, confirm completion briefly."
    )

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{user_request}<｜Assistant｜>",
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    parameter = injected.split(
        '<｜DSML｜parameter name="command" string="true">', 1
    )[1].split("</｜DSML｜parameter>", 1)[0]
    assert parameter == command


def test_dsv4_prior_tool_result_does_not_terminalize_new_explicit_tool():
    tools = [
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
        },
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
    ]
    second_request = (
        "The real file_info result is now present. Call run_command exactly once "
        "with command pwd. Do not repeat file_info and do not answer yet."
    )
    messages = [
        {"role": "user", "content": "Call file_info with path panel/package.json."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "Path: panel/package.json"},
        {"role": "user", "content": second_request},
    ]

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{second_request}<｜Assistant｜>",
        messages,
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    assert "Native DSV4 tool-result continuation" not in injected
    assert "the requested tool already ran" not in injected
    assert 'Tool: run_command' in injected
    assert 'Tool: file_info' not in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "pwd</｜DSML｜parameter>"
    ) in injected
    assert injected.count('<｜DSML｜invoke name="run_command">') == 1


def test_dsv4_client_narrowed_remaining_tool_is_not_terminalized():
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
    current_request = "Execute this exact command: pwd"
    messages = [
        {"role": "user", "content": "Inspect the package metadata."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "Path: panel/package.json"},
        {"role": "user", "content": current_request},
    ]

    injected = check_and_inject_fallback_tools(
        f"<｜User｜>{current_request}<｜Assistant｜>",
        messages,
        tools,
        DSV4LikeTokenizer(),
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": tools,
            "tool_choice": {"type": "function", "name": "run_command"},
        },
        tool_parser_id="dsml",
    )

    assert "Native DSV4 tool-result continuation" not in injected
    assert "the requested tool already ran" not in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "pwd</｜DSML｜parameter>"
    ) in injected


def test_dsv4_fallback_preserves_recent_tool_schema_on_later_user_turn():
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
        },
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
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Use the run_command tool exactly once to create a file named "
                "z.txt. Write the text Z7 into that file."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "printf \"%s\" \"Z7\" > z.txt"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "$ printf Z7\n\nZ7"},
        {"role": "assistant", "content": "Z7"},
        {"role": "user", "content": "Add one to the previous number."},
    ]

    injected = check_and_inject_fallback_tools(
        "<｜User｜>Add one to the previous number.<｜Assistant｜>",
        messages,
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="dsml",
    )

    assert "Tool: run_command" in injected
    assert "The current user explicitly named an available tool." in injected
    assert '<｜DSML｜invoke name="run_command">' in injected
    assert (
        '<｜DSML｜parameter name="command" string="true">'
        "printf %s Z7 > z.txt</｜DSML｜parameter>"
    ) in injected
    assert "Tool: read_file" not in injected
    assert '<｜DSML｜invoke name="read_file">' not in injected


def test_dsv4_broad_catalog_does_not_duplicate_native_history_prompt():
    """A concrete DSML exemplar in history suppresses the duplicate fallback.

    The catalog is broad and stable, so history can only ever exemplify the
    tools that actually ran. One real exemplar demonstrates the grammar; the
    outer gate has already required every authorized name to be present.
    """
    tools = [
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
        },
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
        },
    ]
    messages = [
        {"role": "user", "content": "Call file_info for panel/package.json."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "Size: 5.2 KB"},
        {"role": "assistant", "content": "5.2 KB"},
        {"role": "user", "content": "Repeat the size without calling a tool."},
    ]
    # The catalog is broad and stable, so both authorized names are present,
    # but only the tool that actually ran can appear as a DSML exemplar.
    native_prompt = """
<｜begin▁of▁sentence｜>system
Available tools: file_info, read_file
<｜DSML｜tool_calls>
<｜DSML｜invoke name="file_info">
<｜DSML｜parameter name="path" string="true">panel/package.json</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
<｜User｜>Repeat the size without calling a tool.
<｜Assistant｜>
""".strip()

    checked = check_and_inject_fallback_tools(
        native_prompt,
        messages,
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    assert checked == native_prompt
    assert "Tool: read_file" not in checked


def test_dsv4_broad_catalog_keeps_request_bound_fallback_for_explicit_tool():
    tools = [
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
        },
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
        },
    ]
    user_request = "Call file_info exactly once with path panel/package.json."
    native_prompt = f"""
<｜begin▁of▁sentence｜>system
<｜DSML｜tool_calls>
<｜DSML｜invoke name="file_info">
<｜DSML｜parameter name="path" string="true"></｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
<｜User｜>{user_request}
<｜Assistant｜>
""".strip()

    checked = check_and_inject_fallback_tools(
        native_prompt,
        [{"role": "user", "content": user_request}],
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    assert checked != native_prompt
    assert "The current user explicitly named an available tool." in checked
    assert (
        '<｜DSML｜parameter name="path" string="true">panel/package.json'
        '</｜DSML｜parameter>'
    ) in checked
    assert "Tool: read_file" not in checked


def test_dsv4_new_explicit_tool_turn_does_not_reuse_prior_tool_result():
    tools = [
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
    messages = [
        {"role": "user", "content": "Call file_info with path panel/package.json."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "Path: panel/package.json"},
        {"role": "assistant", "content": "FIRST-DONE"},
        {
            "role": "user",
            "content": "Call file_info exactly once with path README.md.",
        },
    ]

    injected = check_and_inject_fallback_tools(
        "<｜User｜>Call file_info exactly once with path README.md.<｜Assistant｜>",
        messages,
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    current_turn = injected.rsplit("<｜User｜>", 1)[1]
    assert "earlier <tool_result> blocks" in current_turn
    assert "do not satisfy this request" in current_turn
    assert "Do not copy, fabricate, or output a <tool_result> block" in current_turn
    assert '<｜DSML｜invoke name="file_info">' in current_turn
    assert (
        '<｜DSML｜parameter name="path" string="true">README.md'
        '</｜DSML｜parameter>'
    ) in current_turn
    assert "panel/package.json" not in current_turn


def test_dsv4_tool_result_continuation_finishes_instead_of_calling_again():
    tools = [
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
    messages = [
        {
            "role": "user",
            "content": (
                "Call file_info exactly once with path README.md. After the tool "
                "result, reply exactly DSV4-HEAD2-DONE and nothing else."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "README.md"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "Path: README.md"},
    ]

    injected = check_and_inject_fallback_tools(
        "<｜User｜>Call file_info.<｜Assistant｜>",
        messages,
        tools,
        DSV4LikeTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="dsml",
    )

    assert "Native DSV4 tool-result continuation" in injected
    assert "requested tool already ran" in injected
    assert "Do not emit another <｜DSML｜tool_calls> block" in injected
    assert "Your next assistant message must be exactly: DSV4-HEAD2-DONE" in injected
    assert "Your next assistant output must be exactly one canonical DSML tool call" not in injected


def test_lfm2_fallback_for_file_request_forbids_content_only_pseudo_call():
    """LFM2 must not treat a JSON content blob as a tool call substitute.

    A live Responses UI run on LFM2.5 produced ``{"content": "..."}`` prose
    instead of the native Python-call-list shape for a natural file-create
    request. The fallback prompt must explicitly forbid that failure mode while
    still deriving the exact shell command from the user request.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    prompt = "<|im_start|>user\nCreate a file.<|im_end|>\n<|im_start|>assistant\n"
    user_request = (
        "Use the run_command tool exactly once to create a file named "
        "real_ui_tool_probe_1.txt in the configured working directory. "
        "Write the text REAL_UI_LIVE_TOOL_ONE into that file."
    )

    injected = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="lfm2",
    )

    assert "<|tool_call_start|>[run_command(command=" in injected
    assert "printf %s REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt" in injected
    assert 'Do not emit JSON such as {"content": "..."}' in injected


def test_lfm2_direct_run_exactly_binds_command_without_placeholder():
    """Direct shell requests must produce a concrete Liquid call example.

    The real Electron LFM2.5 turn copied the old ``VALUE_HERE`` example and
    executed ``:`` instead of the user's command.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    user_request = (
        "Use run_command exactly once to run exactly: printf LFM_TOOL_519 . "
        "After the tool result, reply exactly LFM-TOOL=OK."
    )

    injected = check_and_inject_fallback_tools(
        "user: run a command\nassistant:",
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert (
        "<|tool_call_start|>[run_command(command='printf LFM_TOOL_519')]"
        "<|tool_call_end|>"
    ) in injected
    assert "run_command(command='VALUE_HERE')" not in injected
    assert "After the tool result" not in injected.split(
        "<|tool_call_start|>", 1
    )[1].split("<|tool_call_end|>", 1)[0]


def test_lfm2_direct_file_info_binds_explicit_path_without_placeholder():
    """Named scalar parameters must not leave VALUE_HERE for LFM2 to copy."""
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
    user_request = (
        "Call the built-in file_info tool exactly once with path "
        "panel/package.json. After the tool result, reply exactly DONE."
    )

    injected = check_and_inject_fallback_tools(
        (
            "<|im_start|>system\n"
            "<|tool_call_start|>[file_info(path='VALUE_HERE')]"
            "<|tool_call_end|><|im_end|>\n"
            "<|im_start|>user\ninspect a file<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert (
        "<|tool_call_start|>[file_info(path='panel/package.json')]"
        "<|tool_call_end|>"
    ) in injected
    assert "For this request, file_info.path must be exactly: panel/package.json" in injected
    assert "file_info(path='VALUE_HERE')" not in injected


def test_lfm2_native_tool_schema_is_not_replaced_by_synthetic_fallback():
    """Liquid's trained native schema must remain the sole tool instruction."""
    tools = _file_info_tool()
    user_request = (
        "Call the built-in file_info tool exactly once with path "
        "panel/package.json."
    )
    prompt = (
        "<|im_start|>system\n"
        f"Today's date: 2026-07-24\n\nList of tools: {json.dumps(tools)}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_request}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    rendered = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert rendered == prompt
    assert "Liquid LFM2 native tools" not in rendered
    assert "<|tool_call_start|>" not in rendered


def test_lfm2_native_tool_schema_honors_required_tool_choice():
    """A native schema alone does not encode the API's required-call contract."""
    tools = _file_info_tool()
    user_request = "Call file_info with path panel/package.json before answering."
    prompt = (
        "<|im_start|>system\n"
        f"List of tools: {json.dumps(tools)}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_request}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    rendered = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": tools,
            "tool_choice": "required",
        },
        tool_parser_id="lfm2",
    )

    assert rendered != prompt
    assert "The current API request requires a tool call." in rendered
    assert "exactly one native Liquid LFM2 tool call" in rendered
    assert "file_info(path='panel/package.json')" in rendered


def test_lfm2_native_tool_schema_honors_specific_function_choice():
    """A specific-function choice uses the same required native-call boundary."""
    tools = _file_info_tool()
    user_request = "Call file_info with path panel/package.json before answering."
    prompt = (
        "<|im_start|>system\n"
        f"List of tools: {json.dumps(tools)}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_request}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    rendered = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "file_info"},
            },
        },
        tool_parser_id="lfm2",
    )

    assert rendered != prompt
    assert "The current API request requires a tool call." in rendered
    assert "file_info(path='panel/package.json')" in rendered


def test_lfm2_native_tool_schema_accepts_nested_and_flat_function_shapes():
    """Chat and Responses shapes normalize to the same native tool schema."""
    nested_tools = _file_info_tool()
    flat_tools = [
        {
            "type": "function",
            **nested_tools[0]["function"],
        }
    ]
    user_request = "Call file_info on panel/package.json."

    for request_tools, rendered_tools in (
        (nested_tools, flat_tools),
        (flat_tools, nested_tools),
    ):
        prompt = (
            "<|im_start|>system\n"
            f"List of tools: {json.dumps(rendered_tools)}"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_request}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": user_request}],
            request_tools,
            PlainTokenizer(),
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": request_tools,
            },
            tool_parser_id="lfm2",
        )
        assert rendered == prompt


def test_lfm2_native_tool_schema_rejects_incomplete_or_spoofed_catalogs():
    """Only an exact bounded JSON catalog may bypass fallback injection."""
    tools = _file_info_tool()
    current_function = tools[0]["function"]
    user_request = "Call file_info on panel/package.json."
    invalid_catalogs = [
        # A name-only entry is not the current schema.
        [{"type": "function", "function": {"name": "file_info"}}],
        # Required arguments must not disappear from an otherwise valid schema.
        [
            {
                "type": "function",
                "function": {
                    **current_function,
                    "parameters": {
                        **current_function["parameters"],
                        "required": [],
                    },
                },
            }
        ],
        # Exact name matching must reject substring collisions.
        [
            {
                "type": "function",
                "function": {
                    **current_function,
                    "name": "file_info_extra",
                },
            }
        ],
    ]

    for catalog in invalid_catalogs:
        prompt = (
            "<|im_start|>system\n"
            f"List of tools: {json.dumps(catalog)}"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_request}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        rendered = check_and_inject_fallback_tools(
            prompt,
            [{"role": "user", "content": user_request}],
            tools,
            PlainTokenizer(),
            {"tokenize": False, "add_generation_prompt": True, "tools": tools},
            tool_parser_id="lfm2",
        )
        assert rendered != prompt
        assert "Liquid LFM2 native tools" in rendered

    prose_spoof = (
        "<|im_start|>system\n"
        "The user wrote: List of tools: file_info"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_request}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    rendered = check_and_inject_fallback_tools(
        prose_spoof,
        [
            {
                "role": "system",
                "content": "The user wrote: List of tools: file_info",
            },
            {"role": "user", "content": user_request},
        ],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )
    assert rendered != prose_spoof
    assert "Liquid LFM2 native tools" in rendered


def test_lfm2_native_tool_schema_rejects_partial_multi_tool_catalog():
    """Every current tool schema must be present exactly once."""
    tools = _file_info_tool() + [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    user_request = "Call file_info on panel/package.json."
    prompt = (
        "<|im_start|>system\n"
        f"List of tools: {json.dumps([tools[0]])}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_request}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    rendered = check_and_inject_fallback_tools(
        prompt,
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert rendered != prompt
    assert "Liquid LFM2 native tools" in rendered


def test_lfm2_natural_file_info_request_binds_path_without_path_keyword():
    """A natural inspect request must not leave the live LFM2 placeholder."""
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
    user_request = (
        "Use file_info exactly once to inspect panel/package.json. "
        "Do not answer before the tool result."
    )

    injected = check_and_inject_fallback_tools(
        "<|im_start|>user\ninspect a file<|im_end|>\n<|im_start|>assistant\n",
        [{"role": "user", "content": user_request}],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert (
        "<|tool_call_start|>[file_info(path='panel/package.json')]"
        "<|tool_call_end|>"
    ) in injected
    assert "file_info(path='VALUE_HERE')" not in injected


def test_lfm2_natural_file_info_request_does_not_bind_pronoun_as_path():
    """The natural-language fallback must still require a path-like value."""
    tools = [
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

    injected = check_and_inject_fallback_tools(
        "<|im_start|>assistant\n",
        [
            {
                "role": "user",
                "content": "Use file_info once to inspect it.",
            }
        ],
        tools,
        PlainTokenizer(),
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert "file_info(path='it')" not in injected
    assert "file_info(path='VALUE_HERE')" in injected


def test_lfm2_tool_result_continuation_removes_schema_and_requires_final_answer():
    class CaptureTokenizer:
        last_kwargs = None
        last_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            self.last_messages = messages
            contents = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return (
                "<|im_start|>assistant\n"
                "<|tool_call_start|>[run_command(command='printf LFM_TOOL_519')]"
                "<|tool_call_end|><|im_end|>\n"
                "<|im_start|>tool\n"
                f"{contents}<|im_end|>\n<|im_start|>assistant\n"
            )

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
    messages = [
        {
            "role": "user",
            "content": (
                "Use run_command exactly once to run exactly: printf LFM_TOOL_519 . "
                "After the tool result, reply exactly LFM-TOOL=OK."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lfm",
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "printf LFM_TOOL_519"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_lfm", "content": "LFM_TOOL_519"},
    ]
    tokenizer = CaptureTokenizer()

    injected = check_and_inject_fallback_tools(
        "user: prior tool result\nassistant:",
        messages,
        tools,
        tokenizer,
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    assert '[{"result": "LFM_TOOL_519"}]' in injected
    assert injected.count("<|tool_call_start|>[run_command") == 1
    assert tokenizer.last_kwargs["tools"] == tools
    assert tokenizer.last_messages[-1]["content"] == '[{"result": "LFM_TOOL_519"}]'


def test_lfm2_shell_tool_result_continuation_uses_structured_json():
    class CaptureTokenizer:
        last_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_messages = messages
            contents = "\n".join(
                str(message.get("content", "")) for message in messages
            )
            return (
                "<|im_start|>assistant\n"
                "<|tool_call_start|>[run_command(command='pwd')]"
                "<|tool_call_end|><|im_end|>\n"
                "<|im_start|>tool\n"
                f"{contents}<|im_end|>\n<|im_start|>assistant\n"
            )

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
    messages = [
        {"role": "user", "content": "Use run_command with command `pwd`."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lfm",
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "pwd"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_lfm",
            "content": "$ pwd\n\nExit code: 0\n\n/Users/example/mlx/vllm-mlx\n",
        },
    ]
    tokenizer = CaptureTokenizer()

    rendered = check_and_inject_fallback_tools(
        "user: prior tool result\nassistant:",
        messages,
        tools,
        tokenizer,
        {"tokenize": False, "add_generation_prompt": True, "tools": tools},
        tool_parser_id="lfm2",
    )

    expected = (
        '[{"exit_code": 0, "stdout": "/Users/example/mlx/vllm-mlx"}]'
    )
    assert expected in rendered
    assert tokenizer.last_messages[-1]["content"] == expected


def test_lfm2_responses_final_pass_normalizes_tool_result_without_current_tools():
    """tool_choice=none must not bypass prior LFM2 tool-result normalization."""

    class CaptureTokenizer:
        last_kwargs = None
        last_messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            self.last_messages = messages
            return "\n".join(
                f"{message.get('role')}: {message.get('content', '')}"
                for message in messages
            )

    messages = [
        {
            "role": "system",
            "content": (
                "After the tool result, reply exactly two lines: "
                "LFM-FINAL-DONE and FILE=panel/package.json SIZE=5.2 KB."
            ),
        },
        {"role": "user", "content": "Inspect panel/package.json with file_info."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lfm_file_info",
                    "function": {
                        "name": "file_info",
                        "arguments": {"path": "panel/package.json"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_lfm_file_info",
            "content": (
                "Path: panel/package.json\n"
                "Type: file\n"
                "Size: 5.2 KB\n"
                "Permissions: 0644"
            ),
        },
    ]
    tokenizer = CaptureTokenizer()

    rendered = check_and_inject_fallback_tools(
        "ORIGINAL RAW TOOL RESULT PROMPT",
        messages,
        None,
        tokenizer,
        {
            "tokenize": False,
            "add_generation_prompt": True,
        },
        tool_parser_id="lfm2",
    )

    expected = (
        '[{"result": "Path: panel/package.json\\nType: file\\n'
        'Size: 5.2 KB\\nPermissions: 0644"}]'
    )
    assert expected in rendered
    assert tokenizer.last_messages[-1]["content"] == expected
    assert "tools" not in tokenizer.last_kwargs
    assert "List of tools:" not in rendered
    assert "tool_choice=required" not in rendered


def test_lfm2_no_schema_final_pass_cannot_reintroduce_stale_template_tools():
    """An inconsistent caller must not re-enable stale schemas during rerender."""

    class CaptureTokenizer:
        last_kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.last_kwargs = kwargs
            return "\n".join(str(message.get("content", "")) for message in messages)

    stale_tools = [
        {
            "type": "function",
            "name": "file_info",
            "parameters": {"type": "object"},
        }
    ]
    tokenizer = CaptureTokenizer()

    rendered = check_and_inject_fallback_tools(
        "ORIGINAL",
        [
            {"role": "user", "content": "Use file_info."},
            {
                "role": "tool",
                "tool_call_id": "call_lfm",
                "content": "Path: panel/package.json\nSize: 5.2 KB",
            },
        ],
        None,
        tokenizer,
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": stale_tools,
        },
        tool_parser_id="lfm2",
    )

    assert '[{"result": "Path: panel/package.json\\nSize: 5.2 KB"}]' in rendered
    assert "tools" not in tokenizer.last_kwargs


def test_non_lfm_no_schema_tool_history_is_not_normalized_or_rerendered():
    """The no-schema early normalization path is strictly Liquid-scoped."""

    class RejectRerenderTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            raise AssertionError("non-LFM history must not be re-rendered")

    prompt = "QWEN ORIGINAL RAW TOOL RESULT PROMPT"
    rendered = check_and_inject_fallback_tools(
        prompt,
        [
            {"role": "user", "content": "Use file_info."},
            {
                "role": "tool",
                "tool_call_id": "call_qwen",
                "content": "Path: panel/package.json\nSize: 5.2 KB",
            },
        ],
        None,
        RejectRerenderTokenizer(),
        {"tokenize": False, "add_generation_prompt": True},
        tool_parser_id="qwen3",
    )

    assert rendered == prompt


def test_lfm2_responses_final_pass_preserves_existing_json_without_rerender():
    """An already-native LFM2 tool result should remain byte-stable."""

    class RejectRerenderTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            raise AssertionError("already-normalized history must not be re-rendered")

    prompt = (
        "<|im_start|>tool\n"
        '[{"result":"already native"}]'
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    messages = [
        {"role": "user", "content": "Use file_info."},
        {
            "role": "tool",
            "tool_call_id": "call_lfm",
            "content": '[{"result":"already native"}]',
        },
    ]

    rendered = check_and_inject_fallback_tools(
        prompt,
        messages,
        [],
        RejectRerenderTokenizer(),
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "tool_choice": "none",
        },
        tool_parser_id="liquid",
    )

    assert rendered == prompt


@pytest.mark.asyncio
async def test_chat_and_responses_tool_choice_template_contracts(monkeypatch):
    """Endpoint kwargs must mirror only required/specific tool choices to templates."""

    import copy
    from types import SimpleNamespace

    from fastapi import HTTPException

    import vmlx_engine.server as server
    from vmlx_engine.api.models import (
        ChatCompletionRequest,
        Message,
        ResponsesRequest,
    )
    from vmlx_engine.engine.base import GenerationOutput

    model_name = "liquid-ai/LFM2.5-8B-A1B"

    class FakeEngine:
        is_mllm = False
        preserve_native_tool_format = False
        tokenizer = SimpleNamespace(has_thinking=False)

    captures: list[dict] = []

    async def fake_await_chat(*_args, **kwargs):
        captures.append(copy.deepcopy(kwargs["chat_kwargs"]))
        return GenerationOutput(
            text="plain answer",
            raw_text="plain answer",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        )

    monkeypatch.setattr(server, "_engine", FakeEngine())
    monkeypatch.setattr(server, "_model_path", None)
    monkeypatch.setattr(server, "_model_name", model_name)
    monkeypatch.setattr(server, "_served_model_name", model_name)
    monkeypatch.setattr(server, "_model_type", "llm")
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_call_parser", "lfm2")
    monkeypatch.setattr(server, "_tool_call_parser_disabled_explicitly", False)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_default_timeout", 5.0)
    monkeypatch.setattr(
        server,
        "_await_chat_with_disconnect_abort",
        fake_await_chat,
    )

    chat_tools = [
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "description": "Inspect a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    response_tools = [
        {
            "type": "function",
            "name": "file_info",
            "description": "Inspect a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]

    with pytest.raises(HTTPException):
        await server.create_chat_completion(
            ChatCompletionRequest(
                model=model_name,
                messages=[Message(role="user", content="Inspect the file.")],
                tools=chat_tools,
                tool_choice="required",
                chat_template_kwargs={
                    "sentinel": "chat-required",
                    "tool_choice": "none",
                },
            ),
            fastapi_request=None,
        )
    chat_required = captures[-1]
    assert chat_required["tool_choice"] == "required"
    assert chat_required["chat_template_kwargs"] == {
        "sentinel": "chat-required",
        "tool_choice": "required",
    }
    assert len(chat_required["tools"]) == 1

    await server.create_chat_completion(
        ChatCompletionRequest(
            model=model_name,
            messages=[Message(role="user", content="Answer directly.")],
            tools=chat_tools,
            tool_choice="none",
            chat_template_kwargs={
                "sentinel": "chat-none",
                "tool_choice": "required",
            },
        ),
        fastapi_request=None,
    )
    chat_none = captures[-1]
    assert chat_none["tool_choice"] == "none"
    assert chat_none["chat_template_kwargs"] == {"sentinel": "chat-none"}
    assert not chat_none.get("tools")

    await server.create_chat_completion(
        ChatCompletionRequest(
            model=model_name,
            messages=[Message(role="user", content="Use a tool if needed.")],
            tools=chat_tools,
            tool_choice="auto",
            chat_template_kwargs={
                "sentinel": "chat-auto",
                "tool_choice": "required",
            },
        ),
        fastapi_request=None,
    )
    chat_auto = captures[-1]
    assert chat_auto["tool_choice"] == "auto"
    assert chat_auto["chat_template_kwargs"] == {"sentinel": "chat-auto"}
    assert len(chat_auto["tools"]) == 1

    specific_choice = {"type": "function", "name": "file_info"}
    with pytest.raises(HTTPException):
        await server.create_response(
            ResponsesRequest(
                model=model_name,
                input="Inspect the file.",
                tools=response_tools,
                tool_choice=specific_choice,
                chat_template_kwargs={
                    "sentinel": "responses-specific",
                    "tool_choice": "none",
                },
            ),
            fastapi_request=None,
        )
    responses_specific = captures[-1]
    assert responses_specific["tool_choice"] == specific_choice
    assert responses_specific["chat_template_kwargs"] == {
        "sentinel": "responses-specific",
        "tool_choice": specific_choice,
    }
    assert len(responses_specific["tools"]) == 1

    await server.create_response(
        ResponsesRequest(
            model=model_name,
            input="Answer directly.",
            tools=response_tools,
            tool_choice="none",
            chat_template_kwargs={
                "sentinel": "responses-none",
                "tool_choice": "required",
            },
        ),
        fastapi_request=None,
    )
    responses_none = captures[-1]
    assert responses_none["tool_choice"] == "none"
    assert responses_none["chat_template_kwargs"] == {
        "sentinel": "responses-none"
    }
    assert not responses_none.get("tools")
