# SPDX-License-Identifier: Apache-2.0
"""
Tests for the chat template tool injection fallback.

Some models (like Qwen 3.5 without reasoning, or base models) have chat templates
that silently drop tool schemas. Our server detects this and forcibly injects them.
"""

import json
from unittest.mock import MagicMock

import pytest

from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools


@pytest.fixture
def mock_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "Lists a directory",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }
        }
    ]


@pytest.fixture
def mock_messages():
    return [{"role": "user", "content": "Hello, please list the directory."}]


def test_fallback_not_triggered_when_tools_present(mock_tools, mock_messages):
    """If the original chat template outputs the tool names, fallback is skipped."""
    mock_tokenizer = MagicMock()
    # The prompt ALREADY contains ALL tools
    original_prompt = "<system>Tools: read_file, list_directory</system><user>Hello</user>"
    
    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=mock_messages,
        template_tools=mock_tools,
        tokenizer=mock_tokenizer,
        template_kwargs={}
    )
    
    # Needs to return original prompt
    assert result == original_prompt
    # Must NOT re-apply template
    mock_tokenizer.apply_chat_template.assert_not_called()


def test_fallback_triggered_when_tools_missing(mock_tools, mock_messages):
    """If original prompt drops tools, fallback is triggered and re-applies template."""
    mock_tokenizer = MagicMock()
    
    # Original prompt completely silent on tools
    original_prompt = "<system>Hello</system><user>Hello</user>"
    
    # Mock the secondary template application
    def mock_apply(messages, **kwargs):
        # Return the system prompt we built
        return messages[0]["content"] if messages[0]["role"] == "system" else ""
        
    mock_tokenizer.apply_chat_template.side_effect = mock_apply
    
    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=mock_messages,
        template_tools=mock_tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"tools": mock_tools}
    )
    
    # Must have triggered apply_chat_template again
    assert mock_tokenizer.apply_chat_template.call_count == 1
    
    # The tools should be removed from kwargs for the second pass
    _, call_kwargs = mock_tokenizer.apply_chat_template.call_args
    assert "tools" not in call_kwargs
    
    # The new prompt must contain the tools in XML format
    assert "You have access to the following tools:" in result
    assert "read_file" in result
    assert "list_directory" in result
    assert "<tool_call>" in result
    assert "FUNCTION_NAME" in result


def test_fallback_with_existing_system_message(mock_tools):
    """Fallback appends to existing system message instead of creating a new one."""
    messages = [
        {"role": "system", "content": "You are a helpful AI."},
        {"role": "user", "content": "Read a file."}
    ]
    mock_tokenizer = MagicMock()
    
    original_prompt = "You are a helpful AI. Read a file."
    
    def mock_apply(modified_messages, **kwargs):
        assert len(modified_messages) == 2
        assert modified_messages[0]["role"] == "system"
        return modified_messages[0]["content"]
        
    mock_tokenizer.apply_chat_template.side_effect = mock_apply
    
    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=messages,
        template_tools=mock_tools,
        tokenizer=mock_tokenizer,
        template_kwargs={}
    )
    
    # Verify original system message is preserved
    assert result.startswith("You are a helpful AI.\n\nYou are an expert assistant")
    assert "read_file" in result


def test_fallback_with_list_system_message_preserves_multimodal_content(mock_tools):
    """Multimodal templates may represent system/user content as content parts."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful AI."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read a file."},
            ],
        },
    ]
    mock_tokenizer = MagicMock()

    def mock_apply(modified_messages, **kwargs):
        assert len(modified_messages) == 2
        system_content = modified_messages[0]["content"]
        assert isinstance(system_content, list)
        assert system_content[0]["text"] == "You are a helpful AI."
        assert "You are an expert assistant" in system_content[-1]["text"]
        assert modified_messages[1]["content"][0]["type"] == "image"
        return "\n".join(part.get("text", "") for part in system_content)

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt="<|im_start|>user\nRead a file.<|im_end|>",
        messages=messages,
        template_tools=mock_tools,
        tokenizer=mock_tokenizer,
        template_kwargs={},
    )

    assert "You are a helpful AI." in result
    assert "read_file" in result


def test_fallback_skips_when_no_tools_requested(mock_messages):
    """If no tools were requested, fallback does nothing."""
    mock_tokenizer = MagicMock()
    
    result = check_and_inject_fallback_tools(
        prompt="prompt",
        messages=mock_messages,
        template_tools=[],  # Empty
        tokenizer=mock_tokenizer,
        template_kwargs={}
    )
    
    assert result == "prompt"
    mock_tokenizer.apply_chat_template.assert_not_called()
    
    result2 = check_and_inject_fallback_tools(
        prompt="prompt",
        messages=mock_messages,
        template_tools=None,  # None
        tokenizer=mock_tokenizer,
        template_kwargs={}
    )

    assert result2 == "prompt"
    mock_tokenizer.apply_chat_template.assert_not_called()


def test_fallback_triggered_when_one_tool_missing(mock_tools, mock_messages):
    """If prompt has one tool but not all, fallback must trigger."""
    mock_tokenizer = MagicMock()

    # Only 'read_file' is in the prompt, 'list_directory' is missing
    original_prompt = "<system>Tools: read_file</system><user>Hello</user>"

    def mock_apply(messages, **kwargs):
        return messages[0]["content"] if messages[0]["role"] == "system" else ""

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=mock_messages,
        template_tools=mock_tools,
        tokenizer=mock_tokenizer,
        template_kwargs={}
    )

    # Fallback should trigger because 'list_directory' is missing
    mock_tokenizer.apply_chat_template.assert_called_once()
    assert "list_directory" in result


def test_fallback_with_empty_name_tools(mock_messages):
    """Tools with empty function names should not cause errors."""
    mock_tokenizer = MagicMock()

    # Tools with empty name
    tools_with_empty = [
        {"type": "function", "function": {"name": "", "description": "No name"}},
    ]

    result = check_and_inject_fallback_tools(
        prompt="prompt",
        messages=mock_messages,
        template_tools=tools_with_empty,
        tokenizer=mock_tokenizer,
        template_kwargs={}
    )

    # Should return prompt unchanged (no valid tool names to check)
    assert result == "prompt"
    mock_tokenizer.apply_chat_template.assert_not_called()


def test_mimo_xml_function_native_template_does_not_trigger_step_fallback(mock_messages):
    """MiMo XML-function templates with concrete examples should be left alone."""
    mock_tokenizer = MagicMock()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_fact",
                "description": "Record a fact.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        },
    ]
    original_prompt = """
<|im_start|>system
You may call one or more functions to assist with the user query.
<tools>
<function>
<name>record_fact</name>
<description>Record a fact.</description>
</function>
</tools>
For each function call, return xml:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>
</tool_call>
Concrete example:
<tool_call>
<function=record_fact>
<parameter=value>blue-cat</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
Record blue-cat.
<|im_end|>
"""

    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=mock_messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"tools": tools},
        tool_parser_id="xml_function",
    )

    assert result == original_prompt
    mock_tokenizer.apply_chat_template.assert_not_called()


def test_mimo_xml_function_schema_only_read_tool_gets_required_filepath_shape():
    """Schema-only XML prompts need the required read(filePath) shape."""
    mock_tokenizer = MagicMock()
    messages = [
        {
            "role": "user",
            "content": (
                "File CardDisplayField.swift. "
                "Write only its review and do not modify contents."
            ),
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "The absolute path to the file or directory to read",
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 9007199254740991,
                            "description": "The line number to start reading from",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 9007199254740991,
                            "description": "The maximum number of lines to read",
                        },
                    },
                    "required": ["filePath"],
                },
            },
        },
    ]
    original_prompt = """
<|im_start|>system
You may call one or more functions to assist with the user query.
<tools>
<function>
<name>read</name>
<description>Read a file.</description>
<parameter>
<name>filePath</name>
<type>string</type>
</parameter>
</function>
</tools>
For each function call, return xml:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
File CardDisplayField.swift. Write only its review and do not modify contents.
<|im_end|>
"""

    def mock_apply(modified_messages, **_kwargs):
        system_text = modified_messages[0]["content"]
        user_text = modified_messages[1]["content"]
        return (
            "<|im_start|>system\n"
            f"{system_text}\n"
            "<tools>\n"
            "<function><name>read</name></function>\n"
            "</tools>\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>value_1</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_text}<|im_end|>\n"
        )

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"tools": tools},
        tool_parser_id="xml_function",
    )

    assert result != original_prompt
    assert "native XML function shape" in result
    assert "<function=read>" in result
    assert "<parameter=filePath>" in result
    assert "filePath (required)" in result
    assert "read parameters:" in result
    assert '"filePath":{"type":"string","description":"The absolute path' in result
    assert '"offset":{"type":"integer","description":"The line number' in result
    assert '"limit":{"type":"integer","description":"The maximum number' in result
    assert "read required: [\"filePath\"]" in result
    assert "Every name in a tool's required array must be emitted" in result
    assert "Never emit empty required tool calls such as <function=read></function>" in result
    assert "REQUIRED_filePath_VALUE" in result
    assert '"name": "FUNCTION_NAME"' not in result


def test_qwen_fallback_teaches_json_required_read_filepath_and_glob_first():
    """Qwen fallback must match its JSON parser and teach required read args."""
    mock_tokenizer = MagicMock()
    messages = [
        {
            "role": "user",
            "content": (
                "Review CardDisplayField.swift. "
                "Write only the review and do not modify files."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": "{\"filePath\":\"/missing/CardDisplayField.swift\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read",
            "content": "File not found: /missing/CardDisplayField.swift",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad_glob",
                    "type": "function",
                    "function": {
                        "name": "glob",
                        "arguments": "{\"pattern\":\":\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bad_glob",
            "content": "No files found",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_good_glob",
                    "type": "function",
                    "function": {
                        "name": "glob",
                        "arguments": "{\"pattern\":\"**/CardDisplayField.swift\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_good_glob",
            "content": (
                "/repo/Dependencies/"
                "Packages/iOS_Inspection/Sources/InspectionSwift/AuxEntities/"
                "CustomField/CardDisplayField.swift"
            ),
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "The absolute path to the file or directory to read",
                        },
                        "offset": {"type": "integer"},
                    },
                    "required": ["filePath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
    ]
    original_prompt = """
<|im_start|>system
# Tools
<tools>
<function><name>read</name></function>
<function><name>glob</name></function>
</tools>
For each function call, return xml:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
Review CardDisplayField.swift.
<|im_end|>
"""

    def mock_apply(modified_messages, **_kwargs):
        return modified_messages[0]["content"]

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt=original_prompt,
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"tools": tools},
        tool_parser_id="qwen",
    )

    assert result != original_prompt
    assert "Qwen JSON tool-call shape" in result
    assert (
        '"name":"read","arguments":{"filePath":'
        '"/repo/Dependencies/Packages/iOS_Inspection/Sources/InspectionSwift/AuxEntities/CustomField/CardDisplayField.swift"}'
    ) in result
    assert '"name":"glob","arguments":{"pattern":"**/CardDisplayField.swift"}' in result
    assert "REQUIRED_filePath_VALUE" not in result
    assert 'required: ["filePath"]' in result
    assert "Every required parameter listed above must be present" in result
    assert "parameters belong only inside the JSON arguments object" in result
    assert 'filePath/path must be an absolute path starting with "/" and must be present' in result
    assert "Relative paths are invalid even when they are relative to the current working directory" in result
    assert "Never shorten, trim, or remove the leading directory prefix" in result
    assert "call glob first with a pattern like **/filename" in result
    assert "the assistant output must start with <tool_call>" in result
    assert '"name":"TOOL_NAME","arguments":{"FIELD":"VALUE"}' in result
    assert "The latest tool result contains an absolute path" in result
    assert "copy this exact Qwen JSON tool call" in result
    assert "Copy the entire filePath byte-for-byte" in result
    assert '{"function=' not in result
    assert "<function=example_function_name>" not in result
    assert "<parameter=example_parameter_1>" not in result
    assert "<function=read>" not in result
    assert "<parameter=filePath>" not in result


def test_qwen_rendered_fallback_strips_legacy_xml_parameter_scaffold():
    from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

    class LegacyTokenizer:
        def __init__(self, rendered):
            self.rendered = rendered

        def apply_chat_template(self, _messages, **_kwargs):
            return self.rendered

    legacy_prompt = """
<|im_start|>system
# Tools
<tools>
<function><name>read</name></function>
</tools>
For each function call, return xml:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>
</tool_call>
<|im_end|>
<|im_start|>user
Read CardDisplayField.swift.
<|im_end|>
<|im_start|>assistant
"""
    messages = [{"role": "user", "content": "Read CardDisplayField.swift."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "The absolute path to the file",
                        }
                    },
                    "required": ["filePath"],
                },
            },
        }
    ]

    result = check_and_inject_fallback_tools(
        prompt=legacy_prompt,
        messages=messages,
        template_tools=tools,
        tokenizer=LegacyTokenizer(legacy_prompt),
        template_kwargs={"tools": tools},
        tool_parser_id="qwen",
    )

    assert "Qwen JSON tool-call shape" in result
    assert '"name":"read","arguments":{"filePath":"REQUIRED_filePath_VALUE"}' in result
    assert "<function=example_function_name>" not in result
    assert "<parameter=example_parameter_1>" not in result
    assert "return xml" not in result.lower()


def test_qwen_fallback_rewrites_legacy_xml_tool_call_history():
    from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

    class HistoryXmlTokenizer:
        def apply_chat_template(self, modified_messages, **_kwargs):
            return (
                modified_messages[0]["content"]
                + "\n<|im_start|>user\nReview CardDisplayField.swift.<|im_end|>\n"
                + "<|im_start|>assistant\n"
                + "<tool_call>\n"
                + "<function=glob>\n"
                + "<parameter=pattern>\n**/CardDisplayField.swift\n</parameter>\n"
                + "</function>\n"
                + "</tool_call><|im_end|>\n"
                + "<|im_start|>user\n"
                + "<tool_response>\n/repo/CardDisplayField.swift\n</tool_response>"
                + "<|im_end|>\n<|im_start|>assistant\n"
            )

    messages = [
        {"role": "user", "content": "Review CardDisplayField.swift."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_glob",
                    "type": "function",
                    "function": {
                        "name": "glob",
                        "arguments": "{\"pattern\":\"**/CardDisplayField.swift\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_glob",
            "content": "/repo/CardDisplayField.swift",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files.",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"],
                },
            },
        },
    ]
    prompt = (
        "<|im_start|>system\n"
        "Tools are available, but no parser-native examples are present."
        "<|im_end|>\n"
        "<|im_start|>user\nReview CardDisplayField.swift.<|im_end|>\n"
    )

    result = check_and_inject_fallback_tools(
        prompt=prompt,
        messages=messages,
        template_tools=tools,
        tokenizer=HistoryXmlTokenizer(),
        template_kwargs={"tools": tools},
        tool_parser_id="qwen",
    )

    assert "Qwen JSON tool-call shape" in result
    assert '"name":"glob","arguments":{"pattern":"**/CardDisplayField.swift"}' in result
    assert (
        '"name":"read","arguments":{"filePath":'
        '"/repo/CardDisplayField.swift"}'
    ) in result
    assert "<function=glob>" not in result
    assert "<parameter=pattern>" not in result


def test_qwen_fallback_prerenders_tool_call_history_as_json():
    from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

    class InspectingTokenizer:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, modified_messages, **_kwargs):
            self.messages = modified_messages
            return "\n".join(
                str(message.get("content") or "")
                for message in modified_messages
            )

    messages = [
        {"role": "user", "content": "Review CardDisplayField.swift."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "\n\n",
            "tool_calls": [
                {
                    "id": "call_glob",
                    "type": "function",
                    "function": {
                        "name": "glob",
                        "arguments": "{\"pattern\":\"**/CardDisplayField.swift\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_glob",
            "content": "/repo/CardDisplayField.swift",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"],
                },
            },
        },
    ]
    tokenizer = InspectingTokenizer()

    result = check_and_inject_fallback_tools(
        prompt="<|im_start|>system\nTools are available.<|im_end|>",
        messages=messages,
        template_tools=tools,
        tokenizer=tokenizer,
        template_kwargs={"tools": tools},
        tool_parser_id="qwen",
    )

    history_message = next(
        message
        for message in tokenizer.messages
        if message.get("role") == "assistant"
    )
    assert "tool_calls" not in history_message
    assert "reasoning_content" not in history_message
    assert (
        history_message["content"]
        == '<tool_call>\n{"name":"glob","arguments":{"pattern":"**/CardDisplayField.swift"}}\n</tool_call>'
    )
    assert (
        '<tool_call>\n{"name":"glob","arguments":{"pattern":"**/CardDisplayField.swift"}}\n</tool_call>'
        in result
    )
    assert "<function=glob>" not in result
    assert "<parameter=pattern>" not in result


def test_qwen_parser_id_forces_json_contract_even_when_tool_names_are_present():
    from vmlx_engine.api.tool_calling import check_and_inject_fallback_tools

    mock_tokenizer = MagicMock()
    messages = [{"role": "user", "content": "Read CardDisplayField.swift."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {
                            "type": "string",
                            "description": "The absolute path to the file",
                        }
                    },
                    "required": ["filePath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files.",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        },
    ]
    prompt = (
        "System instructions mention available tools by name: read and glob. "
        "No parser-native examples are present."
    )

    def mock_apply(modified_messages, **_kwargs):
        return modified_messages[0]["content"]

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt=prompt,
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"tools": tools},
        tool_parser_id="qwen",
    )

    assert result != prompt
    assert "Qwen JSON tool-call shape" in result
    assert '"name":"read","arguments":{"filePath":"REQUIRED_filePath_VALUE"}' in result
    assert '"name":"glob","arguments":{"pattern":"**/CardDisplayField.swift"}' in result
    assert "<parameter=" not in result
    assert '{"function=' not in result


def test_mimo_xml_function_fallback_matches_parser_dialect(mock_messages):
    """If MiMo fallback is needed, it must instruct native XML, not JSON."""
    mock_tokenizer = MagicMock()
    messages = [{"role": "user", "content": "Use record_fact with value blue-cat."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_fact",
                "description": "Record a fact.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        },
    ]

    def mock_apply(modified_messages, **kwargs):
        return modified_messages[0]["content"]

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt="<|im_start|>user\nRecord blue-cat.<|im_end|>",
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"tools": tools},
        tool_parser_id="xml_function",
    )

    assert "native XML function shape" in result
    assert "<tool_call>" in result
    assert "<function=record_fact>" in result
    assert "<parameter=value>" in result
    assert '"name": "FUNCTION_NAME"' not in result


def test_mimo_xml_function_direct_fallback_stays_inside_chatml_system():
    """If MiMo drops injected fallback messages, do not prefix outside ChatML."""
    mock_tokenizer = MagicMock()
    prompt = (
        "<|im_start|>system\n"
        "You are MiMo.<|im_end|>"
        "<|im_start|>user\n"
        "Use the record_fact tool exactly once with value blue-cat.<|im_end|>"
        "<|im_start|>assistant\n"
        "<think></think>"
    )
    mock_tokenizer.apply_chat_template.return_value = prompt
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_fact",
                "description": "Record a fact.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "Use the record_fact tool exactly once with value blue-cat.",
        },
    ]

    result = check_and_inject_fallback_tools(
        prompt=prompt,
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"add_generation_prompt": True, "enable_thinking": False},
        tool_parser_id="xml_function",
    )

    assert result.startswith("<|im_start|>system\nYou are MiMo.")
    assert "MiMo XML function tools" in result
    assert result.index("MiMo XML function tools") < result.index("<|im_end|>")
    assert not result.startswith("MiMo XML function tools")


def test_mimo_xml_function_fallback_prefers_chatml_system_turn():
    """MiMo XML fallback should teach tools in system scope, not user scope."""
    mock_tokenizer = MagicMock()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_fact",
                "description": "Record a fact.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "Use the record_fact tool exactly once with value blue-cat.",
        },
    ]

    def mock_apply(modified_messages, **kwargs):
        system = modified_messages[0]["content"]
        user = modified_messages[1]["content"]
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>"
            "<|im_start|>user\n"
            f"{user}<|im_end|>"
            "<|im_start|>assistant\n<think></think>"
        )

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt="<|im_start|>system\nYou are MiMo.<|im_end|>"
        "<|im_start|>user\nUse the record_fact tool exactly once with value blue-cat.<|im_end|>"
        "<|im_start|>assistant\n<think></think>",
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={"add_generation_prompt": True, "enable_thinking": False},
        tool_parser_id="xml_function",
    )

    system_end = result.index("<|im_end|>")
    user_start = result.index("<|im_start|>user")
    assert "MiMo XML function tools" in result[:system_end]
    assert "<function=record_fact>" in result[:system_end]
    assert "<parameter=value>" in result[:system_end]
    assert "blue-cat" in result[:system_end]
    assert "MiMo XML function tools" not in result[user_start:]


def test_mimo_xml_function_fallback_keeps_tool_schema_compact_for_tight_memory():
    """MiMo JANG_2L must not spend hundreds of tokens on one-tool scaffolding."""
    mock_tokenizer = MagicMock()
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
        },
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Call function record_fact with exactly these JSON arguments "
                'and no other value: {"value":"B7-CAT-09"}. The string '
                "B7-CAT-09 is a literal value; preserve every character."
            ),
        },
    ]

    def mock_apply(modified_messages, **kwargs):
        return (
            "<|im_start|>system\n"
            f"{modified_messages[0]['content']}<|im_end|>"
            "<|im_start|>user\n"
            f"{modified_messages[1]['content']}<|im_end|>"
            "<|im_start|>assistant\n"
        )

    mock_tokenizer.apply_chat_template.side_effect = mock_apply

    result = check_and_inject_fallback_tools(
        prompt="<|im_start|>user\nUse record_fact.<|im_end|><|im_start|>assistant\n",
        messages=messages,
        template_tools=tools,
        tokenizer=mock_tokenizer,
        template_kwargs={
            "add_generation_prompt": True,
            "enable_thinking": False,
            "tool_choice": "required",
        },
        tool_parser_id="xml_function",
    )

    system_text = result.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]
    assert len(system_text) <= 420
    assert "MiMo XML function tools" in system_text
    assert "<tool_call>" in system_text
    assert "<function=record_fact>" in system_text
    assert "<parameter=value>" in system_text
    assert '"name": "FUNCTION_NAME"' not in system_text
