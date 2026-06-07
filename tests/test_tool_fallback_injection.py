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
    """MiMo XML-function templates should not be rewritten as Step/JSON fallback."""
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
    assert "You have access to MiMo XML function tools" in result
    assert result.index("You have access to MiMo XML function tools") < result.index("<|im_end|>")
    assert not result.startswith("You have access to MiMo XML function tools")


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
    assert "You have access to MiMo XML function tools" in result[:system_end]
    assert "<function=record_fact>" in result[:system_end]
    assert "<parameter=value>" in result[:system_end]
    assert "blue-cat" in result[:system_end]
    assert "You have access to MiMo XML function tools" not in result[user_start:]
