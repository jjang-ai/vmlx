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
    # The prompt ALREADY contains the tools
    original_prompt = "<system>Tools: read_file</system><user>Hello</user>"
    
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
