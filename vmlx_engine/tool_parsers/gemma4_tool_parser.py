# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vmlx-engine.

Handles Gemma 4's native tool calling format:
  <|tool_call>call:function_name{key:value,key:value}<tool_call|>

The format uses a custom key:value syntax (not JSON) inside braces.
Values can be quoted with <|"|>...</|"|> or unquoted.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)

# Gemma 4 tool call markers
_STC = "<|tool_call>"    # start-of-tool-call
_ETC = "<tool_call|>"    # end-of-tool-call

# Pattern to extract tool calls: <|tool_call>call:name{...}<tool_call|>
_TOOL_CALL_PATTERN = re.compile(
    r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)

# Gemma 4 escape token for string quoting
_ESCAPE_OPEN = '<|"|>'
_ESCAPE_CLOSE = '<|"|>'


def _parse_gemma4_args(args_str: str) -> dict[str, Any]:
    """Parse Gemma 4's key:value argument format into a dict.

    Format: key1:value1,key2:value2
    Values may be:
    - Quoted strings: <|"|>text<|"|>
    - Nested objects: {key:value}
    - Arrays: [item1,item2]
    - Numbers/booleans: unquoted
    """
    if not args_str.strip():
        return {}

    result: dict[str, Any] = {}

    # Try to parse as JSON first (some models may output JSON-like format)
    try:
        # Convert key:value format to JSON
        json_str = _convert_to_json(args_str)
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: simple key:value parsing
    # Split on commas that aren't inside braces/brackets/quotes
    depth = 0
    current = ""
    pairs: list[str] = []

    for char in args_str:
        if char in ('{', '['):
            depth += 1
            current += char
        elif char in ('}', ']'):
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            pairs.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        pairs.append(current.strip())

    for pair in pairs:
        colon_idx = pair.find(':')
        if colon_idx > 0:
            key = pair[:colon_idx].strip()
            value = pair[colon_idx + 1:].strip()
            # Strip Gemma 4 quote tokens
            value = value.replace(_ESCAPE_OPEN, '').replace(_ESCAPE_CLOSE, '')
            # Try to parse value as JSON
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                result[key] = value
        else:
            # Can't parse — skip
            continue

    return result


def _convert_to_json(args_str: str) -> str:
    """Attempt to convert Gemma 4 key:value format to JSON."""
    s = args_str
    # Replace Gemma 4 escape tokens with double quotes
    s = s.replace(_ESCAPE_OPEN, '"').replace(_ESCAPE_CLOSE, '"')
    # Add quotes around unquoted keys (word before colon)
    s = re.sub(r'(\b\w+)\s*:', r'"\1":', s)
    # Wrap in braces if not already
    s = s.strip()
    if not s.startswith('{'):
        s = '{' + s + '}'
    return s


@ToolParserManager.register_module(["gemma4"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Supports Gemma 4's native tool call format:
    - <|tool_call>call:function_name{key:value,key:value}<tool_call|>

    Also falls back to Hermes format (<tool_call>JSON</tool_call>) and
    raw JSON format for compatibility.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Also match Hermes format as fallback
    HERMES_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Gemma 4 model output."""
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        # Strip think/channel tags
        cleaned_text = self._strip_thought_channel(cleaned_text)
        cleaned_text = self.strip_think_tags(cleaned_text)

        # Parse Gemma 4 native format: <|tool_call>call:name{args}<tool_call|>
        matches = _TOOL_CALL_PATTERN.findall(cleaned_text)
        for name, args_str in matches:
            arguments = _parse_gemma4_args(args_str)
            if name:
                tool_calls.append({
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                })

        if matches:
            cleaned_text = _TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        # Fallback: try Hermes format
        if not tool_calls:
            hermes_matches = self.HERMES_PATTERN.findall(cleaned_text)
            for match in hermes_matches:
                try:
                    data = json.loads(match)
                    name = data.get("name", "")
                    arguments = data.get("arguments", {})
                    if name:
                        tool_calls.append({
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        })
                except json.JSONDecodeError:
                    continue
            if hermes_matches:
                cleaned_text = self.HERMES_PATTERN.sub("", cleaned_text).strip()

        # Strip residual special tokens
        cleaned_text = self._clean_special_tokens(cleaned_text)

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=cleaned_text
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract tool calls from streaming output."""
        # Check for Gemma 4 native tool call end marker
        if _STC in current_text and _ETC in delta_text:
            result = self.extract_tool_calls(current_text, request)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }
            return None

        # Check for Hermes format fallback
        if "<tool_call>" in current_text and "</tool_call>" in delta_text:
            result = self.extract_tool_calls(current_text, request)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }
            return None

        return {"content": delta_text}

    @staticmethod
    def _strip_thought_channel(text: str) -> str:
        """Strip Gemma 4 thought channel markers from text."""
        # Remove <|channel>thought\n...<channel|> blocks
        result = re.sub(
            r'<\|channel>thought\n.*?<channel\|>',
            '',
            text,
            flags=re.DOTALL,
        )
        # Also strip bare markers
        result = result.replace('<|channel>', '').replace('<channel|>', '')
        return result.strip()

    @staticmethod
    def _clean_special_tokens(text: str) -> str:
        """Clean residual Gemma 4 special tokens from text."""
        tokens_to_strip = [
            '<turn|>', '<|turn>', '<eos>',
            '<|tool_call>', '<tool_call|>',
            '<|tool_response>', '<tool_response|>',
            '<|tool>', '<tool|>',
            '<|channel>', '<channel|>',
        ]
        for token in tokens_to_strip:
            text = text.replace(token, '')
        return text.strip()
