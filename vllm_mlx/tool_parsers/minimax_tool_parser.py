# SPDX-License-Identifier: Apache-2.0
"""
MiniMax tool call parser for vllm-mlx.

Handles MiniMax M2/M2.5 tool calling XML format:
- <minimax:tool_call>
    <invoke name="func_name">
      <parameter name="param1">value1</parameter>
      <parameter name="param2">value2</parameter>
    </invoke>
  </minimax:tool_call>

Supports MiniMax-M2.5 and MiniMax-M2 models.
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


def _extract_name(name_str: str) -> str:
    """Extract name from a possibly quoted string.

    Handles: name="func", name='func', name=func
    """
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or (
        name_str.startswith("'") and name_str.endswith("'")
    ):
        return name_str[1:-1]
    return name_str


def _convert_param_value(value: str) -> Any:
    """Convert a parameter value string to an appropriate Python type.

    Tries JSON parsing first (handles arrays, objects, numbers, booleans, null),
    falls back to raw string.
    """
    value = value.strip()
    if not value:
        return value

    # Try JSON parsing (handles null, true, false, numbers, objects, arrays)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


@ToolParserManager.register_module(["minimax", "minimax_m2"])
class MiniMaxToolParser(ToolParser):
    """
    Tool call parser for MiniMax M2/M2.5 models.

    Supports MiniMax's XML-based tool call format:
    <minimax:tool_call>
      <invoke name="get_weather">
        <parameter name="location">San Francisco</parameter>
        <parameter name="unit">celsius</parameter>
      </invoke>
    </minimax:tool_call>

    Multiple invocations can appear in a single <minimax:tool_call> block,
    and content text before the tool call block is preserved.

    Used when --enable-auto-tool-choice --tool-call-parser minimax are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Pattern to match entire <minimax:tool_call>...</minimax:tool_call> blocks
    TOOL_CALL_PATTERN = re.compile(
        r"<minimax:tool_call>(.*?)</minimax:tool_call>",
        re.DOTALL,
    )

    # Pattern to match <invoke name="func_name">...</invoke> within a block
    INVOKE_PATTERN = re.compile(
        r"<invoke name=([^>]+)>(.*?)</invoke>",
        re.DOTALL,
    )

    # Pattern to match <parameter name="key">value</parameter> within an invoke
    PARAM_PATTERN = re.compile(
        r"<parameter name=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    # Fallback: <func_name>content</func_name> (no invoke wrapper)
    XML_FUNC_PATTERN = re.compile(
        r"<([a-zA-Z_]\w*)>(.*?)</\1>",
        re.DOTALL,
    )

    # Fallback: func_name followed by JSON on next line
    FUNC_JSON_PATTERN = re.compile(
        r"([a-zA-Z_]\w*)\s*\n\s*(\{.*?\})\s*$",
        re.DOTALL,
    )

    # Fallback: func_name key="value" key=value (shell-style)
    FUNC_KV_PATTERN = re.compile(
        r'^([a-zA-Z_]\w*)\s+((?:\w+=\S+\s*)+)$',
        re.MULTILINE,
    )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from MiniMax model output.
        """
        # Strip <think> tags first (model uses interleaved thinking)
        cleaned_text = self.strip_think_tags(model_output)

        if "<minimax:tool_call>" not in cleaned_text:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

        tool_calls: list[dict[str, Any]] = []

        # Find all <minimax:tool_call> blocks
        for block_match in self.TOOL_CALL_PATTERN.finditer(cleaned_text):
            block_content = block_match.group(1).strip()

            # Strategy 1: <invoke name="func">...</invoke> (standard format)
            invoke_found = False
            for invoke_match in self.INVOKE_PATTERN.finditer(block_content):
                invoke_found = True
                func_name = _extract_name(invoke_match.group(1))
                invoke_content = invoke_match.group(2)

                # Extract parameters
                params = self.PARAM_PATTERN.findall(invoke_content)
                if params:
                    arguments = {}
                    for param_name, param_value in params:
                        clean_name = _extract_name(param_name)
                        # Strip leading/trailing newlines from values
                        clean_value = param_value.strip()
                        arguments[clean_name] = _convert_param_value(clean_value)

                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": json.dumps(
                                arguments, ensure_ascii=False
                            ),
                        }
                    )
                else:
                    # No parameter tags — try parsing content as JSON directly
                    raw_content = invoke_content.strip()
                    if raw_content:
                        try:
                            json.loads(raw_content)
                            tool_calls.append(
                                {
                                    "id": generate_tool_id(),
                                    "name": func_name,
                                    "arguments": raw_content,
                                }
                            )
                        except json.JSONDecodeError:
                            # Raw content without recognized format
                            tool_calls.append(
                                {
                                    "id": generate_tool_id(),
                                    "name": func_name,
                                    "arguments": json.dumps(
                                        {"raw": raw_content}, ensure_ascii=False
                                    ),
                                }
                            )
                    else:
                        # Empty invoke — no arguments
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": func_name,
                                "arguments": "{}",
                            }
                        )

            # Strategy 2: No <invoke> found — try fallback formats
            if not invoke_found and block_content:
                tc = self._parse_block_fallback(block_content)
                if tc:
                    tool_calls.append(tc)

        # Remove tool call blocks from text to get remaining content
        content_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_text if content_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

    def _parse_block_fallback(
        self, block_content: str
    ) -> dict[str, Any] | None:
        """
        Parse tool call from non-standard formats inside a <minimax:tool_call> block.

        Handles:
        1. <func_name>JSON</func_name> — XML tag with function name
        2. func_name\\nJSON — function name on one line, JSON on next
        3. func_name key="value" key=value — shell-style key=value
        4. Raw JSON with "name"/"arguments" keys
        """
        # 1. <func_name>{...}</func_name>
        xml_match = self.XML_FUNC_PATTERN.search(block_content)
        if xml_match:
            func_name = xml_match.group(1)
            content = xml_match.group(2).strip()
            if content:
                try:
                    json.loads(content)
                    return {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": content,
                    }
                except json.JSONDecodeError:
                    return {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(
                            {"raw": content}, ensure_ascii=False
                        ),
                    }

        # 2. func_name\n{...}
        func_json_match = self.FUNC_JSON_PATTERN.search(block_content)
        if func_json_match:
            func_name = func_json_match.group(1)
            args_json = func_json_match.group(2).strip()
            try:
                json.loads(args_json)
                return {
                    "id": generate_tool_id(),
                    "name": func_name,
                    "arguments": args_json,
                }
            except json.JSONDecodeError:
                pass

        # 3. func_name key="value" key=value (shell-style)
        kv_match = self.FUNC_KV_PATTERN.search(block_content)
        if kv_match:
            func_name = kv_match.group(1)
            kv_str = kv_match.group(2).strip()
            arguments = {}
            for kv in re.finditer(r'(\w+)=(?:"([^"]*?)"|(\S+))', kv_str):
                key = kv.group(1)
                value = kv.group(2) if kv.group(2) is not None else kv.group(3)
                arguments[key] = _convert_param_value(value)
            if arguments:
                return {
                    "id": generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }

        # 4. Raw JSON object with name/arguments structure
        try:
            obj = json.loads(block_content)
            if isinstance(obj, dict):
                if "name" in obj:
                    args = obj.get("arguments", obj.get("parameters", {}))
                    return {
                        "id": generate_tool_id(),
                        "name": obj["name"],
                        "arguments": json.dumps(args, ensure_ascii=False)
                        if isinstance(args, dict)
                        else str(args),
                    }
                # JSON with just the arguments (no function name) — can't parse
        except (json.JSONDecodeError, ValueError):
            pass

        return None

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
        """
        Extract tool calls from streaming MiniMax model output.
        """
        # No tool call markers yet — pass through as content
        if "<minimax:tool_call>" not in current_text:
            return {"content": delta_text}

        # Tool call block just completed — parse the full accumulated text
        if "</minimax:tool_call>" in delta_text:
            result = self.extract_tool_calls(current_text)
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

        # Still accumulating tool call content — suppress output
        return None
