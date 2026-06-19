# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for vmlx-engine.

Handles Qwen's tool calling formats:
- XML style: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket style: [Calling tool: func_name({"arg": "value"})]
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


@ToolParserManager.register_module(["qwen", "qwen3"])
class QwenToolParser(ToolParser):
    """
    Tool call parser for Qwen models.

    Supports multiple Qwen tool call formats:
    - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Bracket: [Calling tool: func_name({"arg": "value"})]

    Used when --enable-auto-tool-choice --tool-call-parser qwen are set.
    """

    # Pattern for XML-style: <tool_call>{"json"}</tool_call>
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # Pattern for bracket-style: [Calling tool: func_name({...})]
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    # Qwen3-Coder / Qwen3.6 XML function-parameter format (issue #192):
    #   <function=name><parameter=arg>value</parameter></function>
    FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
    PARAMETER_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    BARE_ARG_PATTERN = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</\1>", re.DOTALL)

    @classmethod
    def _plain_tool_line_call(
        cls, text: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        lines = [line.strip() for line in text.strip().splitlines()]
        lines = [line for line in lines if line]
        if len(lines) < 2:
            return None
        tool_name = lines[0]
        schema = cls._function_schema_for_tool(request, tool_name)
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            return None
        required_names = [name for name in required if isinstance(name, str)]
        if len(required_names) != 1:
            return None
        param_name = required_names[0]
        param_schema = properties.get(param_name)
        if not isinstance(param_schema, dict):
            return None
        param_type = param_schema.get("type")
        if param_type not in (None, "string"):
            return None
        value = "\n".join(lines[1:]).strip()
        if not value:
            return None
        return {
            "id": generate_tool_id(),
            "name": tool_name,
            "arguments": json.dumps({param_name: value}, ensure_ascii=False),
        }

    @staticmethod
    def _coerce_arg_value(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    @classmethod
    def _parse_function_blocks(cls, text: str) -> list[dict[str, Any]]:
        """Parse Qwen3-Coder XML tool format (issue #192).

        <function=name><parameter=p>v</parameter></function>; falls back to
        bare <p>v</p> params when no <parameter=> tags are present.
        """
        calls: list[dict[str, Any]] = []
        for func_name, body in cls.FUNCTION_PATTERN.findall(text):
            name = func_name.strip()
            if not name:
                continue
            arguments: dict[str, Any] = {}
            params = cls.PARAMETER_PATTERN.findall(body)
            if params:
                for pn, pv in params:
                    arguments[pn.strip()] = cls._coerce_arg_value(pv)
            else:
                for pn, pv in cls.BARE_ARG_PATTERN.findall(body):
                    arguments[pn.strip()] = cls._coerce_arg_value(pv)
            calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Qwen model response.
        """
        tool_calls = []

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(model_output)

        # Try bracket pattern first (Qwen3 style)
        bracket_matches = self.BRACKET_PATTERN.findall(cleaned_text)
        for name, args_str in bracket_matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                continue

        if bracket_matches:
            cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text).strip()

        # Try XML pattern (traditional Qwen style)
        xml_matches = self.XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": self._serialize_tool_arguments(
                                name, arguments, request
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if xml_matches:
            cleaned_text = self.XML_PATTERN.sub("", cleaned_text).strip()

        # Qwen3-Coder / Qwen3.6 XML function-parameter format (issue #192)
        if not tool_calls and "<function=" in cleaned_text:
            func_calls = self._parse_function_blocks(cleaned_text)
            if func_calls:
                tool_calls.extend(func_calls)
                cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text)
                cleaned_text = (
                    cleaned_text.replace("<tool_call>", "")
                    .replace("</tool_call>", "")
                    .strip()
                )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        plain_call = self._plain_tool_line_call(cleaned_text, request)
        if plain_call is not None:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[plain_call],
                content=None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        """
        Extract tool calls from streaming Qwen model output.
        """
        # Check for tool call markers
        has_tool_marker = (
            "<tool_call>" in current_text
            or "[Calling tool:" in current_text
            or "<function=" in current_text  # issue #192
        )

        if not has_tool_marker:
            return {"content": delta_text}

        # If we're in a tool call, accumulate and parse at the end
        # For simplicity, return None during accumulation
        if (
            "</tool_call>" in delta_text
            or ")]" in delta_text
            or "</function>" in delta_text  # issue #192
        ):
            # Tool call complete, parse the whole thing
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

        return None
