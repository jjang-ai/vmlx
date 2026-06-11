# SPDX-License-Identifier: Apache-2.0
"""
Liquid LFM2 tool call parser for vmlx-engine.

LFM2 chat templates render assistant tool calls as a Python-call list:

<|tool_call_start|>[get_weather(city='Paris'), search(query='mlx')]<|tool_call_end|>
"""

import ast
import json
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)


@ToolParserManager.register_module(["lfm2", "liquid"])
class Lfm2ToolParser(ToolParser):
    """Parser for Liquid LFM2 Python-call-list tool calls."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_START = "<|tool_call_start|>"
    TOOL_CALL_END = "<|tool_call_end|>"

    def _strip_control_markers(self, text: str) -> str | None:
        content = (
            text.replace(self.TOOL_CALL_START, "")
            .replace(self.TOOL_CALL_END, "")
            .strip()
        )
        return content or None

    def _literal(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            if isinstance(node, ast.Name):
                return node.id
            return ast.unparse(node)

    def _call_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts: list[str] = [func.attr]
            value = func.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            return ".".join(reversed(parts)).split(".")[-1]
        return ast.unparse(func).split(".")[-1]

    def _parse_calls(
        self, payload: str, request: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        expr = ast.parse(payload.strip(), mode="eval").body
        nodes = expr.elts if isinstance(expr, (ast.List, ast.Tuple)) else [expr]
        tool_calls: list[dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            args: dict[str, Any] = {}
            for index, arg in enumerate(node.args):
                args[f"arg{index}"] = self._literal(arg)
            for keyword in node.keywords:
                if keyword.arg is None:
                    expanded = self._literal(keyword.value)
                    if isinstance(expanded, dict):
                        args.update(expanded)
                    continue
                args[keyword.arg] = self._literal(keyword.value)
            name = self._call_name(node.func).strip()
            if not self._arguments_satisfy_required_schema(name, args, request):
                continue
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
        return tool_calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned_output = self.strip_think_tags(model_output)
        start = cleaned_output.find(self.TOOL_CALL_START)
        if start < 0:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )
        payload_start = start + len(self.TOOL_CALL_START)
        end = cleaned_output.find(self.TOOL_CALL_END, payload_start)
        if end < 0:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._strip_control_markers(cleaned_output),
            )
        content = cleaned_output[:start].strip() or None
        payload = cleaned_output[payload_start:end]
        try:
            tool_calls = self._parse_calls(payload, request=request)
        except SyntaxError:
            tool_calls = []
        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._strip_control_markers(cleaned_output),
            )
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
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
        if self.TOOL_CALL_START not in current_text:
            return {"content": delta_text}
        if self.TOOL_CALL_END in delta_text:
            result = self.extract_tool_calls(current_text, request=request)
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
