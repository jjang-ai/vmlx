# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 tool call parser (<ernie:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></ernie:tool_call>)."""

from __future__ import annotations

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


def _strip_quotes(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _coerce_param(value: str, prop_schema: dict[str, Any] | None = None) -> Any:
    raw = value.strip()
    if prop_schema and isinstance(prop_schema, dict):
        expected = prop_schema.get("type")
        if expected == "string":
            return value
        if expected in ("integer", "number", "boolean", "array", "object"):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return value
    if not raw:
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            return value
        return parsed
    except (json.JSONDecodeError, ValueError):
        return value


@ToolParserManager.register_module(["ernie", "ernie5_5", "ernie5_5_moe"])
class ErnieToolParser(ToolParser):
    """Tool call parser for ERNIE 5.5 native XML invoke/parameter format."""

    NATIVE_MARKERS = ("<ernie:tool_call>", "</ernie:tool_call>")
    SUPPRESS_INVALID_NATIVE_MARKUP = True
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(
        r"<ernie:tool_call>(.*?)</ernie:tool_call>",
        re.DOTALL,
    )
    LENIENT_TOOL_CALL_PATTERN = re.compile(
        r"<ernie:tool_call>(.*?)(?:</ernie:tool_call>|$)",
        re.DOTALL,
    )
    INVOKE_PATTERN = re.compile(
        r"<invoke\s+name=([^>]+)>(.*?)</invoke>",
        re.DOTALL,
    )
    LENIENT_INVOKE_PATTERN = re.compile(
        r"<invoke\s+name=([^>]+)>(.*?)(?=(?:</invoke>|<invoke\s+name=|$))",
        re.DOTALL,
    )
    PARAM_PATTERN = re.compile(
        r"<parameter\s+name=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    def _schema_map(self, request: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if not isinstance(request, dict):
            return {}
        tools = request.get("tools") or []
        out: dict[str, dict[str, Any]] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", tool)
            name = fn.get("name")
            if isinstance(name, str) and name:
                params = fn.get("parameters") or {}
                props = params.get("properties") if isinstance(params, dict) else {}
                out[name] = props if isinstance(props, dict) else {}
        return out

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned = re.sub(
            r"<ernie:think>.*?</ernie:think>", "", model_output, flags=re.DOTALL
        )
        if "</ernie:think>" in cleaned and "<ernie:think>" not in cleaned:
            cleaned = cleaned.split("</ernie:think>", 1)[1]
        cleaned = self.strip_think_tags(cleaned)
        if "<ernie:tool_call>" not in cleaned:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned or None
            )
        schemas = self._schema_map(request)
        blocks = list(self.TOOL_CALL_PATTERN.finditer(cleaned))
        if not blocks:
            blocks = list(self.LENIENT_TOOL_CALL_PATTERN.finditer(cleaned))
        tool_calls: list[dict[str, Any]] = []
        for block in blocks:
            body = block.group(1)
            invokes = list(self.INVOKE_PATTERN.finditer(body))
            if not invokes:
                invokes = list(self.LENIENT_INVOKE_PATTERN.finditer(body))
            for inv in invokes:
                func_name = _strip_quotes(inv.group(1))
                if not func_name:
                    continue
                props = schemas.get(func_name, {})
                args: dict[str, Any] = {}
                for pm in self.PARAM_PATTERN.finditer(inv.group(2)):
                    pname = _strip_quotes(pm.group(1))
                    if pname:
                        args[pname] = _coerce_param(pm.group(2), props.get(pname))
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    }
                )
        content = self.LENIENT_TOOL_CALL_PATTERN.sub("", cleaned).strip()
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content or None,
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
        if "<ernie:tool_call>" not in current_text:
            # Hold trailing prefix of "<ernie:tool_call>" across chunk boundaries.
            marker = "<ernie:tool_call>"
            for k in range(min(len(marker) - 1, len(current_text)), 0, -1):
                if current_text.endswith(marker[:k]):
                    return None
            return {"content": delta_text} if delta_text else None
        if "</ernie:tool_call>" in current_text and "</ernie:tool_call>" not in previous_text:
            parsed = self.extract_tool_calls(current_text, request)
            if parsed.tools_called:
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
                        for i, tc in enumerate(parsed.tool_calls)
                    ]
                }
        return None
