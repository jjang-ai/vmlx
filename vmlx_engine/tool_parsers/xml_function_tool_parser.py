# SPDX-License-Identifier: Apache-2.0
"""Generic XML function-call parser.

Parses templates that emit:

<tool_call>
<function=name>
<parameter=arg>value</parameter>
</function>
</tool_call>
"""

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


@ToolParserManager.register_module(["xml_function", "mimo_xml_function"])
class XMLFunctionToolParser(ToolParser):
    """Parse XML function calls used by MiMo-style chat templates."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    VALUE_WRAPPER_PATTERN = re.compile(
        r"^<value>\s*(.*?)\s*</value>$",
        re.DOTALL,
    )

    @classmethod
    def _coerce_value(cls, value: str) -> Any:
        value = value.strip()
        wrapped = cls.VALUE_WRAPPER_PATTERN.match(value)
        if wrapped:
            value = wrapped.group(1).strip()
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        if "<tool_call>" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[dict[str, Any]] = []
        for block in self.TOOL_CALL_PATTERN.findall(model_output):
            for func_name, body in self.FUNCTION_PATTERN.findall(block):
                arguments: dict[str, Any] = {}
                for param_name, param_value in self.PARAM_PATTERN.findall(body):
                    arguments[param_name.strip()] = self._coerce_value(param_value)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )

        cleaned_text = self.TOOL_CALL_PATTERN.sub("", model_output).strip()
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
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
        if "<tool_call>" not in current_text:
            return {"content": delta_text}
        if "</tool_call>" in delta_text:
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
