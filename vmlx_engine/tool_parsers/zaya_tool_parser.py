# SPDX-License-Identifier: Apache-2.0
"""ZAYA/Zyphra XML tool-call parser."""

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


@ToolParserManager.register_module(["zaya_xml", "zaya", "zyphra"])
class ZayaToolParser(ToolParser):
    """Parse ZAYA's native Zyphra XML tool-call format.

    Format:
    <zyphra_tool_call>
    <function=name>
    <parameter=arg>
    value
    </parameter>
    </function>
    </zyphra_tool_call>
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(
        r"<zyphra_tool_call>\s*(.*?)\s*</zyphra_tool_call>",
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

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if "<zyphra_tool_call>" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[dict[str, Any]] = []
        for block in self.TOOL_CALL_PATTERN.findall(model_output):
            for func_name, body in self.FUNCTION_PATTERN.findall(block):
                arguments: dict[str, Any] = {}
                for param_name, param_value in self.PARAM_PATTERN.findall(body):
                    value = param_value.strip()
                    try:
                        arguments[param_name.strip()] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[param_name.strip()] = value
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
        if "<zyphra_tool_call>" not in current_text:
            return {"content": delta_text}
        if "</zyphra_tool_call>" in delta_text:
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
