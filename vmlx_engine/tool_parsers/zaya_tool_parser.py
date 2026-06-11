# SPDX-License-Identifier: Apache-2.0
"""ZAYA/Zyphra XML tool-call parser."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from html import unescape
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
        r"<parameter=([^>]+)>([\s\S]*?)(?=(?:</parameter>)?\s*<parameter=|</function>|$)",
        re.DOTALL,
    )
    VALUE_WRAPPER_PATTERN = re.compile(
        r"^<value>(.*?)</value>$",
        re.DOTALL,
    )

    def _get_param_schema(
        self,
        func_name: str,
        param_name: str,
        request: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not request:
            return None
        tools = request.get("tools") or []
        for tool in tools:
            if isinstance(tool, dict):
                nested = tool.get("function")
                func = nested if isinstance(nested, dict) else tool
            else:
                nested = getattr(tool, "function", None)
                func = nested if nested is not None else tool
            if isinstance(func, dict):
                name = func.get("name")
                parameters = func.get("parameters", {})
            else:
                name = getattr(func, "name", None)
                parameters = getattr(func, "parameters", {})
            if name != func_name:
                continue
            props = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            schema = props.get(param_name)
            return schema if isinstance(schema, dict) else None
        return None

    @classmethod
    def _clean_parameter_value(cls, value: str) -> str:
        value = re.sub(r"</parameter>\s*$", "", value, flags=re.DOTALL)
        wrapped = cls.VALUE_WRAPPER_PATTERN.match(value.strip())
        if wrapped:
            value = wrapped.group(1)
        return unescape(value)

    @staticmethod
    def _trim_wrapper_newlines_for_string(value: str) -> str:
        stripped = value.strip()
        if (
            ("\n" in value or "\r" in value)
            and "\n" not in stripped
            and "\r" not in stripped
        ):
            return stripped
        return value

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
                    param_name = param_name.strip()
                    value = self._clean_parameter_value(param_value)
                    schema = self._get_param_schema(func_name.strip(), param_name, request)
                    if schema is not None and schema.get("type", "string") == "string":
                        value = self._trim_wrapper_newlines_for_string(value)
                    try:
                        arguments[param_name] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[param_name] = value
                name = func_name.strip()
                if not self._arguments_satisfy_required_schema(
                    name, arguments, request
                ):
                    continue
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name,
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
