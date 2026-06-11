# SPDX-License-Identifier: Apache-2.0
"""
StepFun Step-3.5 tool call parser for vmlx-engine.

Handles StepFun's XML-based tool calling format:
- <tool_call><function=name><parameter=p>value</parameter></function></tool_call>

This is the same XML format used by Nemotron models, with additional
type coercion for parameter values (string → int/float/bool) based
on the tool schema provided in the request.
"""

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


@ToolParserManager.register_module(["step3p5", "stepfun"])
class Step3p5ToolParser(ToolParser):
    """
    Tool call parser for StepFun Step-3.5 models.

    Supports Step-3.5's XML tool call format:
    <tool_call>
    <function=get_weather>
    <parameter=city>Paris</parameter>
    <parameter=units>celsius</parameter>
    </function>
    </tool_call>

    Also supports JSON arguments inside function tags:
    <tool_call><function=get_weather>{"city": "Paris"}</function></tool_call>

    Features:
    - Parameter type coercion: converts string values to int/float/bool
      based on the tool schema in the request (matching vLLM step3p5 behavior)
    - Handles multi-line parameter values

    Used when --enable-auto-tool-choice --tool-call-parser step3p5 are set.
    """

    # Pattern for XML-style with <function=name>...</function> inside <tool_call>
    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )

    # Pattern to extract <parameter=name>value</parameter>
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    def _get_param_schema(
        self, func_name: str, param_name: str, request: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Look up parameter schema from request tools for type coercion."""
        if not request:
            return None
        tools = request.get("tools")
        if not tools:
            return None
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
            if name == func_name:
                props = (
                    parameters.get("properties", {})
                    if isinstance(parameters, dict)
                    else {}
                )
                return props.get(param_name)
        return None

    def _coerce_value(
        self, value: str, schema: dict[str, Any] | None
    ) -> Any:
        """
        Coerce a string parameter value to the correct type based on schema.

        Matches vLLM step3p5 parser behavior: converts string values to
        int, float, or bool when the schema specifies those types.
        """
        stripped = value.strip()
        if not schema:
            return unescape(value)

        param_type = schema.get("type", "string")

        if param_type == "integer":
            try:
                return int(stripped)
            except (ValueError, TypeError):
                return unescape(value)
        elif param_type == "number":
            try:
                return float(stripped)
            except (ValueError, TypeError):
                return unescape(value)
        elif param_type == "boolean":
            lower = stripped.lower()
            if lower in ("true", "1", "yes"):
                return True
            elif lower in ("false", "0", "no"):
                return False
            return unescape(value)
        elif param_type == "array" or param_type == "object":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return unescape(value)
        elif param_type == "string":
            if (
                ("\n" in value or "\r" in value)
                and "\n" not in stripped
                and "\r" not in stripped
            ):
                return unescape(stripped)

        return unescape(value)

    def _coerce_json_arguments(
        self,
        func_name: str,
        arguments: dict[str, Any],
        request: dict[str, Any] | None,
    ) -> dict[str, Any]:
        coerced = dict(arguments)
        for param_name, value in list(coerced.items()):
            if not isinstance(value, str):
                continue
            schema = self._get_param_schema(func_name, str(param_name), request)
            if not isinstance(schema, dict) or schema.get("type", "string") != "string":
                continue
            stripped = value.strip()
            if (
                ("\n" in value or "\r" in value)
                and "\n" not in stripped
                and "\r" not in stripped
            ):
                coerced[param_name] = unescape(stripped)
        return coerced

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from StepFun Step-3.5 model output.
        """
        # Strip think tags first (handles models with reasoning + tool calls)
        cleaned_output = self.strip_think_tags(model_output)

        if "<tool_call>" not in cleaned_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )

        tool_calls = []
        cleaned_text = cleaned_output

        matches = self.TOOL_CALL_PATTERN.findall(cleaned_output)
        for func_name, content in matches:
            func_name = func_name.strip()

            # Try to parse content as JSON first
            content = content.strip()
            if content.startswith("{"):
                try:
                    arguments = json.loads(content)
                    if isinstance(arguments, dict):
                        arguments = self._coerce_json_arguments(
                            func_name,
                            arguments,
                            request,
                        )
                    if not self._arguments_satisfy_required_schema(
                        func_name, arguments, request
                    ):
                        continue
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        }
                    )
                    continue
                except json.JSONDecodeError:
                    pass

            # Parse parameter tags
            params = self.PARAM_PATTERN.findall(content)
            if params:
                arguments = {}
                for param_name, param_value in params:
                    param_name = param_name.strip()

                    # Type coercion based on schema
                    schema = self._get_param_schema(func_name, param_name, request)
                    arguments[param_name] = self._coerce_value(param_value, schema)

                if not self._arguments_satisfy_required_schema(
                    func_name, arguments, request
                ):
                    continue
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            else:
                # Raw content without parameter tags, or empty content
                if not self._arguments_satisfy_required_schema(
                    func_name, content, request
                ):
                    continue
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": content if content else "{}",
                    }
                )

        # Clean the text
        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
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
        Extract tool calls from streaming StepFun model output.
        """
        if "<tool_call>" not in current_text:
            return {"content": delta_text}

        if "</tool_call>" in delta_text:
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
