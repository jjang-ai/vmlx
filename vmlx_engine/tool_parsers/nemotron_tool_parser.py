# SPDX-License-Identifier: Apache-2.0
"""
Nemotron tool call parser for vmlx-engine.

Handles NVIDIA Nemotron models' tool calling format:
- <tool_call><function=name><parameter=p>v</parameter></function></tool_call>

Supports Nemotron-3-Nano-30B-A3B and similar models.
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


@ToolParserManager.register_module(["nemotron", "nemotron3"])
class NemotronToolParser(ToolParser):
    """
    Tool call parser for NVIDIA Nemotron models.

    Supports Nemotron's tool call format:
    <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>

    Also supports JSON arguments:
    <tool_call><function=get_weather>{"city": "Paris"}</function></tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser nemotron are set.
    """

    # Pattern for Nemotron-style with parameters
    TOOL_CALL_PATTERN = re.compile(
        r"(?:<tool_call>\s*)?<function=([^>]+)>(.*?)</function>\s*(?:</tool_call>)?",
        re.DOTALL,
    )

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    @staticmethod
    def _coerce_parameter_value(value: str) -> Any:
        stripped = value.strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Pretty-printed template XML often wraps scalar values as
            # ``<parameter=x>\n.\n</parameter>``. Keep exact same-line string
            # payloads intact, but do not treat those wrapper newlines as part
            # of a one-line scalar argument.
            if ("\n" in value or "\r" in value) and "\n" not in stripped and "\r" not in stripped:
                return unescape(stripped)
            return unescape(value)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Nemotron model output.
        """
        if "<tool_call>" not in model_output and "<function=" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        cleaned_text = model_output

        matches = self.TOOL_CALL_PATTERN.findall(model_output)
        for func_name, content in matches:
            func_name = func_name.strip()

            # Try to parse content as JSON first
            content = content.strip()
            if content.startswith("{"):
                try:
                    arguments = json.loads(content)
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
                    arguments[param_name.strip()] = self._coerce_parameter_value(
                        param_value
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
        Extract tool calls from streaming Nemotron model output.
        """
        if "<tool_call>" not in current_text and "<function=" not in current_text:
            return {"content": delta_text}

        if "</tool_call>" in delta_text or "</function>" in delta_text:
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
