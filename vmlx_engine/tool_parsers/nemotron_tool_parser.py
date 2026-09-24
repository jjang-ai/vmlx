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
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)
from .xml_function_tool_parser import XMLFunctionToolParser


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

    NATIVE_MARKERS = ("<tool_call>", "<function=")

    # Pattern for Nemotron-style with parameters
    TOOL_CALL_PATTERN = re.compile(
        r"(?:<tool_call>\s*)?<function=([^>]+)>(.*?)</function>\s*(?:</tool_call>)?",
        re.DOTALL,
    )

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        # Strip at most ONE framing newline per side. `\s*` ate the payload's
        # own leading indentation, so a code argument came back with its first
        # line unindented and later lines intact — a SyntaxError once written
        # to disk. Same defect as the qwen dialect (9df8c1660).
        r"<parameter=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

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
                    json.loads(content)
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": content,
                        }
                    )
                    continue
                except json.JSONDecodeError:
                    pass

            # Parse parameter tags
            params = self.PARAM_PATTERN.findall(content)
            if params:
                arguments = {}
                properties = self._argument_properties(request, func_name)
                for param_name, param_value in params:
                    param_name = param_name.strip()
                    raw = XMLFunctionToolParser._unframe(param_value)
                    hint = properties.get(param_name)
                    if self._schema_is_string_or_null(hint):
                        arguments[param_name] = (
                            None if self._schema_allows_null(hint) and raw.strip().lower() in self._NULL_SPELLINGS
                            else raw
                        )
                        continue
                    # The native template applies ``| string`` to scalar
                    # None, including nullable numbers/booleans/containers.
                    # Resolve only an explicit non-string nullable hint;
                    # unresolved or mixed string schemas must not guess.
                    hinted_types = hint.get("type") if isinstance(hint, dict) else None
                    if (raw.strip() == "None"
                            and isinstance(hinted_types, (str, list, tuple))
                            and "string" not in hinted_types
                            and self._schema_allows_null(hint)):
                        arguments[param_name] = None
                        continue
                    # The native template stringifies scalar booleans as
                    # True/False, while mappings and sequences use JSON.
                    if raw.strip() in ("True", "False") and XMLFunctionToolParser._schema_is_boolean_or_null(hint):
                        arguments[param_name] = raw.strip() == "True"
                        continue
                    try:
                        arguments[param_name] = json.loads(raw.strip())
                    except json.JSONDecodeError:
                        arguments[param_name] = raw

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            else:
                # Raw content without parameter tags, or empty content
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
