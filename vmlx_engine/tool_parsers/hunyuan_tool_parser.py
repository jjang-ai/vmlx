# SPDX-License-Identifier: Apache-2.0
"""Hunyuan / Tencent Hy3 tool-call parser.

Parses the Hunyuan/Tencent custom XML tool-call format used by
``model_type=hy_v3``. Contract source:

    ~/jang/jang-tools/examples/hy3/python_runtime/hy3_parser_contract.py
    ~/jang/docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md

Format::

    <tool_calls>
    <tool_call>NAME<tool_sep>
    <arg_key>K1</arg_key>
    <arg_value>V1</arg_value>
    <arg_key>K2</arg_key>
    <arg_value>V2</arg_value>
    ...
    </tool_call>
    <tool_call>NAME2<tool_sep>...</tool_call>
    </tool_calls>

Multiple ``<tool_call>`` blocks may appear inside a single
``<tool_calls>`` envelope. ``<tool_sep>`` separates the function name
from the argument body. Argument values are coerced via ``json.loads``
when possible (so ``"true"`` becomes ``True``, ``"42"`` becomes
``42``), otherwise kept as strings.

Reasoning is handled by the separate ``qwen3`` reasoning parser on
``<think>...</think>`` tags — this parser only touches tool calls.
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


@ToolParserManager.register_module(["hunyuan", "hy_v3", "tencent"])
class HunyuanToolParser(ToolParser):
    """Hunyuan/Tencent Hy3 tool-call XML parser."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALLS_PATTERN = re.compile(
        r"<tool_calls>\s*(.*?)\s*</tool_calls>",
        re.DOTALL,
    )
    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    ARG_KEY_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>",
        re.DOTALL,
    )
    ARG_VALUE_PATTERN = re.compile(
        r"<arg_value>\s*(.*?)\s*</arg_value>",
        re.DOTALL,
    )
    TOOL_SEP = "<tool_sep>"

    @staticmethod
    def _coerce(value: str) -> Any:
        """Match the contract's value coercion: try JSON, fall back to string."""
        if not value:
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        if "<tool_calls>" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[dict[str, Any]] = []
        for envelope in self.TOOL_CALLS_PATTERN.findall(model_output):
            for raw_call in self.TOOL_CALL_PATTERN.findall(envelope):
                if self.TOOL_SEP not in raw_call:
                    continue
                name, args_blob = raw_call.split(self.TOOL_SEP, 1)
                # Pair keys with values positionally — order matters in the
                # template emission, both lists should be the same length.
                keys = self.ARG_KEY_PATTERN.findall(args_blob)
                values = self.ARG_VALUE_PATTERN.findall(args_blob)
                arguments: dict[str, Any] = {}
                for key, value in zip(keys, values):
                    arguments[key.strip()] = self._coerce(value.strip())
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )

        cleaned_text = self.TOOL_CALLS_PATTERN.sub("", model_output).strip()
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
        # Stream visible content until we see the envelope close, then
        # extract the full block (mirrors zaya_tool_parser.py's approach).
        if "<tool_calls>" not in current_text:
            return {"content": delta_text}
        if "</tool_calls>" in delta_text:
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
