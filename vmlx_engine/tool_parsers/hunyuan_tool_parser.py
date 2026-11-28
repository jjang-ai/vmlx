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
from the argument body. The native template emits strings verbatim and JSON
encodes non-strings. Declared strings therefore retain their bytes; values with
non-string or ambiguous schemas retain the legacy JSON decoding behavior.

Every tag may carry a ``:variant`` suffix. The handoff doc above shows the
bare form, but the shipped ``tencent/Hy3`` tokenizer defines *only* the
suffixed tokens — ``<tool_calls:opensource>``, ``<tool_sep:opensource>``,
``<arg_key:opensource>``, and so on — so a bare-tag-only parser never fires
on the real model. Both forms are accepted.

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


@ToolParserManager.register_module(["hunyuan", "hy_v3", "hy_v4_flash", "hy4", "tencent"])
class HunyuanToolParser(ToolParser):
    """Hunyuan/Tencent Hy3 & Hy4-flash tool-call XML parser."""

    NATIVE_MARKERS = (
        "<tool_calls>",
        "<tool_calls:opensource>",
        "<tool_sep>",
        "<arg_key>",
        "<arg_key:opensource>",
        "<arg_value>",
        "<arg_value:opensource>",
    )

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Optional ``:variant`` suffix on every tag (the shipped Hy3 tokenizer
    # uses ``:opensource``; the handoff contract uses the bare form).
    _V = r"(?::[A-Za-z0-9_-]+)?"

    TOOL_CALLS_PATTERN = re.compile(
        rf"<tool_calls{_V}>\s*(.*?)\s*</tool_calls{_V}>",
        re.DOTALL,
    )
    TOOL_CALL_PATTERN = re.compile(
        rf"<tool_call{_V}>\s*(.*?)\s*</tool_call{_V}>",
        re.DOTALL,
    )
    ARG_KEY_PATTERN = re.compile(
        rf"<arg_key{_V}>\s*(.*?)\s*</arg_key{_V}>",
        re.DOTALL,
    )
    ARG_VALUE_PATTERN = re.compile(
        rf"<arg_value{_V}>(.*?)</arg_value{_V}>",
        re.DOTALL,
    )
    TOOL_SEP_PATTERN = re.compile(rf"<tool_sep{_V}>")
    # Cheap pre-checks that mirror the full patterns' tag spelling.
    TOOL_CALLS_OPEN = re.compile(rf"<tool_calls{_V}>")
    TOOL_CALLS_CLOSE = re.compile(rf"</tool_calls{_V}>")

    @classmethod
    def _coerce(cls, value: str, schema: Any = None) -> Any:
        """Honor native literal strings before decoding JSON-shaped values."""
        if cls._schema_is_string_or_null(schema):
            if cls._schema_allows_null(schema) and value.strip().lower() in cls._NULL_SPELLINGS:
                return None
            return value
        value = value.strip()
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
        if not self.TOOL_CALLS_OPEN.search(model_output):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[dict[str, Any]] = []
        for envelope in self.TOOL_CALLS_PATTERN.findall(model_output):
            for raw_call in self.TOOL_CALL_PATTERN.findall(envelope):
                parts = self.TOOL_SEP_PATTERN.split(raw_call, maxsplit=1)
                if len(parts) == 2:
                    name, args_blob = parts
                else:
                    # Hy4-flash format omits <tool_sep>: `<tool_call:opensource>NAME<arg_key:opensource>...`
                    first_arg = self.ARG_KEY_PATTERN.search(raw_call)
                    if first_arg is not None:
                        name = raw_call[: first_arg.start()]
                        args_blob = raw_call[first_arg.start() :]
                    else:
                        name = raw_call.strip()
                        args_blob = ""
                    if not name.strip():
                        continue
                # Pair keys with values positionally — order matters in the
                # template emission, both lists should be the same length.
                keys = self.ARG_KEY_PATTERN.findall(args_blob)
                values = self.ARG_VALUE_PATTERN.findall(args_blob)
                properties = self._argument_properties(request, name.strip())
                arguments: dict[str, Any] = {}
                for key, value in zip(keys, values):
                    key = key.strip()
                    arguments[key] = self._coerce(value, properties.get(key))
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
        if not self.TOOL_CALLS_OPEN.search(current_text):
            return {"content": delta_text}
        if self.TOOL_CALLS_CLOSE.search(current_text):
            result = self.extract_tool_calls(current_text, request=request)
            previous = self.extract_tool_calls(previous_text, request=request)
            completed_count = len(previous.tool_calls)
            new_calls = result.tool_calls[completed_count:]
            if new_calls:
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
                        for i, tc in enumerate(new_calls, start=completed_count)
                    ]
                }
        return None
