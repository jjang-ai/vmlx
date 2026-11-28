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
    _OPEN_PREFIXES = ("<tool_calls:opensource>", "<tool_calls>", "<tool_call:opensource>", "<tool_call>")

    @classmethod
    def _trailing_open_prefix_len(cls, text: str) -> int:
        """Length of any trailing partial `<tool_calls...>` opener at the end of `text`."""
        last_lt = text.rfind("<")
        if last_lt < 0:
            return 0
        tail = text[last_lt:]
        if ">" in tail:
            return 0
        for pfx in cls._OPEN_PREFIXES:
            if pfx.startswith(tail):
                return len(tail)
        return 0

    def _parse_single_call(
        self,
        raw_call: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
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
                return None
        keys = self.ARG_KEY_PATTERN.findall(args_blob)
        values = self.ARG_VALUE_PATTERN.findall(args_blob)
        properties = self._argument_properties(request, name.strip())
        arguments: dict[str, Any] = {}
        for key, value in zip(keys, values):
            key = key.strip()
            arguments[key] = self._coerce(value, properties.get(key))
        return {
            "id": generate_tool_id(),
            "name": name.strip(),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _extract_completed_calls_prefix(
        self,
        text: str,
        request: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract all completed `<tool_call...>...</tool_call...>` blocks inside open `<tool_calls...>`."""
        m_open = self.TOOL_CALLS_OPEN.search(text)
        if m_open is None:
            return []
        body = text[m_open.end() :]
        calls: list[dict[str, Any]] = []
        for raw_call in self.TOOL_CALL_PATTERN.findall(body):
            parsed = self._parse_single_call(raw_call, request=request)
            if parsed is not None:
                calls.append(parsed)
        return calls

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
                parsed = self._parse_single_call(raw_call, request=request)
                if parsed is not None:
                    tool_calls.append(parsed)

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
        # Per-tool-call streaming without leaking partial `<tool_calls...>` tags:
        # 1. Before `<tool_calls...>` opens, hold back any trailing partial `<tool_...` prefix.
        # 2. Emit any remaining prose before `<tool_calls...>` when the opener lands mid-delta.
        # 3. Inside `<tool_calls...>`, emit each `<tool_call...>...</tool_call...>` immediately
        #    as soon as its closing tag arrives (without waiting for `</tool_calls...>`).
        cur_open = self.TOOL_CALLS_OPEN.search(current_text)
        prev_open = self.TOOL_CALLS_OPEN.search(previous_text)

        if cur_open is None:
            prev_safe = len(previous_text) - self._trailing_open_prefix_len(previous_text)
            cur_safe = len(current_text) - self._trailing_open_prefix_len(current_text)
            emit = current_text[prev_safe:cur_safe]
            return {"content": emit} if emit else None

        out: dict[str, Any] = {}
        if prev_open is None:
            prev_safe = len(previous_text) - self._trailing_open_prefix_len(previous_text)
            prefix_prose = current_text[prev_safe : cur_open.start()]
            if prefix_prose:
                out["content"] = prefix_prose

        prev_calls = self._extract_completed_calls_prefix(previous_text, request=request) if prev_open else []
        cur_calls = self._extract_completed_calls_prefix(current_text, request=request)
        if len(cur_calls) > len(prev_calls):
            out["tool_calls"] = [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for i, tc in enumerate(cur_calls[len(prev_calls) :], start=len(prev_calls))
            ]
        return out if out else None
