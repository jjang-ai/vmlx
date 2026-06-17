# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 tool call parser for vmlx-engine.

M3's chat template emits Anthropic-style XML where the PARAMETER NAME IS THE TAG
(not a ``name=`` attribute), wrapped in ``<tool_call>...</tool_call>``, with a
literal namespace separator token (``]<]minimax[>[``) prefixed before every
element. Reasoning is emitted in ``<mm:think>...</mm:think>``.

Rendered shape (ns_token shown as · for clarity, stripped before parsing)::

    <tool_call>
    ·<invoke name="get_weather">
    ·<location>San Francisco·</location>
    ·<opts>
    ·<unit>celsius·</unit>
    ·</opts>
    ·<days>
    ·<item>mon·</item>
    ·<item>tue·</item>
    ·</days>
    ·</invoke>
    </tool_call>

Value mapping:
  * scalar param  -> ``<p>value</p>``                       -> coerced scalar
  * nested object -> ``<p><k1>v1</k1><k2>v2</k2></p>``       -> dict
  * array         -> ``<p><item>..</item><item>..</item></p>`` -> list

Multiple ``<invoke>`` may appear inside a single ``<tool_call>`` block. Used when
``--enable-auto-tool-choice --tool-call-parser minimax_m3`` are set.
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

# Namespace separator token the M3 template prefixes before every element.
NS_TOKEN = "]<]minimax[>["

_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Lenient: <tool_call> opened but truncated before </tool_call> (hit max_tokens).
_TOOLCALL_OPEN_RE = re.compile(r"<tool_call>(.*?)(?=</tool_call>|$)", re.DOTALL)
_INVOKE_RE = re.compile(
    r"""<invoke\s+name=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))\s*>(.*?)</invoke>""",
    re.DOTALL,
)
_MM_THINK_RE = re.compile(r"<mm:think>.*?</mm:think>", re.DOTALL)
_OPEN_TAG_RE = re.compile(r"<([A-Za-z_][\w\-.:]*)\s*>")


def _coerce(value: str) -> Any:
    value = value.strip()
    if value == "":
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _next_tag(s: str, pos: int):
    """Return (tag, inner, end) for the next top-level ``<tag>...</tag>`` at/after
    ``pos``, with correct nesting of same-named tags. None if no complete tag."""
    m = _OPEN_TAG_RE.search(s, pos)
    if not m:
        return None
    tag = m.group(1)
    open_tag = "<" + tag + ">"
    close_tag = "</" + tag + ">"
    depth = 1
    scan = m.end()
    while True:
        nxt_open = s.find(open_tag, scan)
        nxt_close = s.find(close_tag, scan)
        if nxt_close == -1:
            return None  # unterminated
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            scan = nxt_open + len(open_tag)
        else:
            depth -= 1
            scan = nxt_close + len(close_tag)
            if depth == 0:
                return (tag, s[m.end():nxt_close], scan)


def _children(s: str):
    """Yield all top-level (tag, inner) pairs in ``s``."""
    pos = 0
    out = []
    while True:
        nxt = _next_tag(s, pos)
        if nxt is None:
            break
        tag, inner, end = nxt
        out.append((tag, inner))
        pos = end
    return out


def _parse_value(inner: str) -> Any:
    kids = _children(inner)
    if not kids:
        return _coerce(inner)
    # array: every top-level child is <item>
    if all(tag == "item" for tag, _ in kids):
        return [_parse_value(ci) for _, ci in kids]
    # object: child tag name -> value (last wins on dup keys)
    obj: dict[str, Any] = {}
    for tag, ci in kids:
        obj[tag] = _parse_value(ci)
    return obj


def _args_from_invoke(body: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for tag, inner in _children(body):
        args[tag] = _parse_value(inner)
    return args


@ToolParserManager.register_module(["minimax_m3"])
class MiniMaxM3ToolParser(ToolParser):
    """Tool call parser for MiniMax-M3 (tag-named-parameter XML)."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def _strip_noise(self, text: str) -> str:
        text = text.replace(NS_TOKEN, "")
        text = _MM_THINK_RE.sub("", text)
        return text

    def _invokes_to_calls(self, search_space: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for m in _INVOKE_RE.finditer(search_space):
            name = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            if not name:
                continue
            args = _args_from_invoke(m.group(4))
            calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
        return calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned = self._strip_noise(model_output)

        if "<invoke" not in cleaned:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned
            )

        blocks = _TOOLCALL_RE.findall(cleaned)
        if blocks:
            search_space = "\n".join(blocks)
        else:
            # truncated <tool_call> with no closing tag, or invoke without wrapper
            open_blocks = _TOOLCALL_OPEN_RE.findall(cleaned)
            search_space = "\n".join(open_blocks) if open_blocks else cleaned

        tool_calls = self._invokes_to_calls(search_space)
        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned
            )

        # Content = everything outside the tool_call wrapper.
        if blocks:
            content = _TOOLCALL_RE.sub("", cleaned).strip()
        else:
            content = re.sub(r"<tool_call>.*$", "", cleaned, flags=re.DOTALL).strip()
            content = _INVOKE_RE.sub("", content).strip()
        return ExtractedToolCallInformation(
            tools_called=True,
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
        # Server uses buffer-then-parse (extract_tool_calls on full text); this is
        # a conservative streaming impl retained for parity/testing: stream visible
        # content deltas until a tool-call marker appears, then withhold.
        cleaned_current = self._strip_noise(current_text)
        if "<tool_call>" in cleaned_current or "<invoke" in cleaned_current:
            return None
        if not delta_text:
            return None
        return {"content": delta_text}
