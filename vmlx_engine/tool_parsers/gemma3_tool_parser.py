# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Gemma 3 and Gemma 3n.

Google's documented function-calling format for Gemma 3 / 3n uses a
``tool_code`` markdown code block with a Python-style function call:

    ```tool_code
    get_weather(city="Paris")
    ```

Multiple calls can appear as multiple code blocks. Argument values can
be string literals ("..." or '...'), numbers, booleans (True/False),
None, or JSON literals. The parser converts them to a JSON-compatible
arguments string that the OpenAI-shape tool call expects.

Reference: https://ai.google.dev/gemma/docs/core/function_calling
"""

from __future__ import annotations

from ast import literal_eval as _parse_literal
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)

logger = logging.getLogger(__name__)

# Match a ```tool_code\n<body>\n``` block. Allow optional whitespace.
_TOOL_CODE_BLOCK = re.compile(
    r"```tool_code\s*\n(?P<body>.*?)\n?```",
    re.DOTALL,
)

# Match a single Python-style call `name(k=v, k2=v2)` — body is everything
# between the outermost parens. Non-greedy up to the matching closer.
_CALL_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>.*?)\)\s*$",
    re.DOTALL,
)


def _parse_py_value(src: str) -> Any:
    """Convert a Python-literal snippet into a JSON-safe value.

    Uses ``ast.literal_eval`` which is sandboxed — it ONLY parses
    literals (strings, numbers, booleans, None, lists, dicts, tuples)
    and NEVER executes arbitrary code. Safe for untrusted model output.
    """
    s = src.strip()
    if not s:
        return ""
    try:
        return _parse_literal(s)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _split_kwargs(args_src: str) -> list[tuple[str, str]]:
    """Split `k1=v1, k2=v2` honoring brackets and quotes."""
    pairs: list[tuple[str, str]] = []
    depth_paren = depth_brack = depth_brace = 0
    in_str: str | None = None
    esc = False
    start = 0
    i = 0
    pieces: list[str] = []
    while i < len(args_src):
        ch = args_src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack -= 1
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "," and depth_paren == depth_brack == depth_brace == 0:
                pieces.append(args_src[start:i])
                start = i + 1
        i += 1
    if start < len(args_src):
        pieces.append(args_src[start:])

    positional = 0
    for piece in pieces:
        p = piece.strip()
        if not p:
            continue
        eq_idx = -1
        in_str = None
        esc = False
        depth = 0
        for j, ch in enumerate(p):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "=" and depth == 0:
                eq_idx = j
                break
        if eq_idx >= 0:
            key = p[:eq_idx].strip()
            val = p[eq_idx + 1 :].strip()
            pairs.append((key, val))
        else:
            pairs.append((f"arg{positional}", p))
            positional += 1
    return pairs


@ToolParserManager.register_module(["gemma3", "gemma3n"])
class Gemma3ToolParser(ToolParser):
    """Parse Gemma 3 / 3n ``tool_code`` code-block tool calls."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        blocks = _TOOL_CODE_BLOCK.findall(model_output)
        if not blocks:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[dict[str, Any]] = []
        for body in blocks:
            body = body.strip()
            if not body:
                continue
            for line in body.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = _CALL_PATTERN.search(line)
                if not match:
                    continue
                name = match.group("name")
                arg_src = match.group("args")
                kwargs = _split_kwargs(arg_src)
                args_obj: dict[str, Any] = {}
                for k, v in kwargs:
                    args_obj[k] = _parse_py_value(v)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name,
                        "arguments": json.dumps(args_obj, ensure_ascii=False),
                    }
                )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        cleaned = _TOOL_CODE_BLOCK.sub("", model_output).strip()
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=cleaned or None,
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
        if "```" in delta_text and "tool_code" in current_text:
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
        return {"content": delta_text}
