# SPDX-License-Identifier: Apache-2.0
"""
DSML tool call parser for DeepSeek V4-Flash / V4-Pro.

DeepSeek V4 emits tool calls in the "DSML" (DeepSeek Markup Language) format.
The DSML delimiter is the fullwidth vertical bar `｜` (U+FF5C) bracketing the
literal string "DSML" — the same character class DeepSeek uses for its other
special tokens (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`).

Example completion:

    <｜DSML｜invoke name="search_web">
    <｜DSML｜parameter name="query" string="true">weather in LA</｜DSML｜parameter>
    <｜DSML｜parameter name="limit" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>

Multiple `<｜DSML｜invoke>` blocks per turn are allowed. Parameters carry a
`string="true"` / `string="false"` attribute — when false, the value is valid
JSON and should be parsed (numbers, booleans, arrays, objects); when true,
it's a raw string. Reference: research/DSV4-RUNTIME-ARCHITECTURE.md §4 and
jang_tools/dsv4/test_chat.py::parse_dsml_tool_calls.

Selected via `--tool-call-parser dsml` or via the deepseek_v4 family config
in model_configs.py.
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


# Fullwidth vertical bar, DSV4's canonical DSML delimiter.
DSML_CHAR = "｜"  # ｜
DSML_PREFIX = f"{DSML_CHAR}DSML{DSML_CHAR}"


@ToolParserManager.register_module(["dsml", "deepseek_v4"])
class DSMLToolParser(ToolParser):
    """
    DeepSeek V4 DSML tool call parser.

    Input pattern:
        <｜DSML｜invoke name="fn">
          <｜DSML｜parameter name="p1" string="true">str_val</｜DSML｜parameter>
          <｜DSML｜parameter name="p2" string="false">42</｜DSML｜parameter>
        </｜DSML｜invoke>

    Output: list of ToolCall with `function.name` = fn and
    `function.arguments` = JSON-encoded object mapping param → value
    (numbers/bools/nested structures parsed when `string="false"`).
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Streaming state markers — we buffer until we see a complete `<invoke …>…</invoke>`
    # and stop emitting content between the opening `<｜DSML｜invoke` and its close.
    INVOKE_OPEN_PREFIX = f"<{DSML_PREFIX}invoke "
    INVOKE_CLOSE = f"</{DSML_PREFIX}invoke>"

    # Top-level regex: find every <｜DSML｜invoke name="…">…</｜DSML｜invoke> block.
    _INVOKE_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}invoke\s+name="([^"]+)"\s*>(.*?)</{re.escape(DSML_PREFIX)}invoke>',
        re.DOTALL,
    )

    # Param regex: <｜DSML｜parameter name="…" string="true|false">value</｜DSML｜parameter>
    _PARAM_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</{re.escape(DSML_PREFIX)}parameter>',
        re.DOTALL,
    )

    def _has_dsml(self, text: str) -> bool:
        return self.INVOKE_OPEN_PREFIX in text

    def _parse_params(self, body: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for m in self._PARAM_RE.finditer(body):
            name, is_string, raw = m.group(1), m.group(2), m.group(3)
            if is_string == "true":
                out[name] = raw
            else:
                # DSV4 emits JSON-serialised values with `string="false"` —
                # attempt json.loads; fall back to the raw string if it's
                # malformed (preserves the original text rather than crashing
                # the whole tool-call round).
                try:
                    out[name] = json.loads(raw)
                except Exception:
                    out[name] = raw
        return out

    def extract_tool_calls(
        self, model_output: str, request: Any | None = None
    ) -> ExtractedToolCallInformation:
        """Non-streaming path — parse entire completion and return tool calls + residue."""
        if not self._has_dsml(model_output):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        for m in self._INVOKE_RE.finditer(model_output):
            name = m.group(1)
            body = m.group(2)
            args = self._parse_params(body)
            tool_calls.append(
                self._make_tool_call(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                    id_=generate_tool_id(),
                )
            )

        # Residue content = everything OUTSIDE the invoke blocks. Strip the
        # matched spans and collapse surrounding whitespace so the chat UI
        # doesn't show a blank paragraph where the tool call used to be.
        residue = self._INVOKE_RE.sub("", model_output).strip()

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=residue if residue else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: Any | None = None,
    ):
        """Streaming path — buffer partial `<｜DSML｜invoke …>` blocks, emit on close.

        Strategy: run the non-streaming regex on `current_text`. If we've seen
        N complete invoke blocks previously and there are N+k now, flush those
        k deltas. Content OUTSIDE invoke blocks streams normally via the
        abstract parser's default path.
        """
        if not self._has_dsml(current_text):
            # No DSML at all → pass through as plain content.
            return self._default_content_delta(delta_text)

        # Count complete blocks up to the previous cursor vs the current one.
        prev_blocks = list(self._INVOKE_RE.finditer(previous_text))
        curr_blocks = list(self._INVOKE_RE.finditer(current_text))

        if len(curr_blocks) == len(prev_blocks):
            # We're mid-invoke (opening tag emitted but close not seen yet),
            # OR we're in plain content between invokes. Suppress the delta
            # while inside an unclosed invoke; otherwise pass through.
            open_tail = current_text.rsplit(self.INVOKE_OPEN_PREFIX, 1)
            if len(open_tail) == 2 and self.INVOKE_CLOSE not in open_tail[1]:
                # We're inside an unclosed invoke — buffer silently.
                return None
            return self._default_content_delta(delta_text)

        # A new invoke block just closed. Emit tool calls for each block
        # that's new since `prev_blocks`.
        new_calls = []
        for m in curr_blocks[len(prev_blocks):]:
            name = m.group(1)
            body = m.group(2)
            args = self._parse_params(body)
            new_calls.append(
                self._make_stream_tool_call_delta(
                    index=len(prev_blocks) + len(new_calls),
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                    id_=generate_tool_id(),
                )
            )
        return self._pack_stream_tool_calls(new_calls)

    # ── Abstract-parser-compatible shims ────────────────────────────────
    # The base ToolParser class in this codebase has varied signatures across
    # releases; these helpers normalise the construction paths. Real impl may
    # override; leaving thin bodies so tests can import + exercise the regex.

    def _make_tool_call(self, *, name: str, arguments: str, id_: str):
        # Use parent class helper if available; fall back to a plain dict
        # that the streaming/assembly code can serialise.
        try:
            return super()._make_tool_call(name=name, arguments=arguments, id_=id_)
        except Exception:
            return {
                "id": id_,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }

    def _make_stream_tool_call_delta(
        self, *, index: int, name: str, arguments: str, id_: str
    ):
        try:
            return super()._make_stream_tool_call_delta(
                index=index, name=name, arguments=arguments, id_=id_
            )
        except Exception:
            return {
                "index": index,
                "id": id_,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }

    def _pack_stream_tool_calls(self, calls: list):
        try:
            return super()._pack_stream_tool_calls(calls)
        except Exception:
            return calls if calls else None

    def _default_content_delta(self, delta_text: str):
        try:
            return super()._default_content_delta(delta_text)
        except Exception:
            return {"content": delta_text} if delta_text else None
