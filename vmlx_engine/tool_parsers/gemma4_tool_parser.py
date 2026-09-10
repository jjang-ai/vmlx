# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vmlx-engine.

Handles Gemma 4's native tool calling format:
  <|tool_call>call:function_name{key:value,key:value}<tool_call|>

The format uses a custom key:value syntax (not JSON) inside braces.
Values can be quoted with <|"|>...</|"|> or unquoted.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    is_valid_tool_name,
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)

# Gemma 4 tool call markers
_STC = "<|tool_call>"    # start-of-tool-call
_ETC = "<tool_call|>"    # end-of-tool-call

# Tool-response role markers. A model must NEVER *generate* these — the
# tool-response role belongs to the client. Degraded low-bit gemma bundles
# sometimes hallucinate a fake `<|tool_response>...` block (with an invented
# answer + orphan `thought` channel-name) AFTER emitting the real tool call.
_TRESP_OPEN = "<|tool_response>"
_TRESP_CLOSE = "<tool_response|>"

# Pattern to extract tool calls: <|tool_call>call:name{...}<tool_call|>
_TOOL_CALL_PATTERN = re.compile(
    r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)

# Gemma 4 escape token for string quoting
_ESCAPE_OPEN = '<|"|>'
_ESCAPE_CLOSE = '<|"|>'


def _parse_gemma4_args(args_str: str) -> dict[str, Any]:
    """Parse Gemma 4's key:value argument format into a dict.

    Format: key1:value1,key2:value2
    Values may be:
    - Quoted strings: <|"|>text<|"|>
    - Nested objects: {key:value}
    - Arrays: [item1,item2]
    - Numbers/booleans: unquoted
    """
    if not args_str.strip():
        return {}

    result: dict[str, Any] = {}

    # Try to parse as JSON first (some models may output JSON-like format)
    try:
        # Convert key:value format to JSON
        json_str = _convert_to_json(args_str)
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: simple key:value parsing
    # Split on commas that aren't inside braces/brackets/quotes
    depth = 0
    current = ""
    pairs: list[str] = []

    for char in args_str:
        if char in ('{', '['):
            depth += 1
            current += char
        elif char in ('}', ']'):
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            pairs.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        pairs.append(current.strip())

    for pair in pairs:
        colon_idx = pair.find(':')
        if colon_idx > 0:
            key = pair[:colon_idx].strip()
            value = pair[colon_idx + 1:].strip()
            # Strip Gemma 4 quote tokens
            value = value.replace(_ESCAPE_OPEN, '').replace(_ESCAPE_CLOSE, '')
            # Try to parse value as JSON
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                result[key] = value
        else:
            # Can't parse — skip
            continue

    return result


def _convert_to_json(args_str: str) -> str:
    """Attempt to convert Gemma 4 key:value format to JSON."""
    s = args_str
    # Replace Gemma 4 escape tokens with double quotes
    s = s.replace(_ESCAPE_OPEN, '"').replace(_ESCAPE_CLOSE, '"')
    # Wrap in braces first so the regex can anchor on { for the first key
    s = s.strip()
    if not s.startswith('{'):
        s = '{' + s + '}'
    # Add quotes around unquoted keys at JSON key positions (after { or ,).
    # This avoids corrupting values like URLs (http://...) or times (14:30).
    s = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', s)
    return s


@ToolParserManager.register_module(["gemma4"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Supports Gemma 4's native tool call format:
    - <|tool_call>call:function_name{key:value,key:value}<tool_call|>

    Also falls back to Hermes format (<tool_call>JSON</tool_call>) and
    raw JSON format for compatibility.
    """

    NATIVE_MARKERS = ("<|tool_call>",)

    # Generic JSON repair does not understand malformed Gemma native envelopes.
    # Keep non-native JSON/Hermes compatibility, but reject a native block that
    # this parser could not decode rather than exposing or repairing its debris.
    SUPPRESS_INVALID_NATIVE_MARKUP = True

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    STREAM_STOPS_AFTER_COMPLETE_CALL = True

    # Also match Hermes format as fallback
    HERMES_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Gemma 4 model output."""
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        # Drop hallucinated tool-response blocks BEFORE any other parsing. A
        # model must never generate a tool response; when a degraded bundle
        # emits `<|tool_call>...<tool_call|><|tool_response>...<fabricated
        # answer>` we must keep the real call but discard the invented result
        # (and its orphan `thought` channel-name) so neither the fake data nor
        # the leftover marker reaches the user. Fixes stream+nonstream alike.
        cleaned_text = self._strip_hallucinated_tool_response(cleaned_text)

        # Strip think/channel tags
        cleaned_text = self._strip_thought_channel(cleaned_text)
        cleaned_text = self.strip_think_tags(cleaned_text)

        # Parse Gemma 4 native format: <|tool_call>call:name{args}<tool_call|>
        matches = _TOOL_CALL_PATTERN.findall(cleaned_text)
        for name, args_str in matches:
            arguments = _parse_gemma4_args(args_str)
            if name:
                tool_calls.append({
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                })

        if matches:
            cleaned_text = _TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        # Fallback: try Hermes format
        if not tool_calls:
            hermes_matches = self.HERMES_PATTERN.findall(cleaned_text)
            for match in hermes_matches:
                try:
                    data = json.loads(match)
                    name = data.get("name", "")
                    arguments = data.get("arguments", {})
                    if is_valid_tool_name(name):
                        tool_calls.append({
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        })
                except json.JSONDecodeError:
                    continue
            if hermes_matches:
                cleaned_text = self.HERMES_PATTERN.sub("", cleaned_text).strip()

        # Strip residual special tokens
        cleaned_text = self._clean_special_tokens(cleaned_text)

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=cleaned_text
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
        """Extract tool calls from streaming output.

        mlxstudio#77: when multiple tool calls arrive sequentially, the
        end-marker detection on the SECOND call was re-extracting from
        current_text which still contains the FIRST call. Each
        extract_tool_calls() call also mints fresh UUIDs, so the first
        call got re-emitted with a NEW id. The client then saw two
        tool_calls for "datetime" with different IDs, routed the
        datetime RESULT to one id, and when the model expected the
        search_web RESULT (under the other id) it saw datetime text
        instead — hence the reporter's "datetime was interrupted" and
        hallucinated search-web content.

        Fix: track how many tool calls have already been emitted this
        streaming session via `self._gemma4_emitted_count`. On each
        end-marker fire, extract the full list but only emit the tail
        that hasn't been streamed yet. Reset on state-machine reset
        (new request).
        """
        # Lazy-init per-parser-instance emitted counter. Base class's
        # current_tool_id/prev_tool_call_arr are used by other parsers;
        # keep this namespaced so we don't collide.
        if not hasattr(self, "_gemma4_emitted_count"):
            self._gemma4_emitted_count = 0
            self._gemma4_emitted_ids: list[str] = []

        # Check for Gemma 4 native tool call end marker
        if _STC in current_text and _ETC in delta_text:
            result = self.extract_tool_calls(current_text, request)
            if result.tools_called:
                # Emit only the tail: tool calls past the ones we already streamed.
                tail = result.tool_calls[self._gemma4_emitted_count:]
                # Preserve prior IDs for the calls we already emitted so
                # downstream re-runs of extract_tool_calls don't surprise
                # clients that cached the first id.
                if not tail:
                    return None
                start_index = self._gemma4_emitted_count
                # Rewrite the first tail call's id to match the originally-
                # emitted id, if extract_tool_calls produced a stale new one
                # (defensive — only fires if the counter desyncs).
                emitted = []
                for offset, tc in enumerate(tail):
                    tc_id = tc["id"]
                    self._gemma4_emitted_ids.append(tc_id)
                    emitted.append({
                        "index": start_index + offset,
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })
                self._gemma4_emitted_count += len(tail)
                return {"tool_calls": emitted}
            return None

        # Check for Hermes format fallback
        if "<tool_call>" in current_text and "</tool_call>" in delta_text:
            result = self.extract_tool_calls(current_text, request)
            if result.tools_called:
                tail = result.tool_calls[self._gemma4_emitted_count:]
                if not tail:
                    return None
                start_index = self._gemma4_emitted_count
                emitted = []
                for offset, tc in enumerate(tail):
                    tc_id = tc["id"]
                    self._gemma4_emitted_ids.append(tc_id)
                    emitted.append({
                        "index": start_index + offset,
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })
                self._gemma4_emitted_count += len(tail)
                return {"tool_calls": emitted}
            return None

        return {"content": delta_text}

    def reset_streaming_state(self) -> None:
        """Reset the emitted-count tracker — call on new request start.

        Without this, a parser instance reused across requests would
        think the first call of the NEW request was already emitted.
        """
        self._gemma4_emitted_count = 0
        self._gemma4_emitted_ids = []

    def stream_tool_calls_complete(self, buffered_text: str) -> bool:
        """True once a complete native Gemma tool call has closed.

        Gemma 4 can continue by hallucinating a client-owned
        ``<|tool_response>`` block after the real call. Once the native call is
        parseable and no new call is opening in the tail, the server should stop
        generation and execute the tool itself.
        """
        matches = list(_TOOL_CALL_PATTERN.finditer(buffered_text))
        if not matches:
            return False
        tail = buffered_text[matches[-1].end():]
        if _STC in tail:
            return False
        for n in range(len(_STC) - 1, 0, -1):
            if tail.endswith(_STC[:n]):
                return False
        return True

    def stream_tool_call_stop_truncate(self, buffered_text: str) -> str:
        """Drop hallucinated post-call text after the last complete call."""
        matches = list(_TOOL_CALL_PATTERN.finditer(buffered_text))
        if not matches:
            return buffered_text
        return buffered_text[: matches[-1].end()]

    @staticmethod
    def _strip_hallucinated_tool_response(text: str) -> str:
        """Drop a model-generated tool-response block.

        The tool-response role belongs to the client — a model must never emit
        `<|tool_response>`. Low-bit gemma bundles sometimes hallucinate one
        right after the real tool call, inventing an answer the model cannot
        actually know (it called the tool precisely because it doesn't). Cut
        from the first tool-response marker to end so the fabricated result —
        and any orphan `thought` channel-name embedded in it — never reaches
        the user. Legitimate prose emitted BEFORE the tool call is preserved.
        """
        idx = text.find(_TRESP_OPEN)
        if idx >= 0:
            return text[:idx].rstrip()
        return text

    @staticmethod
    def _strip_thought_channel(text: str) -> str:
        """Strip Gemma 4 thought channel markers from text.

        Handles both forms:
        1. Full marker: `<|channel>thought\\n...<channel|>`
        2. Degraded form (detokenizer stripped `<|channel>` special token
           but left `thought\\n` as plain text): `thought\\n...<channel|>`.
           Falls back to stripping just `thought\\n` + everything up to
           the first `<channel|>` OR (if no endmarker) a blank line.
        """
        # Remove <|channel>thought\n...<channel|> blocks
        result = re.sub(
            r'<\|channel>thought\n.*?<channel\|>',
            '',
            text,
            flags=re.DOTALL,
        )
        # Degraded form: leading `thought\n...<channel|>` without SOC
        result = re.sub(
            r'^\s*thought\n.*?<channel\|>',
            '',
            result,
            flags=re.DOTALL,
        )
        # Degraded form with no endmarker (truncated or single-channel):
        # strip leading `thought\n` if it's still at the start.
        if result.lstrip().startswith("thought\n"):
            result = result.lstrip()[len("thought\n"):]
        # Also strip bare markers
        result = result.replace('<|channel>', '').replace('<channel|>', '')
        return result.strip()

    @staticmethod
    def _clean_special_tokens(text: str) -> str:
        """Clean residual Gemma 4 special tokens from text."""
        tokens_to_strip = [
            '<turn|>', '<|turn>', '<eos>',
            '<|tool_call>', '<tool_call|>',
            '<|tool_response>', '<tool_response|>',
            '<|tool>', '<tool|>',
            '<|channel>', '<channel|>',
            # Gemma 4 quote/escape token + media placeholders — never valid in
            # user-visible content; 2-bit bundles leak these alongside tool calls.
            '<|"|>',
            '<audio|>', '<|audio>', '<audio>',
            '<image|>', '<|image>', '<image>',
        ]
        for token in tokens_to_strip:
            text = text.replace(token, '')
        # Strip malformed/residual bare tool-call markup the model sometimes
        # echoes OUTSIDE the <|tool_call>...<tool_call|> wrapper (observed:
        # "loader:list_files{path:/tmp}" leaking into content on 2-bit gemma).
        # Only the tool-call prefixes call:/loader:/declaration: are targeted.
        text = re.sub(r'(?:call|loader|declaration):\w+\{[^{}]*\}', '', text)
        return text.strip()
