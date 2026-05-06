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
    _PARTIAL_INVOKE_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}invoke\s+name="([^"]+)"\s*>(.*)',
        re.DOTALL,
    )

    # Param regex: <｜DSML｜parameter name="…" string="true|false">value</｜DSML｜parameter>
    _PARAM_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</{re.escape(DSML_PREFIX)}parameter>',
        re.DOTALL,
    )
    _MALFORMED_PARAM_VALUE_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}parameter\s+name="([^"]+)"[^>]*?\bvalue\s*"?([^">\s]*)"?',
        re.DOTALL,
    )
    _MALFORMED_NAME_RE = re.compile(
        rf'<{re.escape(DSML_PREFIX)}(?:tool_call_type|tool_call|tool_calls)[^>]*?(?:type|name)="([^"]+)"',
        re.DOTALL,
    )
    _ATTR_RE = re.compile(r'"attributes"\s*:\s*"([^"}\n]*)', re.DOTALL)
    _HTMLISH_INVOKE_RE = re.compile(
        r"<invoke_([A-Za-z_][A-Za-z0-9_]*)\b[^>]*>(.*)",
        re.DOTALL,
    )
    _HTMLISH_PARAM_RE = re.compile(
        r'<param\s+name=["\']?([A-Za-z_][A-Za-z0-9_]*)["\']?[^>]*>([^<]*)',
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

    def _tool_schemas(self, request: Any | None) -> dict[str, dict[str, Any]]:
        """Return available tool schemas keyed by function name.

        Parser repair must be schema-gated. DSV4 JANGTQ can emit malformed
        DSML-ish text under tool pressure; we only turn that into a structured
        call when the emitted name is one of the request's available tools.
        """
        tools = []
        if isinstance(request, dict):
            tools = request.get("tools") or []
        elif request is not None:
            tools = getattr(request, "tools", []) or []
        out: dict[str, dict[str, Any]] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                out[name] = fn
        return out

    def _repair_malformed_dsml(
        self, text: str, request: Any | None
    ) -> list[dict[str, Any]]:
        """Best-effort repair for old DSV4 JANGTQ malformed DSML.

        Canonical DSML is still required for normal parsing. This fallback
        handles the observed older-bundle shape:

            <｜DSML｜tool_call_type type="list_directory"... "attributes":"." ...>

        It is intentionally conservative: the function name must exist in the
        request tool schema, and arguments are only inferred for declared
        parameters. If there is one required parameter and the malformed block
        exposes a single attributes value, that value is assigned to it.
        """
        if DSML_PREFIX not in text:
            return []
        schemas = self._tool_schemas(request)
        if not schemas:
            return []

        calls: list[dict[str, Any]] = []
        for m in self._MALFORMED_NAME_RE.finditer(text):
            name = m.group(1)
            schema = schemas.get(name)
            if not schema:
                continue
            params_schema = schema.get("parameters") or {}
            props = (
                params_schema.get("properties") or {}
                if isinstance(params_schema, dict)
                else {}
            )
            required = (
                params_schema.get("required") or []
                if isinstance(params_schema, dict)
                else []
            )
            args: dict[str, Any] = {}
            for p_name in props:
                # Common malformed forms: path=".", "path": ".", path:. 
                pat = re.compile(
                    rf'(?:{re.escape(p_name)}\s*=\s*"([^"]*)"|"{re.escape(p_name)}"\s*:\s*"([^"]*)")'
                )
                pm = pat.search(text)
                if pm:
                    args[p_name] = next(g for g in pm.groups() if g is not None)
            attr = self._ATTR_RE.search(text)
            if attr and len(required) == 1 and required[0] not in args:
                args[required[0]] = attr.group(1)
            if not args and len(props) == 1:
                # Last-resort single-parameter inference from an explicit dot.
                only = next(iter(props))
                if '"."' in text or ">.<" in text or " . " in text:
                    args[only] = "."
            if args or not required:
                calls.append(
                    self._make_tool_call(
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                        id_=generate_tool_id(),
                    )
                )
        return calls

    def _repair_partial_invoke(
        self, text: str, request: Any | None
    ) -> list[dict[str, Any]]:
        """Repair a canonical DSML invoke whose closing tag was truncated.

        DSV4 JANGTQ2 sometimes emits a perfectly valid opening invoke and
        complete parameter tags, then stops after the first characters of the
        closing tag, e.g. ``</``. Treat that as a tool call only when:

        - the tool name exists in the request schema; and
        - at least all required parameters were parsed from complete parameter
          tags.

        This keeps the non-streaming parser from showing raw DSML text while
        avoiding arbitrary conversion of incomplete markup into tool calls.
        """
        if DSML_PREFIX not in text or self.INVOKE_CLOSE in text:
            return []
        schemas = self._tool_schemas(request)
        if not schemas:
            return []
        m = self._PARTIAL_INVOKE_RE.search(text)
        if not m:
            return []
        name = m.group(1)
        schema = schemas.get(name)
        if not schema:
            return []
        params_schema = schema.get("parameters") or {}
        required = (
            params_schema.get("required") or []
            if isinstance(params_schema, dict)
            else []
        )
        args = self._parse_params(m.group(2))
        if any(p not in args for p in required):
            body = m.group(2)
            for pm in self._MALFORMED_PARAM_VALUE_RE.finditer(body):
                p_name, raw = pm.group(1), pm.group(2)
                if p_name not in args and raw:
                    args[p_name] = raw
        if any(p not in args for p in required):
            return []
        return [
            self._make_tool_call(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
                id_=generate_tool_id(),
            )
        ]

    def _repair_htmlish_invoke(
        self, text: str, request: Any | None
    ) -> list[dict[str, Any]]:
        """Repair DSV4's degraded ``<invoke_name><param ...>`` form.

        Live DSV4 JANGTQ can degrade canonical DSML into HTML-ish tags after
        the reasoning parser strips ``</think>``, for example::

            <invoke_list_directory><br />
            <param name="path".">.</br />
            </inv

        This is schema-gated for the same reason as the other repair paths: we
        only emit a tool call when the function and parameters exist in the
        request's tool schema.
        """
        schemas = self._tool_schemas(request)
        if not schemas:
            return []
        m = self._HTMLISH_INVOKE_RE.search(text)
        if not m:
            return []
        name, body = m.group(1), m.group(2)
        schema = schemas.get(name)
        if not schema:
            return []
        params_schema = schema.get("parameters") or {}
        props = (
            params_schema.get("properties") or {}
            if isinstance(params_schema, dict)
            else {}
        )
        required = (
            params_schema.get("required") or []
            if isinstance(params_schema, dict)
            else []
        )
        args: dict[str, Any] = {}
        for pm in self._HTMLISH_PARAM_RE.finditer(body):
            p_name, raw = pm.group(1), pm.group(2)
            if p_name in props:
                value = re.sub(r"<br\s*/?>", "", raw, flags=re.IGNORECASE).strip()
                if value:
                    args[p_name] = value
        if not args and len(props) == 1 and ">.<" in text:
            args[next(iter(props))] = "."
        if any(p not in args for p in required):
            return []
        return [
            self._make_tool_call(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
                id_=generate_tool_id(),
            )
        ]

    def _try_encoding_dsv4_parse(self, model_output: str):
        """Route DSML extraction through the canonical DSV4 chat-template
        encoder when it's loaded. Returns ExtractedToolCallInformation on
        success, None when encoding_dsv4 isn't available or the parse
        produced nothing actionable (caller falls back to regex)."""
        try:
            from vmlx_engine.loaders.dsv4_chat_encoder import (
                _load_encoding_dsv4_module,
            )
        except Exception:
            return None
        try:
            enc = _load_encoding_dsv4_module()
        except Exception:
            return None
        parse_fn = getattr(enc, "parse_message_from_completion_text", None)
        if parse_fn is None:
            return None
        try:
            parsed = parse_fn(model_output)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        raw_calls = parsed.get("tool_calls") or []
        if not raw_calls:
            return None
        tool_calls = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
            if not isinstance(name, str) or not name:
                continue
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            elif isinstance(args, str):
                args_str = args
            else:
                args_str = json.dumps(args or {}, ensure_ascii=False)
            tool_calls.append(
                self._make_tool_call(
                    name=name, arguments=args_str, id_=generate_tool_id()
                )
            )
        if not tool_calls:
            return None
        residue = parsed.get("content") or ""
        if isinstance(residue, list):
            residue = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in residue
            )
        residue = (residue or "").strip() or None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=residue,
        )

    def extract_tool_calls(
        self, model_output: str, request: Any | None = None
    ) -> ExtractedToolCallInformation:
        """Non-streaming path — parse entire completion and return tool calls + residue.

        Routing precedence (added 2026-05-04):
          1. Canonical ``encoding_dsv4.parse_message_from_completion_text``
             when the DSV4 chat-template encoder is loaded — this matches the
             round-trip semantics of the converter exactly and handles
             nested / multi-line / parameter-string-coerced invokes that the
             generic regex misses.
          2. Fallback to the regex-based path below when encoding_dsv4 is
             unavailable (e.g. non-DSV4 bundle still requesting `dsml` parser).
        """
        if DSML_PREFIX not in model_output and "<invoke_" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        canonical = self._try_encoding_dsv4_parse(model_output)
        if canonical is not None:
            return canonical

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

        if not tool_calls:
            tool_calls = self._repair_partial_invoke(model_output, request)

        if not tool_calls:
            tool_calls = self._repair_malformed_dsml(model_output, request)

        if not tool_calls:
            tool_calls = self._repair_htmlish_invoke(model_output, request)

        # Residue content = everything OUTSIDE the invoke blocks. Strip the
        # matched spans and collapse surrounding whitespace so the chat UI
        # doesn't show a blank paragraph where the tool call used to be.
        residue = self._INVOKE_RE.sub("", model_output).strip()
        if tool_calls and residue == model_output.strip():
            # Repaired malformed DSML: hide the malformed raw marker from the
            # client once we successfully emitted a structured tool call.
            residue = ""

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
        # Non-streaming parser contract in this codebase is the flat shape
        # used by qwen/mistral/nemotron parsers. `server._parse_tool_calls_*`
        # wraps this into Chat/Responses API objects. Returning OpenAI-shaped
        # dicts here trips KeyError('name') and falls back to raw DSML text.
        return {"id": id_, "name": name, "arguments": arguments}

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
