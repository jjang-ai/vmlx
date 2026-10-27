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

import inspect
import json
import re
from collections.abc import Sequence
from pathlib import Path
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

    NATIVE_MARKERS = (
        f"<{DSML_PREFIX}tool_calls>",
        f"<{DSML_PREFIX}invoke",
    )

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # DSML is an executable native control protocol.  A malformed DSML-looking
    # completion must never fall through to the generic tool repair parser or
    # be returned verbatim as assistant content.  server.py consults these
    # flags only for this parser family; plain prose remains visible.
    STRICT_NATIVE_TOOL_FORMAT = True
    SUPPRESS_INVALID_NATIVE_MARKUP = True

    # A canonical DSV4 tool turn is complete once its DSML wrapper (or a
    # schema-valid bare invoke used by older bundles) closes. Some low-bit DSV4
    # bundles keep generating after that boundary instead of emitting EOS. The
    # server may stop those streams after its grace window and execute the call;
    # it must not burn the remainder of max_tokens behind a buffering heartbeat.
    STREAM_STOPS_AFTER_COMPLETE_CALL = True

    # Streaming state markers — we buffer until we see a complete `<invoke …>…</invoke>`
    # and stop emitting content between the opening `<｜DSML｜invoke` and its close.
    INVOKE_OPEN_PREFIX = f"<{DSML_PREFIX}invoke "
    DSML_OPEN_PREFIX = f"<{DSML_PREFIX}"
    TOOL_CALLS_OPEN = f"<{DSML_PREFIX}tool_calls>"
    TOOL_CALLS_CLOSE = f"</{DSML_PREFIX}tool_calls>"

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
    _WRAPPER_RESIDUE_RE = re.compile(
        rf'</?{re.escape(DSML_PREFIX)}(?:tool_calls?|tool_call_type|tool_c)[^>]*>?',
        re.DOTALL,
    )

    _NONCANONICAL_PROTOCOL_MARKERS = (
        DSML_OPEN_PREFIX,
        f"</{DSML_PREFIX}",
        "<invoke_",
        "<invoke ",
    )
    _PARTIAL_PROTOCOL_MARKERS = (
        DSML_OPEN_PREFIX,
        f"</{DSML_PREFIX}",
    )
    _MIN_PARTIAL_PROTOCOL_MARKER = 4

    def _has_dsml(self, text: str) -> bool:
        return self.DSML_OPEN_PREFIX in text

    def _native_protocol_start(self, text: str) -> int | None:
        """Return the first full or terminally split native-protocol marker.

        A generation can finish between tokenizer pieces of the fullwidth DSML
        prefix (for example ``<｜DSML``).  That proper-prefix suffix is control
        markup just like a complete marker and must not become assistant text.
        Partial matching is deliberately limited to the DSML namespace markers
        so ordinary prose ending in an HTML-ish ``<inv`` fragment is unaffected.
        """
        starts = [
            position
            for marker in self._NONCANONICAL_PROTOCOL_MARKERS
            if (position := text.find(marker)) >= 0
        ]
        for marker in self._PARTIAL_PROTOCOL_MARKERS:
            max_prefix = min(len(marker) - 1, len(text))
            for length in range(
                max_prefix,
                self._MIN_PARTIAL_PROTOCOL_MARKER - 1,
                -1,
            ):
                if text.endswith(marker[:length]):
                    starts.append(len(text) - length)
                    break
        return min(starts) if starts else None

    def _has_native_protocol_marker(self, text: str) -> bool:
        return self._native_protocol_start(text) is not None

    def _safe_invalid_protocol_content(self, text: str) -> str | None:
        """Return only prose that safely precedes malformed native markup.

        Once a native marker starts, everything after it is an untrusted
        executable envelope.  Keeping a suffix after malformed markup risks
        exposing arguments, DSML tokens, or model-generated fake tool results.
        """
        protocol_start = self._native_protocol_start(text)
        if protocol_start is None:
            return text or None
        prefix = text[:protocol_start].strip()
        return prefix or None

    def _strict_parse_params(self, body: str) -> dict[str, Any] | None:
        """Parse a complete canonical parameter body without repair.

        Every non-whitespace byte in an invoke body must belong to a canonical
        ``parameter name=... string=true|false`` element.  Duplicate names and
        malformed JSON values are protocol errors, not strings to coerce.
        """
        args: dict[str, Any] = {}
        cursor = 0
        for match in self._PARAM_RE.finditer(body):
            if body[cursor : match.start()].strip():
                return None
            name, is_string, raw = match.group(1), match.group(2), match.group(3)
            if name in args:
                return None
            if is_string == "true":
                args[name] = raw
            else:
                try:
                    args[name] = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
            cursor = match.end()
        if body[cursor:].strip():
            return None
        return args

    def _strict_canonical_regex_parse(
        self,
        model_output: str,
        request: Any | None,
    ) -> ExtractedToolCallInformation | None:
        """Parse one fully closed canonical DSML turn.

        This grammar check is shared by streaming and non-streaming paths.  It
        deliberately accepts only canonical invoke/parameter elements, with an
        optional canonical ``tool_calls`` wrapper.  Truncated, typoed,
        HTML-ish, self-closing, or schema-incomplete variants return ``None``.
        """
        matches = list(self._INVOKE_RE.finditer(model_output))
        if not matches:
            return None

        wrapper_open_count = model_output.count(self.TOOL_CALLS_OPEN)
        wrapper_close_count = model_output.count(self.TOOL_CALLS_CLOSE)
        has_wrapper = bool(wrapper_open_count or wrapper_close_count)
        if has_wrapper:
            if wrapper_open_count != 1 or wrapper_close_count != 1:
                return None
            wrapper_open = model_output.find(self.TOOL_CALLS_OPEN)
            wrapper_close = model_output.find(self.TOOL_CALLS_CLOSE)
            if not (
                wrapper_open < matches[0].start()
                and matches[-1].end() < wrapper_close
            ):
                return None
            inner_start = wrapper_open + len(self.TOOL_CALLS_OPEN)
            inner = model_output[inner_start:wrapper_close]
            inner_residue = self._INVOKE_RE.sub("", inner)
            if inner_residue.strip():
                return None
            outside = (
                model_output[:wrapper_open]
                + model_output[wrapper_close + len(self.TOOL_CALLS_CLOSE) :]
            )
            if self._INVOKE_RE.search(outside):
                return None
            visible_content = outside.strip() or None
        else:
            visible_content = self._INVOKE_RE.sub("", model_output).strip() or None

        if visible_content and self._has_native_protocol_marker(visible_content):
            return None

        schemas = self._tool_schemas(request)
        schema_gate_active = request is not None
        tool_calls: list[dict[str, Any]] = []
        for match in matches:
            name, body = match.group(1), match.group(2)
            schema = schemas.get(name) if schemas else None
            if schema_gate_active and schema is None:
                return None
            args = self._strict_parse_params(body)
            if args is None:
                return None
            if schema and not self._arguments_match_schema(args, schema):
                return None
            tool_calls.append(
                self._make_tool_call(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                    id_=generate_tool_id(),
                )
            )

        if not tool_calls:
            return None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=visible_content,
        )

    def _parse_completed_canonical_calls(
        self,
        model_output: str,
        request: Any | None,
    ) -> ExtractedToolCallInformation | None:
        """Single strict completed-call parser used by every DSML path.

        The local grammar validates the bytes before the bundle encoder sees
        them, preventing a permissive/older adapter from repairing malformed
        output.  When the canonical bundle parser is available, both parsers
        must agree on call names and decoded arguments.  The local parse remains
        the returned representation so visible-content sanitization is owned by
        one path.
        """
        strict = self._strict_canonical_regex_parse(model_output, request)
        if strict is None:
            return None
        model_path = (
            request.get("model_path")
            if isinstance(request, dict)
            else getattr(request, "model_path", None)
        )
        if not model_path:
            return strict
        canonical_available, canonical = self._encoding_dsv4_parse_status(
            model_output,
            request=request,
        )
        if not canonical_available:
            # Compatibility for older DSV4 bundles that genuinely ship no
            # canonical completion parser.  Once a selected bundle exposes the
            # parser, its rejection is authoritative and must not be converted
            # into a regex-only executable call.
            return strict
        if canonical is None:
            return None

        def signatures(result: ExtractedToolCallInformation) -> list[tuple[str, Any]]:
            normalized: list[tuple[str, Any]] = []
            for call in result.tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                arguments = call.get("arguments") if isinstance(call, dict) else None
                if not isinstance(name, str) or not isinstance(arguments, str):
                    return []
                try:
                    decoded = json.loads(arguments)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return []
                normalized.append((name, decoded))
            return normalized

        if signatures(canonical) != signatures(strict):
            return None
        return strict

    def _schema_props_required(
        self, schema: dict[str, Any] | None
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(schema, dict):
            return {}, []
        params_schema = schema.get("parameters") or {}
        if not isinstance(params_schema, dict):
            return {}, []
        props = params_schema.get("properties") or {}
        required = params_schema.get("required") or []
        return (
            props if isinstance(props, dict) else {},
            required if isinstance(required, list) else [],
        )

    def _required_satisfied(
        self, args: dict[str, Any], schema: dict[str, Any] | None
    ) -> bool:
        _, required = self._schema_props_required(schema)
        if not all(p in args for p in required):
            return False
        # DSV4 can copy the native schema attribute as the argument value
        # (``string`` / ``string=``).  Presence alone must not make that a
        # schema-valid call: executing it turns a malformed model emission into
        # a hallucinated tool result and poisons the following history turn.
        return not any(
            isinstance(value, str) and value.strip().rstrip("=") == "string"
            for value in args.values()
        )

    def _arguments_match_schema(
        self,
        args: dict[str, Any],
        schema: dict[str, Any],
    ) -> bool:
        """Validate decoded arguments against the request's function schema.

        JSON Schema permits additional properties by default.  Native DSML is
        stricter: once a function explicitly declares a ``properties`` map,
        only those names may execute, including when the map is empty for a
        zero-argument tool.  The shared structured-output validator then owns
        nested types, enums, arrays, and all other JSON-Schema constraints.
        """
        params_schema = schema.get("parameters")
        if params_schema is None:
            return self._required_satisfied(args, schema)
        if not isinstance(params_schema, dict):
            return False

        if "properties" in params_schema:
            props = params_schema.get("properties")
            if not isinstance(props, dict):
                return False
            if any(key not in props for key in args):
                return False

        try:
            from vmlx_engine.api.tool_calling import validate_json_schema

            valid, _error = validate_json_schema(args, params_schema)
        except Exception:
            return False
        return bool(valid) and self._required_satisfied(args, schema)

    def _clean_residue(self, text: str | None) -> str | None:
        """Remove DSML wrapper/invoke residue from visible content."""
        if not text:
            return None
        residue = self._INVOKE_RE.sub("", text)
        residue = self._WRAPPER_RESIDUE_RE.sub("", residue)
        residue = residue.strip()
        if DSML_PREFIX in residue:
            # Any remaining DSML token is malformed wrapper noise, not prose.
            residue = ""
        return residue or None

    def _tool_schemas(self, request: Any | None) -> dict[str, dict[str, Any]]:
        """Return available tool schemas keyed by function name.

        Strict parsing is schema-gated: a canonical call is actionable only
        when its name and required parameters match the request's tools.
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

    def _canonical_arguments_actionable(
        self,
        *,
        name: str,
        arguments: str,
        schemas: dict[str, dict[str, Any]],
    ) -> bool:
        """Validate canonical DSV4 parser output before accepting it.

        Older encoders can recover names while dropping required parameters or
        retaining raw native markup inside arguments. Such output is rejected;
        no secondary repair path may promote it into an executable call.
        """
        raw_markers = (
            f"<{DSML_PREFIX}",
            f"</{DSML_PREFIX}",
            "<param",
            "</param",
            "<parameter",
            "</parameter",
            "<invoke",
            "</inv",
        )
        if any(marker in arguments for marker in raw_markers):
            return False
        if schemas and name not in schemas:
            return False
        schema = schemas.get(name)
        if not schema:
            return True
        try:
            decoded = json.loads(arguments)
        except Exception:
            return False
        if not isinstance(decoded, dict):
            return False
        if any(
            isinstance(value, str) and value.strip().rstrip("=") == "string"
            for value in decoded.values()
        ):
            return False
        return self._arguments_match_schema(decoded, schema)

    def _canonical_encoding_module(self, request: Any | None):
        """The bundle's canonical encoder module (``parse_message_from_completion_text``), or None.
        DSV4: the encoding_dsv4.py located by loaders.dsv4_chat_encoder."""
        try:
            from vmlx_engine.loaders.dsv4_chat_encoder import (
                _load_encoding_dsv4_module,
            )
        except Exception:
            return None
        try:
            if isinstance(request, dict):
                model_path = request.get("model_path")
            else:
                model_path = getattr(request, "model_path", None)
            return _load_encoding_dsv4_module(
                model_path=Path(model_path) if model_path else None
            )
        except Exception:
            return None

    def _encoding_dsv4_parse_status(
        self,
        model_output: str,
        request: Any | None = None,
    ) -> tuple[bool, ExtractedToolCallInformation | None]:
        """Return canonical-parser availability and its validated result.

        ``(False, None)`` means no canonical completion parser is available and
        permits the explicit older-bundle regex compatibility path.
        ``(True, None)`` means the canonical parser was available but rejected
        the bytes or produced a non-actionable call.  That rejection is final;
        callers must not execute a regex-only interpretation.
        """
        enc = self._canonical_encoding_module(request)
        if enc is None:
            return False, None
        parse_fn = getattr(enc, "parse_message_from_completion_text", None)
        if parse_fn is None:
            return False, None

        # The production DSV4 encoder owns two contracts that differ from the
        # older one-argument test adapters:
        #
        # * ``thinking_mode`` is a required second argument; and
        # * the text must include the bundle's EOS token even though the
        #   generation loop removes EOS before tool parsing.
        #
        # Inspect the callable before invoking it so compatibility does not
        # rely on catching TypeError and accidentally hiding a TypeError raised
        # *inside* the canonical parser.
        parser_text = model_output
        canonical_tool_calls_open = self.TOOL_CALLS_OPEN
        if parser_text.startswith(canonical_tool_calls_open):
            # Server parse sites intentionally strip display/reasoning text.
            # The production encoder, however, recognizes a tool-call turn by
            # the canonical separator immediately before the wrapper.
            parser_text = "\n\n" + parser_text
        eos_token = getattr(enc, "eos_token", None)
        if (
            isinstance(eos_token, str)
            and eos_token
            and not parser_text.endswith(eos_token)
        ):
            parser_text += eos_token
        # Parse the fragment we actually received. Most server paths already
        # separated reasoning from content even when the request enabled
        # thinking, so forcing ``thinking`` from request metadata would make
        # the production parser demand a </think> marker that is no longer in
        # this fragment.
        thinking_mode = "thinking" if "</think>" in model_output else "chat"
        try:
            parameters = inspect.signature(parse_fn).parameters
            mode_parameter = parameters.get("thinking_mode")
            accepts_keyword_mode = mode_parameter is not None and (
                mode_parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
            )
            accepts_keyword_mode = accepts_keyword_mode or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_keyword_mode:
                parsed = parse_fn(parser_text, thinking_mode=thinking_mode)
            elif mode_parameter is not None:
                parsed = parse_fn(parser_text, thinking_mode)
            else:
                parsed = parse_fn(parser_text)
        except TypeError:
            # A TypeError from the selected call shape is an implementation
            # defect, not malformed model output. Do not silently retry another
            # signature or hide a TypeError raised inside the bundle parser.
            raise
        except Exception:
            return True, None
        if not isinstance(parsed, dict):
            return True, None
        raw_calls = parsed.get("tool_calls") or []
        if not raw_calls:
            return True, None
        schemas = self._tool_schemas(request)
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
            if not self._canonical_arguments_actionable(
                name=name,
                arguments=args_str,
                schemas=schemas,
            ):
                return True, None
            tool_calls.append(
                self._make_tool_call(
                    name=name, arguments=args_str, id_=generate_tool_id()
                )
            )
        if not tool_calls:
            return True, None
        residue = parsed.get("content") or ""
        if isinstance(residue, list):
            residue = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in residue
            )
        residue = self._clean_residue(residue)
        return (
            True,
            ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=residue,
            ),
        )

    def _try_encoding_dsv4_parse(
        self,
        model_output: str,
        request: Any | None = None,
    ) -> ExtractedToolCallInformation | None:
        """Compatibility wrapper for direct callers of the canonical parser."""
        _available, result = self._encoding_dsv4_parse_status(
            model_output,
            request=request,
        )
        return result

    def extract_tool_calls(
        self, model_output: str, request: Any | None = None
    ) -> ExtractedToolCallInformation:
        """Parse only complete canonical DSML; fail closed on native debris."""
        if not self._has_native_protocol_marker(model_output):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        parsed = self._parse_completed_canonical_calls(model_output, request)
        if parsed is not None:
            return parsed
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=self._safe_invalid_protocol_content(model_output),
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
        """Buffer native markup and emit only shared-parser canonical calls."""
        if not self._has_native_protocol_marker(current_text):
            return self._default_content_delta(delta_text)

        parsed = self._parse_completed_canonical_calls(current_text, request)
        if parsed is None:
            # Incomplete and malformed native protocol are both buffered and
            # never surfaced as assistant content.
            return None

        emitted = len(self.prev_tool_call_arr)
        if emitted >= len(parsed.tool_calls):
            return None
        new_calls = []
        for index, call in enumerate(parsed.tool_calls[emitted:], start=emitted):
            new_calls.append(
                self._make_stream_tool_call_delta(
                    index=index,
                    name=call["name"],
                    arguments=call["arguments"],
                    id_=call["id"],
                )
            )
        self.prev_tool_call_arr.extend(parsed.tool_calls[emitted:])
        return self._pack_stream_tool_calls(new_calls)

    @staticmethod
    def _tail_starts_marker(tail: str, marker: str) -> bool:
        """True when a token-split suffix may be opening another marker."""
        if marker in tail:
            return True
        return any(tail.endswith(marker[:n]) for n in range(len(marker) - 1, 0, -1))

    def _stream_tool_call_stop_end(self, buffered_text: str) -> int | None:
        """Return the safe end offset of a complete schema-valid DSML turn."""
        matches = list(self._INVOKE_RE.finditer(buffered_text))
        if not matches:
            return None

        request = getattr(self, "_stream_stop_request", None)

        first_match = matches[0]
        last_match = matches[-1]
        wrapper_start = buffered_text.rfind(
            self.TOOL_CALLS_OPEN, 0, first_match.start() + 1
        )
        if wrapper_start >= 0:
            wrapper_end_start = buffered_text.find(
                self.TOOL_CALLS_CLOSE, last_match.end()
            )
            if wrapper_end_start < 0:
                return None
            wrapper_end = wrapper_end_start + len(self.TOOL_CALLS_CLOSE)
            wrapper_text = buffered_text[wrapper_start:wrapper_end]
            # Do not stop on the first valid invoke if another canonical invoke
            # has opened but is still incomplete inside the same multi-call
            # wrapper.
            if wrapper_text.count(self.INVOKE_OPEN_PREFIX) != len(
                list(self._INVOKE_RE.finditer(wrapper_text))
            ):
                return None
            tail = buffered_text[wrapper_end:]
            if self._tail_starts_marker(tail, self.TOOL_CALLS_OPEN):
                return None
            if self._parse_completed_canonical_calls(
                buffered_text[:wrapper_end], request
            ) is None:
                return None
            return wrapper_end

        # Older DSV4 encoders may emit a bare invoke without tool_calls wrapper.
        # The server grace window gives an immediately following invoke time to
        # open; once no new marker is opening, the last complete invoke is the
        # native turn boundary.
        bare_end = last_match.end()
        tail = buffered_text[bare_end:]
        if self._tail_starts_marker(tail, self.INVOKE_OPEN_PREFIX):
            return None
        if self._parse_completed_canonical_calls(
            buffered_text[:bare_end], request
        ) is None:
            return None
        return bare_end

    def stream_tool_calls_complete(self, buffered_text: str) -> bool:
        """True once a complete, request-schema-valid DSML turn has closed."""
        return self._stream_tool_call_stop_end(buffered_text) is not None

    def stream_tool_call_stop_truncate(self, buffered_text: str) -> str:
        """Drop post-call rambling after the native DSML turn boundary."""
        end = self._stream_tool_call_stop_end(buffered_text)
        return buffered_text if end is None else buffered_text[:end]

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
