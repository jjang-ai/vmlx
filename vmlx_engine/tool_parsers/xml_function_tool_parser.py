# SPDX-License-Identifier: Apache-2.0
"""Generic XML function-call parser.

Parses templates that emit:

<tool_call>
<function=name>
<parameter=arg>value</parameter>
</function>
</tool_call>
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


@ToolParserManager.register_module(["xml_function", "mimo_xml_function", "qwen3_coder"])
class XMLFunctionToolParser(ToolParser):
    """Parse XML function calls used by MiMo-style chat templates.

    ``qwen3_coder`` is registered here rather than on the Qwen parser because
    the formats differ and the name has to follow the FORMAT: Qwen3.6-27B
    D-series bundles are stamped ``tool parser qwen3_coder`` and their chat
    template emits exactly this parser's shape —
    ``<tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>`` —
    not the plain ``<tool_call>{json}`` the ``qwen``/``qwen3`` parser reads.
    Verified against the bundle's chat_template.jinja, not inferred from the
    name. Before this name existed the engine exited at startup
    ("Tool parser 'qwen3_coder' not found"), which took the whole session down
    for every stamped D-series bundle.
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        template = getattr(tokenizer, "chat_template", None)
        # Naive's native template explicitly declares literal parameter bodies
        # and serializes them inline. Existing Qwen/MiMo templates frame values
        # with newlines. Select per instance; never change shared parser state.
        literal = (
            isinstance(template, str)
            and "The value enclosed between parameter tags is preserved exactly as-is, including newlines and spaces." in template
            and "'<parameter=' + args_name + '>'" in template
        )
        self._parameter_parser = _LiteralXMLFunctionToolParser if literal else type(self)

    NATIVE_MARKERS = ("<function=", "<function_call>", "<tool_call>")

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    # The value is captured RAW. The template frames a value with one newline
    # on each side (``<parameter=K>\nV\n</parameter>``); only that frame is
    # removed, in ``_unframe``. A ``\s*`` on both ends here ate a string
    # argument's own leading indentation and trailing newlines — a file whose
    # first line is indented, or that ends in blank lines, was written wrong
    # (LOSSLESS-API-TOOL-HISTORY-CONTRACT: escaping and whitespace are data).
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )
    # Ornith-1.0 (qwen3.5 + Gemma vocab frankenmerge) emits a malformed
    # variant of the template-specified `<parameter=K>V</parameter>` format:
    #   <arg_key>K</arg_key>
    #   <value>V</value>
    # sometimes followed by a spurious extra `</arg_key>`. Accept this pair
    # as a fallback when the standard pattern matches nothing in the body.
    # Engine tolerance for a model-quality merge artifact.
    ORNITH_ARG_KEY_VALUE_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<value>\s*(.*?)\s*</value>",
        re.DOTALL,
    )
    INVOKE_PATTERN = re.compile(
        r"<invoke>\s*(.*?)\s*</invoke>",
        re.DOTALL,
    )
    TOOL_NAME_PATTERN = re.compile(
        r"<tool_name>\s*(.*?)\s*</tool_name>",
        re.DOTALL,
    )
    ARGUMENTS_PATTERN = re.compile(
        r"<arguments>\s*(.*?)\s*</arguments>",
        re.DOTALL,
    )
    SIMPLE_XML_ARG_PATTERN = re.compile(
        r"<([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</\1>",
        re.DOTALL,
    )
    VALUE_WRAPPER_PATTERN = re.compile(
        r"^<value>\s*(.*?)\s*</value>$",
        re.DOTALL,
    )

    @staticmethod
    def _unframe(value: str) -> str:
        """Drop the template's one-newline frame on each side, nothing more."""
        if value.startswith("\r\n"):
            value = value[2:]
        elif value.startswith("\n"):
            value = value[1:]
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]
        return value

    @classmethod
    def _schema_is_boolean_or_null(cls, prop: Any) -> bool:
        """Only unambiguous boolean schemas may accept Python boolean spelling."""
        if not isinstance(prop, dict):
            return False
        typ = prop.get("type")
        if typ == "boolean":
            return True
        if isinstance(typ, (list, tuple)):
            return "boolean" in typ and all(t in ("boolean", "null") for t in typ)
        for key in ("anyOf", "oneOf"):
            options = prop.get(key)
            if isinstance(options, list) and options:
                booleans = [cls._schema_is_boolean_or_null(o) for o in options]
                if any(booleans) and all(is_bool or (isinstance(o, dict) and o.get("type") == "null")
                                         for o, is_bool in zip(options, booleans)):
                    return True
        return False

    @classmethod
    def _coerce_value(cls, value: str, prop_schema: Any = None) -> Any:
        # Strip only to TEST for a JSON shape (and the <value> wrapper); the
        # string result keeps its own bytes — indentation, trailing newlines,
        # literal backslash sequences — exactly as the model wrote them.
        raw = cls._unframe(value)
        # The native template emits string parameters verbatim, but JSON for
        # non-strings. Decode using the declared type BEFORE json.loads loses
        # string bytes: JSON source code, numeric filenames, "null", quotes,
        # escapes and trailing whitespace can all be legitimate string data.
        if cls._schema_is_string_or_null(prop_schema):
            if cls._schema_allows_null(prop_schema) and raw.strip().lower() in cls._NULL_SPELLINGS:
                return None
            return raw
        candidate = raw.strip()
        wrapped = cls.VALUE_WRAPPER_PATTERN.match(candidate)
        if wrapped:
            raw = wrapped.group(1)
            candidate = raw.strip()
        # XML scalar bodies are not themselves JSON envelopes. Some native
        # generations use unquoted Python True/False; leaving False as a
        # nonempty string can invert a client's boolean branch. Decode only
        # these exact spellings under an explicitly boolean schema. Quoted
        # strings, ambiguous schemas and JSON-native calls stay untouched.
        if candidate in ("True", "False") and cls._schema_is_boolean_or_null(prop_schema):
            return candidate == "True"
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return raw

    # _request_tool_names and the doubled-wrapper recovery live on ToolParser:
    # the malformed shape arrives on both the qwen and xml_function routes.

    # Relaxed function pattern: Ornith-1.0 may emit `<function=name>` without
    # the matching `</function>` close. Body extends to the next `</tool_call>`
    # or end-of-string. Only used as fallback when the strict pattern misses.
    RELAXED_FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>\s*(.*?)\s*(?=</function>|</tool_call>|$)",
        re.DOTALL,
    )

    @classmethod
    def _recovery_arguments_from_body(cls, body: str) -> dict[str, Any]:
        # Route the shared doubled-wrapper recovery through this parser's own
        # argument extraction so the Ornith arg_key/value fallback still works.
        return cls._extract_arguments_from_body(body)

    @classmethod
    def _extract_arguments_from_body(
        cls, body: str, properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract args trying strict `<parameter=K>V</parameter>` first, the
        Ornith `<arg_key>K</arg_key><value>V</value>` fallback, then the
        Qwen3.6 miskeyed `<function=K>V</parameter>` shape (correct function
        name, parameter opened with the wrong tag — observed live on the
        Responses stream path as an empty-args call the required-args
        validator then dropped)."""
        arguments: dict[str, Any] = {}
        properties = properties or {}
        for param_name, param_value in cls.PARAM_PATTERN.findall(body):
            arguments[param_name.strip()] = cls._coerce_value(param_value, properties.get(param_name.strip()))
        if not arguments:
            for k, v in cls._RECOVERY_SPLIT_KEY_PARAM.findall(body):
                arguments[k.strip()] = cls._coerce_value(v, properties.get(k.strip()))
        if not arguments:
            for k, v in cls.ORNITH_ARG_KEY_VALUE_PATTERN.findall(body):
                arguments[k.strip()] = cls._coerce_value(v, properties.get(k.strip()))
        if not arguments:
            for k, v in cls._RECOVERY_MISKEYED_PARAM.findall(body):
                arguments[k.strip()] = cls._coerce_value(v, properties.get(k.strip()))
        if not arguments and body.strip().startswith("{"):
            # A JSON-object body is a template-defined form, not a malformation: MiMo-V2.6's vendor template renders a
            # call's string `arguments` (and custom-tool `input`) verbatim inside <function=NAME>...</function>, so the
            # model emits it after such a history turn.
            try:
                parsed = json.loads(body.strip())
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                arguments = parsed
        return arguments

    @classmethod
    def _parse_functions(
        cls, text: str, *, allowed_names: set[str] | None = None,
        request: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        matches = list(cls.FUNCTION_PATTERN.findall(text))
        if not matches:
            # Fallback: Ornith-1.0 may emit `<function=name>` without `</function>`.
            matches = list(cls.RELAXED_FUNCTION_PATTERN.findall(text))
        for func_name, body in matches:
            name = func_name.strip()
            if allowed_names is not None and name not in allowed_names:
                continue
            arguments = cls._extract_arguments_from_body(body, cls._argument_properties(request, name))
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    @classmethod
    def _parse_nested_invoke_functions(
        cls, text: str, *, allowed_names: set[str] | None = None,
        request: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for body in cls.INVOKE_PATTERN.findall(text):
            name_match = cls.TOOL_NAME_PATTERN.search(body)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            if allowed_names is not None and name not in allowed_names:
                continue
            arguments: dict[str, Any] = {}
            properties = cls._argument_properties(request, name)
            args_match = cls.ARGUMENTS_PATTERN.search(body)
            if args_match:
                args_body = args_match.group(1)
                for arg_name, arg_value in cls.SIMPLE_XML_ARG_PATTERN.findall(args_body):
                    arguments[arg_name.strip()] = cls._coerce_value(arg_value, properties.get(arg_name.strip()))
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    @classmethod
    def _strip_repaired_function_blocks(cls, text: str) -> str:
        cleaned = cls.FUNCTION_PATTERN.sub("", text)
        cleaned = cls.INVOKE_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"</?function_call>", "", cleaned)
        cleaned = cleaned.replace("<tool_call>", "")
        cleaned = cleaned.replace("</tool_call>", "")
        cleaned = re.sub(r"```(?:xml|XML)?", "", cleaned)
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        if "<tool_call>" not in model_output:
            allowed_names = self._request_tool_names(request)
            if (
                allowed_names
                and "</tool_call>" in model_output
                and "<function=" in model_output
            ):
                repaired_calls = self._parameter_parser._parse_functions(
                    model_output, allowed_names=allowed_names, request=request
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[dict[str, Any]] = []
        allowed_names_for_recovery = self._request_tool_names(request)
        for block in self.TOOL_CALL_PATTERN.findall(model_output):
            parsed = self._parameter_parser._parse_functions(block, request=request)
            # A doubled `<function=function>` wrapper parses as one bogus call
            # named "function" with no arguments; recover the real nested call
            # when the request's tool names disambiguate it.
            if allowed_names_for_recovery and parsed and all(
                call.get("name") not in allowed_names_for_recovery
                for call in parsed
            ):
                recovered = self._recover_doubled_wrapper_calls(
                    block, allowed_names=allowed_names_for_recovery,
                    argument_parser=lambda name, body: self._parameter_parser._extract_arguments_from_body(
                        body, self._argument_properties(request, name)),
                )
                if recovered:
                    parsed = recovered
            tool_calls.extend(parsed)
            if not tool_calls:
                tool_calls.extend(self._parameter_parser._parse_nested_invoke_functions(block, request=request))

        cleaned_text = self.TOOL_CALL_PATTERN.sub("", model_output).strip()
        if not tool_calls:
            allowed_names = self._request_tool_names(request)
            if allowed_names and "<function=" in model_output:
                repaired_calls = self._parameter_parser._parse_functions(
                    model_output,
                    allowed_names=allowed_names,
                    request=request,
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
            if allowed_names and "<tool_name>" in model_output:
                repaired_calls = self._parameter_parser._parse_nested_invoke_functions(
                    model_output,
                    allowed_names=allowed_names,
                    request=request,
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
        if tool_calls:
            # A length-limited parallel-call turn may end inside its next
            # envelope. It is neither executable nor visible prose. Keep
            # completed calls, but do not expose the unfinished native XML.
            partial = re.match(r"^<tool_call>\s*<function=([^>]+)>", cleaned_text)
            if partial and partial.group(1).strip() in (allowed_names_for_recovery or set()):
                from ..request_diagnostics import record

                record("xml_function: incomplete trailing tool call omitted; completed calls retained")
                cleaned_text = ""
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=self._coerce_nullable_arguments(tool_calls, request),
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    # XML parameter values are text: a model writes ``None``/``null`` for a
    # parameter the schema declares nullable, and json.loads keeps ``None`` as
    # the STRING "None" (live 2026-09-07, Qwen3.8-27B: search(path="None")
    # searched a directory literally named None). Only a schema that allows
    # null turns those spellings into JSON null; a plain string field keeps
    # the text the model wrote.
    def _coerce_nullable_arguments(
        self, tool_calls: list[dict[str, Any]], request: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not request:
            return tool_calls
        out: list[dict[str, Any]] = []
        for call in tool_calls:
            schema = self._function_schema_for_tool(request, str(call.get("name") or ""))
            props = (schema or {}).get("properties") if isinstance(schema, dict) else None
            raw = call.get("arguments")
            if not isinstance(props, dict) or not isinstance(raw, str):
                out.append(call)
                continue
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                out.append(call)
                continue
            if not isinstance(args, dict):
                out.append(call)
                continue
            changed = False
            for key, value in list(args.items()):
                if (
                    isinstance(value, str)
                    and value.strip().lower() in self._NULL_SPELLINGS
                    and self._schema_allows_null(props.get(key))
                ):
                    args[key] = None
                    changed = True
            if changed:
                call = dict(call)
                call["arguments"] = json.dumps(args, ensure_ascii=False)
            out.append(call)
        return out

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
        if "<tool_call>" not in current_text:
            return {"content": delta_text}
        if "</tool_call>" in delta_text:
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


class _LiteralXMLFunctionToolParser(XMLFunctionToolParser):
    """Native inline parameter bodies have no removable newline frame."""

    @staticmethod
    def _unframe(value: str) -> str:
        return value
