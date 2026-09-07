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
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
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

    @classmethod
    def _coerce_value(cls, value: str) -> Any:
        value = value.strip()
        wrapped = cls.VALUE_WRAPPER_PATTERN.match(value)
        if wrapped:
            value = wrapped.group(1).strip()
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

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
    def _extract_arguments_from_body(cls, body: str) -> dict[str, Any]:
        """Extract args trying strict `<parameter=K>V</parameter>` first, the
        Ornith `<arg_key>K</arg_key><value>V</value>` fallback, then the
        Qwen3.6 miskeyed `<function=K>V</parameter>` shape (correct function
        name, parameter opened with the wrong tag — observed live on the
        Responses stream path as an empty-args call the required-args
        validator then dropped)."""
        arguments: dict[str, Any] = {}
        for param_name, param_value in cls.PARAM_PATTERN.findall(body):
            arguments[param_name.strip()] = cls._coerce_value(param_value)
        if not arguments:
            for k, v in cls._RECOVERY_SPLIT_KEY_PARAM.findall(body):
                arguments[k.strip()] = cls._coerce_value(v)
        if not arguments:
            for k, v in cls.ORNITH_ARG_KEY_VALUE_PATTERN.findall(body):
                arguments[k.strip()] = cls._coerce_value(v)
        if not arguments:
            for k, v in cls._RECOVERY_MISKEYED_PARAM.findall(body):
                arguments[k.strip()] = cls._coerce_value(v)
        return arguments

    @classmethod
    def _parse_functions(
        cls, text: str, *, allowed_names: set[str] | None = None
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
            arguments = cls._extract_arguments_from_body(body)
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
        cls, text: str, *, allowed_names: set[str] | None = None
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
            args_match = cls.ARGUMENTS_PATTERN.search(body)
            if args_match:
                args_body = args_match.group(1)
                for arg_name, arg_value in cls.SIMPLE_XML_ARG_PATTERN.findall(args_body):
                    arguments[arg_name.strip()] = cls._coerce_value(arg_value)
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
                repaired_calls = self._parse_functions(
                    model_output, allowed_names=allowed_names
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
            parsed = self._parse_functions(block)
            # A doubled `<function=function>` wrapper parses as one bogus call
            # named "function" with no arguments; recover the real nested call
            # when the request's tool names disambiguate it.
            if allowed_names_for_recovery and parsed and all(
                call.get("name") not in allowed_names_for_recovery
                for call in parsed
            ):
                recovered = self._recover_doubled_wrapper_calls(
                    block, allowed_names=allowed_names_for_recovery
                )
                if recovered:
                    parsed = recovered
            tool_calls.extend(parsed)
            if not tool_calls:
                tool_calls.extend(self._parse_nested_invoke_functions(block))

        cleaned_text = self.TOOL_CALL_PATTERN.sub("", model_output).strip()
        if not tool_calls:
            allowed_names = self._request_tool_names(request)
            if allowed_names and "<function=" in model_output:
                repaired_calls = self._parse_functions(
                    model_output,
                    allowed_names=allowed_names,
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
            if allowed_names and "<tool_name>" in model_output:
                repaired_calls = self._parse_nested_invoke_functions(
                    model_output,
                    allowed_names=allowed_names,
                )
                if repaired_calls:
                    cleaned_text = self._strip_repaired_function_blocks(model_output)
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=repaired_calls,
                        content=cleaned_text if cleaned_text else None,
                    )
        if tool_calls:
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
    _NULL_SPELLINGS = frozenset({"none", "null", "nil"})

    @classmethod
    def _schema_allows_null(cls, prop: Any) -> bool:
        if not isinstance(prop, dict):
            return False
        if prop.get("nullable") is True:
            return True
        typ = prop.get("type")
        if typ == "null":
            return True
        if isinstance(typ, (list, tuple)) and "null" in typ:
            return True
        for key in ("anyOf", "oneOf"):
            options = prop.get(key)
            if isinstance(options, list) and any(
                isinstance(o, dict) and o.get("type") == "null" for o in options
            ):
                return True
        return False

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
