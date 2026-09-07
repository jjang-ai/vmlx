# SPDX-License-Identifier: Apache-2.0
"""
dots (dots3-note) tool call parser for vmlx-engine.

Handles the dots XML function-call dialect emitted by dots3_note bundles:

- <dots_function_call>
    <invoke name="func_name">
    <parameter name="param1">
    value1
    </parameter>
    <parameter name="param2">
    {"nested": "json for non-string args"}
    </parameter>
    </invoke>
  </dots_function_call>

Dialect contract (from the bundle's chat_template.jinja):
- multiple <invoke> blocks may appear inside one <dots_function_call>;
- the template writes parameter bodies as ``\\n{value}\\n``, so exactly one
  leading and one trailing newline are frame characters — inner whitespace
  (e.g. code indentation) is payload and must survive;
- string-typed arguments are emitted verbatim while non-string arguments are
  serialized with ``|tojson`` — the request tool schema decides which way a
  body is read back.
"""

import json
import re
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)


def _extract_name(name_str: str) -> str:
    """Extract name from a possibly quoted attribute value."""
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or (
        name_str.startswith("'") and name_str.endswith("'")
    ):
        return name_str[1:-1]
    return name_str


def _strip_frame_newlines(value: str) -> str:
    """Remove exactly one leading and one trailing newline frame.

    The dots template writes ``<parameter name=..>\\n{value}\\n</parameter>``;
    those two newlines are markup, everything else — including leading
    indentation on the first payload line — is the argument itself. A plain
    ``str.strip()`` here corrupts code-valued arguments.
    """
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


@ToolParserManager.register_module(["dots", "dots3", "dots3_note"])
class DotsToolParser(ToolParser):
    """Tool call parser for dots3_note models (dots XML dialect)."""

    NATIVE_MARKERS = ("<dots_function_call>",)

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(
        r"<dots_function_call>(.*?)</dots_function_call>",
        re.DOTALL,
    )

    INVOKE_PATTERN = re.compile(
        r"<invoke name=([^>]+)>(.*?)</invoke>",
        re.DOTALL,
    )

    PARAM_PATTERN = re.compile(
        r"<parameter name=([^>]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    # Lenient variants for generations that emit the native dialect but stop
    # (max_tokens) before all closing tags. Scoped to a valid opener so
    # arbitrary visible XML is never promoted to a tool call.
    LENIENT_INVOKE_PATTERN = re.compile(
        r"<invoke name=([^>]+)>(.*?)(?=(?:</invoke>|<invoke name=|$))",
        re.DOTALL,
    )
    LENIENT_PARAM_PATTERN = re.compile(
        r"<parameter name=([^>]+)>(.*?)(?=(?:</parameter>|<parameter name=|</invoke>|$))",
        re.DOTALL,
    )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned_text = self.strip_think_tags(model_output)

        if "<dots_function_call>" not in cleaned_text:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

        tool_calls: list[dict[str, Any]] = []

        for block_match in self.TOOL_CALL_PATTERN.finditer(cleaned_text):
            block_content = block_match.group(1)
            invoke_found = False
            for invoke_match in self.INVOKE_PATTERN.finditer(block_content):
                invoke_found = True
                tool_call = self._tool_call_from_invoke(
                    invoke_match.group(1),
                    invoke_match.group(2),
                    lenient=False,
                    request=request,
                )
                if tool_call:
                    tool_calls.append(tool_call)
            if not invoke_found:
                for invoke_match in self.LENIENT_INVOKE_PATTERN.finditer(
                    block_content
                ):
                    tool_call = self._tool_call_from_invoke(
                        invoke_match.group(1),
                        invoke_match.group(2),
                        lenient=True,
                        request=request,
                    )
                    if tool_call:
                        tool_calls.append(tool_call)

        if not tool_calls:
            # Native opener present but the wrapper never closed (truncated
            # generation): parse the unterminated tail leniently.
            open_idx = cleaned_text.rfind("<dots_function_call>")
            tail = cleaned_text[open_idx + len("<dots_function_call>"):]
            if "</dots_function_call>" not in tail:
                for invoke_match in self.LENIENT_INVOKE_PATTERN.finditer(tail):
                    tool_call = self._tool_call_from_invoke(
                        invoke_match.group(1),
                        invoke_match.group(2),
                        lenient=True,
                        request=request,
                    )
                    if tool_call:
                        tool_calls.append(tool_call)

        content_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text)
        if tool_calls and "<dots_function_call>" in content_text:
            content_text = re.sub(
                r"<dots_function_call>.*$", "", content_text, flags=re.DOTALL
            )
        content_text = content_text.strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_text if content_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=cleaned_text
        )

    def _tool_call_from_invoke(
        self,
        raw_func_name: str,
        invoke_content: str,
        *,
        lenient: bool,
        request: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        func_name = _extract_name(raw_func_name)
        if not func_name:
            return None
        schema = self._function_schema_for_tool(request, func_name)
        properties = (
            schema.get("properties") if isinstance(schema, dict) else None
        )
        if not isinstance(properties, dict):
            properties = {}

        params = (
            self.LENIENT_PARAM_PATTERN.findall(invoke_content)
            if lenient
            else self.PARAM_PATTERN.findall(invoke_content)
        )
        arguments: dict[str, Any] = {}
        for param_name, param_value in params:
            clean_name = _extract_name(param_name)
            arguments[clean_name] = self._coerce_value(
                _strip_frame_newlines(param_value),
                properties.get(clean_name),
            )
        if not arguments and lenient:
            return None
        return {
            "id": generate_tool_id(),
            "name": func_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    @staticmethod
    def _coerce_value(value: str, prop_schema: Any) -> Any:
        """Read a parameter body back through the emit contract.

        The template emits string arguments verbatim and everything else via
        ``|tojson``. With a schema, its declared type is authoritative: a
        string-typed argument stays verbatim even when it happens to look
        like JSON. Without a schema, JSON that parses is taken as JSON.
        """
        declared = (
            prop_schema.get("type") if isinstance(prop_schema, dict) else None
        )
        # A JSON-Schema TYPE LIST (``["string", "null"]`` for a nullable
        # parameter) is a declaration, not a decode hint: a string member keeps
        # the value verbatim, a null spelling decodes to None only when null is
        # a member, and only a list without "string" falls through to JSON.
        if isinstance(declared, (list, tuple)):
            kinds = [str(t) for t in declared if t is not None]
            if "null" in kinds and value.strip().lower() in {"null", "none", "nil"}:
                return None
            if "string" in kinds:
                return value
            declared = kinds[0] if len(kinds) == 1 else "object"
        if declared == "string":
            return value
        if declared is not None:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return value
