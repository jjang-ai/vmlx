# SPDX-License-Identifier: Apache-2.0
"""Spark-X2.5 native XML-argument calls.

The native template emits string values verbatim and JSON-encodes other values.
Never strip file contents or turn a schema-declared string into a number.
Schema enforcement remains the server's shared warn/enforce policy.
"""
import json
import re

from .abstract_tool_parser import (
    ExtractedToolCallInformation, ToolParser, ToolParserManager, generate_tool_id,
)


@ToolParserManager.register_module(["spark25"])
class Spark25ToolParser(ToolParser):
    NATIVE_MARKERS = ("<tool_call>",)
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    # A parsed call rejected by the request's allow-list is protocol, not prose.
    # Preserve extracted surrounding content and the server's drop diagnostic.
    SUPPRESS_INVALID_NATIVE_MARKUP = True
    _CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

    @staticmethod
    def _decode(value, schema):
        kinds = schema.get("type") if isinstance(schema, dict) else None
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if "null" in kinds and value.strip() == "null":
            return None
        if "string" in kinds:
            return value
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value

    def extract_tool_calls(self, model_output, request=None):
        text = self.strip_think_tags(model_output)
        calls = []
        spans = []
        for match in self._CALL.finditer(text):
            block = match.group(1)
            start = block.find("<arg_key>")
            name = (block if start < 0 else block[:start]).strip()
            arguments_text = "" if start < 0 else block[start:]
            if not name or "<" in name or ">" in name:
                raise ValueError("Malformed Spark tool function name")
            schema = self._function_schema_for_tool(request, name) or {}
            properties = schema.get("properties") or {}
            arguments = {}
            cursor = 0
            for arg in self._ARG.finditer(arguments_text):
                if arguments_text[cursor:arg.start()].strip():
                    raise ValueError("Malformed Spark argument envelope")
                key = arg.group(1).strip()
                if not key or key in arguments or "<" in key or ">" in key:
                    raise ValueError("Invalid or duplicate Spark argument key")
                arguments[key] = self._decode(arg.group(2), properties.get(key))
                cursor = arg.end()
            if arguments_text[cursor:].strip():
                raise ValueError("Incomplete Spark argument envelope")
            calls.append({"id": generate_tool_id(), "name": name,
                          "arguments": json.dumps(arguments, ensure_ascii=False)})
            spans.append(match.span())
        remaining = text
        for start, end in reversed(spans):
            remaining = remaining[:start] + remaining[end:]
        if "<tool_call>" in remaining or "</tool_call>" in remaining:
            raise ValueError("Incomplete Spark tool call")
        return ExtractedToolCallInformation(
            tools_called=bool(calls), tool_calls=calls,
            content=remaining if remaining.strip() else None,
        )
