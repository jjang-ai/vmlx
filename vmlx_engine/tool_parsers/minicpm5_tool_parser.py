# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5 function/param XML, including byte-preserving CDATA strings.

Only complete, well-formed calls are executable. No argument guessing, partial
call repair, or tool invocation from a reasoning block. The server buffers this
dialect through the ordinary per-generation durability fence before publication.
"""

import json
import html
import re
import xml.etree.ElementTree as ET

from .abstract_tool_parser import (
    ExtractedToolCallInformation, ToolParser, ToolParserManager, generate_tool_id,
)


def _decode_value(value, schema):
    kind = schema.get("type") if isinstance(schema, dict) else None
    kinds = kind if isinstance(kind, list) else [kind]
    # An absent schema is a string, not permission to infer numbers or booleans.
    if kind is None or kind == "string":
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    if parsed is None and "null" in kinds:
        return None
    if "string" in kinds:
        return value
    if isinstance(parsed, bool):
        return parsed if "boolean" in kinds else value
    if isinstance(parsed, int):
        return parsed if "integer" in kinds or "number" in kinds else value
    if isinstance(parsed, float):
        return parsed if "number" in kinds and parsed not in (float("inf"), float("-inf")) and parsed == parsed else value
    if isinstance(parsed, dict) and "object" in kinds:
        return parsed
    if isinstance(parsed, list) and "array" in kinds:
        return parsed
    return value


@ToolParserManager.register_module(["minicpm5", "minicpm5_xml_function"])
class MiniCPM5ToolParser(ToolParser):
    NATIVE_MARKERS = ("<function",)
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    # CDATA can contain literal closing function/param tags. Match it as an
    # indivisible unit before considering the outer closing delimiter.
    CALL = re.compile(r'<function\s+name="[^"]+"\s*>(?:<!\[CDATA\[.*?\]\]>|(?!</function>).)*</function>', re.DOTALL)

    def extract_tool_calls(self, model_output, request=None):
        # Mask CDATA while stripping reasoning: code strings containing literal
        # <think> tags are data and must not be changed by the shared stripper.
        values = []
        def hide(match):
            values.append(match.group())
            return f"\x00CDATA{len(values) - 1}\x00"
        text = re.sub(r"<!\[CDATA\[.*?\]\]>", hide, model_output, flags=re.DOTALL)
        text = self.strip_think_tags(text)
        text = re.sub(r"\x00CDATA(\d+)\x00", lambda m: values[int(m[1])], text)
        calls, content, end = [], [], 0
        for match in self.CALL.finditer(text):
            raw = match.group()
            try:
                # Function fragments have no document type/entity declaration.
                # XML newline normalization must not change a code argument's
                # CRLF bytes. Escape CDATA as text before handing it to XML.
                xml = re.sub(r"<!\[CDATA\[(.*?)\]\]>",
                             lambda m: html.escape(m[1], quote=False), raw, flags=re.DOTALL)
                node = ET.fromstring(xml.replace("\r", "&#13;"))
                if node.tag != "function" or set(node.attrib) != {"name"} or not node.attrib["name"].strip():
                    continue
                if node.text and node.text.strip():
                    continue
                schema = self._function_schema_for_tool(request, node.attrib["name"]) or {}
                props = schema.get("properties") or {}
                args = {}
                for param in node:
                    if (param.tag != "param" or set(param.attrib) != {"name"}
                            or not param.attrib["name"] or len(param)
                            or param.attrib["name"] in args or (param.tail and param.tail.strip())):
                        raise ValueError("invalid or duplicate parameter")
                    args[param.attrib["name"]] = _decode_value(param.text or "", props.get(param.attrib["name"]))
            except (ET.ParseError, ValueError, TypeError):
                continue
            calls.append({"id": generate_tool_id(), "name": node.attrib["name"],
                          "arguments": json.dumps(args, ensure_ascii=False, allow_nan=False)})
            content.append(text[end:match.start()])
            end = match.end()
        content.append(text[end:])
        return ExtractedToolCallInformation(bool(calls), calls, "".join(content).strip() or None)
