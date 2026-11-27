# SPDX-License-Identifier: Apache-2.0
"""Decode the native JSON tool channel described by the Inkling Mini template."""

import json
import re

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    generate_tool_id,
)

TOOL_CHANNEL = "<|content_invoke_tool_json|>"
MESSAGE_END = "<|end_message|>"
_DECODER = json.JSONDecoder()
_CHANNEL = re.compile(
    r"(?:<\|message_model\|>([A-Za-z0-9_-]{1,64}))?" + re.escape(TOOL_CHANNEL)
)


def read_tool_message(text, start):
    """Decode a payload only when its enclosing message has closed."""
    start += len(text[start:]) - len(text[start:].lstrip())
    try:
        payload, end = _DECODER.raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    end += len(text[end:]) - len(text[end:].lstrip())
    if not text.startswith(MESSAGE_END, end):
        return None
    return payload, end + len(MESSAGE_END)


@ToolParserManager.register_module(["inkling", "inkling2", "inkling2_loop"])
class InklingToolParser(ToolParser):
    NATIVE_MARKERS = (TOOL_CHANNEL,)
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def extract_tool_calls(self, model_output, request=None):
        calls, visible = [], []
        offset = 0
        while match := _CHANNEL.search(model_output, offset):
            visible.append(model_output[offset:match.start()])
            result = read_tool_message(model_output, match.end())
            if result is None:
                return ExtractedToolCallInformation(
                    bool(calls), calls, "".join(visible) or None
                )
            payload, offset = result
            if not isinstance(payload, dict):
                continue
            name, arguments = payload.get("name"), payload.get("args")
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None
                or not isinstance(arguments, dict)
                or (match[1] is not None and match[1] != name)
            ):
                continue
            calls.append({
                "id": generate_tool_id(),
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            })
        visible.append(model_output[offset:])
        return ExtractedToolCallInformation(bool(calls), calls, "".join(visible) or None)
