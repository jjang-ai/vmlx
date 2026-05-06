# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

Supports parsing tool calls from multiple model formats:
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Llama: <function=name>{"arg": "value"}</function>

Also includes structured output (JSON Schema) utilities:
- parse_json_output: Extract JSON from model output
- validate_json_schema: Validate JSON against a schema
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import validate, ValidationError

from .models import FunctionCall, ResponseFormat, ToolCall

logger = logging.getLogger(__name__)


def check_and_inject_fallback_tools(
    prompt: Optional[str],
    messages: List[Dict[str, Any]],
    template_tools: Optional[List[dict]],
    tokenizer: Any,
    template_kwargs: dict
) -> Optional[str]:
    """
    Check if the chat template silently dropped tool definitions, and if so,
    re-apply the template with an injected system message containing the tools.

    This fixes models like Qwen 2.5/3 where disabling reasoning causes the
    chat template to completely ignore the tools kwarg, as well as generic models
    without tool-aware templates.
    """
    if not template_tools or prompt is None:
        return prompt

    # Check if at least one tool name made it into the prompt
    tool_names = [t.get("function", {}).get("name", "") for t in template_tools]
    tool_names = [name for name in tool_names if name]

    if not tool_names:
        return prompt

    # Some templates need more than tool-name visibility. DSV4 may mention
    # schemas without a parser-matching DSML exemplar. Qwen3.5/3.6 MoE's
    # shipped template includes only an `example_function_name` exemplar; live
    # tests showed it could answer with a fake directory listing instead of a
    # native tool call even when the user explicitly said to use the tool.
    is_dsv4_prompt = "<｜User｜>" in prompt or "<｜Assistant｜>" in prompt
    is_qwen_native_tool_prompt = (
        "<|im_start|>" in prompt
        and "<tools>" in prompt
        and "<function=example_function_name>" in prompt
    )

    # If ALL tool names made it into a prompt and the prompt also contains a
    # concrete parser-native exemplar for those names, the template handled
    # tools correctly. Otherwise inject a concrete native exemplar.
    _dsv4_has_native_dsml_schema = (
        is_dsv4_prompt
        and "<｜DSML｜tool_calls>" in prompt
        and all(name in prompt for name in tool_names)
    )
    _dsv4_has_concrete_dsml_examples = (
        is_dsv4_prompt
        and all(f'<｜DSML｜invoke name="{name}"' in prompt for name in tool_names)
    )
    _qwen_has_concrete_tool_examples = (
        is_qwen_native_tool_prompt
        and all(f"<function={name}>" in prompt for name in tool_names)
    )
    if all(name in prompt for name in tool_names) and (
        (not is_dsv4_prompt or _dsv4_has_native_dsml_schema or _dsv4_has_concrete_dsml_examples)
        and (not is_qwen_native_tool_prompt or _qwen_has_concrete_tool_examples)
    ):
        return prompt

    logger.warning("Chat template needs fallback tool schema injection.")

    # Format fallback tool schema
    tool_descs = []
    for tool in template_tools:
        func = tool.get("function", {})
        tool_descs.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {})
        })

    # DSV4's native parser is DSML, not generic <tool_call> JSON. Its shipped
    # templates currently do not render tool schemas, so this fallback is the
    # only place the model sees tool instructions on OpenAI/Responses tool
    # requests. Detect by the canonical DSV4 turn tokens already present in
    # the rendered prompt and inject the parser-matching format.
    if is_dsv4_prompt:
        dsv4_lines = [
            "You have access to these tools. Call them using DSML format.",
            "",
        ]
        first_name = "FUNCTION_NAME"
        first_param = "arg1"
        for idx, tool in enumerate(template_tools):
            func = tool.get("function", {})
            name = func.get("name", "") or "unknown_tool"
            if idx == 0:
                first_name = name
            dsv4_lines.append(f"Tool: {name}")
            desc = func.get("description", "")
            if desc:
                dsv4_lines.append(f"  description: {desc}")
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", []) if isinstance(params, dict) else [])
            if props:
                dsv4_lines.append("  parameters:")
                for p_name, p_schema in props.items():
                    if idx == 0 and first_param == "arg1":
                        first_param = p_name
                    p_type = (
                        p_schema.get("type", "string")
                        if isinstance(p_schema, dict)
                        else "string"
                    )
                    req = "required" if p_name in required else "optional"
                    p_desc = (
                        p_schema.get("description", "")
                        if isinstance(p_schema, dict)
                        else ""
                    )
                    suffix = f": {p_desc}" if p_desc else ""
                    dsv4_lines.append(f"    - {p_name} ({p_type}, {req}){suffix}")
            dsv4_lines.append("")
        tool_prompt = (
            "\n".join(dsv4_lines).rstrip()
            + "\n\nWhen you decide to call a tool, emit ONLY this DSML shape. "
            "Do not emit JSON, markdown, prose, generic XML tool tags, or a DSML wrapper block.\n"
            f"<｜DSML｜invoke name=\"{first_name}\">\n"
            f"  <｜DSML｜parameter name=\"{first_param}\" string=\"true\">VALUE HERE</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n\n"
            "For a request to list the current directory, set the path parameter to \".\" exactly. "
            "Do not explain inability to call tools; emit the DSML call."
        )
    elif is_qwen_native_tool_prompt:
        qwen_lines = [
            "You have access to these tools. When a user asks you to use one, "
            "you must call it instead of fabricating a result.",
            "",
        ]
        first_name = "FUNCTION_NAME"
        first_param = "arg1"
        for idx, tool in enumerate(template_tools):
            func = tool.get("function", {})
            name = func.get("name", "") or "unknown_tool"
            if idx == 0:
                first_name = name
            qwen_lines.append(f"Tool: {name}")
            desc = func.get("description", "")
            if desc:
                qwen_lines.append(f"  description: {desc}")
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", []) if isinstance(params, dict) else [])
            if props:
                qwen_lines.append("  parameters:")
                for p_name, p_schema in props.items():
                    if idx == 0 and first_param == "arg1":
                        first_param = p_name
                    p_type = (
                        p_schema.get("type", "string")
                        if isinstance(p_schema, dict)
                        else "string"
                    )
                    req = "required" if p_name in required else "optional"
                    p_desc = (
                        p_schema.get("description", "")
                        if isinstance(p_schema, dict)
                        else ""
                    )
                    suffix = f": {p_desc}" if p_desc else ""
                    qwen_lines.append(f"    - {p_name} ({p_type}, {req}){suffix}")
            qwen_lines.append("")
        tool_prompt = (
            "\n".join(qwen_lines).rstrip()
            + "\n\nWhen a tool call is needed, emit ONLY this native XML shape. "
            "Do not emit JSON result data, markdown, prose, or a fake directory listing.\n"
            "<tool_call>\n"
            f"<function={first_name}>\n"
            f"<parameter={first_param}>\n"
            "VALUE HERE\n"
            f"</parameter>\n"
            f"</function>\n"
            "</tool_call>\n\n"
            "For a request to list the current directory, set path to \".\" exactly."
        )
    else:
        tool_prompt = (
            "You are an expert assistant with access to tools.\n\n"
            "# Available Tools\n\n"
            "You have access to the following tools:\n\n"
            + json.dumps(tool_descs, indent=2) + "\n\n"
            "When you need to use a tool, you must output a tool call in exactly this XML format:\n"
            "<tool_call>\n"
            '{"name": "FUNCTION_NAME", "arguments": {"arg1": "value"}}\n'
            "</tool_call>"
        )

    # Inject into messages
    messages_copy = [dict(m) for m in messages]
    injected = False
    for msg in messages_copy:
        if msg.get("role") == "system":
            msg["content"] = (msg.get("content") or "") + "\n\n" + tool_prompt
            injected = True
            break

    if not injected:
        messages_copy.insert(0, {"role": "system", "content": tool_prompt})

    # Re-apply template with modified messages
    # Remove tools from kwargs so template doesn't try to format them again
    safe_kwargs = dict(template_kwargs)
    safe_kwargs.pop("tools", None)

    try:
        new_prompt = tokenizer.apply_chat_template(messages_copy, **safe_kwargs)
        return new_prompt
    except Exception as e:
        logger.error(f"Failed to apply template with injected tools: {e}")
        return prompt


def _parse_raw_json_tool_calls(text: str) -> Optional[List[dict]]:
    """
    Parse raw JSON tool calls from model output.

    Handles:
    - Single JSON object: {"name": "func", "arguments": {...}}
    - Multiple objects separated by commas: {...}, {...}
    - JSON array: [{...}, {...}]

    Args:
        text: Raw model output text

    Returns:
        List of tool call dicts with 'name' and 'arguments', or None if no valid tool calls found
    """
    if not text:
        return None

    text = text.strip()

    # Try JSON array first
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(
                isinstance(item, dict) and "name" in item for item in parsed
            ):
                return [
                    {"name": item["name"], "arguments": item.get("arguments", item.get("parameters", {}))}
                    for item in parsed
                ]
        except json.JSONDecodeError:
            pass

    # Find JSON objects with balanced braces (string-aware)
    tool_calls = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '\\' and in_string:
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                json_str = text[start : i + 1]
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict) and "name" in obj:
                        # Accept both "arguments" and "parameters" keys
                        args = obj.get("arguments", obj.get("parameters", {}))
                        tool_calls.append(
                            {"name": obj["name"], "arguments": args}
                        )
                except json.JSONDecodeError:
                    pass
                start = None

    return tool_calls if tool_calls else None


def parse_tool_calls(text: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse tool calls from model output.

    Supports multiple formats:
    - Qwen3 bracket: [Calling tool: function_name({"arg": "value"})]
    - Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Llama: <function=name>{"arg": "value"}</function>
    - Nemotron: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    - Raw JSON: {"name": "...", "arguments": {...}} (single or multiple)

    Args:
        text: Raw model output text

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    tool_calls = []
    cleaned_text = text

    # Pattern for Qwen3 bracket-style: [Calling tool: function_name({...})]
    bracket_pattern = r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]"
    bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)

    for name, args_str in bracket_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove bracket tool calls from cleaned text
    if bracket_matches:
        cleaned_text = re.sub(
            r"\[Calling tool:\s*\w+\(\{.*?\}\)\]", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Laguna / Poolside-style compact XML:
    # <tool_call>list_directory
    # <arg_key>path</arg_key>
    # <arg_value>.</arg_value>
    # </tool_call>
    # This is not the Qwen JSON-in-XML format, so parse it before the generic
    # JSON object fallback. Skip blocks that start with "{" so Qwen remains
    # handled by its stricter branch below.
    laguna_pattern = r"<tool_call>\s*([A-Za-z_][\w.-]*)\s*(.*?)</tool_call>"
    laguna_matches = re.findall(laguna_pattern, text, re.DOTALL)
    for name, body in laguna_matches:
        if name.strip().startswith("{"):
            continue
        keys = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>", body, re.DOTALL)
        vals = re.findall(r"<arg_value>\s*(.*?)\s*</arg_value>", body, re.DOTALL)
        if not keys:
            continue
        arguments = {
            k.strip(): (vals[i].strip() if i < len(vals) else "")
            for i, k in enumerate(keys)
            if k.strip()
        }
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name.strip(),
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )

    if laguna_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*[A-Za-z_][\w.-]*\s*.*?</tool_call>",
            "",
            cleaned_text,
            flags=re.DOTALL,
        ).strip()

    # Pattern for Nemotron-style: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    nemotron_pattern = (
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"
    )
    nemotron_matches = re.findall(nemotron_pattern, text, re.DOTALL)

    for name, params_block in nemotron_matches:
        # Parse parameters from <parameter=name>value</parameter> format
        param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
        params = re.findall(param_pattern, params_block, re.DOTALL)
        arguments = {}
        for p_name, p_value in params:
            v = p_value.strip()
            try:
                arguments[p_name.strip()] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                arguments[p_name.strip()] = v

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name.strip(), arguments=json.dumps(arguments)
                ),
            )
        )

    # Remove Nemotron tool call tags from cleaned text
    if nemotron_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*<function=[^>]+>.*?</function>\s*</tool_call>",
            "",
            cleaned_text,
            flags=re.DOTALL,
        ).strip()

    # Lenient Nemotron-variant: <tool_call> <function=name> <parameter=k> v <parameter=k2> v2 </tool_call>
    # Some models omit </parameter> and </function> closing tags.
    if not nemotron_matches:
        lenient_nemotron_pattern = (
            r"<tool_call>\s*<function=([^>]+)>(.*?)</tool_call>"
        )
        lenient_matches = re.findall(lenient_nemotron_pattern, text, re.DOTALL)
        for name, params_block in lenient_matches:
            # Parse parameters: <parameter=key> value (terminated by next <parameter= or end)
            param_pattern = r"<parameter=([^>]+)>\s*(.*?)(?=\s*<parameter=|\s*$)"
            params = re.findall(param_pattern, params_block, re.DOTALL)
            arguments = {}
            for p_name, p_value in params:
                v = p_value.strip()
                try:
                    arguments[p_name.strip()] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[p_name.strip()] = v
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(), arguments=json.dumps(arguments)
                    ),
                )
            )
        if lenient_matches:
            cleaned_text = re.sub(
                r"<tool_call>\s*<function=[^>]+>.*?</tool_call>",
                "",
                cleaned_text,
                flags=re.DOTALL,
            ).strip()

    # Pattern for Qwen-style tool calls: <tool_call>{"json"}</tool_call>
    qwen_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    qwen_matches = re.findall(qwen_pattern, cleaned_text, re.DOTALL)

    for match in qwen_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove Qwen tool call tags from cleaned text
    if qwen_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Llama-style: <function=name>{"json"}</function>
    llama_pattern = r"<function=([^>]+)>(\{.*?\})</function>"
    llama_matches = re.findall(llama_pattern, cleaned_text, re.DOTALL)

    for name, args_str in llama_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    if llama_matches:
        cleaned_text = re.sub(
            r"<function=[^>]+>\{.*?\}</function>", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Note: We keep <think>...</think> tags for reasoning models
    # The user may want to see the model's reasoning process

    # Pattern for GPT-OSS/Harmony native: to=<name> code{json}
    # The model sometimes uses its native format instead of the template's <tool_call> format.
    harmony_pattern = r"to=(\w[\w.]*)\s+code(\{.*?\})(?:\s|$)"
    harmony_matches = re.findall(harmony_pattern, cleaned_text, re.DOTALL)

    for name, args_str in harmony_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    if harmony_matches:
        cleaned_text = re.sub(
            r"to=\w[\w.]*\s+code\{.*?\}(?:\s|$)", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # DSV4 JANGTQ fallback observed in live Responses auto-tool-choice:
    #
    #   <use_list_directory
    #
    #   {"path": "."}
    #
    # This is neither canonical DSML nor Qwen/Llama XML, but it is still a
    # structured tool-call form: the tool name follows "<use_" and a JSON object
    # immediately follows the marker. Parse it after native formats, before raw
    # JSON fallback. Server-side `_parse_tool_calls_with_parser()` still filters
    # these calls to request.tools, so unavailable names remain visible content.
    use_json_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"<use_([A-Za-z_][\w.-]*)\b", cleaned_text):
        name = m.group(1).strip()
        pos = m.end()
        while pos < len(cleaned_text) and cleaned_text[pos].isspace():
            pos += 1
        if pos < len(cleaned_text) and cleaned_text[pos] == ">":
            pos += 1
            while pos < len(cleaned_text) and cleaned_text[pos].isspace():
                pos += 1
        if pos >= len(cleaned_text) or cleaned_text[pos] != "{":
            continue
        try:
            arguments, end_rel = json.JSONDecoder().raw_decode(cleaned_text[pos:])
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        end = pos + end_rel
        close_pat = re.compile(rf"\s*</use_{re.escape(name)}\s*>", re.DOTALL)
        close = close_pat.match(cleaned_text, end)
        if close:
            end = close.end()
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )
        use_json_spans.append((m.start(), end))

    if use_json_spans:
        for start, end in reversed(use_json_spans):
            cleaned_text = (cleaned_text[:start] + cleaned_text[end:]).strip()

    # Fallback: Raw JSON tool calls (lowest priority)
    # Only try if no other formats matched
    if not tool_calls:
        raw_json_calls = _parse_raw_json_tool_calls(cleaned_text)
        if raw_json_calls:
            for call_data in raw_json_calls:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=call_data["name"],
                            arguments=(
                                json.dumps(call_data["arguments"])
                                if isinstance(call_data["arguments"], dict)
                                else str(call_data["arguments"])
                            ),
                        ),
                    )
                )
            # Clean the JSON from text since we parsed it as tool calls
            cleaned_text = ""

    return cleaned_text, tool_calls if tool_calls else None


def convert_tools_for_template(tools: Optional[List]) -> Optional[List[dict]]:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            tool_func = tool.get("function")
        else:
            tool_type = getattr(tool, "type", None)
            tool_func = getattr(tool, "function", None)

        if tool_type == "function" and tool_func:
            # Handle function as dict or Pydantic model
            if isinstance(tool_func, dict):
                func_name = tool_func.get("name", "")
                func_desc = tool_func.get("description", "")
                func_params = tool_func.get(
                    "parameters", {"type": "object", "properties": {}}
                )
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(
                    tool_func, "parameters", {"type": "object", "properties": {}}
                )

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_desc,
                        "parameters": func_params,
                    },
                }
            )
        elif isinstance(tool, dict) and "name" in tool:
            # Flat Responses API format: {"type":"function","name":"...","parameters":{...}}
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )

    return converted if converted else None


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================


def validate_json_schema(
    data: Any, schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON data against a JSON Schema.

    Args:
        data: The JSON data to validate (dict, list, etc.)
        schema: JSON Schema specification

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data matches schema
        - error_message: Error description if invalid, None if valid
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output text.

    Tries multiple strategies:
    1. Parse entire text as JSON
    2. Extract JSON from markdown code blocks
    3. Find JSON object/array in text

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON data, or None if no valid JSON found
    """
    text = text.strip()

    # Strategy 1: Try to parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object or array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def parse_json_output(
    text: str, response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
) -> Tuple[str, Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Parse JSON from model output when response_format is set.

    Args:
        text: Raw model output text
        response_format: ResponseFormat specification (optional)
            - If type="json_object", extracts any valid JSON
            - If type="json_schema", extracts and validates against schema

    Returns:
        Tuple of (cleaned_text, parsed_json, is_valid, error_message)
        - cleaned_text: Original text (preserved for reference)
        - parsed_json: Extracted JSON data, or None if extraction failed
        - is_valid: True if JSON is valid (and matches schema if specified)
        - error_message: Error description if invalid, None if valid
    """
    # Handle None or text format - just return original
    if response_format is None:
        return text, None, True, None

    # Normalize response_format to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})

        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def build_json_system_prompt(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Build a system prompt instruction for JSON output.

    For models without native JSON mode support, this adds instructions
    to the prompt to encourage proper JSON formatting.

    Args:
        response_format: ResponseFormat specification

    Returns:
        System prompt instruction string, or None if not needed
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type == "text":
        return None

    if format_type == "json_object":
        return (
            "You must respond with valid JSON only. "
            "Do not include any explanation or text outside the JSON object."
        )

    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})
        name = json_schema_spec.get("name", "response")
        description = json_schema_spec.get("description", "")

        prompt = f"You must respond with valid JSON matching the '{name}' schema."
        if description:
            prompt += f" {description}"
        prompt += (
            f"\n\nJSON Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            "Respond with only the JSON object, no additional text or explanation."
        )
        return prompt

    return None
