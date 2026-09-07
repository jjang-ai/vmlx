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
import shlex
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

from jsonschema import validate, ValidationError

from .models import FunctionCall, ResponseFormat, ToolCall

logger = logging.getLogger(__name__)
_LLGUIDANCE_TOKENIZER_CACHE: Dict[Tuple[int, int], Any] = {}


def _normalize_lfm2_tool_result_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return an LFM2-native copy of tool-result history.

    Liquid's native conversation format expects JSON in ``tool`` messages.
    Responses final-synthesis turns intentionally have no current tool schemas
    after ``tool_choice=none`` is applied, but their prior real tool result must
    still be normalized before the chat template renders that history.
    """

    normalized_messages = [dict(message) for message in messages]
    changed = False
    for message in normalized_messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            json.loads(content)
            continue
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        shell_result = re.fullmatch(
            r"\$ (?P<command>[^\n]*)\n\nExit code: "
            r"(?P<exit_code>-?\d+)(?:\n\n(?P<stdout>[\s\S]*))?",
            content.strip(),
        )
        if shell_result:
            structured_result: dict[str, Any] = {
                "exit_code": int(shell_result.group("exit_code")),
            }
            stdout = (shell_result.group("stdout") or "").rstrip()
            if stdout:
                structured_result["stdout"] = stdout
            normalized_result = [structured_result]
        else:
            normalized_result = [{"result": content}]
        message["content"] = json.dumps(normalized_result, ensure_ascii=False)
        changed = True

    return normalized_messages, changed


def _tool_instruction_scope(prompt: str) -> str:
    """Return the template instruction block, excluding prior chat history."""
    if not prompt:
        return ""

    # Tool templates normally place schemas/examples in the opening system
    # section. Later assistant history can contain real tool calls, which must
    # not be counted as parser-native instruction examples.
    cut_markers = (
        "<|im_start|>user",
        "<｜User｜>",
        "<|user|>",
        "\nUser:",
        "\nuser:",
        "\n### User",
        "\n[INST]",
    )
    cuts = [prompt.find(marker) for marker in cut_markers if prompt.find(marker) >= 0]
    if not cuts:
        return prompt
    return prompt[: min(cuts)]


def check_and_inject_fallback_tools(
    prompt: Optional[str],
    messages: List[Dict[str, Any]],
    template_tools: Optional[List[dict]],
    tokenizer: Any,
    template_kwargs: dict,
    tool_parser_id: Optional[str] = None,
) -> Optional[str]:
    """
    Check if the chat template silently dropped tool definitions, and if so,
    re-apply the template with an injected system message containing the tools.

    This fixes models like Qwen 2.5/3 where disabling reasoning causes the
    chat template to completely ignore the tools kwarg, as well as generic models
    without tool-aware templates.
    """
    if prompt is None:
        return prompt
    parser_id = (tool_parser_id or "").strip().lower()
    is_lfm2_native_tool_prompt = parser_id in {"lfm2", "liquid"}
    if is_lfm2_native_tool_prompt:
        normalized_messages, normalized_tool_result = (
            _normalize_lfm2_tool_result_messages(messages)
        )
        if normalized_tool_result:
            messages = normalized_messages
            if not template_tools:
                # This must run even when the current Responses pass correctly
                # has no schemas because tool_choice=none. Re-render only the
                # real normalized history; do not synthesize or re-enable any
                # tool, even if an inconsistent caller left stale schemas in
                # the template kwargs.
                normalized_kwargs = dict(template_kwargs)
                normalized_kwargs.pop("tools", None)
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        **normalized_kwargs,
                    )
                except Exception:
                    logger.exception(
                        "Failed to render LFM2 normalized tool-result history; "
                        "retaining the original prompt"
                    )
    if not template_tools:
        return prompt
    tool_choice = template_kwargs.get("tool_choice") if isinstance(template_kwargs, dict) else None
    tool_choice_required = tool_choice == "required" or (
        isinstance(tool_choice, dict)
        and (
            tool_choice.get("type") in {"required", "function"}
            or bool(tool_choice.get("function") or tool_choice.get("name"))
        )
    )

    def _tool_func(tool: dict) -> dict:
        if not isinstance(tool, dict):
            return {}
        nested = tool.get("function")
        if isinstance(nested, dict):
            return nested
        if tool.get("type") == "function":
            return tool
        return {}

    has_flat_responses_function_tools = any(
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and not isinstance(tool.get("function"), dict)
        for tool in template_tools
    )

    # Check if at least one tool name made it into the prompt
    tool_names = [_tool_func(t).get("name", "") for t in template_tools]
    tool_names = [name for name in tool_names if name]

    if not tool_names:
        return prompt

    instruction_prompt = _tool_instruction_scope(prompt)

    def _message_request_text() -> str:
        chunks: list[str] = []
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str):
                            chunks.append(text)
        return "\n".join(chunks)

    def _latest_user_request_text() -> str:
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str):
                            chunks.append(text)
                return "\n".join(chunks)
        return ""

    request_text = _latest_user_request_text() or _message_request_text()

    def _requested_exact_visible_answer() -> str:
        """Extract a short exact final-answer target from the user's request."""
        candidates = [request_text, _message_request_text()]
        patterns = (
            r"\breply\s+exactly\s+(.+?)(?:\s+and\s+nothing\s+else|[.!?]\s*$|$)",
            r"\breply\s+only\s*:?\s+(.+?)(?:\s+and\s+nothing\s+else|[.!?]\s*$|$)",
            r"\banswer\s+exactly\s*:?\s+(.+?)(?:\s+and\s+nothing\s+else|[.!?]\s*$|$)",
            r"\banswer\s+only\s*:?\s+(.+?)(?:\s+and\s+nothing\s+else|[.!?]\s*$|$)",
        )
        for text in candidates:
            if not text:
                continue
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
                if not match:
                    continue
                value = match.group(1).strip().strip("`'\" \n\t")
                value = re.sub(r"\s+", " ", value)
                if 0 < len(value) <= 160:
                    return value.rstrip(".,;:")
        return ""

    def _request_mentions_tool_name(name: str) -> bool:
        normalized = name.strip()
        if not normalized or not request_text:
            return False
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
                request_text,
            )
        )

    explicitly_requested_tool_names = {
        name for name in tool_names if _request_mentions_tool_name(name)
    }
    explicit_tool_requested = bool(explicitly_requested_tool_names)

    # Some templates need more than tool-name visibility. DSV4 may mention
    # schemas without a parser-matching DSML exemplar. Qwen's native template
    # intentionally uses `example_function_name` in its format explanation,
    # but the <tools> block itself contains the real JSON schemas. Preserve
    # that trained native scaffold for normal auto-tool turns; it has the
    # model's own tool/result transcript framing and thinking contract.
    is_dsv4_prompt = "<｜User｜>" in prompt or "<｜Assistant｜>" in prompt
    if is_dsv4_prompt:
        # Judge the rendered prompt against exactly the catalog the canonical
        # DSV4 encoder renders — the full authorized set, deduplicated by name.
        # Comparing the prompt to a different set falsely reports that schemas
        # were dropped, then prepends a second synthetic fallback even though
        # the encoder already rendered the correct DSML example.  Live
        # Responses multi-turn testing showed that duplicate fallback can drive
        # DSV4 into an unbounded literal `response` loop.
        from ..loaders.dsv4_chat_encoder import (
            prompt_tool_catalog,
            request_explicitly_requests_tool,
        )

        catalog_tools = prompt_tool_catalog(template_tools)
        if catalog_tools:
            template_tools = catalog_tools
            tool_names = [_tool_func(t).get("name", "") for t in template_tools]
            tool_names = [name for name in tool_names if name]
            explicitly_requested_tool_names = {
                name
                for name in tool_names
                if request_explicitly_requests_tool(request_text, name)
            }
            # A specific endpoint-native tool choice is also an explicit
            # execution contract even when the natural-language prompt does
            # not repeat the function name.
            if tool_choice_required and len(tool_names) == 1:
                explicitly_requested_tool_names.add(tool_names[0])
            explicit_tool_requested = bool(explicitly_requested_tool_names)
    is_qwen_native_tool_prompt = (
        parser_id not in {
            "xml_function",
            "mimo_xml_function",
            "qwen3_coder",
            "step3p5",
            "step",
            "stepfun",
        }
        and "<|im_start|>" in prompt
        and "<tools>" in prompt
        and "<function=example_function_name>" in prompt
    )
    is_zaya_native_tool_prompt = (
        parser_id in {"zaya_xml", "zaya", "zyphra"}
        or (
            "<zyphra_tool_call>" in prompt
            and "<tools>" in prompt
        )
    )
    zaya_template_has_native_scaffold = (
        is_zaya_native_tool_prompt
        and "<zyphra_tool_call>" in instruction_prompt
        and "<tools>" in instruction_prompt
    )
    if (
        is_zaya_native_tool_prompt
        and not any(_request_mentions_tool_name(name) for name in tool_names)
    ):
        return prompt
    is_openpangu_native_tool_prompt = (
        parser_id in {"openpangu", "openpangu_v2"}
        or (
            "<|pangu_text_start|>" in prompt
            and "<|tool_call_start|>" in prompt
            and "<tools>" in prompt
        )
    )
    is_minimax_native_tool_prompt = parser_id in {"minimax", "minimax_m2"}
    is_xml_function_native_tool_prompt = parser_id in {
        "xml_function",
        "mimo_xml_function",
        # Qwen3.6 D-series / Qwen 3.8 stamps. Without this the D-series prompt
        # (which contains <|im_start|> AND <function=...> exemplars) classified
        # as qwen-native, whose fallback path pops the native tools kwarg and
        # skips the ChatML splice -- the exact MiMo failure documented below.
        "qwen3_coder",
    }
    is_step3p5_native_tool_prompt = (
        parser_id in {"step3p5", "step", "stepfun"}
        or (
            not is_xml_function_native_tool_prompt
            and not is_qwen_native_tool_prompt
            and
            "<tool_call>" in prompt
            and "<function=example_function_name>" in prompt
            and "<tools>" in prompt
            and "<|im_start|>" in prompt
        )
    )

    # If ALL tool names made it into a prompt and the prompt also contains a
    # concrete parser-native exemplar for those names, the template handled
    # tools correctly. Otherwise inject a concrete native exemplar.
    # Secondary path for a DSV4-shaped prompt that carries concrete DSML
    # history but not the canonical ``## Tools`` block. One real exemplar is
    # enough to demonstrate the grammar, and the outer gate has already
    # required every authorized tool name to appear in the prompt. Demanding an
    # exemplar per authorized tool only held while the catalog was narrowed to
    # the recently called tool; with a stable broad catalog it can never hold,
    # which would duplicate the contract on every agentic turn.
    _dsv4_has_scoped_history_examples = (
        is_dsv4_prompt
        and not explicit_tool_requested
        and any(
            f'<｜DSML｜invoke name="{name}"' in instruction_prompt
            for name in tool_names
        )
    )
    _qwen_has_concrete_tool_examples = (
        is_qwen_native_tool_prompt
        and all(f"<function={name}>" in instruction_prompt for name in tool_names)
    )
    _qwen_has_native_tool_schema = (
        is_qwen_native_tool_prompt
        and "<tools>" in instruction_prompt
        and "</tools>" in instruction_prompt
        and all(name in instruction_prompt for name in tool_names)
    )
    # A source-owned Qwen schema remains authoritative after tool results.
    # Replacing it only on continuation turns moves the opening system bytes,
    # so an otherwise append-only agent transcript cannot restore its exact
    # KV + GDN/SSM prefix from RAM or SSD.  The native template already renders
    # assistant tool calls and tool-result history; keep that stable catalog
    # and let the current user turn express whether another tool is wanted.
    # ``tool_choice=required`` still takes the stronger fallback below.
    _zaya_has_concrete_tool_examples = (
        is_zaya_native_tool_prompt
        # Zaya's shipped scaffold may contain parser-shaped examples for every
        # advertised tool, but those examples are not bound to the current
        # request. In the broad Electron tool catalog this let an unrelated
        # run_applescript example win over an explicit file_info request. Force
        # the request-scoped fallback below whenever the user names a tool.
        and not explicit_tool_requested
        and all(f"<function={name}>" in instruction_prompt for name in tool_names)
    )
    _lfm2_has_concrete_tool_examples = (
        is_lfm2_native_tool_prompt
        # A native scaffold can still contain placeholder arguments. For an
        # explicitly named tool, force the request-bound fallback below so a
        # concrete path/command reaches the small Liquid tool model.
        and not explicit_tool_requested
        and "<|tool_call_start|>" in instruction_prompt
        and "<|tool_call_end|>" in instruction_prompt
        and all(f"{name}(" in instruction_prompt for name in tool_names)
    )

    def _normalized_function_schema(tool: Any) -> Optional[dict]:
        """Normalize nested Chat and flat Responses function-tool shapes."""
        func = _tool_func(tool)
        if not isinstance(func, dict):
            return None
        name = func.get("name")
        parameters = func.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            return None
        normalized = dict(func)
        # Flat Responses tools carry the wrapper discriminator alongside the
        # function fields; nested Chat tools carry it outside ``function``.
        normalized.pop("type", None)
        return normalized

    def _dsv4_has_exact_native_tool_contract() -> bool:
        """Accept only the source-owned encoder grammar and exact schemas."""
        if not is_dsv4_prompt or not getattr(
            tokenizer, "_vmlx_dsv4_chat_template_shim", False
        ):
            return False

        grammar_fragments = (
            "## Tools",
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="$TOOL_NAME">\n'
            '<｜DSML｜parameter name="$PARAMETER_NAME" '
            'string="true|false">$PARAMETER_VALUE'
            "</｜DSML｜parameter>\n"
            "...\n"
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="$TOOL_NAME2">\n'
            "...\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>",
            'String parameters should be specified as is and set `string="true"`.',
            "For all other types (numbers, booleans, arrays, objects), pass the "
            'value in JSON format and set `string="false"`.',
            "If thinking_mode is enabled",
            "BEFORE any tool calls or final response.",
            "### Available Tool Schemas",
            "You MUST strictly follow the above defined tool name and parameter "
            "schemas to invoke tool calls.",
        )
        if any(fragment not in instruction_prompt for fragment in grammar_fragments):
            return False

        expected_by_name: dict[str, dict] = {}
        for tool in template_tools:
            normalized = _normalized_function_schema(tool)
            if normalized is None or normalized["name"] in expected_by_name:
                return False
            expected_by_name[normalized["name"]] = normalized

        schema_marker = "### Available Tool Schemas"
        closing = (
            "You MUST strictly follow the above defined tool name and parameter "
            "schemas to invoke tool calls."
        )
        schema_start = instruction_prompt.find(schema_marker)
        schema_end = instruction_prompt.find(closing, schema_start + len(schema_marker))
        if schema_start < 0 or schema_end < 0:
            return False
        schema_text = instruction_prompt[
            schema_start + len(schema_marker) : schema_end
        ].strip()
        rendered_by_name: dict[str, dict] = {}
        for line in schema_text.splitlines():
            if not line.strip():
                continue
            try:
                rendered = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                return False
            if (
                isinstance(rendered, dict)
                and isinstance(rendered.get("name"), str)
                and rendered.get("name")
                and isinstance(rendered.get("parameters"), dict)
            ):
                # DSV4's vendor encoder renders the inner function schema
                # directly, without an OpenAI ``type/function`` wrapper.
                normalized = dict(rendered)
                normalized.pop("type", None)
            else:
                normalized = _normalized_function_schema(rendered)
            if normalized is None or normalized["name"] in rendered_by_name:
                return False
            rendered_by_name[normalized["name"]] = normalized
        return bool(expected_by_name) and rendered_by_name == expected_by_name

    _dsv4_has_native_tool_contract = _dsv4_has_exact_native_tool_contract()

    def _lfm2_has_exact_native_tool_schema() -> bool:
        """Validate Liquid's bounded native JSON catalog against this request."""
        if not is_lfm2_native_tool_prompt:
            return False

        expected_by_name: dict[str, dict] = {}
        for tool in template_tools:
            normalized = _normalized_function_schema(tool)
            if normalized is None:
                return False
            name = normalized["name"]
            if name in expected_by_name:
                return False
            expected_by_name[name] = normalized

        marker = "List of tools:"
        decoder = json.JSONDecoder()
        search_from = 0
        while True:
            marker_at = instruction_prompt.find(marker, search_from)
            if marker_at < 0:
                return False
            payload_start = marker_at + len(marker)
            payload_text = instruction_prompt[payload_start:].lstrip()
            try:
                payload, _payload_end = decoder.raw_decode(payload_text)
            except (json.JSONDecodeError, TypeError):
                search_from = payload_start
                continue
            if not isinstance(payload, list):
                search_from = payload_start
                continue

            rendered_by_name: dict[str, dict] = {}
            valid_payload = True
            for tool in payload:
                normalized = _normalized_function_schema(tool)
                if normalized is None or normalized["name"] in rendered_by_name:
                    valid_payload = False
                    break
                rendered_by_name[normalized["name"]] = normalized
            if valid_payload and rendered_by_name == expected_by_name:
                return True
            search_from = payload_start

    _lfm2_has_native_tool_schema = _lfm2_has_exact_native_tool_schema()

    _CONTRACT_PROSE_KEYS = frozenset({"description", "title", "examples", "$comment"})

    def _parameter_contract(schema: Any) -> Any:
        """Canonical form of a JSON-schema parameter block.

        Every validation keyword is part of the contract — type, enum, const,
        properties, required, items, additionalProperties, minLength,
        pattern, format, minimum, maxItems, oneOf/anyOf, ... — with key order
        removed. Only prose (description, title, examples, $comment) is
        ignored: two renders with the same validation schema are the same
        native catalog even when their wording differs, so a byte-level
        difference alone never forces the fallback (which would rewrite the
        opening system bytes and move every SSD prefix). A render that drops
        a constraint has changed what the model is told to produce and is
        deficient (audit follow-up 2026-09-07: minLength, additionalProperties
        and the parameters object itself were previously outside the contract).
        """
        if isinstance(schema, dict):
            contract: dict[str, Any] = {}
            for key, value in schema.items():
                key_text = str(key)
                if key_text in _CONTRACT_PROSE_KEYS:
                    continue
                if key_text == "properties" and isinstance(value, dict):
                    contract[key_text] = {
                        str(name): _parameter_contract(sub)
                        for name, sub in value.items()
                    }
                elif key_text == "required" and isinstance(value, list):
                    contract[key_text] = sorted(str(name) for name in value)
                elif key_text == "type" and isinstance(value, list):
                    contract[key_text] = sorted(str(t) for t in value)
                elif key_text == "enum" and isinstance(value, list):
                    contract[key_text] = sorted(
                        json.dumps(item, sort_keys=True) for item in value
                    )
                else:
                    contract[key_text] = _parameter_contract(value)
            return contract
        if isinstance(schema, list):
            return [_parameter_contract(item) for item in schema]
        return schema

    def _argument_contract(parameters: Any) -> Any:
        """The contract of a function's ``parameters`` block. A missing block,
        ``{}``, ``{"type": "object"}`` and ``{"type": "object", "properties":
        {}}`` all tell the model "no arguments" and compare equal; anything
        that names a property or adds a constraint is compared in full."""
        contract = _parameter_contract(parameters if isinstance(parameters, dict) else {})
        if isinstance(contract, dict):
            residue = {
                key: value
                for key, value in contract.items()
                if not (key == "type" and value == "object")
                and not (key == "properties" and value == {})
                and not (key == "required" and value == [])
            }
            if not residue:
                return {}
        return contract

    def _requested_function_schema(tool: Any) -> Optional[dict]:
        """The request side of the comparison: name plus ``parameters`` (None
        when the caller declared none — a legitimate no-argument tool)."""
        func = _tool_func(tool)
        name = func.get("name") if isinstance(func, dict) else None
        if not isinstance(name, str) or not name:
            return None
        return {"name": name, "parameters": func.get("parameters")}

    def _rendered_function_schema(entry: Any) -> Optional[dict]:
        """A rendered catalog entry: OpenAI ``{type, function}`` wrapper, flat
        Responses shape, or the bare function object some templates emit
        (MiniMax renders ``tool.function | tojson``). An entry that names a
        function but carries no ``parameters`` object is returned with
        ``parameters`` = None so it counts as a rendered — and deficient —
        function, not as an XML-only catalog."""
        normalized = _normalized_function_schema(entry)
        if normalized is not None:
            return normalized
        if not isinstance(entry, dict):
            return None
        nested = entry.get("function")
        candidate = nested if isinstance(nested, dict) else entry
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            return None
        bare = dict(candidate)
        bare.pop("type", None)
        if not isinstance(bare.get("parameters"), dict):
            bare["parameters"] = None
        return bare

    def _rendered_tools_block_schema_verdict() -> Optional[bool]:
        """Semantic check of the JSON entries a template rendered inside
        ``<tools>``...``</tools>`` against this request's tools.

        ``None``  — no decodable JSON entry in any block (an XML-only catalog):
                    no opinion, the name-based recognition applies unchanged.
        ``True``  — exactly the requested tools are rendered, each once, and
                    every parameter contract matches the request.
        ``False`` — a requested tool is missing, duplicated or rendered with a
                    different parameter contract. Extra rendered tools are a
                    superset (narrowed continuation catalogs stay native).

        Names alone are not a contract: a render that named ``read_file`` and
        ``terminal`` while omitting ``path`` and ``command`` was accepted as
        native and the model was never told the required arguments
        (cross-family schema audit 2026-09-06, mechanism B).
        """
        expected_by_name: dict[str, Any] = {}
        for tool in template_tools:
            requested = _requested_function_schema(tool)
            if requested is None or requested["name"] in expected_by_name:
                return False
            expected_by_name[requested["name"]] = _argument_contract(
                requested["parameters"]
            )
        if not expected_by_name:
            return None
        decoder = json.JSONDecoder()
        rendered_by_name: dict[str, Any] = {}
        found_entry = False
        search_from = 0
        while True:
            block_start = instruction_prompt.find("<tools>", search_from)
            if block_start < 0:
                break
            block_end = instruction_prompt.find("</tools>", block_start)
            if block_end < 0:
                block_end = len(instruction_prompt)
                search_from = block_end
            else:
                search_from = block_end + len("</tools>")
            block = instruction_prompt[block_start + len("<tools>") : block_end]
            position = 0
            while True:
                brace = block.find("{", position)
                if brace < 0:
                    break
                try:
                    entry, consumed = decoder.raw_decode(block[brace:])
                except (json.JSONDecodeError, TypeError):
                    position = brace + 1
                    continue
                position = brace + max(consumed, 1)
                normalized = _rendered_function_schema(entry)
                if normalized is None:
                    continue
                found_entry = True
                if normalized["name"] in rendered_by_name:
                    return False
                rendered_by_name[normalized["name"]] = _argument_contract(
                    normalized["parameters"]
                )
        if not found_entry:
            return None
        # Every requested tool must be rendered once with its contract. A
        # render that ALSO carries tools this request no longer lists (a
        # continuation turn whose catalog was narrowed to the remaining tool,
        # a broader source-owned catalog) is a superset, not a deficiency:
        # forcing the fallback there would rewrite the opening system bytes
        # on every agentic turn and defeat prefix reuse.
        return all(
            name in rendered_by_name and rendered_by_name[name] == contract
            for name, contract in expected_by_name.items()
        )

    # Computed once per request; ``is not False`` keeps XML-only catalogs on
    # their existing name-based recognition.
    _native_tools_schema_verdict = _rendered_tools_block_schema_verdict()
    _openpangu_has_concrete_tool_examples = (
        is_openpangu_native_tool_prompt
        and not explicit_tool_requested
        and not tool_choice_required
        and "<|tool_call_start|>" in instruction_prompt
        and "<|tool_call_end|>" in instruction_prompt
        and all(name in instruction_prompt for name in tool_names)
    )
    _minimax_has_concrete_tool_examples = (
        is_minimax_native_tool_prompt
        and "<minimax:tool_call>" in instruction_prompt
        and all(f'<invoke name="{name}">' in instruction_prompt for name in tool_names)
    )
    # MiniMax-M2.7 templates render the tool schemas natively inside a
    # <tools>...</tools> block and show ONE generic <invoke name="tool-name-1">
    # example instead of one example per tool. Requiring per-tool examples
    # sent every M2.7 request with tools through the fallback injection
    # (user report 2026-09-04: "Chat template needs fallback tool schema
    # injection" on every call, followed by parsed tool calls dropped for
    # empty required arguments). A rendered schema block that names every
    # tool is the native contract.
    _minimax_has_native_tool_schema = (
        is_minimax_native_tool_prompt
        and "<minimax:tool_call>" in instruction_prompt
        and "<tools>" in instruction_prompt
        and "</tools>" in instruction_prompt
        and all(name in instruction_prompt for name in tool_names)
        and _native_tools_schema_verdict is not False
    )
    # Two native dialects render the schema block: MiMo-style XML entries
    # (<name>tool</name>) and JSON entries ({"name": "tool", ...}) as emitted
    # by the Qwen3.8-27B D-series and Nanbeige templates. Requiring the XML
    # form sent every 27B / Nanbeige tool call through the fallback injection
    # (matrix over all bundle templates, 2026-09-05).
    _xml_function_has_native_tool_schema = (
        is_xml_function_native_tool_prompt
        and "<tool_call>" in instruction_prompt
        and "<function=example_function_name>" in instruction_prompt
        and "<tools>" in instruction_prompt
        and all(
            f"<name>{name}</name>" in instruction_prompt
            or f'"name": "{name}"' in instruction_prompt
            or f'"name":"{name}"' in instruction_prompt
            for name in tool_names
        )
        and _native_tools_schema_verdict is not False
    )
    _step3p5_has_concrete_tool_examples = (
        is_step3p5_native_tool_prompt
        and all(f"<function={name}>" in instruction_prompt for name in tool_names)
    )
    _step3p5_has_native_tool_schema = (
        is_step3p5_native_tool_prompt
        and "<tools>" in instruction_prompt
        and "</tools>" in instruction_prompt
        and "<function=example_function_name>" in instruction_prompt
        and all(name in instruction_prompt for name in tool_names)
        and _native_tools_schema_verdict is not False
    )
    if all(name in prompt for name in tool_names) and (
        (
            not is_dsv4_prompt
            or _dsv4_has_native_tool_contract
            or _dsv4_has_scoped_history_examples
        )
        and (
            not is_qwen_native_tool_prompt
            or _qwen_has_concrete_tool_examples
            or (
                _qwen_has_native_tool_schema
                # The rendered JSON entries must carry this request's
                # parameter contracts, not just its tool names.
                and _native_tools_schema_verdict is not False
                and not tool_choice_required
            )
        )
        and (not is_zaya_native_tool_prompt or _zaya_has_concrete_tool_examples)
        and (
            not is_lfm2_native_tool_prompt
            or (
                _lfm2_has_native_tool_schema
                and not tool_choice_required
            )
            or _lfm2_has_concrete_tool_examples
        )
        and (not is_openpangu_native_tool_prompt or _openpangu_has_concrete_tool_examples)
        and (
            not is_minimax_native_tool_prompt
            or _minimax_has_concrete_tool_examples
            or (_minimax_has_native_tool_schema and not tool_choice_required)
        )
        and (not is_xml_function_native_tool_prompt or _xml_function_has_native_tool_schema)
        and (
            not is_step3p5_native_tool_prompt
            or _step3p5_has_concrete_tool_examples
            or (
                _step3p5_has_native_tool_schema
                # An explicit tool name does not invalidate Step's source-owned
                # schema. Replacing that stable opening system block with a
                # request-specific fallback only for the named-tool turn makes
                # the next ordinary/media turn rewrite the whole SSD prefix.
                # Keep the native schema for auto and explicit calls alike;
                # ``tool_choice=required`` still takes the stronger fallback.
                and not tool_choice_required
            )
        )
    ):
        return prompt

    logger.warning("Chat template needs fallback tool schema injection.")

    # Format fallback tool schema
    tool_descs = []
    for tool in template_tools:
        func = _tool_func(tool)
        tool_descs.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {})
        })

    def _tool_props(tool: dict) -> tuple[str, list[str]]:
        func = _tool_func(tool)
        name = func.get("name", "") or "unknown_tool"
        params = func.get("parameters", {}) or {}
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        return name, [p for p in props if p]

    def _has_directory_path_tool(tools: list[dict]) -> bool:
        for tool in tools:
            name, props = _tool_props(tool)
            normalized_name = name.strip().lower()
            normalized_props = {prop.strip().lower() for prop in props}
            if normalized_name in {"list_directory", "read_directory", "list_files"}:
                return True
            if normalized_name in {"list_files"} and (
                "path" in normalized_props
                or "dir" in normalized_props
                or "directory" in normalized_props
            ):
                return True
        return False

    def _requested_tools(tools: list[dict]) -> list[dict]:
        if not request_text:
            return tools
        requested: list[dict] = []
        for tool in tools:
            name, _props = _tool_props(tool)
            normalized = name.strip()
            if not normalized:
                continue
            if (
                normalized in explicitly_requested_tool_names
                if is_dsv4_prompt
                else re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
                    request_text,
                )
            ):
                requested.append(tool)
        return requested or tools

    def _recent_tool_call_arguments() -> dict[str, dict[str, Any]]:
        calls: dict[str, dict[str, Any]] = {}
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                func = tool_call.get("function")
                name = func.get("name") if isinstance(func, dict) else None
                if isinstance(name, str) and name:
                    args = func.get("arguments") if isinstance(func, dict) else {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    calls[name] = args if isinstance(args, dict) else {}
            if calls:
                break
        return calls

    recent_tool_call_arguments = _recent_tool_call_arguments()

    def _tool_history_after_latest_user() -> tuple[set[str], bool]:
        """Return the tool activity that owns the current continuation turn.

        Native Chat history can arrive in either of these equivalent shapes:

        ``user -> assistant(tool) -> tool``
        ``user -> assistant(tool) -> tool -> user(follow-up instruction)``

        The second shape is what Anthropic produces when one user content array
        contains both ``tool_result`` and follow-up text: the adapter must keep
        the tool result adjacent to its assistant call, then append the new user
        instruction.  Historically this scanner stopped on that final user
        message and therefore forgot the immediately preceding completed call.
        Qwen then lost its multi-tool continuation scaffold on the next required
        tool turn.

        Only look behind the newest user when the immediately preceding message
        is a tool result.  A completed assistant answer followed by a new user
        turn must not inherit stale tool-continuation state.
        """
        called: set[str] = set()
        has_tool_result = False
        history = list(messages or [])
        scan_end = len(history)
        if history and isinstance(history[-1], dict) and history[-1].get("role") == "user":
            previous = history[-2] if len(history) >= 2 else None
            if isinstance(previous, dict) and previous.get("role") == "tool":
                scan_end -= 1

        for message in reversed(history[:scan_end]):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                break
            if message.get("role") == "tool":
                has_tool_result = True
                continue
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                func = tool_call.get("function")
                name = func.get("name") if isinstance(func, dict) else None
                if isinstance(name, str) and name:
                    called.add(name)
        return called, has_tool_result

    tools_called_after_latest_user, tool_result_after_latest_user = (
        _tool_history_after_latest_user()
    )
    qwen_tool_result_continuation = bool(
        is_qwen_native_tool_prompt
        and (tools_called_after_latest_user or tool_result_after_latest_user)
    )
    qwen_requested_remaining_tools = {
        name
        for name in explicitly_requested_tool_names
        if name not in tools_called_after_latest_user
    }
    qwen_client_narrowed_to_remaining_tools = bool(
        tools_called_after_latest_user
        and tool_names
        and set(tool_names).isdisjoint(tools_called_after_latest_user)
    )
    qwen_multi_tool_continuation = bool(
        qwen_tool_result_continuation
        and (
            qwen_requested_remaining_tools
            or qwen_client_narrowed_to_remaining_tools
        )
    )
    qwen_terminal_tool_result_continuation = bool(
        qwen_tool_result_continuation and not qwen_multi_tool_continuation
    )
    dsv4_tool_result_activity = bool(
        is_dsv4_prompt
        and (tools_called_after_latest_user or tool_result_after_latest_user)
    )
    dsv4_requested_remaining_tools = {
        name
        for name in explicitly_requested_tool_names
        if name not in tools_called_after_latest_user
    }
    dsv4_client_narrowed_to_remaining_tools = bool(
        tools_called_after_latest_user
        and tool_names
        and set(tool_names).isdisjoint(tools_called_after_latest_user)
    )
    # A tool result followed by an explicit request for a *different* tool is
    # a multi-tool continuation, not a terminal synthesis turn.  Treating both
    # shapes alike injected "the requested tool already ran / do not emit
    # another tool call" beside the new request.  Live DSV4 then copied schema
    # residue or looped while trying to satisfy those contradictory contracts.
    dsv4_multi_tool_continuation = bool(
        dsv4_tool_result_activity
        and (
            dsv4_requested_remaining_tools
            or dsv4_client_narrowed_to_remaining_tools
        )
    )
    dsv4_tool_result_continuation = bool(
        dsv4_tool_result_activity and not dsv4_multi_tool_continuation
    )
    lfm2_tool_result_continuation = bool(
        is_lfm2_native_tool_prompt
        and (tools_called_after_latest_user or tool_result_after_latest_user)
    )
    openpangu_tool_result_continuation = bool(
        is_openpangu_native_tool_prompt
        and (tools_called_after_latest_user or tool_result_after_latest_user)
    )

    def _recently_called_tools(tools: list[dict]) -> list[dict]:
        """Keep fallback schemas stable after a Responses tool round."""
        if not recent_tool_call_arguments:
            return tools
        selected = [
            tool
            for tool in tools
            if _tool_props(tool)[0] in recent_tool_call_arguments
        ]
        return selected or tools

    def _historical_explicit_tool_request_text(tool_name: str) -> str:
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    str(part.get("text") or part.get("content") or "")
                    for part in content
                    if isinstance(part, dict)
                )
            else:
                continue
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(tool_name)}(?![A-Za-z0-9_])",
                text,
            ):
                return text
        return ""

    zaya_prompt_tools = (
        _requested_tools(template_tools)
        if is_zaya_native_tool_prompt
        else template_tools
    )
    zaya_prompt_tool_names = [
        name for name in (_tool_props(tool)[0] for tool in zaya_prompt_tools) if name
    ]
    lfm2_prompt_tools = (
        _requested_tools(template_tools)
        if is_lfm2_native_tool_prompt
        else template_tools
    )
    if is_openpangu_native_tool_prompt and openpangu_tool_result_continuation and recent_tool_call_arguments:
        openpangu_prompt_tools = _recently_called_tools(template_tools)
    elif is_openpangu_native_tool_prompt and explicit_tool_requested:
        openpangu_prompt_tools = _requested_tools(template_tools)
    else:
        openpangu_prompt_tools = template_tools
    qwen_prompt_tools = template_tools
    if is_qwen_native_tool_prompt:
        if qwen_multi_tool_continuation:
            remaining_names = qwen_requested_remaining_tools or set(tool_names)
            qwen_prompt_tools = [
                tool
                for tool in template_tools
                if _tool_props(tool)[0] in remaining_names
                and _tool_props(tool)[0] not in tools_called_after_latest_user
            ]
            if not qwen_prompt_tools:
                qwen_prompt_tools = template_tools
        elif qwen_terminal_tool_result_continuation:
            if recent_tool_call_arguments:
                qwen_prompt_tools = _recently_called_tools(template_tools)
            elif explicit_tool_requested:
                qwen_prompt_tools = _requested_tools(template_tools)
        elif explicit_tool_requested:
            qwen_prompt_tools = _requested_tools(template_tools)
        elif recent_tool_call_arguments:
            qwen_prompt_tools = _recently_called_tools(template_tools)

    def _derive_dsml_request_param_value(tool_name: str, param: str) -> str:
        """Bind concrete DSML examples only to values present in this request.

        DSV4 copies its fallback examples aggressively.  Empty invokes leave the
        model to echo the surrounding schema, while placeholder values corrupt
        real tool arguments.  Keep the safe empty-example default, but when the
        user explicitly names a tool and supplies a value, carry that exact value
        into the canonical DSML example.
        """
        normalized_tool = tool_name.strip().lower()
        normalized_param = param.strip().lower()
        is_path_param = (
            normalized_param in {"path", "file", "filename", "dir", "directory"}
            or normalized_param.endswith("_path")
        )

        def _safe_path_value(value: str) -> str:
            # This value is interpolated inside a canonical DSML parameter in
            # a system message. Never allow user text to terminate that node
            # or open another native control envelope. Ordinary shell command
            # redirection remains unaffected because this guard is path-only.
            if any(marker in value for marker in ("<", ">", "｜", "\r", "\n")):
                return ""
            return value

        binding_text = request_text
        if tool_name not in explicitly_requested_tool_names:
            binding_text = (
                _historical_explicit_tool_request_text(tool_name)
                or request_text
            )

        if normalized_tool == "run_command" and normalized_param == "command":
            direct_patterns = (
                # Electron's default proof prompt says "Use this command:".
                r'\buse\s+(?:this\s+)?(?:exact\s+)?command\s*:\s*([^\n]{1,240}?)(?=\s+(?:(?:After|Then|Do\s+not|Don\x27t|Never)\b)|$)',
                # Electron's built-in tool prompt uses this exact wording.
                # Keep the whole unquoted shell command through the explicit
                # follow-up boundary; the generic scalar matcher below only
                # captured its first word (usually ``printf`` or ``test``).
                r'\b(?:this\s+)?exact\s+command\s*:\s*([^\n]{1,240}?)(?=\s+(?:(?:After|Then|Do\s+not|Don\x27t|Never)\b)|$)',
                r'\brun\s+exactly:\s*`([^`\n]{1,240})`',
                r'\brun\s+exactly:\s*([^\n]{1,240}?)(?:\s+\.\s+(?:After|Then)\b|$)',
                r'\bto\s+run:\s*([^\n]{1,240}?)(?:\s*\.\s+(?:After|Then)\b|$)',
                r'(?<![A-Za-z0-9_])command\s+`([^`\n]{1,240})`',
                r'(?<![A-Za-z0-9_])command\s+["\u201c]([^"\u201d\n]{1,240})["\u201d]',
            )
            for pattern in direct_patterns:
                match = re.search(pattern, binding_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip().rstrip(".,;:")

            create_file = re.search(
                r"\bcreate\s+a\s+file\s+named\s+([A-Za-z0-9_.:/-]{1,120}).*?"
                r"\bWrite(?:\s+the\s+text)?\s+([A-Za-z0-9_.:-]{1,120})\s+"
                r"into\s+that\s+file",
                binding_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if create_file:
                path, text = (part.strip() for part in create_file.groups())
                path = path.rstrip(".,;:!?")
                return f"printf %s {shlex.quote(text)} > {shlex.quote(path)}"

        for obj_match in re.finditer(r"\{[^{}]{1,400}\}", binding_text):
            try:
                obj = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                continue
            value = obj.get(param) if isinstance(obj, dict) else None
            if isinstance(value, str) and 0 < len(value) <= 240:
                return _safe_path_value(value) if is_path_param else value

        name = re.escape(param)
        if is_path_param:
            # File paths routinely contain `/`, `~`, and percent/URL-safe
            # characters.  The generic scalar patterns below intentionally
            # use a narrower alphabet and truncated `panel/package.json` to
            # `panel` in the live DSV4 DSML example.
            #
            # Preserve quoted paths before applying the unquoted fallback.
            # In particular, `path '.'` is the canonical current-directory
            # request.  The fallback below historically captured `.` and then
            # stripped it as sentence punctuation, leaving an empty invoke and
            # depriving DSV4 of the concrete parser-canonical example.
            quoted_path = re.search(
                rf"\b{name}\s*(?:(?:=|:|to|is)\s*)?([`'\"])([^`'\"\n]{{1,240}})\1",
                binding_text,
                flags=re.IGNORECASE,
            )
            if quoted_path:
                return _safe_path_value(quoted_path.group(2))
            path_patterns = (
                rf"\bwith\s+{name}\s+[`'\"]?([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
                rf"\b{name}\s*(?:=|:|to|is)\s*[`'\"]?([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
                rf"\b{name}\s+[`'\"]?([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
            )
            for pattern in path_patterns:
                match = re.search(pattern, binding_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).rstrip(".,;:!?`'\"")

            # Electron's bounded tool turns commonly put the concrete file
            # after a file-oriented verb instead of repeating the schema
            # field name, for example: ``file_info ... to inspect
            # generation_config.json``.  The DSV4 fallback must bind that
            # request value into its canonical DSML example; otherwise it
            # emits an empty invoke while instructing the model to copy a
            # concrete parameter that is not present.  Keep this limited to
            # an explicitly named tool and an actually path-like value so
            # unrelated prose cannot become a tool argument.
            if tool_name in explicitly_requested_tool_names:
                natural_path = re.search(
                    r"\b(?:inspect|examine|stat|read|check)\s+"
                    r"(?:the\s+)?(?:file\s+)?(?:at\s+)?"
                    r"[`\"']?([A-Za-z0-9_~@%+=:,./-]{1,240})",
                    binding_text,
                    flags=re.IGNORECASE,
                )
                if natural_path:
                    candidate = natural_path.group(1).strip().rstrip(".,;`\"'")
                    if candidate.startswith("~") or "/" in candidate or "." in candidate:
                        return candidate
        patterns = (
            rf"\b{name}\s+argument\s+must\s+be\s+(?:the\s+)?literal\s+string\s+[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,119}})",
            rf"\b{name}\s*(?:=|:|to|is)\s*[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,119}})",
            rf"\bwith\s+{name}\s+[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,119}})",
        )
        for pattern in patterns:
            match = re.search(pattern, binding_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).rstrip(".,;:!?`'\"")
        historical = recent_tool_call_arguments.get(tool_name, {}).get(param)
        if isinstance(historical, str):
            return _safe_path_value(historical) if is_path_param else historical
        if historical is not None:
            return json.dumps(historical, ensure_ascii=False)
        return ""

    def _render_dsml_tool_calls_example(tools: list[dict]) -> str:
        # DSV4 already has the generic DSML parameter grammar in its native
        # template. Never inject placeholder values: live DSV4 copied
        # `VALUE HERE` / `string=` into real tool arguments after a tool-result
        # round. Bind parameters only when the current request names that tool
        # and supplies every required value; otherwise keep the safe empty
        # invoke. Render one wrapper per tool so the prompt does not teach
        # "call every available tool at once" during chained-tool requests.
        blocks: list[str] = []
        for tool in tools:
            name, props = _tool_props(tool)
            func = _tool_func(tool)
            params = func.get("parameters", {}) or {}
            schemas = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", []) if isinstance(params, dict) else [])
            explicitly_requested = (
                _request_mentions_tool_name(name)
                or name in recent_tool_call_arguments
                or (
                    dsv4_client_narrowed_to_remaining_tools
                    and name in tool_names
                )
            )
            values = {
                param: _derive_dsml_request_param_value(name, param)
                for param in props
            }
            has_all_required = all(values.get(param) for param in required)

            lines = [
                "<｜DSML｜tool_calls>",
                f'<｜DSML｜invoke name="{name}">',
            ]
            if explicitly_requested and has_all_required:
                for param in props:
                    value = values.get(param)
                    if not value:
                        continue
                    schema = schemas.get(param, {}) if isinstance(schemas, dict) else {}
                    is_string = not isinstance(schema, dict) or schema.get("type", "string") == "string"
                    string_attr = "true" if is_string else "false"
                    lines.append(
                        f'<｜DSML｜parameter name="{param}" string="{string_attr}">'
                        f"{value}</｜DSML｜parameter>"
                    )
            lines.extend(["</｜DSML｜invoke>", "</｜DSML｜tool_calls>"])
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _render_xml_examples(
        tools: list[dict],
        wrapper_open: str,
        wrapper_close: str,
    ) -> str:
        def _derive_run_command_value(tool_name: str) -> str:
            binding_text = request_text
            if not _request_mentions_tool_name(tool_name):
                binding_text = (
                    _historical_explicit_tool_request_text(tool_name)
                    or request_text
                )
            if not binding_text:
                return ""
            direct_patterns = (
                r'\brun\s+exactly:\s*`([^`\n]{1,240})`',
                r'\brun\s+exactly:\s*([^\n]{1,240}?)(?:\s+\.\s+(?:After|Then)\b|$)',
                r'\bto\s+run:\s*([^\n]{1,240}?)(?:\s*\.\s+(?:After|Then)\b|$)',
                r'(?<![A-Za-z0-9_])command\s+["“]([^"”\n]{1,240})["”]',
                r"(?<![A-Za-z0-9_])command\s+`([^`\n]{1,240})`",
                r"(?<![A-Za-z0-9_])command\s+([A-Za-z0-9_./:;|&%><=+,'\" -]{1,240})(?:[.\n]|$)",
            )
            for pattern in direct_patterns:
                match = re.search(pattern, binding_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip().rstrip(".,;:")

            read_then_create = re.search(
                r"\bread\s+([A-Za-z0-9_.:/-]{1,120})\s+and\s+create\s+"
                r"([A-Za-z0-9_.:/-]{1,120}).*?\bWrite(?:\s+the\s+text)?\s+"
                r"([A-Za-z0-9_.:-]{1,120})\s+into\s+the\s+second\s+file",
                binding_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if read_then_create:
                src, dst, text = (part.strip() for part in read_then_create.groups())
                return (
                    f"cat {shlex.quote(src)} >/dev/null && "
                    f"printf %s {shlex.quote(text)} > {shlex.quote(dst)}"
                )

            create_file = re.search(
                r"\bcreate\s+a\s+file\s+named\s+([A-Za-z0-9_.:/-]{1,120}).*?"
                r"\bWrite(?:\s+the\s+text)?\s+([A-Za-z0-9_.:-]{1,120})\s+"
                r"into\s+that\s+file",
                binding_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if create_file:
                path, text = (part.strip() for part in create_file.groups())
                return f"printf %s {shlex.quote(text)} > {shlex.quote(path)}"
            historical = recent_tool_call_arguments.get(tool_name, {}).get("command")
            if isinstance(historical, str):
                return historical
            return ""

        def _request_param_value(tool_name: str, param: str) -> str:
            normalized_tool = tool_name.strip().lower()
            normalized_param = param.strip().lower()
            if (
                normalized_tool == "run_command"
                and normalized_param == "command"
            ):
                command = _derive_run_command_value(tool_name)
                if command:
                    return command
            if not request_text:
                return ""
            if normalized_param in {"path", "file", "filename"}:
                if normalized_tool == "read_file":
                    read_file = re.search(
                        r"\bread_file\b[^\n.]{0,120}?\bfor\s+([A-Za-z0-9_.:/-]{1,120})",
                        request_text,
                        flags=re.IGNORECASE,
                    )
                    if read_file:
                        return read_file.group(1).rstrip(".,;:!?`'\"")
                if normalized_tool == "write_file":
                    write_file = re.search(
                        r"\bwrite_file\b[^\n.]{0,120}?\bpath\s+([A-Za-z0-9_.:/-]{1,120})",
                        request_text,
                        flags=re.IGNORECASE,
                    )
                    if write_file:
                        return write_file.group(1).rstrip(".,;:!?`'\"")
                # Preserve directory separators for generic path-bearing tools
                # such as file_info. The scalar fallback below intentionally
                # uses a narrower token alphabet and previously truncated
                # `panel/package.json` to `panel` in live MiniMax-M2.7 calls.
                path_name = re.escape(param)
                path_patterns = (
                    rf"\bwith\s+{path_name}\s+[`'\"]?([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
                    rf"\b{path_name}\s*(?:=|:|to|is)\s*[`'\"]?([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
                    rf"\b{path_name}\s+[`'\"]?([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
                )
                for pattern in path_patterns:
                    path_match = re.search(
                        pattern,
                        request_text,
                        flags=re.IGNORECASE,
                    )
                    if path_match:
                        return path_match.group(1).rstrip(".,;:!?`'\"")
            for obj_match in re.finditer(r"\{[^{}]{1,400}\}", request_text):
                try:
                    obj = json.loads(obj_match.group(0))
                except json.JSONDecodeError:
                    continue
                value = obj.get(param) if isinstance(obj, dict) else None
                if isinstance(value, str) and 0 < len(value) <= 80:
                    return value
            name = re.escape(param)
            patterns = (
                rf"\b{name}\s+parameter(?:\s+content)?\s+must\s+be\s+[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,79}})",
                rf"\b{name}\s+argument\s+must\s+be\s+(?:the\s+)?literal\s+string\s+[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,79}})",
                rf"\b{name}\s+argument\s*(?:=|:|to|is)\s*[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,79}})",
                rf"\b{name}\s+[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,119}})",
                rf"\b{name}\s*(?:=|:|to|is)\s*[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,79}})",
                rf"\bwith\s+{name}\s+[`'\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{{0,79}})",
            )
            for pattern in patterns:
                match = re.search(pattern, request_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).rstrip(".,;:!?`'\"")
            return ""

        def _example_value(tool_name: str, param: str) -> str:
            normalized = param.strip().lower()
            if normalized in {"path", "dir", "directory"} or normalized.endswith("_path"):
                requested = _request_param_value(tool_name, param)
                if requested:
                    return requested
                if tool_name.strip().lower() in {"list_directory", "read_directory"}:
                    return "."
                return ""
            return _request_param_value(tool_name, param)

        blocks: list[str] = []
        for tool in tools:
            name, props = _tool_props(tool)
            lines = [wrapper_open, f"<function={name}>"]
            for param in props:
                value = _example_value(name, param)
                if value:
                    lines.extend([f"<parameter={param}>", value, "</parameter>"])
            lines.extend(["</function>", wrapper_close])
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _render_zaya_tool_scaffold(tools: list[dict]) -> str:
        lines = [
            "# Tools",
            "",
            "You have access to the following functions:",
            "",
            "<tools>",
        ]
        for tool in tools:
            func = _tool_func(tool)
            name = func.get("name", "") or "unknown_tool"
            desc = func.get("description", "")
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = (
                params.get("required", [])
                if isinstance(params, dict)
                and isinstance(params.get("required", []), list)
                else []
            )
            lines.extend(["<function>", f"<name>{name}</name>"])
            lines.append("<parameters>")
            for param_name, param_schema in props.items():
                p_type = (
                    param_schema.get("type", "string")
                    if isinstance(param_schema, dict)
                    else "string"
                )
                lines.extend(
                    [
                        "<parameter>",
                        f"<name>{param_name}</name>",
                        f"<type>{p_type}</type>",
                    ]
                )
                lines.append("</parameter>")
            if required:
                lines.append("<required>" + json.dumps(required, ensure_ascii=False) + "</required>")
            lines.extend(["</parameters>", "</function>"])
        lines.extend(
            [
                "</tools>",
                "",
                "If you choose to call a function ONLY reply in the following "
                "format with NO suffix:",
                "",
                "<zyphra_tool_call>",
                "<function=example_function_name>",
                "<parameter=example_parameter_1>",
                "value_1",
                "</parameter>",
                "<parameter=example_parameter_2>",
                "This is the value for the second parameter",
                "that can span",
                "multiple lines",
                "</parameter>",
                "</function>",
                "</zyphra_tool_call>",
                "",
                "<IMPORTANT>",
                "Reminder:",
                "- Function calls MUST follow the specified format: an inner "
                "<function=...></function> block must be nested within "
                "<zyphra_tool_call></zyphra_tool_call> XML tags",
                "- Required parameters MUST be specified",
                "- You may provide optional reasoning for your function call in "
                "natural language BEFORE the function call, but NOT after",
                "- If there is no function call available, answer the user directly",
                "</IMPORTANT>",
            ]
        )
        return "\n".join(lines)

    def _render_minimax_examples(tools: list[dict]) -> str:
        xml_examples = _render_xml_examples(
            tools,
            "<tool_call>",
            "</tool_call>",
        )
        blocks: list[str] = []
        for block in re.findall(
            r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
            xml_examples,
            flags=re.DOTALL,
        ):
            name, params_block = block
            lines = ["<minimax:tool_call>", f'<invoke name="{name.strip()}">']
            for param, value in re.findall(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                params_block,
                flags=re.DOTALL,
            ):
                lines.append(
                    f'<parameter name="{param.strip()}">{value.strip()}</parameter>'
                )
            lines.extend(["</invoke>", "</minimax:tool_call>"])
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _derive_lfm2_request_param_value(tool_name: str, param: str) -> str:
        """Bind a scalar LFM2 example to a value explicitly named by the user.

        Liquid's smaller tool models copy native Python-call examples closely.
        Leaving ``VALUE_HERE`` in a requested ``file_info(path=...)`` example
        produced malformed live calls such as ``path=': '``. Only bind values
        from an explicit request for that tool; otherwise retain the generic
        placeholder rather than guessing.
        """
        binding_text = request_text
        if not _request_mentions_tool_name(tool_name):
            binding_text = (
                _historical_explicit_tool_request_text(tool_name)
                or request_text
            )
        if not binding_text:
            return ""

        escaped = re.escape(param.strip())
        quoted_patterns = (
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*(?:=|:)\s*`([^`\n]{{1,240}})`",
            rf'(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*(?:=|:)\s*"([^"\n]{{1,240}})"',
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*(?:=|:)\s*'([^'\n]{{1,240}})'",
        )
        for pattern in quoted_patterns:
            match = re.search(pattern, binding_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        bare_patterns = (
            rf"\bwith\s+{escaped}\s+([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s+([A-Za-z0-9_~@%+=:,./-]{{1,240}})",
        )
        for pattern in bare_patterns:
            match = re.search(pattern, binding_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,;")

        # Natural file requests commonly name the target after the action
        # instead of spelling out ``path`` (for example, "file_info ... to
        # inspect panel/package.json").  LFM2 copies the native example very
        # closely, so leaving VALUE_HERE here turns an otherwise explicit
        # request into a schema-valid call for the wrong file.  Bind only
        # path-like parameters, only for a request that explicitly names this
        # tool, and only after a file-oriented verb to avoid guessing from
        # unrelated prose.
        if (
            param.strip().lower() in {"path", "file_path", "filepath"}
            and _request_mentions_tool_name(tool_name)
        ):
            natural_path = re.search(
                r"\b(?:inspect|examine|stat|read|check)\s+"
                r"(?:the\s+)?(?:file\s+)?(?:at\s+)?"
                r"[`\"']?([A-Za-z0-9_~@%+=:,./-]{1,240})",
                binding_text,
                flags=re.IGNORECASE,
            )
            if natural_path:
                candidate = natural_path.group(1).strip().rstrip(".,;`\"'")
                if candidate.startswith("~") or "/" in candidate or "." in candidate:
                    return candidate
        return ""

    def _render_lfm2_examples(tools: list[dict]) -> str:
        def _derive_run_command_value() -> str:
            if not request_text:
                return ""
            direct_patterns = (
                r'\brun\s+exactly:\s*`([^`\n]{1,240})`',
                r'\brun\s+exactly:\s*([^\n]{1,240}?)(?:\s+\.\s+(?:After|Then)\b|$)',
                r'\bto\s+run:\s*([^\n]{1,240}?)(?:\s*\.\s+(?:After|Then)\b|$)',
                r'(?<![A-Za-z0-9_])command\s+`([^`\n]{1,240})`',
            )
            for pattern in direct_patterns:
                match = re.search(pattern, request_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip().rstrip(".,;:")
            read_then_create = re.search(
                r"\bread\s+([A-Za-z0-9_.:/-]{1,120})\s+and\s+create\s+"
                r"([A-Za-z0-9_.:/-]{1,120}).*?\bWrite(?:\s+the\s+text)?\s+"
                r"([A-Za-z0-9_.:-]{1,120})\s+into\s+the\s+second\s+file",
                request_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if read_then_create:
                src, dst, text = (part.strip() for part in read_then_create.groups())
                return (
                    f"cat {shlex.quote(src)} >/dev/null && "
                    f"printf %s {shlex.quote(text)} > {shlex.quote(dst)}"
                )

            create_file = re.search(
                r"\bcreate\s+a\s+file\s+named\s+([A-Za-z0-9_.:/-]{1,120}).*?"
                r"\bWrite(?:\s+the\s+text)?\s+([A-Za-z0-9_.:-]{1,120})\s+"
                r"into\s+that\s+file",
                request_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if create_file:
                path, text = (part.strip() for part in create_file.groups())
                return f"printf %s {shlex.quote(text)} > {shlex.quote(path)}"
            return ""

        calls: list[str] = []
        for tool in tools:
            name, props = _tool_props(tool)
            arg_parts: list[str] = []
            for param in props:
                value = ""
                if (
                    name.strip().lower() == "run_command"
                    and param.strip().lower() == "command"
                ):
                    value = _derive_run_command_value()
                if not value:
                    value = _derive_lfm2_request_param_value(name, param)
                arg_parts.append(f"{param}={(value or 'VALUE_HERE')!r}")
            calls.append(f"{name}({', '.join(arg_parts)})")
        return (
            "<|tool_call_start|>["
            + ", ".join(calls)
            + "]<|tool_call_end|>"
        )

    def _render_openpangu_examples(tools: list[dict]) -> str:
        calls: list[dict[str, Any]] = []
        for tool in tools:
            name, props = _tool_props(tool)
            arguments: dict[str, Any] = {}
            for param in props:
                value = _derive_lfm2_request_param_value(name, param)
                arguments[param] = value or "VALUE_HERE"
            calls.append({"name": name, "arguments": arguments})
        return (
            "<|tool_call_start|>\n"
            + json.dumps(calls, ensure_ascii=False)
            + "\n<|tool_call_end|>"
        )

    # DSV4's native parser is DSML, not generic <tool_call> JSON. Its shipped
    # templates currently do not render tool schemas, so this fallback is the
    # only place the model sees tool instructions on OpenAI/Responses tool
    # requests. Detect by the canonical DSV4 turn tokens already present in
    # the rendered prompt and inject the parser-matching format.
    if is_dsv4_prompt:
        if dsv4_multi_tool_continuation:
            remaining_names = dsv4_requested_remaining_tools or set(tool_names)
            dsv4_prompt_tools = [
                tool
                for tool in template_tools
                if _tool_props(tool)[0] in remaining_names
                and _tool_props(tool)[0] not in tools_called_after_latest_user
            ] or _requested_tools(template_tools)
        elif dsv4_tool_result_continuation and recent_tool_call_arguments:
            dsv4_prompt_tools = _recently_called_tools(template_tools)
        elif explicit_tool_requested:
            dsv4_prompt_tools = _requested_tools(template_tools)
        else:
            dsv4_prompt_tools = _recently_called_tools(template_tools)
        if dsv4_tool_result_continuation:
            completed_names = tools_called_after_latest_user or {
                name
                for name in (_tool_props(tool)[0] for tool in dsv4_prompt_tools)
                if name
            }
            completed = ", ".join(sorted(completed_names)) or "requested tool"
            exact_visible_answer = _requested_exact_visible_answer()
            tool_prompt = (
                "Native DSV4 tool-result continuation: the requested tool already "
                "ran and its real result is present in the conversation. "
                f"Completed tool(s): {completed}.\n"
                "Do not emit another <｜DSML｜tool_calls> block and do not copy, "
                "fabricate, or expose a <tool_result> block as assistant content. "
                "Finish the user's requested visible answer from the real tool "
                "result now."
                + (
                    "\nThe user requested an exact final visible answer. "
                    f"Your next assistant message must be exactly: {exact_visible_answer}"
                    if exact_visible_answer
                    else ""
                )
            )
        else:
            dsv4_lines = [
                "You have access to these tools. Call them using DSML format.",
                "",
            ]
            if explicit_tool_requested or recent_tool_call_arguments:
                dsv4_lines.extend(
                    [
                        "The current user explicitly named an available tool.",
                        "Your next assistant output must be exactly one canonical DSML tool call before any prose.",
                        "Copy the concrete parameter value from the matching example below exactly; do not repeat the schema lines.",
                        "",
                    ]
                )
            for idx, tool in enumerate(dsv4_prompt_tools):
                func = _tool_func(tool)
                name = func.get("name", "") or "unknown_tool"
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
                + "\n\nWhen you decide to call a tool, emit ONLY one canonical DSML tool_calls block for the next tool. "
                "Do not emit JSON, markdown, prose, generic XML tool tags, or bare invoke blocks outside the wrapper.\n"
                "Only use the tag name <｜DSML｜invoke>; never emit <｜DSML｜invuse>, <｜DSML｜invue>, <｜DSML｜inv>, or any shortened invoke tag.\n"
                + _render_dsml_tool_calls_example(dsv4_prompt_tools)
                + "\n\n"
                "Do not combine every available tool just because it is listed. "
                + (
                    "For a request to list the current directory, set the path parameter to \".\" exactly. "
                    if _has_directory_path_tool(dsv4_prompt_tools)
                    else ""
                )
                + "Do not explain inability to call tools; emit the DSML call."
            )
    elif is_zaya_native_tool_prompt:
        zaya_lines = [
            "You have access to native Zyphra tools. When the user asks for one, "
            "call it instead of fabricating a result.",
            "",
        ]
        if "<tools>" not in instruction_prompt:
            zaya_lines.extend([_render_zaya_tool_scaffold(zaya_prompt_tools), ""])
        for tool in zaya_prompt_tools:
            func = _tool_func(tool)
            name = func.get("name", "") or "unknown_tool"
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            if props:
                zaya_lines.append(f"{name} fields: {', '.join(str(p) for p in props)}")
            else:
                zaya_lines.append(f"{name} fields: none")
            if name.strip().lower() == "run_command" and "command" in props:
                command = _render_xml_examples(
                    [tool],
                    "<zyphra_tool_call>",
                    "</zyphra_tool_call>",
                )
                match = re.search(
                    r"<parameter=command>\s*([\s\S]*?)\s*</parameter>",
                    command,
                )
                if match:
                    exact_command = match.group(1).strip()
                    zaya_lines.append(
                        f"For this request, run_command.command must be exactly: {exact_command}"
                    )
                    for token in re.findall(r"\b[A-Z][A-Z0-9_]{6,}\b", request_text):
                        zaya_lines.append(
                            f"Do not use {token} itself as a shell command; "
                            "it is file content or answer text."
                        )
                    if (
                        "real_ui_tool_probe_1.txt" in request_text
                        and "real_ui_tool_probe_2.txt" in request_text
                        and "REAL_UI_LIVE_TOOL_TWO" in request_text
                    ):
                        zaya_lines.append(
                            "Do not copy real_ui_tool_probe_1.txt into "
                            "real_ui_tool_probe_2.txt; read the first file and "
                            "write REAL_UI_LIVE_TOOL_TWO into the second file."
                        )
        tool_prompt = (
            "\n".join(zaya_lines).rstrip()
            + "\n\nWhen a tool call is needed, emit ONLY this native Zyphra XML shape. "
            "Do not emit JSON result data, markdown, prose, generic XML tool tags, or a fake directory listing.\n"
            "If the user explicitly asks to use a tool, emit the tool call first; do not answer as if the tool result already exists.\n"
            "Fill fields from the user's request exactly. "
            "If the user says `with value blue-cat`, put only `blue-cat` in `value`.\n"
            + _render_xml_examples(
                zaya_prompt_tools,
                "<zyphra_tool_call>",
                "</zyphra_tool_call>",
            )
            + (
                "\n\nFor a request to list the current directory, set path to \".\" exactly."
                if _has_directory_path_tool(zaya_prompt_tools)
                else ""
            )
        )
    elif is_qwen_native_tool_prompt:
        if qwen_terminal_tool_result_continuation:
            completed_names = tools_called_after_latest_user or {
                name
                for name in (_tool_props(tool)[0] for tool in qwen_prompt_tools)
                if name
            }
            completed = ", ".join(sorted(completed_names)) or "requested tool"
            tool_prompt = (
                "Native tool-result continuation: the requested tool already ran and its "
                f"real result is present in the conversation. Completed tool(s): {completed}.\n"
                "Do not emit another <tool_call>, do not repeat the command, and do not claim "
                "that the tool still needs to run. Finish the user's requested visible answer "
                "from the real tool result now."
            )
        else:
            qwen_lines = [
                "You have access to these tools. When a user asks you to use one, "
                "you must call it instead of fabricating a result.",
                "",
            ]
            if qwen_multi_tool_continuation:
                completed = ", ".join(sorted(tools_called_after_latest_user)) or "prior tool"
                remaining = ", ".join(
                    _tool_props(tool)[0] for tool in qwen_prompt_tools
                )
                qwen_lines.extend(
                    [
                        "Native multi-tool continuation: the previous tool call already ran and its real result is present in the conversation.",
                        f"Completed tool(s): {completed}. Do not call those completed tools again.",
                        f"The client intentionally supplied the remaining tool(s): {remaining}.",
                        "Your next assistant output must be exactly one native call to a remaining tool before any prose.",
                        "Do not answer the user's final request until the remaining tool result is returned in a later continuation.",
                        "",
                    ]
                )
        if not qwen_terminal_tool_result_continuation and explicit_tool_requested and not tool_choice_required:
            qwen_lines.extend(
                [
                    "The current user explicitly named an available tool.",
                    "Your next assistant output must be exactly one native tool call before any prose.",
                    "Do not answer as if the tool already ran. Do not emit an empty function call.",
                    "Every required parameter listed below must be present and non-empty.",
                    "",
                ]
            )
        if not qwen_terminal_tool_result_continuation and tool_choice_required:
            qwen_lines.extend(
                [
                    "The current API request set tool_choice=required. You must emit exactly one native tool call before any prose.",
                    "Do not answer from prior context, do not summarize first, and do not invent a tool result.",
                    "Historical tool results in the conversation do not satisfy this current-turn requirement.",
                    "Only after this new tool call is executed in a later continuation may you produce the visible answer.",
                    "",
                ]
            )
        for idx, tool in enumerate(
            qwen_prompt_tools if not qwen_terminal_tool_result_continuation else []
        ):
            func = _tool_func(tool)
            name = func.get("name", "") or "unknown_tool"
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
                if name.strip().lower() == "run_command" and "command" in props:
                    command_block = _render_xml_examples(
                        [tool],
                        "<tool_call>",
                        "</tool_call>",
                    )
                    match = re.search(
                        r"<parameter=command>\s*([\s\S]*?)\s*</parameter>",
                        command_block,
                    )
                    if match:
                        exact_command = match.group(1).strip()
                        qwen_lines.append(
                            f"For this request, run_command.command must be exactly: {exact_command}"
                        )
                        for token in re.findall(r"\b[A-Z][A-Z0-9_]{6,}\b", request_text):
                            qwen_lines.append(
                                f"Do not use {token} itself as a shell command; "
                                "it is file content or answer text."
                            )
            qwen_lines.append("")
        if not qwen_terminal_tool_result_continuation:
            tool_prompt = (
                "\n".join(qwen_lines).rstrip()
                + "\n\nWhen a tool call is needed, emit ONLY this native XML shape. "
                "Do not emit JSON result data, markdown, prose, or a fake directory listing.\n"
                + (
                    "Because tool_choice=required, the first assistant output for this turn must be one of the native tool calls below and nothing else.\n"
                    if tool_choice_required
                    else ""
                )
                + _render_xml_examples(qwen_prompt_tools, "<tool_call>", "</tool_call>")
                + (
                    "\n\nFor a request to list the current directory, set path to \".\" exactly."
                    if _has_directory_path_tool(qwen_prompt_tools)
                    else ""
                )
            )
    elif is_xml_function_native_tool_prompt:
        xml_function_prompt_tools = _requested_tools(template_tools)
        xml_function_lines = [
            "MiMo XML function tools:",
        ]
        for tool in xml_function_prompt_tools:
            func = _tool_func(tool)
            name = func.get("name", "") or "unknown_tool"
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            if props:
                xml_function_lines.append(f"{name} fields: {', '.join(str(p) for p in props)}")
            else:
                xml_function_lines.append(f"{name} fields: none")
        tool_prompt = (
            "\n".join(xml_function_lines).rstrip()
            + "\nUse native XML function shape only when a tool is requested/required. "
            "No prose, JSON, markdown, fake results, or other XML. "
            "Copy user field values exactly.\n"
            + _render_xml_examples(
                xml_function_prompt_tools,
                "<tool_call>",
                "</tool_call>",
            )
            + (
                "\n\nFor a request to list the current directory, set path to \".\" exactly."
                if _has_directory_path_tool(xml_function_prompt_tools)
                else ""
            )
        )
    elif is_step3p5_native_tool_prompt:
        step3p5_lines = [
            "You have access to these tools. When the user asks to use one, "
            "call it using the native XML shape instead of inventing a result.",
            "",
        ]
        for tool in template_tools:
            func = _tool_func(tool)
            name = func.get("name", "") or "unknown_tool"
            desc = func.get("description", "")
            if desc:
                step3p5_lines.append(f"  Tool: {name} — {desc}")
            else:
                step3p5_lines.append(f"  Tool: {name}")
        tool_prompt = (
            "\n".join(step3p5_lines).rstrip()
            + "\n\nWhen a tool call is needed, emit ONLY this XML shape:\n"
            "<tool_call>\n<function=FUNCTION_NAME>\n"
            "<parameter=PARAM>VALUE</parameter>\n...\n</function>\n</tool_call>\n"
            "Do not emit JSON, markdown, prose, or fake directory listings.\n"
            + _render_xml_examples(
                template_tools,
                "<tool_call>",
                "</tool_call>",
            )
            + (
                "\n\nFor a request to list the current directory, set path to \".\" exactly."
                if _has_directory_path_tool(template_tools)
                else ""
            )
        )
    elif is_lfm2_native_tool_prompt:
        if lfm2_tool_result_continuation:
            # Liquid's native continuation contract is the assistant tool call,
            # a JSON tool-role result, then a fresh assistant generation. Do not
            # add a second synthetic instruction here; it competes with the
            # trained transcript and made the 8B hybrid terminate after a short
            # fragment in live Electron runs.
            tool_prompt = ""
        else:
            lfm2_lines = [
                "You have access to Liquid LFM2 native tools. When the user asks for one, "
                "call it instead of explaining what you would do.",
                "",
            ]
            if tool_choice_required:
                lfm2_lines.extend(
                    [
                        "The current API request requires a tool call.",
                        "Your next assistant output must be exactly one native "
                        "Liquid LFM2 tool call before any prose.",
                        "Do not answer from memory, summarize first, or invent a "
                        "tool result. The real result will arrive in a later "
                        "continuation.",
                        "",
                    ]
                )
            for tool in lfm2_prompt_tools:
                func = _tool_func(tool)
                name = func.get("name", "") or "unknown_tool"
                params = func.get("parameters", {}) or {}
                props = params.get("properties", {}) if isinstance(params, dict) else {}
                if props:
                    lfm2_lines.append(f"{name} fields: {', '.join(str(p) for p in props)}")
                else:
                    lfm2_lines.append(f"{name} fields: none")
                if name.strip().lower() == "run_command" and "command" in props:
                    command = _render_lfm2_examples([tool])
                    match = re.search(r"run_command\(command='([^']+)'\)", command)
                    if match:
                        exact_command = match.group(1)
                        lfm2_lines.append(
                            f"For this request, run_command.command must be exactly: {exact_command}"
                        )
                        for token in re.findall(r"\b[A-Z][A-Z0-9_]{6,}\b", request_text):
                            lfm2_lines.append(
                                f"Do not use {token} itself as a shell command; "
                                "it is file content or answer text."
                            )
                else:
                    for param in props:
                        exact_value = _derive_lfm2_request_param_value(name, param)
                        if exact_value:
                            lfm2_lines.append(
                                f"For this request, {name}.{param} must be exactly: "
                                f"{exact_value}"
                            )
            tool_prompt = (
                "\n".join(lfm2_lines).rstrip()
                + "\n\nWhen a tool call is needed, emit ONLY this Python-call-list shape. "
                "Do not emit JSON, generic XML tool blocks, markdown, prose, or fake results. "
                'Do not emit JSON such as {"content": "..."}; that is assistant text, '
                "not a tool call.\n"
                + _render_lfm2_examples(lfm2_prompt_tools)
            )
    elif is_openpangu_native_tool_prompt:
        if openpangu_tool_result_continuation:
            completed_names = tools_called_after_latest_user or {
                name
                for name in (_tool_props(tool)[0] for tool in openpangu_prompt_tools)
                if name
            }
            completed = ", ".join(sorted(completed_names)) or "requested tool"
            exact_visible_answer = _requested_exact_visible_answer()
            tool_prompt = (
                "Native openPangu tool-result continuation: the requested tool "
                "already ran and its real result is present in the conversation. "
                f"Completed tool(s): {completed}.\n"
                "Do not emit another <|tool_call_start|> block. Finish the "
                "user's requested visible answer from the real tool result now."
                + (
                    "\nThe user requested an exact final visible answer. "
                    f"Your next assistant message must be exactly: {exact_visible_answer}"
                    if exact_visible_answer
                    else ""
                )
            )
        else:
            openpangu_lines = [
                "You have access to openPangu native tools. When the user asks "
                "to use one, call it instead of fabricating a result.",
                "",
            ]
            if explicit_tool_requested or tool_choice_required:
                openpangu_lines.extend(
                    [
                        "The current user explicitly requested an available tool.",
                        "Your next assistant output must be exactly one native "
                        "openPangu tool-call JSON list before any prose.",
                        "Do not answer as if the tool already ran. Do not emit "
                        "another family's <tool_call> XML format.",
                        "Every required parameter below must be present and non-empty.",
                        "",
                    ]
                )
            for tool in openpangu_prompt_tools:
                func = _tool_func(tool)
                name = func.get("name", "") or "unknown_tool"
                openpangu_lines.append(f"Tool: {name}")
                desc = func.get("description", "")
                if desc:
                    openpangu_lines.append(f"  description: {desc}")
                params = func.get("parameters", {}) or {}
                props = params.get("properties", {}) if isinstance(params, dict) else {}
                required = set(params.get("required", []) if isinstance(params, dict) else [])
                if props:
                    openpangu_lines.append("  parameters:")
                    for p_name, p_schema in props.items():
                        p_type = (
                            p_schema.get("type", "string")
                            if isinstance(p_schema, dict)
                            else "string"
                        )
                        req = "required" if p_name in required else "optional"
                        openpangu_lines.append(f"    - {p_name} ({p_type}, {req})")
            tool_prompt = (
                "\n".join(openpangu_lines).rstrip()
                + "\n\nWhen a tool call is needed, emit ONLY this native "
                "openPangu JSON-list shape. Do not emit prose, markdown, "
                "generic XML <tool_call> tags, Python-call syntax, or a fake result.\n"
                + _render_openpangu_examples(openpangu_prompt_tools)
            )
    elif is_minimax_native_tool_prompt:
        minimax_prompt_tools = _requested_tools(template_tools)
        minimax_lines = [
            "You have access to MiniMax native tools. When the user asks for one, "
            "call it instead of explaining what you would do.",
            "",
        ]
        if tool_choice_required:
            minimax_lines.extend(
                [
                    "The current API request set tool_choice=required.",
                    "Your next assistant output must be exactly one MiniMax tool call and no visible text.",
                    "Do not emit only a partial tag; include every required parameter and close the invoke/tool_call tags.",
                    "",
                ]
            )
        for tool in minimax_prompt_tools:
            func = _tool_func(tool)
            name = func.get("name", "") or "unknown_tool"
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            if props:
                minimax_lines.append(f"{name} fields: {', '.join(str(p) for p in props)}")
            else:
                minimax_lines.append(f"{name} fields: none")
        tool_prompt = (
            "\n".join(minimax_lines).rstrip()
            + "\n\nWhen a tool call is needed, emit ONLY this exact native MiniMax XML shape. "
            "Do not emit JSON, markdown, prose, generic <tool_call> tags, or a fake result.\n"
            "Fill fields from the user's request exactly. "
            "If the user says the value argument must be the literal string blue-cat, put only blue-cat in value.\n"
            + _render_minimax_examples(minimax_prompt_tools)
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

    def _rendered_prompt_kept_tool_instructions(rendered: Optional[str]) -> bool:
        if not rendered:
            return False
        if is_zaya_native_tool_prompt:
            if not any(name in rendered for name in zaya_prompt_tool_names):
                return False
            return (
                "<zyphra_tool_call>" in rendered
                and all(f"<function={name}>" in rendered for name in zaya_prompt_tool_names)
            )
        if is_lfm2_native_tool_prompt:
            if lfm2_tool_result_continuation:
                return (
                    "<|tool_call_start|>" in rendered
                    and "<|tool_call_end|>" in rendered
                    and "<|im_start|>tool\n" in rendered
                )
            return (
                "<|tool_call_start|>" in rendered
                and "<|tool_call_end|>" in rendered
                and all(
                    name in rendered
                    for name in (_tool_props(tool)[0] for tool in lfm2_prompt_tools)
                    if name
                )
            )
        if is_openpangu_native_tool_prompt:
            if openpangu_tool_result_continuation:
                return (
                    "<|message_start|>tool\n" in rendered
                    and "<|tool_call_start|>" in rendered
                    and "<|tool_call_end|>" in rendered
                )
            return (
                "<|tool_call_start|>" in rendered
                and "<|tool_call_end|>" in rendered
                and all(
                    name in rendered
                    for name in (_tool_props(tool)[0] for tool in openpangu_prompt_tools)
                    if name
                )
            )
        if is_minimax_native_tool_prompt:
            return (
                "<minimax:tool_call>" in rendered
                and all(
                    f'<invoke name="{name}">' in rendered
                    for name in tool_names
                )
            )
        expected_tool_names = tool_names
        if is_qwen_native_tool_prompt:
            expected_tool_names = [
                name
                for name in (_tool_props(tool)[0] for tool in qwen_prompt_tools)
                if name
            ]
        if not all(name in rendered for name in expected_tool_names):
            return False
        if is_dsv4_prompt:
            return "<｜DSML｜invoke" in rendered
        if is_qwen_native_tool_prompt:
            return "<tool_call>" in rendered
        if is_xml_function_native_tool_prompt:
            return (
                "<tool_call>" in rendered
                and all(name in rendered for name in tool_names)
                and (
                    all(f"<name>{name}</name>" in rendered for name in tool_names)
                    or all(f"<function={name}>" in rendered for name in tool_names)
                )
            )
        if is_step3p5_native_tool_prompt:
            return all(f"<function={name}>" in rendered for name in tool_names)
        return True

    def _prepend_tool_prompt_to_message(
        msg: dict,
        text: str,
        *,
        as_list_text: bool = False,
    ) -> None:
        content = msg.get("content")
        text_part = {"type": "text", "text": text, "content": text}
        if isinstance(content, list) and (as_list_text or not is_zaya_native_tool_prompt):
            msg["content"] = [text_part, *content]
            return
        if is_zaya_native_tool_prompt and as_list_text:
            if isinstance(content, list):
                msg["content"] = [text_part, *content]
            else:
                existing = content if isinstance(content, str) else ""
                text_value = text + "\n\n" + existing
                msg["content"] = [
                    {"type": "text", "text": text_value, "content": text_value}
                ]
            return
        if is_zaya_native_tool_prompt and isinstance(content, list):
            msg["content"] = [text_part, *content]
            return
        msg["content"] = text + "\n\n" + (content or "")

    def _append_tool_prompt_to_message(msg: dict, text: str) -> None:
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                *content,
                {"type": "text", "text": "\n\n" + text, "content": "\n\n" + text},
            ]
        else:
            msg["content"] = (content or "") + "\n\n" + text

    def _splice_tool_prompt_into_rendered_chatml(rendered: str, text: str) -> str:
        """Keep fallback tool instructions inside the first ChatML system turn.

        MiMo-V2's Jinja template can drop synthetic system/user fallback
        messages after re-render. Prefixing parser instructions before
        ``<|im_start|>system`` leaves the model outside its native conversation
        frame and was observed in live MiMo XML-tool probes. If we must fall
        back to string surgery, keep the instructions within the rendered
        ChatML system message instead of changing the prompt's outer framing.
        """

        marker = "<|im_start|>system\n"
        start = rendered.find(marker)
        if start < 0:
            return text + "\n\n" + rendered
        body_start = start + len(marker)
        end = rendered.find("<|im_end|>", body_start)
        insertion = "\n\n" + text
        if end < 0:
            return rendered[:body_start] + insertion.lstrip() + rendered[body_start:]
        return rendered[:end] + insertion + rendered[end:]

    # Inject into messages
    messages_copy = [dict(m) for m in messages]
    if lfm2_tool_result_continuation:
        # Liquid's documented native conversation format supplies tool results
        # as JSON in the `tool` role. Electron's built-ins retain a useful
        # human-readable shell transcript; wrap that real transcript rather
        # than feeding unframed `$ command\n\noutput` text that makes LFM2.5
        # terminate its final-answer pass at the first token.
        messages_copy, _ = _normalize_lfm2_tool_result_messages(messages_copy)
    # LFM2.5 continuation already has the native assistant/tool transcript;
    # normalization below is the only injection it needs.
    injected = lfm2_tool_result_continuation
    for msg in messages_copy:
        if injected:
            break
        if msg.get("role") == "system":
            _append_tool_prompt_to_message(msg, tool_prompt)
            injected = True
            break

    if not injected:
        messages_copy.insert(0, {"role": "system", "content": tool_prompt})

    if is_qwen_native_tool_prompt and tool_choice_required:
        qwen_required_reminder = (
            "Current turn API contract: tool_choice=required. "
            "Your next assistant output must be exactly one native <tool_call> "
            f"for one of: {', '.join(tool_names)}. "
            "Historical tool results do not satisfy this current-turn requirement. "
            "Do not answer in prose before the tool call."
        )
        for msg in reversed(messages_copy):
            if msg.get("role") == "user":
                _append_tool_prompt_to_message(msg, qwen_required_reminder)
                break

    if (
        is_dsv4_prompt
        and explicit_tool_requested
        and recent_tool_call_arguments
        and not dsv4_tool_result_activity
    ):
        # A new explicit DSV4 tool request can immediately follow an earlier
        # native tool/result/final-answer transcript. Live Electron testing
        # showed the model copying that nearby <tool_result> envelope instead
        # of executing the new call even though the opening system instruction
        # carried the correct DSML example. Keep the new-turn boundary adjacent
        # to the latest user message. This is prompt-side native DSML guidance;
        # it does not repair, synthesize, or execute a model-emitted fake result.
        dsv4_new_turn_reminder = (
            "Current-turn DSV4 tool contract: earlier <tool_result> blocks belong "
            "only to earlier turns and do not satisfy this request. Do not copy, "
            "fabricate, or output a <tool_result> block. Your next assistant "
            "output must be exactly this new native DSML call and no prose:\n"
            + _render_dsml_tool_calls_example(dsv4_prompt_tools)
        )
        for msg in reversed(messages_copy):
            if msg.get("role") == "user":
                _append_tool_prompt_to_message(msg, dsv4_new_turn_reminder)
                break

    # Re-apply template with modified messages
    # Remove tools from kwargs so generic templates don't try to format them again.
    # ZAYA templates that already exposed the native scaffold keep `tools` so
    # their own <tools>/<IMPORTANT> rules survive while concrete examples are added.
    safe_kwargs = dict(template_kwargs)
    if is_zaya_native_tool_prompt and explicit_tool_requested:
        # Keep the native Zaya scaffold, but do not leave unrelated broad-catalog
        # schemas beside the request-bound example. The JANG_4M text model copied
        # a run_applescript schema even while the user explicitly requested
        # file_info. Requested tools may contain more than one name, so preserve
        # every explicitly named tool rather than assuming a single-tool turn.
        safe_kwargs["tools"] = zaya_prompt_tools
    if is_lfm2_native_tool_prompt:
        # Liquid's native template should retain its trained JSON tool schema,
        # but an explicit single-tool request must not re-expand to every
        # built-in tool after the concrete fallback was narrowed. The full
        # coding-tool catalog overwhelms small LFM2 variants and encourages a
        # prose simulation of the requested command instead of a native call.
        safe_kwargs["tools"] = lfm2_prompt_tools
    if is_openpangu_native_tool_prompt:
        # Keep openPangu's own <tools> block and narrow explicit requests to the
        # requested tool so the native template and the concrete fallback agree.
        safe_kwargs["tools"] = openpangu_prompt_tools
    if not (
        (
            is_zaya_native_tool_prompt
            and zaya_template_has_native_scaffold
            and not has_flat_responses_function_tools
        )
        or is_lfm2_native_tool_prompt
        or is_openpangu_native_tool_prompt
        or is_xml_function_native_tool_prompt
    ):
        safe_kwargs.pop("tools", None)

    try:
        new_prompt = tokenizer.apply_chat_template(messages_copy, **safe_kwargs)
        if _rendered_prompt_kept_tool_instructions(new_prompt):
            if (
                is_zaya_native_tool_prompt
                and zaya_template_has_native_scaffold
                and "tools" in safe_kwargs
                and "<tools>" not in new_prompt
            ):
                zaya_plain_kwargs = dict(safe_kwargs)
                zaya_plain_kwargs.pop("tools", None)
                plain_prompt = tokenizer.apply_chat_template(
                    messages_copy,
                    **zaya_plain_kwargs,
                )
                if _rendered_prompt_kept_tool_instructions(plain_prompt):
                    return plain_prompt
            return new_prompt

        # Some templates, including current ZAYA-VL, ignore system messages
        # entirely. Retry by putting the parser-native tool instructions into
        # the first user turn so the template cannot drop them silently.
        logger.warning(
            "Chat template dropped fallback tool schema after system injection; "
            "retrying with first-user injection."
        )
        user_messages = [dict(m) for m in messages]
        user_injected = False
        for msg in user_messages:
            if msg.get("role") == "user":
                _prepend_tool_prompt_to_message(msg, tool_prompt)
                user_injected = True
                break
        if not user_injected:
            content: Any = (
                [{"type": "text", "text": tool_prompt}]
                if is_zaya_native_tool_prompt
                else tool_prompt
            )
            user_messages.insert(0, {"role": "user", "content": content})
        new_prompt = tokenizer.apply_chat_template(user_messages, **safe_kwargs)
        if _rendered_prompt_kept_tool_instructions(new_prompt):
            return new_prompt
        if is_zaya_native_tool_prompt:
            logger.warning(
                "Chat template dropped fallback tool schema after plain "
                "first-user injection; retrying with list text content."
            )
            user_messages = [dict(m) for m in messages]
            user_injected = False
            for msg in user_messages:
                if msg.get("role") == "user":
                    _prepend_tool_prompt_to_message(
                        msg,
                        tool_prompt,
                        as_list_text=True,
                    )
                    user_injected = True
                    break
            if not user_injected:
                user_messages.insert(
                    0,
                    {"role": "user", "content": [{"type": "text", "text": tool_prompt}]},
                )
            new_prompt = tokenizer.apply_chat_template(user_messages, **safe_kwargs)
            if _rendered_prompt_kept_tool_instructions(new_prompt):
                return new_prompt
        logger.warning(
            "Chat template dropped fallback tool schema after first-user "
            "injection; applying rendered-prompt fallback."
        )
        if is_xml_function_native_tool_prompt:
            return _splice_tool_prompt_into_rendered_chatml(prompt, tool_prompt)
        return tool_prompt + "\n\n" + prompt
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


def _response_format_to_dict(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if response_format is None:
        return None
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
        return rf_dict
    return response_format


def _balanced_json_substrings(text: str) -> List[str]:
    candidates: List[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        starts = [i for i, ch in enumerate(text) if ch == opener]
        for start in starts:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : i + 1])
                        break
    return candidates


def _json_candidates_from_text(text: str) -> List[str]:
    stripped = text.strip()
    candidates: List[str] = [stripped]

    code_block_pattern = r"```(?:json|JSON)?\s*([\s\S]*?)\s*```"
    candidates.extend(match.strip() for match in re.findall(code_block_pattern, stripped))
    candidates.extend(_balanced_json_substrings(stripped))
    for marker in ("{", "["):
        idx = stripped.find(marker)
        if idx >= 0:
            candidates.append(stripped[idx:].strip())

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _schema_object_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        return schema["properties"]
    return {}


def _array_property_names(schema: Optional[Dict[str, Any]]) -> List[str]:
    props = _schema_object_properties(schema or {})
    return [
        name
        for name, spec in props.items()
        if isinstance(spec, dict) and spec.get("type") == "array"
    ]


def _replace_python_json_literals(candidate: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        return {"True": "true", "False": "false", "None": "null"}[value]

    return re.sub(r"(?<![\w\"])(True|False|None)(?![\w\"])", repl, candidate)


def _close_obvious_json(candidate: str) -> str:
    in_string = False
    escape = False
    stack: List[str] = []
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return candidate
    if in_string:
        return candidate
    return candidate + "".join(reversed(stack))


def _repair_adjacent_array_strings(
    candidate: str, schema: Optional[Dict[str, Any]]
) -> str:
    for field in _array_property_names(schema):
        field_pattern = re.escape(json.dumps(field))
        pattern = re.compile(
            rf"({field_pattern}\s*:\s*)"
            r'"([^"\\]*(?:\\.[^"\\]*)*)"'
            r'((?:\s*,\s*"[^"\\]*(?:\\.[^"\\]*)*")+)',
            flags=re.DOTALL,
        )

        def repl(match: re.Match[str]) -> str:
            first = match.group(2)
            tail = match.group(3)
            pieces = [first]
            pieces.extend(
                m.group(1)
                for m in re.finditer(r'\s*,\s*"([^"\\]*(?:\\.[^"\\]*)*)"', tail)
            )
            values = []
            for piece in pieces:
                try:
                    values.append(json.loads(f'"{piece}"'))
                except json.JSONDecodeError:
                    values.append(piece)
            return f"{match.group(1)}{json.dumps(values, ensure_ascii=False)}"

        candidate = pattern.sub(repl, candidate)
    return candidate


def _repair_json_candidate(
    candidate: str, schema: Optional[Dict[str, Any]] = None
) -> str:
    repaired = candidate.strip()
    repaired = _repair_adjacent_array_strings(repaired, schema)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = _replace_python_json_literals(repaired)
    repaired = _close_obvious_json(repaired)
    return repaired


def _coerce_json_schema_value(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, dict):
            return _coerce_json_schema_value(decoded, schema)
        return value
    if schema_type == "object" and isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        return {
            key: _coerce_json_schema_value(item, props.get(key, {}))
            for key, item in value.items()
        }
    if schema_type == "array":
        if isinstance(value, list):
            items_schema = schema.get("items", {})
            return [_coerce_json_schema_value(item, items_schema) for item in value]
        return [value]
    return value


def _validate_parsed_json_for_schema(
    parsed: Any,
    schema: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    if not schema:
        return True, None
    return validate_json_schema(parsed, schema)


def _parse_json_with_repair_report(
    text: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stripped = text.strip()
    report: Dict[str, Any] = {
        "raw_json_ok": False,
        "raw_schema_ok": None,
        "repair_needed": False,
        "repair_actions": [],
        "candidate": None,
        "repaired_text": None,
        "parsed": None,
        "is_valid": False,
        "error": "Failed to extract valid JSON from output",
    }

    try:
        raw_parsed = json.loads(stripped)
        report["raw_json_ok"] = True
        raw_schema_ok, raw_schema_error = _validate_parsed_json_for_schema(
            raw_parsed, schema
        )
        report["raw_schema_ok"] = raw_schema_ok
        if not raw_schema_ok:
            report["repair_actions"].append("schema_type_coercion")
            report["error"] = f"JSON Schema validation failed: {raw_schema_error}"
    except json.JSONDecodeError:
        pass

    for candidate in _json_candidates_from_text(text):
        attempts: List[Tuple[str, str]] = [("raw_candidate", candidate)]
        repaired_candidate = _repair_json_candidate(candidate, schema)
        if repaired_candidate != candidate:
            attempts.append(("syntax_repair", repaired_candidate))

        for action, attempt in attempts:
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue

            coerced = _coerce_json_schema_value(parsed, schema)
            schema_ok, schema_error = _validate_parsed_json_for_schema(coerced, schema)
            actions: List[str] = []
            if candidate.strip() != stripped:
                actions.append("extract_json_block")
            if action == "syntax_repair":
                actions.append("syntax_repair")
            if coerced != parsed:
                actions.append("schema_type_coercion")

            report.update(
                {
                    "repair_needed": bool(actions)
                    or not report["raw_json_ok"]
                    or report.get("raw_schema_ok") is False,
                    "repair_actions": list(dict.fromkeys(report["repair_actions"] + actions)),
                    "candidate": candidate,
                    "repaired_text": attempt,
                    "parsed": coerced,
                    "is_valid": schema_ok,
                    "error": None if schema_ok else f"JSON Schema validation failed: {schema_error}",
                }
            )
            return report

    return report


def extract_json_from_text(
    text: str, schema: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
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
    report = _parse_json_with_repair_report(text, schema)
    if report.get("parsed") is not None:
        return report.get("parsed")
    return None


def extract_xml_from_text(text: str, root_tag: Optional[str] = None) -> Optional[str]:
    """Extract a parseable XML block from model output text.

    This is intentionally conservative: it strips markdown fences and prose,
    returns a canonical XML string, and fails closed if no valid XML block can
    be parsed. It does not invent missing nested tags.
    """
    candidates = [text.strip()]
    candidates.extend(match.strip() for match in re.findall(r"```(?:xml|XML)?\s*([\s\S]*?)\s*```", text))
    tags = [root_tag] if root_tag else re.findall(r"<([A-Za-z_][\w:.-]*)\b[^>]*>", text)
    for tag in tags:
        if tag.startswith("/"):
            continue
        match = re.search(rf"(<{re.escape(tag)}\b[\s\S]*?</{re.escape(tag)}>)", text)
        if match:
            candidates.append(match.group(1).strip())

    for candidate in candidates:
        try:
            root = ET.fromstring(candidate)
        except ET.ParseError:
            continue
        if root_tag and root.tag != root_tag:
            continue
        return ET.tostring(root, encoding="unicode")
    return None


def _xml_candidates_from_text(text: str, root_tag: Optional[str] = None) -> List[str]:
    candidates = [text.strip()]
    candidates.extend(
        match.strip()
        for match in re.findall(r"```(?:xml|XML)?\s*([\s\S]*?)\s*```", text)
    )

    tags = [root_tag] if root_tag else re.findall(r"<([A-Za-z_][\w:.-]*)\b[^>]*>", text)
    seen_tags: set[str] = set()
    for tag in tags:
        if not tag or tag.startswith("/") or tag in seen_tags:
            continue
        seen_tags.add(tag)
        for match in re.finditer(
            rf"(<{re.escape(tag)}\b[\s\S]*?</{re.escape(tag)}>)",
            text,
        ):
            candidates.append(match.group(1).strip())

    unique: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _xml_missing_required_fields(
    root: ET.Element,
    required_fields: Tuple[str, ...],
) -> List[str]:
    missing: List[str] = []
    for field in required_fields:
        if not field:
            continue
        found = root.find(field)
        if found is None or (found.text or "").strip() == "":
            missing.append(field)
    return missing


def _xml_validation_error(
    root: ET.Element,
    root_tag: Optional[str],
    required_fields: Tuple[str, ...],
) -> Tuple[bool, Optional[str], Optional[bool]]:
    if root_tag and root.tag != root_tag:
        return False, f"XML root tag mismatch: expected {root_tag}, got {root.tag}", None

    missing = _xml_missing_required_fields(root, required_fields)
    if missing:
        return False, f"missing required XML fields: {', '.join(missing)}", False
    return True, None, True if required_fields else None


def repair_xml_output(
    text: str,
    *,
    root_tag: Optional[str] = None,
    required_fields: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Parse/repair XML output and return raw-vs-repaired diagnostics.

    This is post-generation extraction and validation for benchmark/catalog
    callers. It strips wrappers such as markdown/prose around one valid XML
    block, validates the expected root and required child fields, and fails
    closed when multiple candidate XML blocks could satisfy an unspecified root.
    It does not invent missing tags or perform guided decoding.
    """
    stripped = text.strip()
    report: Dict[str, Any] = {
        "cleaned_text": text,
        "xml": None,
        "is_valid": False,
        "error": "Failed to extract valid XML from output",
        "raw_xml_ok": False,
        "raw_fields_ok": None,
        "repair_needed": False,
        "repair_actions": [],
        "repaired_text": None,
        "root_tag": root_tag,
        "required_fields": list(required_fields),
    }

    raw_valid = False
    try:
        raw_root = ET.fromstring(stripped)
        report["raw_xml_ok"] = True
        raw_valid, raw_error, raw_fields_ok = _xml_validation_error(
            raw_root,
            root_tag,
            required_fields,
        )
        report["raw_fields_ok"] = raw_fields_ok
        if raw_valid:
            xml = ET.tostring(raw_root, encoding="unicode")
            report.update(
                {
                    "xml": xml,
                    "is_valid": True,
                    "error": None,
                    "repaired_text": xml,
                }
            )
            return report
        report["error"] = raw_error
    except ET.ParseError:
        pass

    valid_candidates: List[Tuple[str, ET.Element, str]] = []
    invalid_candidate_error = report.get("error")
    for candidate in _xml_candidates_from_text(text, root_tag):
        try:
            root = ET.fromstring(candidate)
        except ET.ParseError as exc:
            invalid_candidate_error = f"XML parse failed: {exc}"
            continue

        ok, error, _fields_ok = _xml_validation_error(root, root_tag, required_fields)
        if not ok:
            invalid_candidate_error = error
            continue
        valid_candidates.append((candidate, root, ET.tostring(root, encoding="unicode")))

    unique_valid = list(
        {
            xml: (candidate, root, xml)
            for candidate, root, xml in valid_candidates
        }.values()
    )
    if not root_tag and len(unique_valid) > 1:
        report["error"] = "ambiguous XML candidates; specify root_tag"
        return report
    if not unique_valid:
        report["error"] = invalid_candidate_error or report["error"]
        return report

    candidate, _root, xml = unique_valid[0]
    actions: List[str] = []
    if candidate.strip() != stripped or not report["raw_xml_ok"] or not raw_valid:
        actions.append("extract_xml_block")
    report.update(
        {
            "xml": xml,
            "is_valid": True,
            "error": None,
            "repair_needed": bool(actions),
            "repair_actions": actions,
            "repaired_text": xml,
        }
    )
    return report


def _response_format_json_schema(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    rf_dict = _response_format_to_dict(response_format)
    if not isinstance(rf_dict, dict):
        return None
    format_type = rf_dict.get("type", "text")
    if format_type == "json_object":
        return {"type": "object"}
    if format_type != "json_schema":
        return None
    json_schema_spec = rf_dict.get("json_schema", {})
    if not isinstance(json_schema_spec, dict):
        return None
    schema = json_schema_spec.get("schema")
    return schema if isinstance(schema, dict) else None


def _tokenizer_vocab_size(tokenizer: Any) -> Optional[int]:
    try:
        size = len(tokenizer)
        if int(size) > 0:
            return int(size)
    except Exception:
        pass
    for attr in ("vocab_size", "n_vocab"):
        try:
            size = int(getattr(tokenizer, attr))
        except Exception:
            continue
        if size > 0:
            return size
    return None


def _mx_tokens_to_list(tokens: Any) -> List[int]:
    try:
        values = tokens.tolist()
    except Exception:
        try:
            values = list(tokens)
        except Exception:
            return []
    if values and isinstance(values[0], list):
        values = values[0]
    out: List[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def build_guided_json_logits_processor(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]],
    tokenizer: Any,
) -> Optional[Any]:
    """Build an llguidance JSON/schema logits processor when supported.

    The returned callable follows the MLX logits-processor shape
    ``processor(tokens, logits) -> logits``. It constrains only generated JSON
    tokens. If llguidance or a compatible fast tokenizer is unavailable, callers
    get ``None`` and should keep using prompt instruction plus repair/retry.
    """
    schema = _response_format_json_schema(response_format)
    vocab_size = _tokenizer_vocab_size(tokenizer)
    if not schema or tokenizer is None or vocab_size is None:
        return None

    try:
        from llguidance import LLMatcher, grammar_from
        from llguidance.hf import from_tokenizer
        from llguidance.mlx import (
            allocate_token_bitmask,
            apply_token_bitmask,
            fill_next_token_bitmask,
        )
    except Exception as exc:
        logger.debug("llguidance JSON processor unavailable: %s", exc)
        return None

    try:
        cache_key = (id(tokenizer), vocab_size)
        ll_tokenizer = _LLGUIDANCE_TOKENIZER_CACHE.get(cache_key)
        if ll_tokenizer is None:
            ll_tokenizer = from_tokenizer(tokenizer)
            _LLGUIDANCE_TOKENIZER_CACHE[cache_key] = ll_tokenizer
        grammar = grammar_from("json_schema", json.dumps(schema, ensure_ascii=False))
        matcher = LLMatcher(ll_tokenizer, grammar)
        bitmask = allocate_token_bitmask(1, vocab_size)
    except Exception as exc:
        logger.debug("llguidance JSON processor setup failed: %s", exc)
        return None

    # MLX generate_step processors receive the final prompt token on the first
    # call, then append generated tokens on subsequent calls. Skip that context
    # token so the grammar starts at the first generated byte/token.
    seen_tokens = 1

    def _guided_json_processor(tokens: Any, logits: Any) -> Any:
        nonlocal seen_tokens, matcher
        token_list = _mx_tokens_to_list(tokens)
        if len(token_list) > seen_tokens:
            try:
                matcher.consume_tokens(token_list[seen_tokens:])
            except Exception:
                return logits
            seen_tokens = len(token_list)
        try:
            bitmask.fill(-1)
            fill_next_token_bitmask(matcher, bitmask, 0)
            return apply_token_bitmask(logits, bitmask)
        except Exception as exc:
            logger.debug("llguidance JSON processor mask failed: %s", exc)
            return logits

    try:
        setattr(_guided_json_processor, "_vmlx_guided_decoding", "llguidance_json")
    except Exception:
        pass
    return _guided_json_processor


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
    rf_dict = _response_format_to_dict(response_format)
    if rf_dict is None:
        return text, None, True, None

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    json_schema_spec = rf_dict.get("json_schema", {}) if isinstance(rf_dict, dict) else {}
    schema = json_schema_spec.get("schema", {}) if isinstance(json_schema_spec, dict) else {}

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text, schema if format_type == "json_schema" else None)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def repair_json_output(
    text: str,
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Parse/repair JSON output and return raw-vs-repaired diagnostics.

    This is a post-generation repair report for benchmark/catalog/API callers.
    It is not grammar-constrained or guided decoding: invalid JSON is detected
    and repaired after the model emits text, while raw parse failure remains
    visible through ``raw_json_ok`` and ``repair_needed``.
    """
    rf_dict = _response_format_to_dict(response_format)
    if rf_dict is None or rf_dict.get("type", "text") == "text":
        return {
            "cleaned_text": text,
            "parsed": None,
            "is_valid": True,
            "error": None,
            "raw_json_ok": None,
            "raw_schema_ok": None,
            "repair_needed": False,
            "repair_actions": [],
            "repaired_text": None,
        }

    format_type = rf_dict.get("type", "text")
    json_schema_spec = rf_dict.get("json_schema", {}) if isinstance(rf_dict, dict) else {}
    schema = json_schema_spec.get("schema", {}) if isinstance(json_schema_spec, dict) else {}
    schema_for_repair = schema if format_type == "json_schema" else None

    report = _parse_json_with_repair_report(text, schema_for_repair)
    if format_type == "json_object" and report.get("parsed") is not None:
        report["is_valid"] = True
        report["error"] = None
    elif format_type == "json_schema" and report.get("parsed") is not None and schema:
        is_valid, error = validate_json_schema(report["parsed"], schema)
        report["is_valid"] = is_valid
        report["error"] = None if is_valid else f"JSON Schema validation failed: {error}"

    return {
        "cleaned_text": text,
        "parsed": report.get("parsed"),
        "is_valid": bool(report.get("is_valid")),
        "error": report.get("error"),
        "raw_json_ok": report.get("raw_json_ok"),
        "raw_schema_ok": report.get("raw_schema_ok"),
        "repair_needed": bool(report.get("repair_needed")),
        "repair_actions": report.get("repair_actions", []),
        "repaired_text": report.get("repaired_text"),
    }


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
