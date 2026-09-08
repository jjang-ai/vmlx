# SPDX-License-Identifier: Apache-2.0
"""
Abstract tool parser base class and manager for vmlx-engine.

Inspired by vLLM's tool parser architecture but simplified for MLX backend.
"""

import importlib
import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Pattern to match and strip think tags (both <think> and [THINK] formats)
# Handles two cases per format:
# 1. Full tags: <think>...</think> or [THINK]...[/THINK]
# 2. Only closing tag: ...content before...</think> or ...[/THINK] (when think is in prompt)
THINK_TAG_PATTERN = re.compile(r"(?:<think>.*?</think>|\[THINK\].*?\[/THINK\])", re.DOTALL)
IMPLICIT_THINK_PATTERN = re.compile(r"^.*?(?:</think>|\[/THINK\])", re.DOTALL)
UNCLOSED_THINK_PATTERN = re.compile(r"(?:<think>|\[THINK\]).*$", re.DOTALL)


def generate_tool_id() -> str:
    """Generate a unique tool call ID (OpenAI format: call_<8hex>)."""
    return f"call_{uuid.uuid4().hex[:8]}"


def is_valid_tool_name(name: Any) -> bool:
    """A tool name must be a non-empty string.

    JSON tool markup can carry any type in ``name``. A bare truthiness check
    lets ``{"name": 12345}`` through, and response validation then nulls the
    non-string name — so the turn emits a tool call whose name is ``None``.
    That consumes a tool iteration and dispatches nothing, which is worse than
    treating the emission as plain content.

    Only applies where ``name`` comes from decoded JSON; sites that slice it
    out of a regex match are already strings.
    """
    return isinstance(name, str) and bool(name.strip())


@dataclass
class ExtractedToolCallInformation:
    """Information extracted from model output about tool calls."""

    tools_called: bool
    """Whether any tool calls were detected."""

    tool_calls: list[dict[str, Any]]
    """List of tool calls with 'name' and 'arguments' fields."""

    content: str | None = None
    """Any content that wasn't part of tool calls."""


class ToolParser(ABC):
    """
    Abstract base class for tool call parsers.

    Each parser implementation handles a specific model's tool calling format.
    """

    # Stream-visible opening markers of this parser's native dialect. The
    # server's _TOOL_CALL_MARKERS visibility guard must cover every entry
    # (prefix match) or the dialect's control payload can leak into visible
    # text - the exact class that let Muse's <atem: markup reach users. A
    # contract test enforces coverage for every registered parser.
    NATIVE_MARKERS: tuple = ()

    # Class attribute to declare native format support.
    # Set to True in subclasses whose corresponding model chat templates
    # can handle role="tool" messages and tool_calls fields directly,
    # without needing conversion to text format.
    SUPPORTS_NATIVE_TOOL_FORMAT: bool = False

    @classmethod
    def supports_native_format(cls) -> bool:
        """
        Check if this parser supports native tool message format.

        Native format means the parser's corresponding model chat template
        can handle:
        - role="tool" messages directly (not converted to role="user")
        - tool_calls field on assistant messages (not converted to text)

        Returns:
            True if native format is supported
        """
        return cls.SUPPORTS_NATIVE_TOOL_FORMAT

    # Streaming early-stop convention (opt-in per parser family).
    #
    # Most tool-calling models emit EOS right after their tool-call block, so
    # the server's streaming loop simply buffers until the model stops and
    # parses the buffer at end of stream. Some models (live-proven: degraded
    # 2-bit openPangu-2.0 JANG bundles) keep narrating after the closing
    # tool-call tag instead of stopping, burning the rest of max_tokens and
    # ending the stream with finish_reason="length" even though complete tool
    # calls were produced.
    #
    # A parser whose FORMAT makes "the tool-call turn is over" detectable
    # mid-stream can set STREAM_STOPS_AFTER_COMPLETE_CALL = True and implement
    # stream_tool_calls_complete(). The server then aborts generation after a
    # short grace window once the turn is complete, and the normal post-stream
    # extraction emits the tool_calls chunks with finish_reason="tool_calls"
    # (#46 contract). Parsers whose models legitimately continue after tool
    # calls (interleaved content, sequential call blocks without a terminal
    # marker) MUST keep the default False — the server never early-stops them.
    STREAM_STOPS_AFTER_COMPLETE_CALL: bool = False

    def stream_tool_calls_complete(self, buffered_text: str) -> bool:
        """
        Return True when ``buffered_text`` contains a fully closed tool-call
        turn (at least one complete, parseable tool call and no tool-call
        block still open at the end of the text).

        Only consulted when STREAM_STOPS_AFTER_COMPLETE_CALL is True. The
        default implementation never requests an early stop.
        """
        return False

    def stream_tool_call_stop_truncate(self, buffered_text: str) -> str:
        """
        Truncate ``buffered_text`` just past the last complete tool-call block.

        Called by the server when it early-stops generation after
        stream_tool_calls_complete() held through the grace window, so that
        post-call rambling (the reason the stop was needed) never reaches the
        tool parser as content or leaks to the client. Default: no-op.
        """
        return buffered_text

    @staticmethod
    def strip_think_tags(text: str) -> str:
        """
        Strip think tags from text.

        Handles two scenarios:
        1. Full tags: <think>...</think> in output
        2. Only closing tag: ...</think> when <think> was in prompt

        Used as fallback when no reasoning parser is configured but the model
        produces thinking tags. This prevents tool parsing failures with
        models that use thinking tags (e.g., Ring-Mini-Linear-2.0 with hermes).

        Args:
            text: Model output that may contain think tags

        Returns:
            Text with think tags removed
        """
        # First try to strip full tags
        result = THINK_TAG_PATTERN.sub("", text)

        # If no full tags found but </think> exists, strip implicit think
        # (when <think> was injected in the prompt)
        if result == text and ("</think>" in text or "[/THINK]" in text):
            result = IMPLICIT_THINK_PATTERN.sub("", text)

        # Finally, if an open tag remains without a close tag (e.g. hit max_tokens),
        # strip everything from the open tag to the end of string to prevent 
        # evaluating hallucinated models inside the brainstorming monologue.
        if "<think>" in result or "[THINK]" in result:
            result = UNCLOSED_THINK_PATTERN.sub("", result)

        return result.strip()

    def __init__(self, tokenizer: "PreTrainedTokenizerBase | None" = None):
        """
        Initialize the tool parser.

        Args:
            tokenizer: The tokenizer for the model (optional, some parsers need it)
        """
        self.model_tokenizer = tokenizer
        # State for streaming parsing
        self.current_tool_id: int = -1
        self.prev_tool_call_arr: list[dict] = []

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the tokenizer vocabulary."""
        if self.model_tokenizer is None:
            return {}
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response.

        Args:
            model_output: The complete model output string
            request: Optional request context (for tool definitions, etc.)

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        ...

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
        """
        Extract tool calls from streaming model output.

        NOTE: Not called by the server at runtime. The server uses a
        buffer-then-parse strategy (accumulate full output, then call
        extract_tool_calls on complete text). Subclass implementations
        are retained for testing and potential future streaming-native
        tool parsing.

        Args:
            previous_text: Text before this delta
            current_text: Complete text so far
            delta_text: New text in this chunk
            previous_token_ids: Token IDs before this delta
            current_token_ids: All token IDs so far
            delta_token_ids: New token IDs in this chunk
            request: Optional request context

        Returns:
            Delta message dict with content and/or tool_calls, or None
        """
        return None

    def reset(self) -> None:
        """Reset parser state for a new request."""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []

    @staticmethod
    def _function_schema_for_tool(
        request: dict[str, Any] | None, tool_name: str
    ) -> dict[str, Any] | None:
        if not request:
            return None
        for tool in request.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            # Chat Completions templates expose tools as
            # {"type":"function","function":{"name":...,"parameters":...}}.
            # The Responses API exposes function tools as
            # {"type":"function","name":...,"parameters":...}. Parsers must
            # see the same schema in both protocols; otherwise a schema-gated
            # repair can pass in Electron/Chat and fail in raw Responses.
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name") == tool_name:
                parameters = function.get("parameters")
                return parameters if isinstance(parameters, dict) else None
            if tool.get("type") == "function" and tool.get("name") == tool_name:
                parameters = tool.get("parameters")
                return parameters if isinstance(parameters, dict) else None
        return None

    @staticmethod
    def _request_tool_names(request: dict[str, Any] | None) -> set[str]:
        """Advertised function-tool names, reading BOTH protocol shapes
        (Chat Completions nests under "function", Responses is flat) for the
        same reason as _function_schema_for_tool above."""
        if not isinstance(request, dict):
            return set()
        names: set[str] = set()
        for tool in request.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                names.add(fn["name"])
            elif tool.get("type") == "function" and isinstance(tool.get("name"), str):
                names.add(tool["name"])
        return names

    # Qwen3.6-35B at temp 0 emits a malformed doubled-wrapper nesting
    # (observed verbatim in the 2026-08-15 smoke on BOTH the qwen and
    # xml_function parser routes — the recovery lives here so every parser
    # that can receive the shape shares one implementation):
    #   <tool_call>
    #   <function=function>
    #   <function=record_fact>
    #   <function=value>
    #   blue-cat
    #   </parameter>
    #   </function>
    #   </tool_call>
    # The real function name and every parameter are unambiguous when the
    # request's tool names are known: the only opener matching a requested
    # tool is the call, and `<function=KEY>V</parameter>` inside it is a
    # miskeyed `<parameter=KEY>V</parameter>`.
    _RECOVERY_FUNCTION_OPENER = re.compile(r"<function=([^>]+)>")
    _RECOVERY_STRICT_PARAM = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
    )
    _RECOVERY_MISKEYED_PARAM = re.compile(
        r"<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>", re.DOTALL
    )
    # Third live variant (captured verbatim via the raw-suffix log on the
    # Responses stream): the parameter KEY floats outside a bare opener —
    #   <parameter>
    #   command>
    #   printf %s ... > file.txt
    #   </parameter>
    # The key is the first identifier terminated by `>` inside the bare
    # parameter block; everything after it is the value.
    _RECOVERY_SPLIT_KEY_PARAM = re.compile(
        r"<parameter>\s*([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    @staticmethod
    def _recovery_coerce_value(value: str) -> Any:
        try:
            return json.loads(value.strip())
        except (json.JSONDecodeError, ValueError):
            return value

    @classmethod
    def _recovery_arguments_from_body(cls, body: str) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for param_name, param_value in cls._RECOVERY_STRICT_PARAM.findall(body):
            arguments[param_name.strip()] = cls._recovery_coerce_value(param_value)
        if not arguments:
            for param_name, param_value in cls._RECOVERY_SPLIT_KEY_PARAM.findall(body):
                arguments[param_name.strip()] = cls._recovery_coerce_value(param_value)
        return arguments

    @classmethod
    def _recover_doubled_wrapper_calls(
        cls, text: str, *, allowed_names: set[str] | None
    ) -> list[dict[str, Any]]:
        if not allowed_names:
            return []
        tool_calls: list[dict[str, Any]] = []
        for match in cls._RECOVERY_FUNCTION_OPENER.finditer(text):
            name = match.group(1).strip()
            if name not in allowed_names:
                continue
            body = text[match.end():]
            next_real = None
            for later in cls._RECOVERY_FUNCTION_OPENER.finditer(body):
                if later.group(1).strip() in allowed_names:
                    next_real = later.start()
                    break
            if next_real is not None:
                body = body[:next_real]
            arguments = cls._recovery_arguments_from_body(body)
            if not arguments:
                for key, value in cls._RECOVERY_MISKEYED_PARAM.findall(body):
                    key = key.strip()
                    if key in allowed_names:
                        continue
                    arguments[key] = cls._recovery_coerce_value(value)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    @classmethod
    def _serialize_tool_arguments(
        cls,
        tool_name: str,
        arguments: Any,
        request: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(arguments, dict):
            return json.dumps(arguments, ensure_ascii=False)
        if not isinstance(arguments, str):
            return str(arguments)

        schema = cls._function_schema_for_tool(request, tool_name)
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict):
            return arguments

        xml_arg = arguments.strip()
        match = re.fullmatch(r"<([A-Za-z_][\w.-]*)>\s*(.*?)\s*</\1>", xml_arg, re.DOTALL)
        if not match:
            return arguments
        param_name, param_value = match.groups()
        if param_name not in properties:
            return arguments
        return json.dumps({param_name: param_value.strip()}, ensure_ascii=False)


class ToolParserManager:
    """
    Central registry for ToolParser implementations.

    Supports both eager and lazy registration of tool parsers.
    """

    tool_parsers: dict[str, type[ToolParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def load_plugin(cls, spec: str) -> list[str]:
        """Import a tool-parser plugin and return the names it registered.

        ``spec`` is a ``.py`` file path or a dotted module name. The module
        registers parsers itself through ``register_module`` (a plugin may
        re-register an existing name — ``force`` defaults to True — which is
        how a controlled fault-injection parser stands in for a real one in a
        live proof without shipping test code in the engine).
        """
        import importlib.util as _ilu
        import os as _os

        before = set(cls.tool_parsers)
        if spec.endswith(".py") or _os.sep in spec:
            mod_name = "vmlx_tool_parser_plugin_" + re.sub(r"[^A-Za-z0-9_]", "_", _os.path.basename(spec))
            if not _os.path.isfile(spec):
                raise ImportError(f"tool parser plugin not found: {spec}")
            module_spec = _ilu.spec_from_file_location(mod_name, spec)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"tool parser plugin not loadable: {spec}")
            module = _ilu.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            importlib.import_module(spec)
        return sorted(set(cls.tool_parsers) - before) or sorted(
            n for n in cls.tool_parsers if getattr(cls.tool_parsers[n], "__module__", "").startswith("vmlx_tool_parser_plugin_")
        )

    @classmethod
    def get_tool_parser(cls, name: str) -> type[ToolParser]:
        """
        Retrieve a registered ToolParser class by name.

        Args:
            name: Parser name (e.g., 'mistral', 'qwen', 'llama')

        Returns:
            The ToolParser class

        Raises:
            KeyError: If parser not found
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        raise KeyError(
            f"Tool parser '{name}' not found. "
            f"Available parsers: {cls.list_registered()}"
        )

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ToolParser]:
        """Import and register a lazily loaded parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ToolParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ToolParser subclass."
                )
            cls.tool_parsers[name] = parser_cls
            return parser_cls
        except Exception as e:
            raise ImportError(
                f"Failed to import tool parser '{name}' from {module_path}: {e}"
            ) from e

    @classmethod
    def register_module(
        cls,
        name: str | list[str],
        module: type[ToolParser] | None = None,
        force: bool = True,
    ) -> type[ToolParser] | None:
        """
        Register a ToolParser class.

        Can be used as a decorator or direct call.

        Usage:
            @ToolParserManager.register_module("my_parser")
            class MyToolParser(ToolParser):
                ...

            # Or direct registration:
            ToolParserManager.register_module("my_parser", MyToolParser)
        """
        names = [name] if isinstance(name, str) else name

        if module is not None:
            # Direct registration
            if not issubclass(module, ToolParser):
                raise TypeError(
                    f"module must be subclass of ToolParser, got {type(module)}"
                )
            for n in names:
                if not force and n in cls.tool_parsers:
                    raise KeyError(f"Parser '{n}' is already registered")
                cls.tool_parsers[n] = module
            return module

        # Decorator usage
        def decorator(parser_cls: type[ToolParser]) -> type[ToolParser]:
            for n in names:
                if not force and n in cls.tool_parsers:
                    raise KeyError(f"Parser '{n}' is already registered")
                cls.tool_parsers[n] = parser_cls
            return parser_cls

        return decorator  # type: ignore

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping for deferred loading.

        Args:
            name: Parser name to register
            module_path: Full module path (e.g., 'vmlx_engine.tool_parsers.mistral')
            class_name: Class name within the module
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all registered tool parsers."""
        return sorted(set(cls.tool_parsers.keys()) | set(cls.lazy_parsers.keys()))
