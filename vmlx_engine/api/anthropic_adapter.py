# SPDX-License-Identifier: Apache-2.0
"""
Anthropic Messages API adapter.

Converts Anthropic /v1/messages wire format to/from the internal Chat
Completions pipeline, enabling Claude Code and other Anthropic SDK clients
to use vMLX as a local inference backend.

Request flow:
    POST /v1/messages
    → AnthropicRequest (Pydantic validation)
    → to_chat_completion() → ChatCompletionRequest
    → existing stream_chat_completion() / non-stream path
    → AnthropicStreamAdapter / to_anthropic_response()
    → Anthropic SSE events / JSON response

Supported Anthropic features:
- Content blocks (text, tool_use, tool_result, thinking)
- System prompt (top-level field)
- Tool definitions and tool calling round-trips
- Streaming with proper SSE event types
- Thinking/extended thinking blocks
- Usage reporting
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, field_validator, model_validator

from ..video_controls import VIDEO_CONTROL_FIELDS, validate_video_controls
from ..image_controls import IMAGE_CONTROL_FIELDS, validate_image_controls

from .models import (
    ChatCompletionRequest,
    Message,
    StreamOptions,
    ToolDefinition,
)


# ─── Anthropic Request Models ──────────────────────────────────────────


class AnthropicThinking(BaseModel):
    type: str = "enabled"  # "enabled" or "disabled"
    budget_tokens: int | None = None


class AnthropicToolInput(BaseModel):
    """Anthropic tool definition format."""
    name: str
    description: str | None = None
    input_schema: dict = Field(default_factory=dict)


class AnthropicRequest(BaseModel):
    # Non-Anthropic passthrough: some vMLX clients send chat_template_kwargs
    # even on /v1/messages so they can pick reasoning on/off without the
    # Anthropic-native `thinking: {type: "enabled"}` schema. Accept it and
    # fold into the internal ChatCompletionRequest below.
    """Anthropic Messages API request."""
    model: str
    messages: list[dict]
    system: str | list[dict] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[AnthropicToolInput | dict] | None = None
    tool_choice: dict | None = None
    thinking: AnthropicThinking | dict | None = None
    metadata: dict | None = None
    # vMLX extension: per-request prompt/context admission cap. The engine
    # treats this as a request-local cap and will not let it exceed the
    # server/session --max-prompt-tokens ceiling.
    max_prompt_tokens: int | None = None
    max_context_tokens: int | None = None
    max_context: int | None = None
    # Non-Anthropic passthrough for chat_template_kwargs (vMLX extension).
    # Clients targeting /v1/messages but unaware of Anthropic's
    # `thinking: {type}` schema can still forward enable_thinking, etc.
    chat_template_kwargs: dict | None = None
    # Non-Anthropic passthrough for explicit enable_thinking bool.
    enable_thinking: bool | None = None
    # Non-Anthropic passthrough for top-level reasoning_effort (vMLX
    # extension). The chat, responses, and ollama dialects all accept it
    # top-level; without this field pydantic silently DROPPED it here and
    # /v1/messages resolved effort as None while the other three routes
    # resolved the requested tier (measured live: qwen3.8 low/medium/xhigh
    # all divergent on exactly this field).
    reasoning_effort: str | None = None
    seed: int | None = None
    # Non-Anthropic passthrough, same class of drift as reasoning_effort above.
    # server.py already resolves all four off the converted ChatCompletionRequest
    # (_set_resolved_min_p, _resolve_repetition_penalty,
    # _compute_bypass_prefix_cache) — they were simply never ACCEPTED here, so
    # pydantic dropped them and /v1/messages silently ignored sampling and
    # cache controls that the chat, responses and ollama dialects all honour.
    min_p: float | None = None
    repetition_penalty: float | None = None
    cache_salt: str | None = None
    skip_prefix_cache: bool | None = None
    # vMLX extension: per-request video preprocessing controls, same names
    # and validation as the chat/responses dialects.
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_max_pixels: int | None = None
    video_min_pixels: int | None = None
    video_total_pixels: int | None = None
    video_resized_height: int | None = None
    video_resized_width: int | None = None
    video_token_budget: int | None = None
    image_max_pixels: int | None = None
    image_min_pixels: int | None = None
    image_resized_height: int | None = None
    image_resized_width: int | None = None
    media_controls_strict: bool | None = None

    @model_validator(mode="after")
    def validate_video_controls(self):
        validate_image_controls({f: getattr(self, f, None) for f in IMAGE_CONTROL_FIELDS})
        validate_video_controls({f: getattr(self, f, None) for f in VIDEO_CONTROL_FIELDS})
        return self

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_tokens must be at least 1")
        return value


# ─── Request Conversion ────────────────────────────────────────────────


def to_chat_completion(req: AnthropicRequest) -> ChatCompletionRequest:
    """Convert Anthropic Messages request to Chat Completions request."""
    messages = []

    # System message (Anthropic puts it top-level)
    if req.system:
        if isinstance(req.system, str):
            messages.append(Message(role="system", content=req.system))
        elif isinstance(req.system, list):
            # List of content blocks — extract text
            text_parts = []
            for block in req.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                messages.append(Message(role="system", content="\n".join(text_parts)))

    # Convert message history
    for msg in req.messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "assistant":
            messages.append(_convert_assistant_message(msg))
        elif role == "user":
            result = _convert_user_message(msg)
            if isinstance(result, list):
                messages.extend(result)
            else:
                messages.append(result)
        else:
            # Pass through unknown roles
            messages.append(Message(role=role, content=content if isinstance(content, str) else str(content)))

    # Convert tools
    tools = None
    if req.tools:
        tools = []
        for tool in req.tools:
            if isinstance(tool, dict):
                tools.append(ToolDefinition(
                    type="function",
                    function={
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                ))
            else:
                tools.append(ToolDefinition(
                    type="function",
                    function={
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                ))

    # Tool choice mapping
    tool_choice = None
    if req.tool_choice:
        tc_type = req.tool_choice.get("type", "auto")
        if tc_type == "any":
            tool_choice = "required"
        elif tc_type == "auto":
            tool_choice = "auto"
        elif tc_type == "none":
            tool_choice = "none"
        elif tc_type == "tool":
            tool_choice = {"type": "function", "function": {"name": req.tool_choice.get("name", "")}}

    # Thinking/reasoning — three sources, precedence order:
    #   1. req.enable_thinking (vMLX extension, explicit bool)
    #   2. req.thinking (Anthropic-native {type: enabled/disabled})
    #   3. req.chat_template_kwargs.enable_thinking (vMLX extension fallback)
    #
    # Omitted thinking controls stay Auto. The shared server resolver leaves
    # enable_thinking unset so the model's native tokenizer/template/runtime
    # default decides; explicit Anthropic/vMLX controls still opt in/out.
    enable_thinking: bool | None = None
    chat_template_kwargs = None
    max_thinking_tokens: int | None = None
    _thinking_source_seen = False
    # Start with the client's chat_template_kwargs passthrough (lowest prio)
    if req.chat_template_kwargs:
        chat_template_kwargs = dict(req.chat_template_kwargs)
        if "enable_thinking" in chat_template_kwargs:
            v = chat_template_kwargs.get("enable_thinking")
            if isinstance(v, bool):
                enable_thinking = v
                _thinking_source_seen = True
    # Anthropic-native thinking field (mid prio)
    if req.thinking:
        thinking = req.thinking if isinstance(req.thinking, dict) else req.thinking.model_dump()
        if thinking.get("type") == "enabled":
            enable_thinking = True
            _thinking_source_seen = True
            # Forward budget_tokens as thinking_budget for Qwen3 models.
            # For DSV4 (research/DSV4-RUNTIME-ARCHITECTURE.md §4): a large
            # budget_tokens threshold selects "max" mode over plain thinking
            # (DSV4 uses a discrete "max" tier rather than a token budget).
            # Threshold ≥ 32768 matches our _EFFORT_THINKING_BUDGET["high"]
            # ceiling so clients that explicitly request >32k chains land
            # on the deeper reasoning path.
            if thinking.get("budget_tokens"):
                if chat_template_kwargs is None:
                    chat_template_kwargs = {}
                chat_template_kwargs["thinking_budget"] = thinking["budget_tokens"]
                # Arm the actual reasoning cap. The template kwarg above only
                # informs models that read `thinking_budget`; the runtime clamp
                # lives on `max_thinking_tokens`, exactly as the OpenAI Responses
                # path maps `reasoning.budget_tokens` (models.py). Without this
                # the Anthropic budget capped nothing.
                if isinstance(thinking["budget_tokens"], int):
                    max_thinking_tokens = thinking["budget_tokens"]
                if thinking["budget_tokens"] >= 32768:
                    chat_template_kwargs.setdefault("reasoning_effort", "max")
        elif thinking.get("type") == "disabled":
            enable_thinking = False
            _thinking_source_seen = True
    # Explicit enable_thinking (highest prio)
    if req.enable_thinking is not None:
        enable_thinking = req.enable_thinking
        _thinking_source_seen = True
    # If client asserted any thinking intent, honour it. Otherwise preserve
    # Auto and let the shared server/runtime path decide.
    _ = _thinking_source_seen  # (retained for debuggability / future logging)

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_prompt_tokens=(
            req.max_prompt_tokens
            if req.max_prompt_tokens is not None
            else req.max_context_tokens
            if req.max_context_tokens is not None
            else req.max_context
        ),
        stop=req.stop_sequences,
        stream=req.stream,
        # ms#79: always request usage in chunks. The non-streaming Anthropic
        # path at server.py internally calls stream_chat_completion() and
        # accumulates chunks — without include_usage=True here, the inner
        # chunks never emit usage and the final /v1/messages response
        # returns `{input_tokens: 0, output_tokens: 0}`. Claude Code uses
        # these counts for rate-limit accounting and progress display, so
        # zeroed usage looks like a broken request.
        stream_options=StreamOptions(include_usage=True),
        tools=tools,
        tool_choice=tool_choice,
        enable_thinking=enable_thinking,
        max_thinking_tokens=max_thinking_tokens,
        seed=req.seed,
        min_p=req.min_p,
        repetition_penalty=req.repetition_penalty,
        cache_salt=req.cache_salt,
        skip_prefix_cache=req.skip_prefix_cache,
        chat_template_kwargs=chat_template_kwargs,
        # Forward reasoning_effort: explicit top-level field first (parity
        # with the chat/responses/ollama dialects), then the ct_kwargs copy
        # (DSV4 "max" gets set via thinking.budget_tokens≥32768 above).
        # Keeps the Anthropic → OpenAI conversion honest with the shared
        # OpenAI-path auto-mapping block at server.py:5186 / :3122.
        reasoning_effort=(
            req.reasoning_effort
            if req.reasoning_effort is not None
            else (chat_template_kwargs or {}).get("reasoning_effort")
        ),
        **{f: getattr(req, f) for f in VIDEO_CONTROL_FIELDS},
        **{f: getattr(req, f) for f in IMAGE_CONTROL_FIELDS},
        media_controls_strict=req.media_controls_strict,
    )


def _convert_assistant_message(msg: dict) -> Message:
    """Convert Anthropic assistant message to Chat Completions format."""
    content = msg.get("content")

    if isinstance(content, str):
        return Message(role="assistant", content=content)

    if isinstance(content, list):
        text_parts = []
        thinking_parts = []
        tool_calls = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "text")

            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                # Anthropic requires prior assistant thinking blocks to be sent
                # back during tool-result continuation.  Preserve the plain-text
                # reasoning privately; never concatenate it into visible content.
                _thinking = block.get("thinking", "")
                if isinstance(_thinking, str) and _thinking:
                    thinking_parts.append(_thinking)
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        # When the assistant message ONLY has tool_calls (no text), some chat
        # templates (Qwen3, others that do `{{ message.content }}` directly)
        # fail with UndefinedError because exclude_none drops the content key
        # entirely. Emit empty string instead of None so the template sees a
        # defined content attribute. Zero downstream cost — tool-call-only
        # assistant messages still get their tool_calls rendered correctly.
        _content = "\n".join(text_parts) if text_parts else ("" if tool_calls else None)
        return Message(
            role="assistant",
            content=_content,
            reasoning_content="\n".join(thinking_parts) if thinking_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )

    return Message(role="assistant", content=content)


def _convert_user_message(msg: dict) -> Message | list[Message]:
    """Convert Anthropic user message to Chat Completions format.

    Returns a single Message or a list of Messages (when tool_result blocks
    are present — each becomes a separate tool response message).
    """
    content = msg.get("content")

    if isinstance(content, str):
        return Message(role="user", content=content)

    if isinstance(content, list):
        # Check for tool_result blocks — these become tool response messages
        # Anthropic puts tool results in user messages; OpenAI uses role="tool"
        result_messages: list[Message] = []
        content_parts: list[dict] = []
        has_media = False

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "text")

            if block_type == "tool_result":
                # Convert to tool response message
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Extract text from content blocks
                    result_content = "\n".join(
                        b.get("text", "") for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                result_messages.append(Message(
                    role="tool",
                    content=str(result_content),
                    tool_call_id=block.get("tool_use_id", ""),
                ))
            elif block_type == "text":
                content_parts.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                # Convert Anthropic image to OpenAI image_url format for VLM support
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                    has_media = True
                elif source.get("type") == "url":
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": source.get("url", "")},
                    })
                    has_media = True
            elif block_type in {"video", "video_url"}:
                # vMLX extension: Anthropic has no native video block today,
                # but local multimodal clients use the same source envelope as
                # image blocks. Preserve it for the shared Chat media path.
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "video/mp4")
                    data = source.get("data", "")
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                    has_media = True
                elif source.get("type") == "url":
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {"url": source.get("url", "")},
                    })
                    has_media = True
            elif block_type in {"audio", "input_audio"}:
                # vMLX extension matching the image/video source envelope.
                # Base64 audio uses the OpenAI input_audio shape so native
                # audio towers receive data plus the declared format.
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = str(source.get("media_type", "audio/wav"))
                    audio_format = media_type.split("/", 1)[-1].lower()
                    audio_format = {"mpeg": "mp3", "x-wav": "wav"}.get(
                        audio_format, audio_format
                    )
                    content_parts.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": source.get("data", ""),
                            "format": audio_format,
                        },
                    })
                    has_media = True
                elif source.get("type") == "url":
                    content_parts.append({
                        "type": "audio_url",
                        "audio_url": {"url": source.get("url", "")},
                    })
                    has_media = True

        if result_messages:
            # OpenAI-compatible history requires every tool result to remain
            # adjacent to the assistant tool call it answers.  Anthropic permits
            # a tool_result and follow-up text in the same user content array;
            # flatten that shape as tool result(s) FIRST, then the new user
            # instruction.  Reversing those messages breaks tool-call adjacency
            # and also makes exact/required selection gates mistake the previous
            # result for one belonging to the new instruction.
            if content_parts:
                # Text/media + tool results: append the new user instruction
                # after every result from the prior assistant tool call.
                user_content: Any = content_parts if has_media else "\n".join(
                    p["text"] for p in content_parts if p["type"] == "text"
                )
                return result_messages + [Message(role="user", content=user_content)]
            return result_messages if len(result_messages) > 1 else result_messages[0]

        # No tool results — return as user message
        if has_media:
            return Message(role="user", content=content_parts)
        text = "\n".join(p["text"] for p in content_parts if p["type"] == "text")
        return Message(role="user", content=text if text else "")

    return Message(role="user", content=str(content) if content else "")


# ─── Response Conversion ───────────────────────────────────────────────


def to_anthropic_response(
    chat_response: dict,
    model: str,
    request_id: str | None = None,
) -> dict:
    """Convert Chat Completions response to Anthropic Messages response."""
    msg_id = request_id or f"msg_{uuid.uuid4().hex[:12]}"

    content = []
    stop_reason = "end_turn"

    choices = chat_response.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")

        # Map finish reason
        if finish == "tool_calls":
            stop_reason = "tool_use"
        elif finish == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        # Reasoning/thinking
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            content.append({
                "type": "thinking",
                "thinking": reasoning,
            })

        # Text content
        text = message.get("content")
        if text:
            content.append({
                "type": "text",
                "text": text,
            })

        # Tool calls
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed tool arguments for {func.get('name', '?')}: {e}")
                input_data = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
                "input": input_data,
            })

    # Usage
    usage = chat_response.get("usage", {})

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content if content else [{"type": "text", "text": ""}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage(usage),
    }


def _cached_prompt_tokens(usage: dict) -> int:
    """Return prefix-cache hits from a Chat Completions usage block."""
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return 0
    cached = details.get("cached_tokens")
    return cached if isinstance(cached, int) and cached > 0 else 0


def _anthropic_usage(usage: dict) -> dict:
    """Map Chat Completions usage onto Anthropic's usage shape.

    Anthropic clients read ``cache_read_input_tokens`` to show how much of the
    prompt was served from cache. Without it a fully reused prefix looks like a
    cold request, so the surface under-reports its own cache. vMLX has no
    separate cache-creation billing step, so ``cache_creation_input_tokens`` is
    reported as 0 whenever any cache read occurred.
    """
    out = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    cached = _cached_prompt_tokens(usage)
    if cached:
        out["cache_read_input_tokens"] = cached
        out["cache_creation_input_tokens"] = 0
    return out


# ─── Streaming Adapter ─────────────────────────────────────────────────


class AnthropicStreamAdapter:
    """Converts Chat Completions SSE chunks to Anthropic SSE events.

    Usage:
        adapter = AnthropicStreamAdapter(model, request_id)
        async for chunk_line in chat_completion_stream:
            for event in adapter.process_chunk(chunk_line):
                yield event
        for event in adapter.finalize():
            yield event
    """

    def __init__(self, model: str, request_id: str | None = None):
        self.model = model
        self.msg_id = request_id or f"msg_{uuid.uuid4().hex[:12]}"
        self._content_index = 0
        self._started = False
        self._thinking_block_open = False
        self._text_block_open = False
        # Chat Completions may split one function call across two deltas:
        # first the id with an empty function name, then the name+arguments
        # without the id. Anthropic requires the name in content_block_start,
        # so retain incomplete metadata until both fields are available.
        self._pending_tool_calls: dict[int, dict[str, str]] = {}
        self._tool_block_open = False
        self._active_tool_index: int | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_tokens = 0
        self._finish_reason: str | None = None
        self._errored = False
        self._finalized = False

    def _sse(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"

    # Placeholder base64 signature so strict Anthropic clients (Claude Code,
    # Anthropic SDK) accept thinking blocks on replay. vMLX does not verify
    # signatures on input, so a constant placeholder round-trips safely.
    _THINKING_SIGNATURE = "dm1seA=="

    def _close_thinking(self, events: list[str]) -> None:
        """Close an open thinking block, emitting the required signature_delta first."""
        if not self._thinking_block_open:
            return
        events.append(self._sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self._content_index,
            "delta": {"type": "signature_delta", "signature": self._THINKING_SIGNATURE},
        }))
        events.append(self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self._content_index,
        }))
        self._content_index += 1
        self._thinking_block_open = False

    def _close_text(self, events: list[str]) -> None:
        """Close the current text block and advance to the next block index."""
        if not self._text_block_open:
            return
        events.append(self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self._content_index,
        }))
        self._content_index += 1
        self._text_block_open = False

    def _close_tool(self, events: list[str]) -> None:
        """Close the current tool block and advance to the next block index."""
        if not self._tool_block_open:
            return
        events.append(self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self._content_index,
        }))
        self._content_index += 1
        self._tool_block_open = False
        self._active_tool_index = None

    def _close_open_blocks(self, events: list[str]) -> None:
        """Close the one active Anthropic content block before a rail transition."""
        self._close_thinking(events)
        self._close_text(events)
        self._close_tool(events)

    def process_chunk(self, chunk_line: str) -> list[str]:
        """Process a single SSE line from Chat Completions stream.

        Returns list of Anthropic SSE event strings.
        """
        events = []

        # A terminal/error event owns the remainder of the stream. Ignore any
        # trailing upstream data so Anthropic clients never see post-terminal
        # blocks or a second terminal sequence.
        if self._finalized or self._errored:
            return events

        # Skip non-data lines and keep-alive comments
        if not chunk_line.startswith("data: "):
            return events

        data_str = chunk_line[6:].strip()
        if data_str == "[DONE]":
            return events

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            return events

        # Bug 5 relay: upstream Chat Completions sometimes emits a chunk shaped
        # {choices:[], warnings:["..."]} carrying engine diagnostics (dropped
        # tool call, reasoning-only truncation, etc.). Anthropic has no native
        # warnings field, so surface the diagnostic as a text_delta in a fresh
        # text content block so the user sees the explanation instead of an
        # empty response. Must come BEFORE the regular choices-extraction
        # branch since a warnings chunk has choices=[].
        if isinstance(chunk, dict) and chunk.get("warnings") and not chunk.get("choices"):
            warnings_list = chunk["warnings"]
            if isinstance(warnings_list, list) and warnings_list:
                notice_text = "\n\n[vMLX notice] " + "; ".join(
                    str(w) for w in warnings_list if w
                )
                # Ensure message_start fired first so the block index sequence is valid.
                if not self._started:
                    self._started = True
                    events.append(self._sse("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": self.msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    }))
                # Close the active block before opening the notice block.
                self._close_open_blocks(events)
                # Open a fresh text block for the notice + emit + close.
                events.append(self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._content_index,
                    "content_block": {"type": "text", "text": ""},
                }))
                events.append(self._sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self._content_index,
                    "delta": {"type": "text_delta", "text": notice_text},
                }))
                events.append(self._sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._content_index,
                }))
                self._content_index += 1
            return events

        # Surface mid-stream engine errors as an Anthropic error event instead
        # of silently ending the stream (harnesses cannot recover from silent EOF).
        if isinstance(chunk, dict) and chunk.get("error"):
            err = chunk["error"]
            ecode = None
            if isinstance(err, dict):
                etype = err.get("type", "api_error")
                emsg = err.get("message", str(err))
                ecode = err.get("code")
            else:
                etype, emsg = "api_error", str(err)
            events.append(self._sse("error", {
                "type": "error",
                "error": {"type": etype, "message": emsg, **({"code": ecode} if ecode else {})},
            }))
            self._errored = True
            return events

        # Emit message_start on first chunk
        if not self._started:
            self._started = True
            events.append(self._sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }))

        # Extract delta from choices
        choices = chunk.get("choices", [])
        if not choices:
            # Usage-only chunk
            usage = chunk.get("usage")
            if usage:
                self._input_tokens = usage.get("prompt_tokens", self._input_tokens)
                self._output_tokens = usage.get("completion_tokens", self._output_tokens)
                self._cached_tokens = (
                    _cached_prompt_tokens(usage) or self._cached_tokens
                )
            return events

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        # Store finish_reason for use in finalize() stop_reason mapping
        if finish_reason:
            self._finish_reason = finish_reason

        # Track usage from streaming chunks
        usage = chunk.get("usage")
        if usage:
            self._input_tokens = usage.get("prompt_tokens", self._input_tokens)
            self._output_tokens = usage.get("completion_tokens", self._output_tokens)
            self._cached_tokens = _cached_prompt_tokens(usage) or self._cached_tokens

        # Handle reasoning/thinking content
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if not self._thinking_block_open:
                # Some families (including Gemma4) may emit a late thought
                # after visible text. Anthropic permits sequential content
                # blocks, not overlapping blocks or index reuse.
                self._close_text(events)
                self._close_tool(events)
                self._thinking_block_open = True
                events.append(self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._content_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }))
            events.append(self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._content_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            }))

        # Handle text content
        text = delta.get("content")
        if text:
            # Close any open non-text blocks when transitioning to text
            if not self._text_block_open:
                self._close_thinking(events)
                self._close_tool(events)

            if not self._text_block_open:
                self._text_block_open = True
                events.append(self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._content_index,
                    "content_block": {"type": "text", "text": ""},
                }))
            events.append(self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._content_index,
                "delta": {"type": "text_delta", "text": text},
            }))

        # Handle tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            function = tc.get("function", {}) or {}
            args_delta = function.get("arguments", "") or ""

            # Once a block is open, later argument fragments for that same
            # call can be forwarded immediately.
            if self._tool_block_open and self._active_tool_index == tc_index:
                if args_delta:
                    events.append(self._sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args_delta,
                        },
                    }))
                continue

            pending = self._pending_tool_calls.setdefault(
                tc_index,
                {"id": "", "name": "", "arguments": ""},
            )
            if tc.get("id"):
                pending["id"] = str(tc["id"])
            name_delta = function.get("name", "") or ""
            if name_delta:
                # vMLX emits the complete name in the later delta. Preserve
                # normal OpenAI split-name streams too, without duplicating an
                # identical repeated name.
                if not pending["name"]:
                    pending["name"] = str(name_delta)
                elif pending["name"] != str(name_delta):
                    pending["name"] += str(name_delta)
            if args_delta:
                pending["arguments"] += str(args_delta)

            # Anthropic has no name-delta event. Do not open a malformed
            # tool_use block until both the id and non-empty name are known.
            if pending["id"] and pending["name"]:
                # Close any open blocks before starting tool block
                self._close_open_blocks(events)

                events.append(self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": pending["id"],
                        "name": pending["name"],
                        "input": {},
                    },
                }))
                self._tool_block_open = True
                self._active_tool_index = tc_index
                if pending["arguments"]:
                    events.append(self._sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": pending["arguments"],
                        },
                    }))
                self._pending_tool_calls.pop(tc_index, None)

        # Handle finish
        if finish_reason:
            pass  # Finalize handles closing blocks

        return events

    def finalize(self) -> list[str]:
        """Generate closing events for the stream."""
        if self._finalized:
            return []
        # Guard: if stream errored before any data was emitted, don't send
        # orphaned message_delta/message_stop without a preceding message_start
        if not self._started:
            return []
        self._finalized = True
        events = []

        # If an error event was already emitted, end without a normal terminal.
        if self._errored:
            return events

        # Determine stop reason from Chat Completions finish_reason. Tool calls
        # must map to tool_use even if the tool block was already closed.
        if self._finish_reason == "tool_calls" or self._tool_block_open:
            stop_reason = "tool_use"
        elif self._finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        # Close the one active block before the message terminal sequence.
        self._close_open_blocks(events)

        # message_delta with final usage (include input_tokens since message_start
        # emits 0 — prompt tokens aren't known until the final streaming chunk)
        usage = {"output_tokens": self._output_tokens}
        if self._input_tokens > 0:
            usage["input_tokens"] = self._input_tokens
        if self._cached_tokens > 0:
            # Anthropic clients read this to show how much of the prompt was
            # served from cache; omitting it makes a reused prefix look cold.
            usage["cache_read_input_tokens"] = self._cached_tokens
            usage["cache_creation_input_tokens"] = 0
        events.append(self._sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        }))

        # message_stop
        events.append(self._sse("message_stop", {
            "type": "message_stop",
        }))

        return events
