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

from pydantic import BaseModel, Field

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
    """Anthropic Messages API request."""
    model: str
    messages: list[dict]
    system: str | list[dict] | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[AnthropicToolInput | dict] | None = None
    tool_choice: dict | None = None
    thinking: AnthropicThinking | dict | None = None
    metadata: dict | None = None


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

    # Thinking/reasoning
    enable_thinking = None
    chat_template_kwargs = None
    if req.thinking:
        thinking = req.thinking if isinstance(req.thinking, dict) else req.thinking.model_dump()
        if thinking.get("type") == "enabled":
            enable_thinking = True
            # Forward budget_tokens as thinking_budget for Qwen3 models
            if thinking.get("budget_tokens"):
                chat_template_kwargs = {"thinking_budget": thinking["budget_tokens"]}
        elif thinking.get("type") == "disabled":
            enable_thinking = False

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop_sequences,
        stream=req.stream,
        stream_options=StreamOptions(include_usage=True) if req.stream else None,
        tools=tools,
        tool_choice=tool_choice,
        enable_thinking=enable_thinking,
        chat_template_kwargs=chat_template_kwargs,
    )


def _convert_assistant_message(msg: dict) -> Message:
    """Convert Anthropic assistant message to Chat Completions format."""
    content = msg.get("content")

    if isinstance(content, str):
        return Message(role="assistant", content=content)

    if isinstance(content, list):
        text_parts = []
        tool_calls = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "text")

            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                # Thinking blocks from previous turns are dropped — local models
                # don't need prior reasoning context (matches Anthropic guidance)
                pass
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        return Message(
            role="assistant",
            content="\n".join(text_parts) if text_parts else None,
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
        has_images = False

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
                    has_images = True
                elif source.get("type") == "url":
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": source.get("url", "")},
                    })
                    has_images = True

        if result_messages:
            # Return all tool result messages as a list
            # Caller (to_chat_completion) flattens these into the messages array
            if content_parts:
                # Text/images + tool results: prepend content as user message
                user_content: Any = content_parts if has_images else "\n".join(
                    p["text"] for p in content_parts if p["type"] == "text"
                )
                return [Message(role="user", content=user_content)] + result_messages
            return result_messages if len(result_messages) > 1 else result_messages[0]

        # No tool results — return as user message
        if has_images:
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
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


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
        self._tool_blocks: list[dict] = []
        self._tool_block_open = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._finish_reason: str | None = None

    def _sse(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"

    def process_chunk(self, chunk_line: str) -> list[str]:
        """Process a single SSE line from Chat Completions stream.

        Returns list of Anthropic SSE event strings.
        """
        events = []

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

        # Handle reasoning/thinking content
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if not self._thinking_block_open:
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
            if self._thinking_block_open and not self._text_block_open:
                events.append(self._sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._content_index,
                }))
                self._content_index += 1
                self._thinking_block_open = False

            if self._tool_block_open and not self._text_block_open:
                events.append(self._sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._content_index,
                }))
                self._content_index += 1
                self._tool_block_open = False

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

            # Start new tool block
            if tc.get("id"):
                # Close any open blocks before starting tool block
                if self._thinking_block_open:
                    events.append(self._sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._content_index,
                    }))
                    self._content_index += 1
                    self._thinking_block_open = False

                if self._tool_block_open:
                    events.append(self._sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._content_index,
                    }))
                    self._content_index += 1
                    self._tool_block_open = False

                if self._text_block_open:
                    events.append(self._sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._content_index,
                    }))
                    self._content_index += 1
                    self._text_block_open = False

                events.append(self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._content_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc.get("function", {}).get("name", ""),
                        "input": {},
                    },
                }))
                self._tool_block_open = True

            # Stream tool arguments
            args_delta = tc.get("function", {}).get("arguments", "")
            if args_delta and self._tool_block_open:
                events.append(self._sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self._content_index,
                    "delta": {"type": "input_json_delta", "partial_json": args_delta},
                }))

        # Handle finish
        if finish_reason:
            pass  # Finalize handles closing blocks

        return events

    def finalize(self) -> list[str]:
        """Generate closing events for the stream."""
        # Guard: if stream errored before any data was emitted, don't send
        # orphaned message_delta/message_stop without a preceding message_start
        if not self._started:
            return []
        events = []

        # Determine stop reason from Chat Completions finish_reason
        if self._tool_block_open:
            stop_reason = "tool_use"
        elif self._finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        # Close any open blocks
        if self._thinking_block_open:
            events.append(self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self._content_index,
            }))
            self._content_index += 1
            self._thinking_block_open = False

        if self._text_block_open:
            events.append(self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self._content_index,
            }))
            self._content_index += 1
            self._text_block_open = False

        if self._tool_block_open:
            events.append(self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self._content_index,
            }))
            self._content_index += 1
            self._tool_block_open = False

        # message_delta with final usage (include input_tokens since message_start
        # emits 0 — prompt tokens aren't known until the final streaming chunk)
        usage = {"output_tokens": self._output_tokens}
        if self._input_tokens > 0:
            usage["input_tokens"] = self._input_tokens
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
