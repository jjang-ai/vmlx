# SPDX-License-Identifier: Apache-2.0
"""
Ollama API compatibility layer for vmlx-engine.

Translates between Ollama wire format and internal vMLX OpenAI format.
Used by server.py to serve /api/chat, /api/generate, /api/tags, /api/show
for CLI users running `vmlx-serve` directly (without the Electron gateway).

Ollama wire format differences:
  - NDJSON streaming (one JSON per line), not SSE (data: prefix)
  - done: true/false instead of finish_reason
  - message.content instead of choices[0].delta.content
  - /api/tags returns {models: [...]} not {data: [...]}
  - Model names use :tag format (e.g., "qwen3.5:latest")
"""

import json
import time
from typing import Any


def ollama_chat_to_openai(body: dict) -> dict:
    """Convert Ollama /api/chat request to OpenAI /v1/chat/completions."""
    opts = body.get("options", {})
    req: dict[str, Any] = {
        "model": body.get("model", "default"),
        "messages": body.get("messages", []),
        "stream": body.get("stream", True),
        # Always request usage so Ollama clients get eval_count/prompt_eval_count
        "stream_options": {"include_usage": True},
    }
    if opts.get("num_predict") is not None:
        req["max_tokens"] = opts["num_predict"]
    if opts.get("temperature") is not None:
        req["temperature"] = opts["temperature"]
    if opts.get("top_p") is not None:
        req["top_p"] = opts["top_p"]
    if opts.get("top_k") is not None:
        req["top_k"] = opts["top_k"]
    if opts.get("stop"):
        req["stop"] = opts["stop"]
    if opts.get("repeat_penalty") is not None:
        req["repetition_penalty"] = opts["repeat_penalty"]
    # Forward tools if present (Ollama tool calling)
    if body.get("tools"):
        req["tools"] = body["tools"]
    # Ollama 0.7+ supports think at top level: {"model": "...", "think": true}
    if body.get("think") is not None:
        req["enable_thinking"] = body["think"]
    return req


def ollama_generate_to_openai(body: dict) -> dict:
    """Convert Ollama /api/generate request to OpenAI /v1/completions."""
    opts = body.get("options", {})
    req: dict[str, Any] = {
        "model": body.get("model", "default"),
        "prompt": body.get("prompt", ""),
        "stream": body.get("stream", True),
    }
    if opts.get("num_predict") is not None:
        req["max_tokens"] = opts["num_predict"]
    if opts.get("temperature") is not None:
        req["temperature"] = opts["temperature"]
    if opts.get("top_p") is not None:
        req["top_p"] = opts["top_p"]
    if opts.get("stop"):
        req["stop"] = opts["stop"]
    return req


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def openai_chat_response_to_ollama(openai_resp: dict, model: str) -> dict:
    """Convert non-streaming OpenAI chat response to Ollama format."""
    choices = openai_resp.get("choices", [])
    # A thinking model may omit `content` entirely if every token was reasoning
    # (rare but observed on Qwen3 with small max_tokens). Use .get() so we
    # don't KeyError — `thinking` mapping below carries the reasoning across.
    content = (choices[0].get("message", {}).get("content") if choices else "") or ""
    usage = openai_resp.get("usage", {})
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    # Ollama 0.3.12+ `thinking` field in message.
    # Without this mapping, thinking models (Qwen3 auto, MiniMax, DeepSeek-R1)
    # produced empty content for Ollama clients because their reasoning was
    # routed to `reasoning_content` which the adapter never forwarded. Copilot
    # and Continue.dev show nothing when the assistant message has empty content.
    if choices:
        _reasoning = (
            choices[0].get("message", {}).get("reasoning_content")
            or choices[0].get("message", {}).get("reasoning")
        )
        if _reasoning:
            msg["thinking"] = _reasoning
    # Forward tool calls if present. mlxstudio#72: Ollama's tool_calls schema
    # expects `arguments` as an object, while OpenAI emits a JSON-encoded
    # string. Parse it so Copilot / Continue.dev / other Ollama clients can
    # consume it directly. Also preserve done_reason="tool_calls" — prior
    # code was collapsing it to "stop", which hid the tool-call signal from
    # clients that gate tool execution on that field.
    if choices:
        oai_tcs = choices[0].get("message", {}).get("tool_calls")
        if oai_tcs:
            _out_tcs: list[dict[str, Any]] = []
            for tc in oai_tcs:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                elif args is None:
                    args = {}
                _out_tcs.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            if _out_tcs:
                msg["tool_calls"] = _out_tcs
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": msg,
        "done": True,
        "done_reason": finish_reason or "stop",
        "total_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }


def openai_chat_chunk_to_ollama_ndjson(sse_line: str, model: str) -> str | None:
    """Convert a single SSE line to Ollama NDJSON line. Returns None to skip."""
    if not sse_line.startswith("data: "):
        return None
    payload = sse_line[6:].strip()
    if payload == "[DONE]":
        # If we already sent a done chunk (from usage-bearing chunk), skip
        # Otherwise emit a minimal done
        return json.dumps({
            "model": model, "created_at": _now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": "stop",
        }) + "\n"
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None

    choices = chunk.get("choices", [])
    usage = chunk.get("usage", {})
    content = ""
    done = False
    done_reason = None

    # Usage-only chunk (choices empty, usage present) — emit as done with metrics
    if not choices and usage:
        return json.dumps({
            "model": model, "created_at": _now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": "stop",
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
        }) + "\n"

    tool_calls_data = None
    thinking_delta = ""
    if choices:
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        # Map delta.reasoning → message.thinking. Without this every streaming
        # chunk of a thinking model produced empty content for Ollama clients
        # (Copilot, Continue.dev) — the model was generating reasoning but the
        # adapter dropped it. Ollama 0.3.12+ wire format uses `thinking`.
        _r = delta.get("reasoning") or delta.get("reasoning_content")
        if _r:
            thinking_delta = _r
        fr = choices[0].get("finish_reason")
        if fr is not None:
            done = True
            done_reason = fr
        # Capture tool calls from delta. mlxstudio#72: parse stringified
        # arguments into objects so Ollama clients (GitHub Copilot, Continue)
        # can consume them. Skip entries with no name — those are OpenAI
        # delta fragments carrying only partial arguments, which our engine
        # shouldn't produce but we guard against anyway.
        oai_tcs = delta.get("tool_calls")
        if oai_tcs:
            _out_tcs: list[dict[str, Any]] = []
            for tc in oai_tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                if not name:
                    continue
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                elif args is None:
                    args = {}
                _out_tcs.append({"function": {"name": name, "arguments": args}})
            if _out_tcs:
                tool_calls_data = _out_tcs

    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking_delta:
        msg["thinking"] = thinking_delta
    if tool_calls_data:
        msg["tool_calls"] = tool_calls_data
    # Skip fully-empty deltas (no content, no thinking, no tool calls, not done).
    # Prior code emitted chunks with content="" for every reasoning token and
    # every heartbeat — Ollama clients (Copilot) handle these as "nothing new"
    # but ollama's own CLI ignores them too, and they inflate NDJSON bandwidth.
    if not done and not content and not thinking_delta and not tool_calls_data:
        return None
    result: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "message": msg,
        "done": done,
    }
    if done:
        # Preserve done_reason="tool_calls" — clients like Copilot gate
        # tool execution on this. Prior code collapsed it to "stop".
        result["done_reason"] = done_reason or "stop"
        usage = chunk.get("usage", {})
        if usage:
            result["eval_count"] = usage.get("completion_tokens", 0)
            result["prompt_eval_count"] = usage.get("prompt_tokens", 0)
    return json.dumps(result) + "\n"


def openai_completion_chunk_to_ollama_ndjson(sse_line: str, model: str) -> str | None:
    """Convert a single SSE line from /v1/completions to Ollama /api/generate NDJSON."""
    if not sse_line.startswith("data: "):
        return None
    payload = sse_line[6:].strip()
    if payload == "[DONE]":
        return json.dumps({
            "model": model, "created_at": _now_iso(),
            "response": "", "done": True, "done_reason": "stop",
        }) + "\n"
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None

    choices = chunk.get("choices", [])
    text = ""
    done = False
    done_reason = None

    if choices:
        text = choices[0].get("text", "")
        fr = choices[0].get("finish_reason")
        if fr is not None:
            done = True
            done_reason = fr

    result: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "response": text,
        "done": done,
    }
    if done:
        result["done_reason"] = done_reason or "stop"
        usage = chunk.get("usage", {})
        if usage:
            result["eval_count"] = usage.get("completion_tokens", 0)
            result["prompt_eval_count"] = usage.get("prompt_tokens", 0)
    return json.dumps(result) + "\n"


def build_tags_response(model_name: str, model_path: str) -> dict:
    """Build Ollama /api/tags response."""
    return {
        "models": [{
            "name": model_name,
            "model": model_path,
            "modified_at": _now_iso(),
            "size": 0,
            "digest": "",
            "details": {
                "format": "mlx",
                "family": "",
                "parameter_size": "",
                "quantization_level": "",
            },
        }]
    }
