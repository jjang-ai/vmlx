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

    # Ollama convention for VL models: each message may have an
    # `images: [<base64>, ...]` field alongside `content: <string>`.
    # The OpenAI multimodal schema instead embeds images as inline
    # content parts. Translate so VL models (Qwen VL, Gemma 4 VL, etc.)
    # see the image through the normal /v1/chat/completions path.
    #
    # Without this translation `prompt_eval_count` shows only the text
    # tokens, the model reports "I cannot see the image", and the
    # reporter has no indication of why. Surfaced during live VL test
    # against Qwen3.5-VL-4B-JANG_4S-CRACK.
    src_messages = body.get("messages", [])
    translated_messages = []
    for msg in src_messages:
        images = msg.get("images") if isinstance(msg, dict) else None
        if not images:
            translated_messages.append(msg)
            continue
        text = msg.get("content", "") or ""
        parts: list[dict] = []
        if text:
            parts.append({"type": "text", "text": text})
        for img in images:
            if not isinstance(img, str):
                continue
            # Ollama accepts either raw base64 or a data URL — normalize
            # to data URL so the OpenAI content_part handler (which
            # inspects the dataUrl mime prefix) can decode.
            url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        new_msg = {k: v for k, v in msg.items() if k != "images" and k != "content"}
        new_msg["role"] = msg.get("role", "user")
        new_msg["content"] = parts
        translated_messages.append(new_msg)

    req: dict[str, Any] = {
        "model": body.get("model", "default"),
        "messages": translated_messages,
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
    # vMLX also accepts enable_thinking as an extension for clients that share
    # request builders across OpenAI and Ollama surfaces. `think` wins when both
    # are present because it is the native Ollama field.
    if body.get("think") is not None:
        req["enable_thinking"] = body["think"]
    elif body.get("enable_thinking") is not None:
        req["enable_thinking"] = body["enable_thinking"]
    # vMLX extensions on Ollama-shaped bodies: clients that set reasoning_effort
    # (Mistral 4 / GPT-OSS: "none"/"low"/"medium"/"high") or supply custom
    # chat_template_kwargs must reach the parser. Without this passthrough,
    # Mistral 4 on the Ollama adapter loses reasoning-effort level entirely.
    if body.get("reasoning_effort") is not None:
        req["reasoning_effort"] = body["reasoning_effort"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        req["chat_template_kwargs"] = body["chat_template_kwargs"]
    # Ollama's `format` field → OpenAI `response_format`. Two shapes:
    #   "format": "json"     → {"type": "json_object"}
    #   "format": <schema>   → {"type": "json_schema", "json_schema": {...}}
    # Without this translation the model emits ```json ``` fences around
    # its output and Ollama clients that parse .message.content as JSON
    # blow up.
    _fmt = body.get("format")
    if _fmt == "json":
        req["response_format"] = {"type": "json_object"}
    elif isinstance(_fmt, dict):
        req["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ollama_schema",
                "strict": False,
                "schema": _fmt,
            },
        }
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
    # /api/generate also forwards format=json → response_format
    _fmt = body.get("format")
    if _fmt == "json":
        req["response_format"] = {"type": "json_object"}
    elif isinstance(_fmt, dict):
        req["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ollama_schema", "strict": False, "schema": _fmt},
        }
    return req


def ollama_generate_to_openai_chat(body: dict) -> dict:
    """Convert Ollama /api/generate to chat-completions for templated models.

    Ollama's generate endpoint applies the model template by default. `raw:
    true` is the opt-out. vMLX previously always routed /api/generate through
    raw completions, which breaks instruction-tuned/chat-template families.
    """
    opts = body.get("options", {})
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": body.get("prompt", "") or ""})

    req: dict[str, Any] = {
        "model": body.get("model", "default"),
        "messages": messages,
        "stream": body.get("stream", True),
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
    if body.get("think") is not None:
        req["enable_thinking"] = body["think"]
    elif body.get("enable_thinking") is not None:
        req["enable_thinking"] = body["enable_thinking"]
    if body.get("reasoning_effort") is not None:
        req["reasoning_effort"] = body["reasoning_effort"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        req["chat_template_kwargs"] = body["chat_template_kwargs"]

    _fmt = body.get("format")
    if _fmt == "json":
        req["response_format"] = {"type": "json_object"}
    elif isinstance(_fmt, dict):
        req["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ollama_schema",
                "strict": False,
                "schema": _fmt,
            },
        }
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


def openai_chat_response_to_ollama_generate(openai_resp: dict, model: str) -> dict:
    """Convert non-streaming chat-completions response to /api/generate shape."""
    choices = openai_resp.get("choices", [])
    usage = openai_resp.get("usage", {})
    msg = choices[0].get("message", {}) if choices else {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
    result: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "response": content,
        "done": True,
        "done_reason": finish_reason or "stop",
        "total_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }
    if reasoning:
        result["thinking"] = reasoning
    return result


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


def openai_chat_chunk_to_ollama_generate_ndjson(
    sse_line: str, model: str
) -> str | None:
    """Convert chat-completions SSE into Ollama /api/generate NDJSON."""
    chat_line = openai_chat_chunk_to_ollama_ndjson(sse_line, model)
    if not chat_line:
        return None
    try:
        chat_obj = json.loads(chat_line)
    except json.JSONDecodeError:
        return None

    message = chat_obj.get("message") or {}
    result: dict[str, Any] = {
        "model": chat_obj.get("model", model),
        "created_at": chat_obj.get("created_at", _now_iso()),
        "response": message.get("content", ""),
        "done": bool(chat_obj.get("done", False)),
    }
    if message.get("thinking"):
        result["thinking"] = message["thinking"]
    if result["done"]:
        result["done_reason"] = chat_obj.get("done_reason", "stop")
        if "eval_count" in chat_obj:
            result["eval_count"] = chat_obj.get("eval_count", 0)
        if "prompt_eval_count" in chat_obj:
            result["prompt_eval_count"] = chat_obj.get("prompt_eval_count", 0)
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


def build_tags_response(model_name: str, model_path: str, extra_models: list[str] | None = None) -> dict:
    """Build Ollama /api/tags response.

    extra_models: optional additional model IDs to advertise (e.g. the
    loaded embedding model). Clients probing /api/tags can discover
    auxiliary models without having to know full paths.
    """
    entries = [{
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
    seen = {model_name, model_path}
    for extra in extra_models or []:
        if not extra or extra in seen:
            continue
        seen.add(extra)
        entries.append({
            "name": extra,
            "model": extra,
            "modified_at": _now_iso(),
            "size": 0,
            "digest": "",
            "details": {
                "format": "mlx",
                "family": "",
                "parameter_size": "",
                "quantization_level": "",
            },
        })
    return {"models": entries}
