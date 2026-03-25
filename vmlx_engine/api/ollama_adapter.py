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
    content = choices[0]["message"]["content"] if choices else ""
    usage = openai_resp.get("usage", {})
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": content or ""},
        "done": True,
        "done_reason": choices[0].get("finish_reason", "stop") if choices else "stop",
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
    content = ""
    done = False
    done_reason = None

    if choices:
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        fr = choices[0].get("finish_reason")
        if fr is not None:
            done = True
            done_reason = fr

    result: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
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
