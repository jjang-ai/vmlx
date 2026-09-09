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


def _should_forward_reasoning_effort(body: dict, req: dict[str, Any]) -> bool:
    """Reasoning effort is only meaningful when thinking is not explicitly off."""
    if req.get("enable_thinking") is False:
        return False
    ct_kwargs = body.get("chat_template_kwargs")
    if isinstance(ct_kwargs, dict) and ct_kwargs.get("enable_thinking") is False:
        return False
    return body.get("reasoning_effort") is not None


def _normalize_ollama_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


# Recent Ollama clients pass an effort *level* through the same ``think`` field
# instead of a bare boolean. A level both turns thinking on and selects the
# effort; ``none`` is an explicit opt-out. Unknown strings fall through and are
# ignored, exactly as before.
_OLLAMA_THINK_EFFORT_LEVELS = {"minimal", "low", "medium", "high", "xhigh", "max"}
_OLLAMA_THINK_OFF_LEVELS = {"none", "off"}


def _apply_ollama_thinking(body: dict, req: dict[str, Any]) -> None:
    """Normalize Ollama thinking controls into vMLX's canonical field.

    Omitted thinking controls stay omitted so the model's native
    tokenizer/template/runtime default decides. Native Ollama ``think:false``
    is an explicit opt-out and must not be overwritten. A string effort level
    (``think:"high"``) enables thinking and selects the effort, forwarded via
    the shared ``reasoning_effort`` passthrough below.
    """
    raw_think = body.get("think")
    think = _normalize_ollama_bool(raw_think)
    enable_thinking = _normalize_ollama_bool(body.get("enable_thinking"))
    think_level = raw_think.strip().lower() if isinstance(raw_think, str) else None
    if think is not None:
        req["enable_thinking"] = think
    elif think_level in _OLLAMA_THINK_EFFORT_LEVELS:
        req["enable_thinking"] = True
        # Don't clobber an explicit body-level reasoning_effort.
        body.setdefault("reasoning_effort", think_level)
    elif think_level in _OLLAMA_THINK_OFF_LEVELS:
        req["enable_thinking"] = False
    elif enable_thinking is not None:
        req["enable_thinking"] = enable_thinking
    elif isinstance(body.get("chat_template_kwargs"), dict) and (
        body["chat_template_kwargs"].get("enable_thinking") is False
    ):
        return


def _apply_ollama_prompt_context_limit(body: dict, req: dict[str, Any]) -> None:
    """Forward Ollama/vMLX prompt context caps to the internal API shape."""
    opts = body.get("options", {})
    for key in (
        "num_ctx",
        "num_context",
        "max_prompt_tokens",
        "max_context_tokens",
        "max_context",
    ):
        if opts.get(key) is not None:
            req["max_prompt_tokens"] = opts[key]
            return
    for key in ("max_prompt_tokens", "max_context_tokens", "max_context"):
        if body.get(key) is not None:
            req["max_prompt_tokens"] = body[key]
            return


def _apply_ollama_top_k(opts: dict, req: dict[str, Any]) -> None:
    """Preserve omission versus an explicit top_k value.

    The adapter only translates request shapes. It must not truncate fractional
    values or hide negative values before the OpenAI-compatible request models
    validate them.
    """
    if "top_k" in opts:
        req["top_k"] = opts["top_k"]


def _apply_ollama_num_predict(opts: dict, req: dict[str, Any]) -> None:
    """Forward only active output caps; Ollama <=0 sentinels are model-owned."""
    value = opts.get("num_predict")
    if value is None:
        return
    try:
        max_tokens = int(value)
    except (TypeError, ValueError, OverflowError):
        req["max_tokens"] = value
        return
    # Preserve the existing non-positive integer sentinels. All active caps
    # and malformed values belong to the shared request validator: truncating
    # floats or silently dropping bad input would activate a different budget.
    if max_tokens <= 0 and (not isinstance(value, float) or value.is_integer()):
        return
    req["max_tokens"] = value


def _apply_ollama_seed(body: dict, opts: dict, req: dict[str, Any]) -> None:
    """Forward Ollama's request-local seed to the internal text API."""
    value = opts.get("seed", body.get("seed"))
    if value is not None:
        req["seed"] = value


def _ollama_media_content_parts(source: dict, text: str) -> list[dict] | None:
    """Translate Ollama-convention media arrays into OpenAI content parts.

    ``source`` is any dict carrying Ollama-style ``images`` (plus the vMLX
    ``videos``/``audio``/``audios`` extensions): a chat *message* for
    /api/chat, or the top-level request *body* for /api/generate, where
    Ollama places ``images`` alongside ``prompt``. Both Ollama entry points
    go through this one helper so they cannot drift apart.

    Returns None when the source carries no media so callers can leave a
    plain-string prompt untouched (identical tokenization). Base64 payloads
    are not validated here — exactly like /api/chat, strings pass through
    (raw base64 gets a data-URL prefix; existing ``data:`` URLs are kept)
    and the downstream content-part decoder surfaces malformed data.
    Non-string entries are skipped.
    """
    images = source.get("images")
    videos = source.get("videos")
    audio = source.get("audio", source.get("audios"))
    if isinstance(images, str):
        images = [images]
    if isinstance(videos, str):
        videos = [videos]
    if isinstance(audio, str):
        audio = [audio]
    if not images and not videos and not audio:
        return None
    parts: list[dict] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in images or []:
        if not isinstance(img, str):
            continue
        # Ollama accepts either raw base64 or a data URL — normalize
        # to data URL so the OpenAI content_part handler (which
        # inspects the dataUrl mime prefix) can decode.
        url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    for video in videos or []:
        if not isinstance(video, str):
            continue
        url = video if video.startswith("data:") else f"data:video/mp4;base64,{video}"
        parts.append({"type": "video_url", "video_url": {"url": url}})
    for audio_item in audio or []:
        if not isinstance(audio_item, str):
            continue
        url = (
            audio_item
            if audio_item.startswith("data:")
            else f"data:audio/wav;base64,{audio_item}"
        )
        parts.append({"type": "audio_url", "audio_url": {"url": url}})
    return parts



def _apply_ollama_video_controls(body: dict, req: dict) -> None:
    """vMLX extension on Ollama-shaped bodies: the per-request video controls
    (video_fps, video_max_frames, pixel budgets, explicit frame size) may sit
    at the top level or inside ``options``; forward whichever is set so the
    Ollama dialect honours them like chat/responses/anthropic do."""
    from ..video_controls import VIDEO_CONTROL_FIELDS
    from ..image_controls import IMAGE_CONTROL_FIELDS

    opts = body.get("options") if isinstance(body.get("options"), dict) else {}
    for field in (*VIDEO_CONTROL_FIELDS, *IMAGE_CONTROL_FIELDS, "media_controls_strict"):
        value = body.get(field)
        if value is None:
            value = opts.get(field)
        if value is not None:
            req[field] = value


def ollama_chat_to_openai(body: dict) -> dict:
    """Convert Ollama /api/chat request to OpenAI /v1/chat/completions."""
    opts = body.get("options", {})

    # Ollama convention for VL models: each message may have an
    # `images: [<base64>, ...]` field alongside `content: <string>`.
    # vMLX also accepts symmetric `videos` and `audio`/`audios` extension
    # arrays for local runtimes that advertise those modalities. The OpenAI
    # multimodal schema embeds all media as inline content parts.
    #
    # Without this translation `prompt_eval_count` shows only the text
    # tokens, the model reports "I cannot see the image", and the
    # reporter has no indication of why. Surfaced during live VL test
    # against Qwen3.5-VL-4B-JANG_4S-CRACK.
    src_messages = body.get("messages", [])
    translated_messages = []
    for msg in src_messages:
        if not isinstance(msg, dict):
            translated_messages.append(msg)
            continue
        # Ollama exposes private reasoning as ``message.thinking``.  Normalize
        # it into the shared Chat history field before either the text-only or
        # multimodal path runs, keeping it out of visible ``content``.
        normalized_msg = dict(msg)
        if (
            normalized_msg.get("role") == "assistant"
            and "thinking" in normalized_msg
        ):
            thinking = normalized_msg.pop("thinking")
            existing_reasoning = normalized_msg.get("reasoning_content")
            if (
                isinstance(thinking, str)
                and thinking
                and not (
                    isinstance(existing_reasoning, str)
                    and existing_reasoning
                )
            ):
                normalized_msg["reasoning_content"] = thinking
        msg = normalized_msg
        parts = _ollama_media_content_parts(msg, msg.get("content", "") or "")
        if parts is None:
            translated_messages.append(msg)
            continue
        new_msg = {
            k: v
            for k, v in msg.items()
            if k not in {"images", "videos", "audio", "audios", "content"}
        }
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
    _apply_ollama_num_predict(opts, req)
    _apply_ollama_seed(body, opts, req)
    if opts.get("temperature") is not None:
        req["temperature"] = opts["temperature"]
    if opts.get("top_p") is not None:
        req["top_p"] = opts["top_p"]
    _apply_ollama_top_k(opts, req)
    if opts.get("min_p") is not None:
        req["min_p"] = opts["min_p"]
    if opts.get("stop"):
        req["stop"] = opts["stop"]
    if opts.get("repeat_penalty") is not None:
        req["repetition_penalty"] = opts["repeat_penalty"]
    _apply_ollama_prompt_context_limit(body, req)
    # Forward tools if present (Ollama tool calling)
    if body.get("tools"):
        req["tools"] = body["tools"]
    _apply_ollama_thinking(body, req)
    _apply_ollama_video_controls(body, req)
    # vMLX extensions on Ollama-shaped bodies: clients that set reasoning_effort
    # (Mistral 4 / GPT-OSS: "none"/"low"/"medium"/"high") or supply custom
    # chat_template_kwargs must reach the parser. Without this passthrough,
    # Mistral 4 on the Ollama adapter loses reasoning-effort level entirely.
    if _should_forward_reasoning_effort(body, req):
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
    """Convert Ollama /api/generate request to OpenAI /v1/completions.

    Used only for ``raw: true`` requests (server.py routes the templated
    default through :func:`ollama_generate_to_openai_chat`). The internal
    target, ``CompletionRequest``, is text-only — ``prompt: str | list[str]``
    with no content-part carrier — so top-level ``images`` cannot be
    delivered on this path today. Rejecting them loudly (or rerouting) is a
    server.py/models.py decision, not an adapter translation.
    """
    opts = body.get("options", {})
    req: dict[str, Any] = {
        "model": body.get("model", "default"),
        "prompt": body.get("prompt", ""),
        "stream": body.get("stream", True),
    }
    _apply_ollama_num_predict(opts, req)
    _apply_ollama_seed(body, opts, req)
    if opts.get("temperature") is not None:
        req["temperature"] = opts["temperature"]
    if opts.get("top_p") is not None:
        req["top_p"] = opts["top_p"]
    _apply_ollama_top_k(opts, req)
    if opts.get("min_p") is not None:
        req["min_p"] = opts["min_p"]
    if opts.get("repeat_penalty") is not None:
        req["repetition_penalty"] = opts["repeat_penalty"]
    if opts.get("stop"):
        req["stop"] = opts["stop"]
    _apply_ollama_prompt_context_limit(body, req)
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
    prompt = body.get("prompt", "") or ""
    # Ollama's /api/generate carries media as TOP-LEVEL arrays alongside
    # `prompt` (`images: [<base64>, ...]`), unlike /api/chat's per-message
    # convention. Translate through the same helper as /api/chat so both
    # Ollama entry points produce the identical multimodal content-part
    # shape. Previously the array was silently dropped: a vision request
    # became text-only and the model answered about nothing, with no error.
    media_parts = _ollama_media_content_parts(body, prompt)
    if media_parts is None:
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append({"role": "user", "content": media_parts})

    req: dict[str, Any] = {
        "model": body.get("model", "default"),
        "messages": messages,
        "stream": body.get("stream", True),
        "stream_options": {"include_usage": True},
    }
    _apply_ollama_num_predict(opts, req)
    _apply_ollama_seed(body, opts, req)
    if opts.get("temperature") is not None:
        req["temperature"] = opts["temperature"]
    if opts.get("top_p") is not None:
        req["top_p"] = opts["top_p"]
    _apply_ollama_top_k(opts, req)
    if opts.get("min_p") is not None:
        req["min_p"] = opts["min_p"]
    if opts.get("stop"):
        req["stop"] = opts["stop"]
    if opts.get("repeat_penalty") is not None:
        req["repetition_penalty"] = opts["repeat_penalty"]
    _apply_ollama_prompt_context_limit(body, req)
    _apply_ollama_thinking(body, req)
    if _should_forward_reasoning_effort(body, req):
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


def ollama_duration_fields(
    started_ns: int | None,
    first_content_ns: int | None,
    ended_ns: int | None,
    load_ns: int = 0,
) -> dict[str, int]:
    """Ollama's nanosecond timing fields, from measurements only.

    Ollama clients render throughput as ``eval_count / eval_duration * 1e9``
    (Open WebUI, Continue.dev, ``ollama run --verbose``). vMLX used to send
    ``total_duration: 0`` on the non-streaming path and no duration fields at
    all on the streaming one, so every such client showed nothing — while a
    comment in the chat handler claimed the streaming path carried tok/s.

    ``prompt_eval_duration`` is the measured time to the first content token,
    which is what prefill actually cost; ``eval_duration`` is the remainder.
    When no first-token timestamp is available (the non-streaming path can't
    see one) the split is OMITTED rather than guessed — reporting total time
    as decode time would understate tok/s with a number that looks precise.

    ``load_ns`` is the time this request spent waiting for a sleeping model
    to become servable (JIT wake reload, stamped by the wake middleware as
    ``request.state.vmlx_wake_ns``). Real Ollama reports model load in
    ``load_duration`` and includes it in ``total_duration``; a deep-sleep
    wake of a 100 GB model is tens of seconds, and dropping it made a 51 s
    request report ``total_duration`` of 2.8 s. When no wake happened the
    stamp is absent and ``load_duration`` is a measured 0. ``load_ns`` never
    contaminates the prefill/decode split — tok/s stays wake-independent.
    """
    if started_ns is None or ended_ns is None:
        return {}
    load = max(0, int(load_ns or 0))
    gen_total = max(0, int(ended_ns) - int(started_ns))
    fields: dict[str, int] = {
        "total_duration": gen_total + load,
        "load_duration": load,
    }
    if first_content_ns is not None:
        prompt_eval = max(0, int(first_content_ns) - int(started_ns))
        fields["prompt_eval_duration"] = min(prompt_eval, gen_total)
        fields["eval_duration"] = max(0, gen_total - fields["prompt_eval_duration"])
    return fields


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def openai_chat_response_to_ollama(
    openai_resp: dict, model: str, started_ns: int | None = None,
    load_ns: int = 0,
) -> dict:
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
    result: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "message": msg,
        "done": True,
        "done_reason": finish_reason or "stop",
        "total_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }
    # No first-token timestamp exists on the non-streaming path, so only the
    # total is reported; the prefill/decode split is omitted rather than
    # guessed.
    result.update(
        ollama_duration_fields(started_ns, None, time.perf_counter_ns(), load_ns)
    )
    return result


def openai_chat_response_to_ollama_generate(
    openai_resp: dict, model: str, started_ns: int | None = None,
    load_ns: int = 0,
) -> dict:
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
    result.update(
        ollama_duration_fields(started_ns, None, time.perf_counter_ns(), load_ns)
    )
    if reasoning:
        result["thinking"] = reasoning
    return result


def _openai_stream_error_to_ollama(chunk: dict[str, Any]) -> str | None:
    """Return Ollama's native mid-stream error message, if present.

    OpenAI-compatible vMLX streams carry structured failures as
    ``{"error": {"message": ...}}``.  Ollama's streaming contract instead
    requires a terminal NDJSON row shaped exactly as ``{"error": "..."}``.
    """
    error = chunk.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail") or error.get("type")
        # Ollama's row carries one string: keep the typed code in front of the
        # message so a client can still tell a strict media-control rejection
        # (media_controls_unmeetable) from a prompt-length or server failure.
        code = error.get("code")
        if code and message and error.get("type") == "invalid_request_error" and str(code) not in str(message):
            message = f"{code}: {message}"
    else:
        message = error
    if message is None:
        return None
    message = str(message).strip()
    return message or "the model failed to generate a response"


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

    stream_error = _openai_stream_error_to_ollama(chunk)
    if stream_error is not None:
        return json.dumps({"error": stream_error}) + "\n"

    choices = chunk.get("choices", [])
    usage = chunk.get("usage", {})
    content = ""
    done = False
    done_reason = None

    # Bug 5 relay: upstream Chat Completions sometimes emits a chunk shaped
    # {choices:[], warnings:["..."]} carrying engine diagnostics that explain
    # why the response is empty (dropped tool call, reasoning-only truncation,
    # etc.). Ollama clients have no native warnings field, so surface the
    # diagnostic prose as a content chunk so the user sees the explanation
    # instead of an empty response.
    warnings = chunk.get("warnings")
    if not choices and isinstance(warnings, list) and warnings:
        notice_text = "\n\n[vMLX notice] " + "; ".join(str(w) for w in warnings if w)
        return json.dumps({
            "model": model,
            "created_at": _now_iso(),
            "message": {"role": "assistant", "content": notice_text},
            "done": False,
        }) + "\n"

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


def merge_ollama_stream_terminal(
    pending: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Merge deferred Ollama terminal chunks into one final done line.

    Chat streaming can emit a first-pass ``length`` terminal, then run a
    bounded visible-answer pass, then emit a ``stop`` terminal and a separate
    usage terminal. Ollama requires exactly one ``done:true`` line after every
    content chunk. Preserve tool calls and usage while allowing the answer-pass
    ``stop`` reason to replace the provisional first-pass ``length`` reason.
    """
    merged: dict[str, Any] = dict(pending or {})
    merged.update(
        {
            "model": current.get("model", merged.get("model", "")),
            "created_at": current.get("created_at", merged.get("created_at", _now_iso())),
            "done": True,
        }
    )

    previous_message = dict((pending or {}).get("message") or {})
    current_message = dict(current.get("message") or {})
    message: dict[str, Any] = {
        "role": current_message.get(
            "role",
            previous_message.get("role", "assistant"),
        ),
        "content": current_message.get("content")
        or previous_message.get("content")
        or "",
    }
    for key in ("thinking", "tool_calls"):
        value = current_message.get(key) or previous_message.get(key)
        if value:
            message[key] = value
    merged["message"] = message

    for key in ("eval_count", "prompt_eval_count", "total_duration"):
        if key in current:
            merged[key] = current[key]

    current_reason = current.get("done_reason")
    previous_reason = merged.get("done_reason", "stop")
    current_is_usage_only = (
        not current_message.get("content")
        and not current_message.get("thinking")
        and not current_message.get("tool_calls")
        and any(key in current for key in ("eval_count", "prompt_eval_count", "total_duration"))
    )

    if message.get("tool_calls"):
        merged["done_reason"] = "tool_calls"
    elif (
        previous_reason == "length"
        and current_is_usage_only
        and current_reason in (None, "stop")
    ):
        # Usage arrives after the finish chunk on the OpenAI SSE path. The
        # adapter fabricates a done row for that usage with done_reason="stop";
        # do not let that accounting-only row hide a prior max-token terminal.
        merged["done_reason"] = "length"
    else:
        merged["done_reason"] = current_reason or previous_reason
    return merged


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

    if "error" in chat_obj:
        return json.dumps({"error": str(chat_obj["error"])}) + "\n"

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


def merge_ollama_generate_stream_terminal(
    pending: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Merge deferred templated ``/api/generate`` terminal rows.

    Chat Completions sends finish reason and usage as separate SSE events.
    Ollama permits one final ``done:true`` row, so the generate adapter must
    retain the finish event until the later usage event arrives.
    """
    merged: dict[str, Any] = dict(pending or {})
    merged.update(
        {
            "model": current.get("model", merged.get("model", "")),
            "created_at": current.get(
                "created_at",
                merged.get("created_at", _now_iso()),
            ),
            "done": True,
        }
    )
    merged["response"] = current.get("response") or merged.get("response") or ""
    if current.get("thinking") or merged.get("thinking"):
        merged["thinking"] = current.get("thinking") or merged.get("thinking")
    for key in ("eval_count", "prompt_eval_count", "total_duration"):
        if key in current:
            merged[key] = current[key]
    current_reason = current.get("done_reason")
    previous_reason = merged.get("done_reason", "stop")
    current_is_usage_only = (
        not current.get("response")
        and not current.get("thinking")
        and any(key in current for key in ("eval_count", "prompt_eval_count", "total_duration"))
    )
    if (
        previous_reason == "length"
        and current_is_usage_only
        and current_reason in (None, "stop")
    ):
        merged["done_reason"] = "length"
    else:
        merged["done_reason"] = current_reason or previous_reason
    return merged


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

    stream_error = _openai_stream_error_to_ollama(chunk)
    if stream_error is not None:
        return json.dumps({"error": stream_error}) + "\n"

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
