# SPDX-License-Identifier: Apache-2.0
# Created by Jinho Jang (eric@jangq.ai) for vMLX / mlxstudio
"""
Unified OpenAI-compatible API server for vmlx-engine.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports two modes:
- Simple mode (default): Maximum throughput for single-user scenarios
- Batched mode: Continuous batching for multiple concurrent users

Features:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)

Usage:
    # Simple mode (maximum throughput)
    python -m vmlx_engine.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vmlx_engine.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vmlx_engine.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import atexit
import asyncio
import base64
import functools
import json
import logging
import os
import platform
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.models import (
    AssistantMessage,  # noqa: F401
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,  # noqa: F401
    CompletionRequest,
    CompletionResponse,
    ContentPart,  # noqa: F401
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageUrl,  # noqa: F401
    AudioSpeechRequest,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,  # noqa: F401
    MCPServersResponse,
    MCPToolInfo,  # noqa: F401
    MCPToolsResponse,
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    InputTokensDetails,
    ModelsResponse,
    PromptTokensDetails,
    ResponsesFunctionCall,
    ResponsesObject,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesToolDefinition,
    ResponsesUsage,
    ToolCall,
    ToolDefinition,
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
    parse_tool_calls,
)
from .api.utils import (
    clean_output_text,
    strip_marker_tokens_delta,
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine
from .errors import PromptTooLongError, VLMImagePrefillBudgetError
from .logprobs import (
    format_chat_logprobs as _format_chat_logprobs,
    format_completion_logprobs as _format_completion_logprobs,
)
from .mlx_memory import clear_mlx_memory_cache
from .reasoning.gptoss_parser import GptOssReasoningParser
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: BaseEngine | None = None
_standby_state: str | None = (
    None  # None = active, 'soft' = caches cleared, 'deep' = model unloaded
)
_pre_sleep_cache_limit: int | None = None  # Saved cache limit before soft sleep
_cli_args: dict = {}  # Saved CLI args for model reload on wake
_wake_lock: asyncio.Lock | None = (
    None  # Lazy-init mutex for JIT wake (prevents double load_model)
)
_model_name: str | None = None
_model_path: str | None = None  # Full local path for config.json lookups
_served_model_name: str | None = None  # Custom name for API (--served-model-name)
_FALLBACK_MAX_OUTPUT_TOKENS = 4096
_default_max_tokens: int = _FALLBACK_MAX_OUTPUT_TOKENS
_default_max_tokens_explicit: bool = False
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_default_top_k: int | None = None  # Set via --default-top-k
_default_min_p: float | None = None  # Set via --default-min-p
_default_repetition_penalty: float | None = None  # Set via --default-repetition-penalty
_default_enable_thinking: bool | None = (
    None  # Set via --default-enable-thinking or --chat-template-kwargs
)
_native_mtp_sampling_policy: str = "compatible-only"
_default_chat_template_kwargs: dict | None = None  # Set via --chat-template-kwargs
_custom_chat_template: str | None = None  # Set via --chat-template
_max_prompt_tokens: int = 0  # Set at startup based on available memory (0 = no limit)
_model_name_mismatch_warned: bool = (
    False  # Suppress repeated model name mismatch warnings
)
# v1.3.67 mlxstudio#79: track distinct (requested_model, served_model) pairs so we
# log at INFO once per distinct pair instead of silently going DEBUG after the
# first one. Claude Code / opencode / CCSwitch users were seeing their models
# silently substituted with zero operator visibility when the requested name
# didn't match the loaded model — lowered log level hid the bug.
_model_name_mismatch_seen: set[tuple[str, str]] = set()
_metal_na_status_cache: dict[str, dict] = {}


def _argv_has_option(argv: list[str], option: str) -> bool:
    """Return whether an argparse option was supplied by the caller."""
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def _estimate_max_prompt_tokens() -> int:
    """Estimate max safe prompt length based on available GPU memory.

    KV cache memory per token = n_layers * 2 * n_kv_heads * head_dim * 2 bytes.
    We allow KV to use at most 60% of free memory (after model weights).
    Returns 0 if estimation fails (no limit enforced).
    """
    try:
        import mlx.core as mx

        _device_info = getattr(mx, "device_info", None) or mx.metal.device_info
        _get_active = (
            getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        )
        max_ws = _device_info()["max_recommended_working_set_size"]
        active = _get_active()
        free = max_ws - active
        # Allow KV to use at most 60% of free memory
        kv_budget = int(free * 0.6)
        # Estimate bytes per token from engine config
        engine = get_engine()
        if hasattr(engine, "model"):
            model = engine.model
            config = getattr(model, "config", None) or getattr(model, "args", None)
            if config:
                n_layers = getattr(config, "num_hidden_layers", 0) or getattr(
                    config, "n_layers", 0
                )
                n_kv_heads = getattr(config, "num_key_value_heads", 0) or getattr(
                    config, "n_kv_heads", 0
                )
                head_dim = getattr(config, "head_dim", 0)
                if not head_dim:
                    hidden = getattr(config, "hidden_size", 0)
                    n_heads = getattr(config, "num_attention_heads", 0) or getattr(
                        config, "n_heads", 0
                    )
                    head_dim = hidden // n_heads if n_heads else 128
                if n_layers and n_kv_heads and head_dim:
                    bytes_per_token = (
                        n_layers * 2 * n_kv_heads * head_dim * 2
                    )  # K+V, float16
                    max_tokens = (
                        kv_budget // bytes_per_token if bytes_per_token > 0 else 0
                    )
                    return max(1024, max_tokens)  # floor at 1K tokens
        return 0
    except Exception:
        return 0


def _resolve_max_prompt_tokens(
    auto_estimate: int,
    explicit_limit: int | None,
) -> int:
    """Resolve the prompt/context token cap enforced before prefill.

    ``--max-tokens`` controls generated output length. This resolver is for
    prompt/context length only: an explicit ``--max-prompt-tokens`` value wins;
    otherwise the engine uses the automatic memory-safe estimate.
    """
    if explicit_limit is not None and explicit_limit > 0:
        return int(explicit_limit)
    return int(auto_estimate or 0)


def _effective_max_prompt_tokens(request_or_value: Any | int | None = None) -> int:
    """Return the prompt/context cap for one request.

    The server/session cap from ``--max-prompt-tokens`` is the upper bound.
    A per-request ``max_prompt_tokens``/``max_context*`` API value may lower
    that cap for a single request, but it cannot raise above the session cap.
    """
    if isinstance(request_or_value, int):
        request_value = request_or_value
    else:
        request_value = getattr(request_or_value, "max_prompt_tokens", None)
    try:
        request_limit = int(request_value or 0)
    except (TypeError, ValueError):
        request_limit = 0
    session_limit = int(_max_prompt_tokens or 0)
    if request_limit > 0 and session_limit > 0:
        return min(request_limit, session_limit)
    if request_limit > 0:
        return request_limit
    return session_limit


def _text_prompt_token_estimate(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(str(text)) + 2) // 3)


def _configured_media_prompt_token_floor(media_type: str) -> int:
    """Best-effort pre-tokenization floor for media placeholders.

    Exact VLM media token counts are processor-dependent and are only known
    after the VLM processor expands the prompt. This guard is route-level
    preflight, so it counts text exactly by character estimate and includes a
    conservative media floor instead of treating images/videos/audio as zero.
    """
    media_type = (media_type or "").lower()
    try:
        cfg = getattr(get_engine(), "model", None)
        cfg = getattr(cfg, "config", None) or cfg
        candidates = [cfg]
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        vision_cfg = getattr(cfg, "vision_config", None) if cfg is not None else None
        candidates.extend([text_cfg, vision_cfg])
        keys = (
            "image_seq_length",
            "image_token_length",
            "num_image_tokens",
            "image_tokens_per_image",
            "tokens_per_image",
        )
        for obj in candidates:
            if obj is None:
                continue
            for key in keys:
                value = getattr(obj, key, None)
                if isinstance(value, int) and value > 0:
                    image_tokens = value
                    if "video" in media_type:
                        return max(image_tokens, image_tokens * 8)
                    return image_tokens
    except Exception:
        pass
    if "video" in media_type:
        return 8192
    return 1024


def _prompt_content_token_estimate(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _text_prompt_token_estimate(content)
    if hasattr(content, "model_dump"):
        try:
            content = content.model_dump(exclude_none=True)
        except Exception:
            pass
    if isinstance(content, list):
        return sum(_prompt_content_token_estimate(part) for part in content)
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").lower()
        if part_type in {"text", "input_text", "output_text"}:
            return _text_prompt_token_estimate(content.get("text"))
        if part_type in {
            "image",
            "image_url",
            "input_image",
            "video",
            "video_url",
            "input_video",
            "audio",
            "audio_url",
            "input_audio",
        }:
            return _configured_media_prompt_token_floor(part_type)
        # Unknown dict-shaped content is rare. Count user-visible string values
        # but do not count base64/media payload bytes as language tokens.
        total = 0
        for key, value in content.items():
            if key in {"image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"}:
                total += _configured_media_prompt_token_floor(key)
            elif isinstance(value, str):
                total += _text_prompt_token_estimate(value)
            elif isinstance(value, (list, dict)):
                total += _prompt_content_token_estimate(value)
        return total
    return _text_prompt_token_estimate(str(content))


def _message_prompt_token_estimate(messages: list[Any] | None) -> int:
    total = 0
    for msg in messages or []:
        if hasattr(msg, "model_dump"):
            try:
                msg = msg.model_dump(exclude_none=True)
            except Exception:
                pass
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        total += _prompt_content_token_estimate(content)
    return total


def _prompt_too_long_response(
    estimated_tokens: int,
    *,
    max_prompt_tokens: int | None = None,
    source: str = "prompt",
):
    from starlette.responses import JSONResponse

    limit = int(max_prompt_tokens or _max_prompt_tokens or 0)
    _est_gb = (estimated_tokens * 0.4) / 1024  # ~0.4MB per token KV
    return JSONResponse(
        status_code=413,
        content={
            "error": {
                "message": f"Prompt too long ({source}: ~{estimated_tokens:,} tokens). "
                f"This would need ~{_est_gb:.1f}GB of KV cache memory, "
                f"exceeding the configured prompt/context limit of "
                f"~{limit:,} tokens. Shorten the prompt, raise "
                "--max-prompt-tokens, or use a model/session with a larger "
                "prompt/context cap.",
                "type": "invalid_request_error",
                "code": "prompt_too_long",
            }
        },
    )


def _prompt_too_long_response_from_error(exc: PromptTooLongError):
    return _prompt_too_long_response(
        exc.prompt_tokens,
        max_prompt_tokens=exc.max_prompt_tokens,
        source=exc.source,
    )


def _vlm_image_prefill_budget_response_from_error(exc: VLMImagePrefillBudgetError):
    from starlette.responses import JSONResponse

    detail = str(exc)
    if "Reduce image resolution" not in detail:
        detail = (
            f"{detail}. Reduce image resolution/prompt length, use a smaller "
            "model, or disable VMLX_VLM_IMAGE_PREFILL_GUARD=0 at OOM risk."
        )
    return JSONResponse(
        status_code=413,
        content={
            "error": {
                "message": detail,
                "type": "invalid_request_error",
                "code": VLMImagePrefillBudgetError.code,
            }
        },
    )


def _reject_if_prompt_too_long_for_messages(
    messages: list[Any] | None,
    *,
    max_prompt_tokens: int | None = None,
):
    limit = int(_max_prompt_tokens if max_prompt_tokens is None else max_prompt_tokens)
    if limit <= 0 or not messages:
        return None
    estimated_tokens = _message_prompt_token_estimate(messages)
    if estimated_tokens > limit:
        return _prompt_too_long_response(estimated_tokens, max_prompt_tokens=limit)
    return None


def _reject_if_prompt_too_long_for_prompts(
    prompts: list[str] | None,
    *,
    max_prompt_tokens: int | None = None,
):
    limit = int(_max_prompt_tokens if max_prompt_tokens is None else max_prompt_tokens)
    if limit <= 0 or not prompts:
        return None
    estimated_tokens = sum(_text_prompt_token_estimate(str(prompt)) for prompt in prompts)
    if estimated_tokens > limit:
        return _prompt_too_long_response(estimated_tokens, max_prompt_tokens=limit)
    return None


def _merge_ct_kwargs(request_kwargs: dict | None) -> dict:
    """Merge server-wide default chat_template_kwargs with per-request overrides.

    Server defaults (from --chat-template-kwargs) are used as the base layer.
    Per-request chat_template_kwargs override any matching keys.

    Normalizes enable_thinking to a real bool — guards against bool("false") == True
    when API clients send string values instead of JSON booleans.
    """
    base = dict(_default_chat_template_kwargs) if _default_chat_template_kwargs else {}
    if request_kwargs:
        base.update(request_kwargs)
    # Normalize enable_thinking: reject non-bool values that bool() would mishandle
    if "enable_thinking" in base:
        val = base["enable_thinking"]
        if val is None:
            # JSON null means "no preference" — remove so downstream defaults apply
            del base["enable_thinking"]
        elif isinstance(val, bool):
            pass  # Already correct
        elif isinstance(val, str) and val.lower() in ("false", "0", "no", "off"):
            base["enable_thinking"] = False
        elif isinstance(val, str) and val.lower() in ("true", "1", "yes", "on"):
            base["enable_thinking"] = True
        elif isinstance(val, (int, float)):
            base["enable_thinking"] = bool(val)
        else:
            # dict, list, or other non-scalar — remove to avoid silent misinterpretation
            logger.warning(
                f"Invalid enable_thinking value {val!r} in chat_template_kwargs, ignoring"
            )
            del base["enable_thinking"]
    return base


_last_request_time: float = (
    0.0  # Epoch timestamp of last API request (for idle sleep timer)
)
_inference_endpoints: list[str] = [
    "/v1/chat/",
    "/v1/completions",
    "/v1/images/",
    "/v1/mcp/execute",
    "/v1/messages",
    "/v1/responses",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
    "/v1/audio/speech",
    "/v1/rerank",
    "/api/chat",
    "/api/generate",
    "/api/embed",  # Ollama inference endpoints
]
_wake_timeout: int = 300
_model_load_error: str | None = None  # Surfaced via /health when model fails to load
_smelt_enabled: bool = False  # --smelt: partial expert loading for MoE
_smelt_experts: int = 50  # --smelt-experts: percentage of experts per layer
_flash_moe_enabled: bool = False  # --flash-moe: SSD expert streaming
_flash_moe_loader = None  # FlashMoEExpertLoader instance (set after model load)
_distributed_enabled: bool = False  # --distributed: pipeline/tensor parallelism
_distributed_mode: str = "pipeline"  # --distributed-mode: pipeline or tensor
_distributed_coordinator = None  # Coordinator instance when distributed is active

_FALLBACK_TEMPERATURE = 0.0
_FALLBACK_TOP_P = 1.0

# Family temperature/top_p/repetition fallbacks are intentionally empty.
# Defaults come from explicit request values, explicit CLI flags, bundle
# metadata, or generic server fallback.
_FAMILY_FALLBACK_DEFAULTS: dict[str, tuple[float | None, float | None, float | None]] = {
}

# Ling/Bailing bundles on this host omit top_k metadata, but live multilingual
# audit shows unconstrained stochastic sampling can leak CJK into non-CJK
# prompts. Keep the bounded fallback at request resolution so it is logged in
# resolved kwargs and remains overrideable by request/CLI/bundle metadata.
_FAMILY_TOP_K_DEFAULTS: dict[str, int] = {
    "ling": 20,
}


def _family_fallback_for(model_name: str = "") -> tuple[float | None, float | None, float | None]:
    """Look up family-specific fallback (temp, top_p, rep_pen). All-None tuple
    when the family has no override. Resolves via model_config_registry — uses
    `_model_path` (the actual loaded model directory, set at server startup)
    for the registry lookup since `model_name` from chat requests is typically
    a short alias that fails registry's config.json probe."""
    normalized = _model_family_for_defaults(model_name)
    if normalized:
        configured = _FAMILY_FALLBACK_DEFAULTS.get(normalized)
        if configured is not None:
            return configured

    candidates: list[str] = []
    if model_name:
        candidates.append(model_name)
    if _model_path:
        candidates.append(_model_path)
    if not candidates:
        return (None, None, None)
    try:
        from .model_config_registry import get_model_config_registry
        registry = get_model_config_registry()
        family = ""
        for cand in candidates:
            cfg = registry.lookup(cand)
            family = getattr(cfg, "family_name", "") or ""
            if family and family != "unknown":
                break
    except Exception:
        return (None, None, None)
    return _FAMILY_FALLBACK_DEFAULTS.get(family, (None, None, None))
_THINK_STRIP_RE = re.compile(
    r"(?:<think>.*?</think>|\[THINK\].*?\[/THINK\])\s*", re.DOTALL
)
_JANGTQ_PROFILE_BITS_RE = re.compile(r"^JANGTQ([124])(?:$|[_-])", re.IGNORECASE)
_THINK_MARKER_RE = re.compile(
    r"[ \t]*(?:\n[ \t]*)?(?:</?think>|\[/?THINK\])[ \t]*(?:\n[ \t]*)?",
    re.IGNORECASE,
)


def _strip_think_markers_keep_spacing(text: str) -> str:
    def _replacement(match: re.Match[str]) -> str:
        matched = match.group(0)
        if "\n" in matched:
            return "\n"
        if matched[:1].isspace() or matched[-1:].isspace():
            return " "
        return ""

    return _THINK_MARKER_RE.sub(_replacement, text)


def _strip_think_for_tool_parse(text: str) -> str:
    """Strip residual think tags before tool call parsing.

    Handles both <think>...</think> and [THINK]...[/THINK] forms, plus
    the truncated case where the opening tag was consumed by streaming
    but the closing tag remains (partition on </think> or [/THINK]).

    Shared by every tool-call parse site so the behavior stays identical
    across chat completions, Responses API, and their streaming paths.
    """
    if not text:
        return ""
    stripped = _THINK_STRIP_RE.sub("", text)
    if stripped == text and ("</think>" in text or "[/THINK]" in text):
        for end_tag in ("</think>", "[/THINK]"):
            if end_tag in text:
                _, _, stripped = text.partition(end_tag)
                break
    stripped = _strip_think_markers_keep_spacing(stripped)
    return stripped.strip()


def _jangtq_bits_from_profile(profile: Any) -> int | None:
    """Infer uniform routed-expert JANGTQ bits from a profile name.

    Explicit per-role or per-projection `mxtq_bits` metadata remains the source
    of truth. This is only a fallback for early bundles whose sidecar stamps
    `quantization.profile=JANGTQ1|2|4` before adding `mxtq_bits`.
    """
    if not isinstance(profile, str):
        return None
    match = _JANGTQ_PROFILE_BITS_RE.match(profile.strip())
    if not match:
        return None
    return int(match.group(1))


def _strip_residual_think_markup_for_display(text: str) -> str:
    """Remove stale think markup after reasoning was intentionally suppressed.

    `clean_output_text()` deliberately keeps `<think>` markers for normal
    reasoning-mode display, including reopening a missing start tag when only
    `</think>` appears.  That is wrong for product paths where thinking has
    been resolved off: residual model tags are not user-visible content and
    must be removed before display cleaning runs.
    """
    if not text:
        return ""
    stripped = _THINK_STRIP_RE.sub("", text)
    stripped = _strip_think_markers_keep_spacing(stripped)
    return stripped.strip()


def _dsv4_split_reasoning_from_token_ids(
    token_ids: list[int] | tuple[int, ...] | None,
    tokenizer: Any,
) -> tuple[str | None, str | None]:
    """Split DSV4 output on the real ``</think>`` token id.

    DSV4 marks reasoning with dedicated special token IDs. The MLX streaming
    detokenizer can drop those markers from text, leaving non-stream parsers
    unable to distinguish "unclosed reasoning" from "reasoning closed, now
    visible answer". When cumulative output token IDs are available, use the
    token boundary as source of truth.
    """
    if not token_ids:
        return None, None
    ids = [int(t) for t in token_ids]
    think_close = 128822
    if think_close not in ids:
        return None, None
    think_open = 128821
    structural = {1, 128803, 128804, think_open, think_close}
    close_idx = ids.index(think_close)

    def decode(part: list[int]) -> str:
        clean_ids = [t for t in part if t not in structural]
        if not clean_ids:
            return ""
        decode_fn = getattr(tokenizer, "decode", None)
        if decode_fn is None and hasattr(tokenizer, "tokenizer"):
            decode_fn = getattr(tokenizer.tokenizer, "decode", None)
        if decode_fn is None:
            return ""
        try:
            text = decode_fn(clean_ids, skip_special_tokens=True)
        except TypeError:
            try:
                text = decode_fn(clean_ids)
            except Exception:
                return ""
        except Exception:
            return ""
        return clean_output_text(str(text or ""))

    reasoning = decode(ids[:close_idx]).strip() or None
    content = decode(ids[close_idx + 1:]).strip() or None
    return reasoning, content


def _output_token_ids_for_reasoning(output: Any) -> list[int] | None:
    """Return cumulative generated token IDs from any engine output shape."""
    for attr in ("output_token_ids", "tokens"):
        token_ids = getattr(output, attr, None)
        if token_ids:
            try:
                return [int(t) for t in token_ids]
            except Exception:
                return None
    return None


_SINGLE_MARKDOWN_CODE_FENCE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+.#-]*[ \t]*\n(?P<body>.*?)\n?```\s*$",
    re.DOTALL,
)


def _request_text_fragments(request: Any) -> list[str]:
    fragments: list[str] = []

    def add_content(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            fragments.append(value)
            return
        if isinstance(value, list):
            for item in value:
                add_content(item)
            return
        if isinstance(value, dict):
            for key in ("text", "content", "input_text", "output_text"):
                if key in value:
                    add_content(value.get(key))
            return
        for attr in ("text", "content"):
            if hasattr(value, attr):
                add_content(getattr(value, attr))

    prompt = getattr(request, "prompt", None)
    if isinstance(prompt, list):
        for item in prompt:
            add_content(item)
    else:
        add_content(prompt)

    for attr in ("messages", "input"):
        value = getattr(request, attr, None)
        add_content(value)
    return fragments


def _strict_exact_no_markdown_requested(request: Any) -> bool:
    text = "\n".join(_request_text_fragments(request)).lower()
    if not text:
        return False
    if "markdown" not in text and "code fence" not in text and "fenced" not in text:
        return False
    if not any(marker in text for marker in ("exact", "copy", "only")):
        return False
    return True


def _unwrap_single_markdown_code_fence(text: str) -> str:
    match = _SINGLE_MARKDOWN_CODE_FENCE_RE.match(text or "")
    if not match:
        return text
    body = match.group("body")
    if "```" in body:
        return text
    return body.strip()


def _finalize_visible_text_for_request(text: str, request: Any) -> str:
    if not text:
        return ""
    text = _strip_visual_grounding_markup_for_display(text)
    if not text:
        return ""
    if _strict_exact_no_markdown_requested(request):
        return _unwrap_single_markdown_code_fence(text)
    return text


_RESPONSES_EXACT_REPLY_RE = re.compile(
    r"reply exactly:\s*[\"'“”`]?([A-Za-z0-9_=-]+)[\"'“”`]?",
    re.IGNORECASE,
)


def _responses_exact_reply_target(request: Any) -> str | None:
    text = "\n".join(_request_text_fragments(request))
    matches = list(_RESPONSES_EXACT_REPLY_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _responses_messages_have_tool_result_after_latest_user(messages: list[dict]) -> bool:
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return False
        if message.get("role") == "tool":
            return True
        if message.get("type") == "function_call_output":
            return True
    return False


def _responses_fast_path_visible_text(output: Any, request: Any) -> str:
    text = getattr(output, "raw_text", "") or getattr(output, "text", "") or ""
    if getattr(request, "enable_thinking", None) is False and "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    else:
        text = getattr(output, "text", "") or text
    return _finalize_visible_text_for_request(
        clean_output_text(text),
        request,
    )


def _visible_text_for_dsv4_completion(output: Any, engine: Any, request: Any) -> str:
    """Return the visible DSV4 completion text from chat-rail output."""
    raw_text = getattr(output, "raw_text", "") or getattr(output, "text", "") or ""
    visible_text = raw_text

    token_reasoning, token_content = _dsv4_split_reasoning_from_token_ids(
        _output_token_ids_for_reasoning(output),
        getattr(engine, "tokenizer", None),
    )
    if token_content:
        visible_text = token_content
    elif _reasoning_parser is not None:
        try:
            parser = _reasoning_parser.__class__()
            parser.reset_state(think_in_prompt=True)
            _, parsed_content = parser.extract_reasoning(raw_text)
            if parsed_content is not None:
                visible_text = parsed_content
        except Exception:
            if "</think>" in raw_text:
                _, _, visible_text = raw_text.partition("</think>")
    elif "</think>" in raw_text:
        _, _, visible_text = raw_text.partition("</think>")

    return _finalize_visible_text_for_request(
        clean_output_text(visible_text),
        request,
    )


def _dsv4_completion_prompt_for_chat_rail(prompt: str, request: Any) -> str:
    if _strict_exact_no_markdown_requested(request):
        return prompt.rstrip()
    return prompt


def _dsv4_completion_internal_max_tokens(max_tokens: int, request: Any) -> int:
    if _strict_exact_no_markdown_requested(request):
        return max(int(max_tokens), 512)
    return max_tokens


_jang_sampling_defaults_cache: dict[str, dict] = {}
_generation_defaults_cache: dict[str, dict] = {}


def _nearly_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= eps


def _jang_chat_sampling_default(model_name: str, key: str) -> float | None:
    """Read a numeric default from the loaded bundle's
    ``jang_config.json::chat.sampling_defaults.<key>``. Cached per-path.

    The JANG runtime stamp carries per-bundle chat defaults. Reading these
    server-side means panels / CLI users do not need to know per-family
    sampling values; the bundle declares them.

    Returns None when the bundle has no jang_config or the key is absent.
    """
    if not model_name:
        return None
    cache = _jang_sampling_defaults_cache
    if model_name not in cache:
        try:
            from pathlib import Path as _P
            import json as _json
            _p = _P(model_name) / "jang_config.json"
            if _p.is_file():
                _doc = _json.loads(_p.read_text())
                _chat = _doc.get("chat") or {}
                _sd = _chat.get("sampling_defaults") or {}
                cache[model_name] = _sd if isinstance(_sd, dict) else {}
            else:
                cache[model_name] = {}
        except Exception:
            cache[model_name] = {}
    val = cache[model_name].get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _generation_config_default(model_name: str, key: str) -> float | None:
    """Read a numeric generation default from ``generation_config.json``."""
    if not model_name:
        return None
    cache = _generation_defaults_cache
    if model_name not in cache:
        try:
            from pathlib import Path as _P
            import json as _json
            _p = _P(model_name) / "generation_config.json"
            if _p.is_file():
                _doc = _json.loads(_p.read_text())
                cache[model_name] = _doc if isinstance(_doc, dict) else {}
            else:
                cache[model_name] = {}
        except Exception:
            cache[model_name] = {}
    val = cache[model_name].get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _jang_chat_default_mode(bundle_path: str) -> str | None:
    """Return ``jang_config.chat.reasoning.default_mode`` or None."""
    if not bundle_path:
        return None
    try:
        from pathlib import Path as _P
        _p = _P(bundle_path) / "jang_config.json"
        if not _p.exists():
            return None
        import json as _json
        d = _json.loads(_p.read_text())
        m = d.get("chat", {}).get("reasoning", {}).get("default_mode")
        return m if isinstance(m, str) else None
    except Exception:
        return None


def _bundle_repetition_penalty_keys(
    bundle_path: str,
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, ...]:
    """Mode-aware lookup order for `_bundle_sampling_default`.

    Bundles split `repetition_penalty_thinking` vs `_chat` because the
    correct value can differ by reasoning mode. DSV4 routes explicit thinking
    requests through the thinking rail even when bundle metadata declares a
    chat default, so the lookup follows the resolved rail instead of applying
    a generic family-level sampling policy.
    """
    family = _model_family_for_defaults(bundle_path)
    if family == "deepseek_v4":
        if enable_thinking is False:
            # Direct/no-thinking DSV4 exact-code probes corrupt identifiers
            # with the chat-specific 1.05 penalty. Prefer the bundle's neutral
            # generic value when present; fall back to chat metadata only for
            # older bundles that lack it.
            return (
                "repetition_penalty",
                "repetition_penalty_chat",
                "repetition_penalty_thinking",
            )
        return (
            "repetition_penalty_thinking",
            "repetition_penalty",
            "repetition_penalty_chat",
        )

    mode = _jang_chat_default_mode(bundle_path)
    if mode == "thinking":
        return (
            "repetition_penalty_thinking",
            "repetition_penalty_chat",
            "repetition_penalty",
        )
    return (
        "repetition_penalty_chat",
        "repetition_penalty_thinking",
        "repetition_penalty",
    )


def _bundle_sampling_default(model_name: str, key: str) -> float | None:
    """Bundle-declared sampling default, preferring JANG chat metadata.

    ``jang_config.chat.sampling_defaults`` is intentionally higher priority
    than ``generation_config.json`` because it is the JANG runtime stamp for
    chat behavior. DSV4, for example, ships generation_config temperature/top_p
    as 1.0/1.0 but declares the audited chat defaults as 0.6/0.95 in
    jang_config.
    """
    bundle_path = _model_path or model_name
    if not bundle_path:
        return None
    v = _jang_chat_sampling_default(bundle_path, key)
    if v is not None:
        return v
    gen_key = {
        "top_p": "top_p",
        "temperature": "temperature",
        "min_p": "min_p",
        "repetition_penalty": "repetition_penalty",
        "max_new_tokens": "max_new_tokens",
    }.get(key, key)
    return _generation_config_default(bundle_path, gen_key)


def _model_family_for_defaults(model_name: str = "") -> str:
    bundle_path = _model_path or model_name
    if bundle_path:
        try:
            from pathlib import Path as _P
            import json as _json

            _cfg_path = _P(bundle_path) / "config.json"
            if _cfg_path.is_file():
                _doc = _json.loads(_cfg_path.read_text())
                _model_type = _doc.get("model_type")
                if not isinstance(_model_type, str):
                    _text_cfg = _doc.get("text_config")
                    if isinstance(_text_cfg, dict):
                        _model_type = _text_cfg.get("model_type")
                if _model_type == "deepseek_v4":
                    return "deepseek_v4"
                if _model_type in {"minimax", "minimax_m2", "minimax_m2_5"}:
                    return "minimax_m2"
        except Exception:
            pass
    try:
        from .model_config_registry import get_model_config_registry
        _cfg = get_model_config_registry().lookup(bundle_path) if bundle_path else None
        return getattr(_cfg, "family_name", "") if _cfg is not None else ""
    except Exception:
        return ""


def _resolve_temperature(request_value: float | None, model_name: str = "") -> float:
    """Resolve temperature: request > explicit CLI/session > bundle > fallback."""
    if request_value is not None:
        return request_value
    if _default_temperature is not None:
        return _default_temperature
    v = _bundle_sampling_default(model_name, "temperature")
    if v is not None:
        return v
    fam_temp, _, _ = _family_fallback_for(model_name)
    if fam_temp is not None:
        return fam_temp
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None, model_name: str = "") -> float:
    """Resolve top_p: request > explicit CLI/session > bundle > fallback."""
    if request_value is not None:
        return request_value
    if _default_top_p is not None:
        return _default_top_p
    v = _bundle_sampling_default(model_name, "top_p")
    if v is not None:
        return v
    _, fam_top_p, _ = _family_fallback_for(model_name)
    if fam_top_p is not None:
        return fam_top_p
    return _FALLBACK_TOP_P


def _resolve_top_k(request_value: int | None, model_name: str = "") -> int:
    """Resolve top_k: request > explicit CLI/session > bundle > family > disabled."""
    if request_value is not None:
        return max(0, int(request_value))
    if _default_top_k is not None:
        return max(0, int(_default_top_k))
    v = _bundle_sampling_default(model_name, "top_k")
    if v is not None:
        return max(0, int(v))
    family = _model_family_for_defaults(model_name)
    if family in _FAMILY_TOP_K_DEFAULTS:
        return max(0, int(_FAMILY_TOP_K_DEFAULTS[family]))
    return 0


def _set_resolved_top_k(
    target: dict,
    request_value: int | None,
    model_name: str = "",
) -> None:
    top_k = _resolve_top_k(request_value, model_name)
    if top_k > 0:
        target["top_k"] = top_k
    else:
        target.pop("top_k", None)


def _resolve_min_p(request_value: float | None, model_name: str = "") -> float:
    """Resolve min_p: request > explicit CLI/session > bundle > disabled."""
    if request_value is not None:
        return max(0.0, float(request_value))
    if _default_min_p is not None:
        return max(0.0, float(_default_min_p))
    v = _bundle_sampling_default(model_name, "min_p")
    if v is not None:
        return max(0.0, float(v))
    return 0.0


def _set_resolved_min_p(
    target: dict,
    request_value: float | None,
    model_name: str = "",
) -> None:
    min_p = _resolve_min_p(request_value, model_name)
    if min_p > 0:
        target["min_p"] = min_p
    else:
        target.pop("min_p", None)


def _resolve_max_tokens(request_value: int | None, model_name: str = "") -> int:
    """Resolve max output tokens.

    Precedence is request > explicit CLI/session override > bundle
    max_new_tokens > bounded engine fallback. The final fallback must stay
    modest because many local JANG bundles intentionally omit max_new_tokens;
    treating that omission as 32K lets normal chats drift into loops.
    """
    if request_value is not None:
        return request_value
    if _default_max_tokens_explicit:
        return _default_max_tokens
    v = _bundle_sampling_default(model_name, "max_new_tokens")
    if v is not None and v > 0:
        return int(v)
    return _FALLBACK_MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class _DSV4ThinkingDecision:
    enable_thinking: bool
    reasoning_effort_allowed: bool
    reason: str


def _resolve_dsv4_thinking_policy(
    *,
    requested_enable_thinking: bool | None,
    effort_requested: bool = False,
    tools_present: bool,
    tool_choice: Any,
    default_mode: str | None = None,
) -> _DSV4ThinkingDecision:
    """Resolve DSV4's public thinking toggle without hidden force-on behavior."""
    if requested_enable_thinking is True or effort_requested:
        return _DSV4ThinkingDecision(
            enable_thinking=True,
            reasoning_effort_allowed=True,
            reason="requested_thinking",
        )
    if requested_enable_thinking is False:
        return _DSV4ThinkingDecision(
            enable_thinking=False,
            reasoning_effort_allowed=False,
            reason="thinking_disabled",
        )
    _default_mode = str(default_mode or "").strip().lower()
    if _default_mode in {"thinking", "reasoning", "reasoning_on", "on", "true"}:
        return _DSV4ThinkingDecision(
            enable_thinking=True,
            reasoning_effort_allowed=True,
            reason="bundle_default_thinking",
        )
    if _default_mode in {"chat", "direct", "instruct", "off", "false"}:
        return _DSV4ThinkingDecision(
            enable_thinking=False,
            reasoning_effort_allowed=False,
            reason="bundle_default_chat",
        )
    return _DSV4ThinkingDecision(
        enable_thinking=False,
        reasoning_effort_allowed=False,
        reason="thinking_not_requested",
    )


def _normalize_dsv4_reasoning_effort(effort: str | None) -> str | None:
    """Map public effort names onto the actual DSV4 encoder vocabulary."""
    if effort == "max":
        return "max"
    if effort in ("low", "medium", "high"):
        return "high"
    return None


def _is_loaded_dsv4_model(model: str = "") -> bool:
    """Return whether the current loaded model resolves to the DSV4 family."""
    try:
        from .model_config_registry import get_model_config_registry

        key = _model_path or _model_name or model or ""
        cfg = get_model_config_registry().lookup(key)
        return getattr(cfg, "family_name", "") == "deepseek_v4"
    except Exception:
        return False


def _is_hy3_model(model_key: str) -> bool:
    """Return True when the current registry contract resolves to Hy3."""
    try:
        from .model_config_registry import get_model_config_registry

        cfg = get_model_config_registry().lookup(_registry_model_key(model_key))
        return getattr(cfg, "family_name", None) == "hy_v3"
    except Exception:
        return False


def _normalize_hy3_reasoning_effort(
    effort: str | None,
    *,
    enable_thinking: bool | None,
) -> str | None:
    """Map public thinking controls to Hy3's template vocabulary.

    Hy3's template ignores ``enable_thinking`` directly. It opens the thinking
    rail only for ``reasoning_effort in {"low", "high"}``; invalid values
    silently fall back to ``no_think`` inside the template. Normalize here so
    UI/API "reasoning" modes cannot accidentally render the no-thinking rail.
    """
    if enable_thinking is False:
        return "no_think"

    if effort is not None:
        raw = str(effort).strip().lower()
        if raw in {"", "none", "off", "false", "instruct", "chat", "no_think"}:
            return "no_think"
        if raw in {"low", "high"}:
            return raw
        if raw in {"medium", "max", "reasoning", "thinking", "on", "true"}:
            return "high"

    if enable_thinking is True:
        return "high"
    return None


def _apply_hy3_reasoning_policy(
    chat_kwargs: dict,
    ct_kwargs: dict,
    *,
    model_key: str,
    enable_thinking: bool | None,
) -> None:
    """Inject Hy3's request-level ``reasoning_effort`` when needed."""
    if not _is_hy3_model(model_key):
        return
    requested = chat_kwargs.get("reasoning_effort") or ct_kwargs.get("reasoning_effort")
    normalized = _normalize_hy3_reasoning_effort(
        requested,
        enable_thinking=enable_thinking,
    )
    if normalized is None:
        return
    chat_kwargs["reasoning_effort"] = normalized
    ct_kwargs["reasoning_effort"] = normalized


def _hy3_prompt_starts_in_reasoning(
    *,
    model_key: str,
    enable_thinking: bool | None,
    ct_kwargs: dict,
) -> bool:
    """Hy3 opens ``<think>`` only for low/high reasoning effort."""
    if enable_thinking is False or not _is_hy3_model(model_key):
        return False
    effort = _normalize_hy3_reasoning_effort(
        ct_kwargs.get("reasoning_effort"),
        enable_thinking=enable_thinking,
    )
    return effort in {"low", "high"}


def _synthetic_mllm_think_prompt_starts(
    *,
    model_key: str,
    enable_thinking: bool | None,
    engine=None,
) -> bool:
    """Return True when vMLX opens a qwen3 think rail for a plain MLLM template.

    ZAYA1-VL's processor template can end with ``assistant: `` even though the
    model bundle declares qwen3 reasoning support. The MLLM wrapper opens
    ``<think>`` only for explicit thinking-on requests, so the streaming and
    non-streaming parsers must seed ``think_in_prompt`` for that same bounded
    case. This is not a generic VLM heuristic.
    """
    if enable_thinking is not True:
        return False
    if engine is not None and not bool(getattr(engine, "is_mllm", False)):
        return False
    try:
        from .model_config_registry import get_model_config_registry

        cfg = get_model_config_registry().lookup(_registry_model_key(model_key))
    except Exception:
        return False
    family = str(getattr(cfg, "family_name", "") or "").lower()
    if family not in {"zaya", "zaya1_vl", "zaya1-vl"}:
        return False
    if getattr(cfg, "supports_thinking", None) is False:
        return False
    if getattr(cfg, "think_in_template", False):
        return False
    return str(getattr(cfg, "reasoning_parser", "") or "").lower() == "qwen3"


def _resolve_repetition_penalty(
    request_value: float | None,
    model_name: str = "",
    *,
    enable_thinking: bool | None = None,
) -> float | None:
    """Resolve repetition penalty: request > explicit CLI/session > bundle > None.

    Returns None when nothing matches, which means "don't pass
    repetition_penalty at all to the engine" — mlx-lm then applies its
    implicit 1.0 (no penalty). When a value is returned, the caller should
    include it in chat_kwargs.
    """
    _bundle_path = _model_path or model_name

    if request_value is not None:
        return request_value
    if _default_repetition_penalty is not None:
        return _default_repetition_penalty
    for _key in _bundle_repetition_penalty_keys(
        _bundle_path,
        enable_thinking=enable_thinking,
    ):
        _v = _bundle_sampling_default(_bundle_path, _key)
        if _v is not None:
            return _v
    return None


def _set_resolved_repetition_penalty(
    target: dict,
    request_value: float | None,
    model_name: str = "",
    *,
    enable_thinking: bool | None = None,
) -> None:
    """Write the final repetition penalty into generation kwargs.

    Some families, notably DSV4, have mode-specific bundle defaults. API
    handlers initially build generic kwargs before family routing is finalized,
    so DSV4 blocks call this again after the final enable_thinking rail is
    known.
    """
    _rp = _resolve_repetition_penalty(
        request_value,
        model_name,
        enable_thinking=enable_thinking,
    )
    if _rp is not None:
        target["repetition_penalty"] = _rp
    else:
        target.pop("repetition_penalty", None)


def _log_resolved_sampling_kwargs(route: str, model_name: str, kwargs: dict) -> None:
    """Emit the live sampling kwargs that are about to enter generation."""
    keys = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "max_tokens",
        "enable_thinking",
        "reasoning_effort",
    )
    sample = {key: kwargs[key] for key in keys if key in kwargs}
    ct_kwargs = kwargs.get("chat_template_kwargs")
    if isinstance(ct_kwargs, dict) and ct_kwargs:
        sample["chat_template_kwargs"] = {
            key: ct_kwargs[key]
            for key in ("reasoning_effort", "thinking_budget", "enable_thinking")
            if key in ct_kwargs
        }
    logger.info(
        "Resolved sampling kwargs route=%s model=%s kwargs=%s",
        route,
        model_name,
        sample,
    )


def _compute_bypass_prefix_cache(request_obj) -> bool:
    """Read request-level cache bypass flag.

    A request bypasses every prefix-cache layer (paged, memory-aware,
    legacy prefix, disk L2, block disk, SSM companion, multimodal
    pixel_values) when the client sends either:
        cache_salt: "<any non-empty string>"
        skip_prefix_cache: true

    cache_salt is a vLLM-compatible namespacing field. For vMLX we treat
    any non-empty salt as "please give me fresh state" — no cache hits,
    no cache stores for this request. The salt itself is not stored; if
    users want namespaced caches within a run, they should call
    DELETE /v1/cache between runs instead.

    DSV4 note: DSV4 is not bypassed here. Its paged cache path stores a
    dedicated ``deepseek_v4`` composite block record (local SWA plus CSA/HCA
    compressor/indexer pool state) and block disk L2 round-trips that nested
    state. Only explicit request fields bypass it.
    """
    if request_obj is None:
        return False
    if getattr(request_obj, "skip_prefix_cache", None) is True:
        return True
    salt = getattr(request_obj, "cache_salt", None)
    if isinstance(salt, str) and salt:
        return True
    return False


_MULTIMODAL_CONTENT_TYPES = {
    "image_url",
    "image",
    "input_image",
    "video_url",
    "video",
    "input_video",
    "input_audio",
    "audio",
}


def _content_has_multimodal(content) -> bool:
    if isinstance(content, list):
        for part in content:
            if hasattr(part, "model_dump"):
                part = part.model_dump(exclude_none=True)
            elif hasattr(part, "dict"):
                part = {k: v for k, v in part.dict().items() if v is not None}
            if isinstance(part, dict) and part.get("type") in _MULTIMODAL_CONTENT_TYPES:
                return True
    return False


def _content_multimodal_summary(content) -> dict:
    """Return redacted media counts for a message/content-part list.

    This intentionally never logs URL strings or base64 payloads. The goal is
    to make exported user logs answer whether media reached the server and what
    route/runtime saw it, without leaking image/audio/video contents.
    """
    summary = {
        "total": 0,
        "types": {},
        "data_url": 0,
        "url_or_path": 0,
        "missing_source": 0,
    }
    if not isinstance(content, list):
        return summary

    def _as_clean_dict(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if hasattr(obj, "dict"):
            return {k: v for k, v in obj.dict().items() if v is not None}
        return obj if isinstance(obj, dict) else None

    def _source_for_part(part: dict):
        ptype = part.get("type")
        if ptype in ("image_url", "image"):
            src = part.get("image_url") or part.get("image")
            return src.get("url") if isinstance(src, dict) else src
        if ptype in ("video_url", "video"):
            src = part.get("video_url") or part.get("video")
            return src.get("url") if isinstance(src, dict) else src
        if ptype in ("input_image",):
            src = part.get("image_url") or part.get("url") or part.get("file_id")
            return src.get("url") if isinstance(src, dict) else src
        if ptype in ("input_video",):
            src = part.get("video_url") or part.get("url") or part.get("file_id")
            return src.get("url") if isinstance(src, dict) else src
        if ptype in ("input_audio", "audio"):
            src = part.get("input_audio") or part.get("audio")
            if isinstance(src, dict):
                return src.get("url") or ("<inline_audio_data>" if src.get("data") else None)
            return src
        return None

    for raw_part in content:
        part = _as_clean_dict(raw_part)
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype not in _MULTIMODAL_CONTENT_TYPES:
            continue
        summary["total"] += 1
        summary["types"][ptype] = int(summary["types"].get(ptype, 0)) + 1
        src = _source_for_part(part)
        if isinstance(src, str) and src.startswith("data:"):
            summary["data_url"] += 1
        elif src:
            summary["url_or_path"] += 1
        else:
            summary["missing_source"] += 1
    return summary


def _merge_multimodal_summaries(target: dict, child: dict) -> None:
    target["total"] += int(child.get("total") or 0)
    target["data_url"] += int(child.get("data_url") or 0)
    target["url_or_path"] += int(child.get("url_or_path") or 0)
    target["missing_source"] += int(child.get("missing_source") or 0)
    for key, value in (child.get("types") or {}).items():
        target["types"][key] = int(target["types"].get(key, 0)) + int(value)


def _messages_multimodal_summary(messages) -> dict:
    summary = {
        "total": 0,
        "types": {},
        "data_url": 0,
        "url_or_path": 0,
        "missing_source": 0,
        "roles": {},
        "messages": 0,
        "content_arrays": 0,
    }
    for msg in messages or []:
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump(exclude_none=True)
        elif hasattr(msg, "dict"):
            msg = {k: v for k, v in msg.dict().items() if v is not None}
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            summary["content_arrays"] += 1
        child = _content_multimodal_summary(content)
        if child.get("total"):
            summary["messages"] += 1
            role_key = str(role or "unknown")
            summary["roles"][role_key] = int(summary["roles"].get(role_key, 0)) + 1
            _merge_multimodal_summaries(summary, child)
    return summary


def _messages_have_multimodal(messages) -> bool:
    for msg in messages or []:
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump(exclude_none=True)
        elif hasattr(msg, "dict"):
            msg = {k: v for k, v in msg.dict().items() if v is not None}
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if _content_has_multimodal(content):
            return True
    return False


def _requested_modalities_from_summary(summary: dict) -> set[str]:
    types = set((summary.get("types") or {}).keys()) if isinstance(summary, dict) else set()
    modalities: set[str] = set()
    if types & {"image_url", "image", "input_image"}:
        modalities.add("image")
    if types & {"video_url", "video", "input_video"}:
        modalities.add("video")
    if types & {"input_audio", "audio"}:
        modalities.add("audio")
    return modalities


def _messages_requested_modalities(messages) -> set[str]:
    return _requested_modalities_from_summary(_messages_multimodal_summary(messages))


def _responses_input_multimodal_summary(input_data) -> dict:
    summary = {
        "total": 0,
        "types": {},
        "data_url": 0,
        "url_or_path": 0,
        "missing_source": 0,
        "items": 0,
        "content_arrays": 0,
    }
    if isinstance(input_data, str):
        return summary
    for item in input_data or []:
        if hasattr(item, "model_dump"):
            item = item.model_dump(exclude_none=True)
        elif hasattr(item, "dict"):
            item = {k: v for k, v in item.dict().items() if v is not None}
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in _MULTIMODAL_CONTENT_TYPES:
            child = _content_multimodal_summary([item])
            if child.get("total"):
                summary["items"] += 1
                _merge_multimodal_summaries(summary, child)
        content = item.get("content")
        if isinstance(content, list):
            summary["content_arrays"] += 1
            child = _content_multimodal_summary(content)
            if child.get("total"):
                summary["items"] += 1
                _merge_multimodal_summaries(summary, child)
    return summary


def _responses_input_has_multimodal(input_data) -> bool:
    if isinstance(input_data, str):
        return False
    for item in input_data or []:
        if hasattr(item, "model_dump"):
            item = item.model_dump(exclude_none=True)
        elif hasattr(item, "dict"):
            item = {k: v for k, v in item.dict().items() if v is not None}
        if not isinstance(item, dict):
            continue
        if item.get("type") in _MULTIMODAL_CONTENT_TYPES:
            return True
        if _content_has_multimodal(item.get("content")):
            return True
    return False


def _responses_input_requested_modalities(input_data) -> set[str]:
    return _requested_modalities_from_summary(_responses_input_multimodal_summary(input_data))


def _log_multimodal_request_shape(route: str, model_name: str, summary: dict) -> None:
    """Log redacted media request shape for user-exported diagnostics."""
    if not summary or int(summary.get("total") or 0) <= 0:
        return
    try:
        from .model_config_registry import get_model_config_registry

        cfg = get_model_config_registry().lookup(_model_path or model_name or "")
        family = getattr(cfg, "family_name", None)
        registry_is_mllm = bool(getattr(cfg, "is_mllm", False))
    except Exception:
        family = None
        registry_is_mllm = False
    engine_is_mllm = bool(
        getattr(_engine, "is_mllm", False) or getattr(_engine, "_is_mllm", False)
    ) if _engine is not None else None
    payload = {
        "route": route,
        "model": Path(str(model_name)).name if model_name else None,
        "family": family,
        "engine_is_mllm": engine_is_mllm,
        "registry_is_mllm": registry_is_mllm,
        "omni_modalities": _loaded_omni_modalities(),
        "media": summary,
    }
    logger.info("[MEDIA_DIAG] request_shape=%s", json.dumps(payload, sort_keys=True))


def _loaded_omni_modalities() -> list[str] | None:
    try:
        from .omni_multimodal import omni_multimodal_component_status
        _omni_path = _model_path or _model_name
        if _omni_path:
            status = omni_multimodal_component_status(_omni_path)
            if status.get("bundle_compatible"):
                return list(status.get("modalities") or ["text", "audio", "image"])
    except Exception:
        pass
    return None


def _bundle_declares_native_video(bundle_path: str | None) -> bool:
    """Return True when the loaded non-Omni VLM bundle declares video support.

    Do not infer video from tokenizer-only markers. ZAYA1-VL ships a
    ``<video>`` tokenizer token but its local processor/chat-template path only
    expands image placeholders, so advertising video there produces a runtime
    crash instead of usable video understanding.
    """
    cfg = _read_bundle_json(bundle_path, "config.json")
    if not cfg:
        return False
    if cfg.get("video_config") is not None:
        return True
    if Path(str(bundle_path or "")).joinpath("video_preprocessor_config.json").is_file():
        return True
    model_type = str(cfg.get("model_type") or "").lower()
    if model_type in {"gemma4", "gemma4_unified"}:
        proc = _read_bundle_json(bundle_path, "processor_config.json")
        return bool(
            cfg.get("video_token_id") is not None
            and isinstance(proc.get("video_processor") if proc else None, dict)
        )
    for obj in (cfg, cfg.get("text_config"), cfg.get("vision_config")):
        if not isinstance(obj, dict):
            continue
        for key in ("video_token_id", "video_token_index"):
            if obj.get(key) is not None:
                return True
    return False


def _bundle_supports_video_frame_fallback(bundle_path: str | None) -> bool:
    """Return True when video is safe to expose via sampled image frames.

    This is deliberately narrower than "has vision". ZAYA1-VL, for example,
    has an image path but must continue rejecting video because its template
    does not support the current frame-fallback route safely.
    """
    cfg = _read_bundle_json(bundle_path, "config.json")
    if not cfg:
        return False
    model_type = str(cfg.get("model_type") or "").lower()
    if model_type != "step3p7":
        return False
    return bool(
        cfg.get("image_token_id") is not None
        and isinstance(cfg.get("vision_config"), dict)
    )


def _bundle_declares_native_audio(bundle_path: str | None) -> bool:
    cfg = _read_bundle_json(bundle_path, "config.json")
    if not cfg:
        return False
    jang = _read_bundle_json(bundle_path, "jang_config.json")
    model_type = str(cfg.get("model_type") or "").lower()
    weight_format = str((jang or {}).get("weight_format") or "").lower()
    profile = str((jang or {}).get("profile") or "").lower()
    quant = (jang or {}).get("quantization") if isinstance(jang, dict) else {}
    selective = quant.get("selective_passthrough") if isinstance(quant, dict) else {}
    has_audio_safe_mxfp_repair = bool(
        model_type == "gemma4_unified"
        and weight_format == "mxfp8"
        and "attnfp16" in profile
        and isinstance(selective, dict)
        and selective.get("preserve_attention_fp16") is True
        and (
            os.environ.get("VMLX_ALLOW_EXPERIMENTAL_MXFP_AUDIO") == "1"
            or os.environ.get("VMLINUX_ALLOW_EXPERIMENTAL_MXFP_AUDIO") == "1"
        )
    )
    if model_type == "gemma4_unified" and (
        weight_format in {"mxfp4", "mxfp8"} or profile in {"mxfp4", "mxfp8"}
    ):
        if has_audio_safe_mxfp_repair:
            return True
        # The Gemma4 12B MXFP bundles route input_audio through the processor
        # and reach generation, but current live proof shows broken speech
        # quality (`The transcriptionno` / `Mario presents.` / refusals).
        # Keep server capabilities honest until repaired artifacts are
        # stamped/proven. A direct MXFP8 probe can sometimes produce recoverable
        # raw `Audio present.` content, but the API path is not stable enough to
        # advertise audio.
        return False
    if cfg.get("audio_config") is not None:
        return True
    if model_type == "gemma4_unified":
        return True
    for obj in (cfg, cfg.get("text_config")):
        if not isinstance(obj, dict):
            continue
        for key in ("audio_token_id", "audio_token_index"):
            if obj.get(key) is not None:
                return True
    return False


def _loaded_mllm_modalities() -> list[str] | None:
    engine = _engine
    engine_is_mllm = bool(
        getattr(engine, "is_mllm", False) or getattr(engine, "_is_mllm", False)
    ) if engine is not None else False
    if not engine_is_mllm:
        return None
    modalities = ["text", "vision"]
    if _bundle_declares_native_audio(_model_path or _model_name):
        modalities.append("audio")
    if _bundle_declares_native_video(_model_path or _model_name):
        modalities.append("video")
    elif _bundle_supports_video_frame_fallback(_model_path or _model_name):
        modalities.append("video")
    return modalities


def _loaded_runtime_modalities() -> list[str]:
    modalities = _loaded_omni_modalities()
    if modalities is not None:
        return modalities
    modalities = _loaded_mllm_modalities()
    if modalities is not None:
        return modalities
    return ["text"]


def _normalize_modality_set(modalities: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    normalized = {str(m).lower() for m in modalities or []}
    if "vision" in normalized:
        normalized.add("image")
    if "image" in normalized:
        normalized.add("vision")
    return normalized


def _reject_unsupported_multimodal(
    endpoint: str,
    requested_modalities: set[str] | None = None,
) -> None:
    if requested_modalities:
        supported = set(_loaded_runtime_modalities())
        unsupported = sorted(
            {str(m).lower() for m in requested_modalities}
            - _normalize_modality_set(supported)
        )
        if not unsupported:
            return
        if _loaded_runtime_modalities() == ["text"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{endpoint} received unsupported media modality "
                    f"{', '.join(unsupported)} because the loaded runtime is "
                    "text-only. Supported modalities: text."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"{endpoint} received unsupported media modality "
                f"{', '.join(unsupported)}. Supported modalities: "
                f"{', '.join(_loaded_runtime_modalities())}."
            ),
        )
    if _loaded_omni_modalities() is not None:
        # Omni requests should have been routed through OmniMultimodalDispatcher
        # before the text path. If dispatch failed, do not silently drop media.
        raise HTTPException(
            status_code=503,
            detail=(
                f"{endpoint} received multimodal input for an Omni bundle, but "
                "the Omni multimodal dispatcher did not complete. The request "
                "was rejected to avoid silently ignoring image/audio/video data."
            ),
        )
    raise HTTPException(
        status_code=400,
        detail=(
            f"{endpoint} received image/video/audio content, but the loaded "
            "runtime is text-only. Start a VLM/Omni session that reports "
            "vision/audio/video in /v1/models/{id}/capabilities, or send a "
            "text-only request. This guard prevents silent media drop."
        ),
    )


def _grouped_conv1d_layout_error_detail(exc: Exception) -> str | None:
    """Return an actionable message for escaped grouped Conv1d layout errors."""
    msg = str(exc)
    if (
        "Given groups=" not in msg
        or "weights of shape" not in msg
        or "input channels" not in msg
    ):
        return None
    return (
        "Grouped Conv1d layout mismatch in a hybrid SSM/Mamba model. "
        "This usually means the bundle was converted by an older JANG build "
        "that wrote Conv1d weights in HF layout instead of MLX layout. vMLX "
        "now applies runtime grouped-Conv1d layout backstops for JANG/JANGTQ "
        "load paths; if this error still appears, re-convert with a newer "
        "jang build and include config.json, jang_config.json, and the full "
        f"original error in the report. Original error: {msg}"
    )


def _generation_error_detail(exc: Exception, *, prefix: str = "Generation failed") -> str:
    detail = _grouped_conv1d_layout_error_detail(exc)
    if detail:
        return detail
    return f"{prefix}: {type(exc).__name__}: {exc}"


def _resolve_enable_thinking(
    request_value: bool | None,
    ct_kwargs: dict,
    tools_present: bool,
    model_key: str,
    engine=None,
    auto_detect: bool = False,
) -> bool | None:
    """Resolve enable_thinking using the shared precedence chain.

    Precedence: unsupported-family guard > per-request > chat_template_kwargs >
    explicit server default > Auto.

    Native API client defaults such as Ollama ``think:false`` must not force
    reasoning off globally. Omitted controls must also not force reasoning on:
    Auto is represented as ``None`` so the native tokenizer/template/runtime
    default decides. Explicit user/default settings and hard family contracts
    such as Ling/Bailing's supports_thinking=False still produce a concrete
    bool.
    """
    _mc = None
    try:
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(_registry_model_key(model_key))
    except Exception:
        _mc = None
    _family = getattr(_mc, "family_name", None) or getattr(_mc, "model_type", None)
    if not _family or str(_family).lower() == "unknown":
        _family = str(model_key or "")
    _family_l = str(_family).lower()
    if _mc is not None and getattr(_mc, "supports_thinking", None) is False:
        # The registry is the compatibility boundary. If a model family or
        # quant profile cannot safely expose a reasoning rail, stale UI state
        # and raw API kwargs must not resurrect it.
        return False
    if request_value is not None:
        return request_value
    if "enable_thinking" in ct_kwargs:
        return bool(ct_kwargs["enable_thinking"])
    if _default_enable_thinking is True:
        return True
    if _default_enable_thinking is False:
        return False

    default_hint = None
    try:
        default_hint = (getattr(_mc, "architecture_hints", None) or {}).get(
            "default_enable_thinking"
        )
    except Exception:
        default_hint = None
    if isinstance(default_hint, bool):
        return default_hint

    return None


# Global MCP manager
_mcp_manager = None
_mcp_policy = None


def _load_mcp_policy_from_env():
    """Build the effective MCP policy from per-session CLI/env values."""
    from vmlx_engine.mcp import MCPPolicy

    return MCPPolicy.from_values(
        enabled_servers=os.environ.get("VLLM_MLX_MCP_ENABLED_SERVERS"),
        disabled_servers=os.environ.get("VLLM_MLX_MCP_DISABLED_SERVERS"),
        enabled_tools=os.environ.get("VLLM_MLX_MCP_ENABLED_TOOLS"),
        disabled_tools=os.environ.get("VLLM_MLX_MCP_DISABLED_TOOLS"),
    )


def _tool_definition_name(tool: Any) -> str | None:
    """Return a function/tool name from either flat or nested API tool shapes."""
    if hasattr(tool, "function"):
        function = getattr(tool, "function", None)
        if isinstance(function, dict):
            return function.get("name")
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return tool.get("name")
    return None


def _effective_tools_for_tool_parsing(request: Any) -> Any:
    """Return the tool set actually shown to the model for parser schema gates."""
    if request is None:
        return None
    effective = getattr(request, "_vmlx_effective_tools", None)
    if effective is not None:
        return effective
    return getattr(request, "tools", None)


def _attach_effective_tools_for_tool_parsing(request: Any, tools: list[Any]) -> None:
    """Attach merged request+MCP tools without mutating the public API payload."""
    if request is None:
        return
    try:
        object.__setattr__(request, "_vmlx_effective_tools", list(tools or []))
    except Exception:
        logger.debug("Failed to attach effective tool schema set for parser context")


def _filter_tools_for_specific_choice(
    tools: list[Any],
    tool_choice: Any,
) -> list[Any]:
    """Apply explicit tool_choice filtering to MCP tools as well as user tools."""
    if not isinstance(tool_choice, dict):
        return tools
    function = tool_choice.get("function")
    target_name = function.get("name") if isinstance(function, dict) else None
    target_name = target_name or tool_choice.get("name")
    if not target_name:
        return tools
    filtered = [tool for tool in tools if _tool_definition_name(tool) == target_name]
    return filtered


def _drop_colliding_mcp_tools(
    mcp_tools: list[dict[str, Any]],
    request_tools: Any,
) -> list[dict[str, Any]]:
    """Keep request/native tools authoritative when names collide with MCP."""
    user_names = {
        name
        for tool in (request_tools or [])
        for name in [_tool_definition_name(tool)]
        if name
    }
    if not user_names:
        return mcp_tools
    return [tool for tool in mcp_tools if _tool_definition_name(tool) not in user_names]

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used
_embedding_lock: asyncio.Lock | None = (
    None  # Lazy-init to avoid binding to wrong event loop
)

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

# Reasoning parser (for models like Qwen3, DeepSeek-R1)
_reasoning_parser = None  # ReasoningParser instance when enabled

# Cache: does a model's template inject <think> even when enable_thinking=False?
# Some templates (e.g., MiniMax M2.5) unconditionally inject <think> regardless.
_template_always_thinks_cache: dict[str, bool] = {}
_template_starts_reasoning_cache: dict[tuple[str, bool], bool] = {}

# Tool call markers to detect in streaming output for buffering
_TOOL_CALL_MARKERS = [
    "<tool_call",
    "<tool_calls>",
    "<tool_call>",
    "<tool_sep>",
    "<arg_key>",
    "<arg_value>",
    "<zyphra_tool_call",
    "<|tool_call>",  # Gemma 4 native tool call format
    "<|tool_call|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "[TOOL_CALLS]",
    "<function",
    "<function=",
    "<minimax:tool_call>",
    "[Calling tool:",
    "<|recipient|>",
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "<\uff5ctool\u2581calls\u2581begin\uff5c>",  # DeepSeek Unicode variant (U+FF5C, U+2581)
    "<\uff5ctool\u2581call\u2581begin\uff5c>",  # DeepSeek singular begin (GLM-5.1 emits this)
    "<｜DSML｜tool",  # DSV4 can flush this short prefix before `_c...`
    "<｜DSML｜tool_c",  # DSV4 DSML wrapper prefix before inner invoke
    "<｜DSML｜invoke",  # DSV4 native tool-call format
    "<|python_tag|>",  # Llama 3.1+ code interpreter / tool call
    "```tool_code",  # Gemma 3 / 3n function-call code block (Google docs format)
]

_TOOL_MARKUP_RESIDUE_PATTERNS = []
_TOOL_MARKUP_STRIP_TO_EOL_MARKERS = {
    "<tool_call",
    "<zyphra_tool_call",
    "<function",
    "<function=",
}
for _marker in _TOOL_CALL_MARKERS:
    if _marker == "```tool_code":
        _TOOL_MARKUP_RESIDUE_PATTERNS.append(
            re.escape(_marker) + r"(?:\r?\n[\s\S]*?(?:```|$))"
        )
    if _marker.startswith("<") and not _marker.endswith(">"):
        if _marker in _TOOL_MARKUP_STRIP_TO_EOL_MARKERS:
            _TOOL_MARKUP_RESIDUE_PATTERNS.append(
                re.escape(_marker) + r"[^>\n]*(?:>|$)"
            )
        _TOOL_MARKUP_RESIDUE_PATTERNS.append(re.escape(_marker) + r"[^>\n]*>")
    elif _marker == "[Calling tool:":
        _TOOL_MARKUP_RESIDUE_PATTERNS.append(re.escape(_marker) + r"[^\]\n]*(?:\])?")
    _TOOL_MARKUP_RESIDUE_PATTERNS.append(re.escape(_marker))
_TOOL_MARKUP_RESIDUE_PATTERNS.extend(
    [
        r"</?｜DSML｜(?:tool_calls?|tool_call_type|tool_c|tool|invoke|parameter)?[^>\n]*>?",
        r"</?tool_calls?>?",
        r"</?tool_call>?",
        r"</?arg_key>?",
        r"</?arg_value>?",
        r"</?minimax:tool_call>?",
        r"</?zyphra_tool_call[^>\n]*>?",
        r"</?function[^>\n]*>?",
        r"</?\|tool_calls_section_begin\|>?",
        r"</?\|tool_call_begin\|>?",
        r"</?\|tool_call\|>?",
        r"</?\|tool_call_start\|>?",
        r"</?\|tool_call_end\|>?",
        r"</?｜tool▁calls?▁begin｜>?",
        r"```tool_code",
    ]
)
_TOOL_MARKUP_RESIDUE_RE = re.compile(
    "|".join(_TOOL_MARKUP_RESIDUE_PATTERNS),
    re.DOTALL,
)

_VISUAL_GROUNDING_MARKERS = (
    "<|point_start|>",
    "<|point_end|>",
    "<|box_start|>",
    "<|box_end|>",
)
_VISUAL_GROUNDING_SPAN_RE = re.compile(
    r"<\|(?:point|box)_start\|>[\s\S]*?(?:<\|(?:point|box)_end\|>|$)",
    re.DOTALL,
)
_VISUAL_GROUNDING_END_RE = re.compile(r"<\|(?:point|box)_end\|>")


def _strip_partial_visual_grounding_suffix(text: str) -> str:
    if not text:
        return text
    for marker in _VISUAL_GROUNDING_MARKERS:
        max_prefix = min(len(marker) - 1, len(text))
        for n in range(max_prefix, 0, -1):
            if text.endswith(marker[:n]):
                return text[:-n]
    return text


def _strip_visual_grounding_markup_for_display(text: str) -> str:
    """Remove VL point/box control spans from user-visible assistant text."""
    if not text:
        return text
    cleaned = _VISUAL_GROUNDING_SPAN_RE.sub("", text)
    cleaned = _VISUAL_GROUNDING_END_RE.sub("", cleaned)
    return _strip_partial_visual_grounding_suffix(cleaned)


def _visual_grounding_display_delta(
    accumulated_text: str,
    streamed_display_text: str,
) -> str | None:
    """Return the next visible delta after hiding VL point/box markup."""
    cleaned = _strip_visual_grounding_markup_for_display(accumulated_text)
    if not cleaned:
        return None
    if streamed_display_text and cleaned.startswith(streamed_display_text):
        delta = cleaned[len(streamed_display_text) :]
    elif streamed_display_text == cleaned:
        delta = ""
    else:
        delta = cleaned
    return delta if delta else None


def _strip_tool_markup_residue_for_display(text: str) -> str:
    """Remove native tool wrapper fragments from visible fallback text.

    This is intentionally used only after tool-call buffering/cleanup has
    identified a tool-markup path. It catches short mid-stream fragments like
    DSV4's ``<｜DSML｜tool`` and Hy3 envelope tags without changing ordinary
    assistant prose globally.
    """
    if not text:
        return text
    cleaned = _TOOL_MARKUP_RESIDUE_RE.sub("", text)
    return re.sub(r"[ \t]*\n[ \t]*\n[ \t]*", "\n", cleaned).strip()


def _has_tool_marker_or_partial_suffix(text: str) -> bool:
    """Return True for full native tool markers or a marker split at stream tail."""
    if not text:
        return False
    for marker in _TOOL_CALL_MARKERS:
        if marker in text:
            return True
    for marker in _TOOL_CALL_MARKERS:
        max_prefix = min(len(marker) - 1, len(text))
        for n in range(max_prefix, 0, -1):
            if text.endswith(marker[:n]):
                return True
    return False


def _rendered_prompt_starts_in_reasoning(rendered: str, marker: str = "__test__") -> bool:
    """Return whether a rendered assistant prefix leaves ``<think>`` open."""
    after_user = str(rendered).rsplit(marker, 1)[-1]
    # A closed sentinel (`<think></think>`) means "thinking produced
    # nothing; visible answer starts now", not "parser starts in reasoning".
    cleaned = re.sub(r"<think>\s*</think>", "", after_user)
    return "<think>" in cleaned and "</think>" not in cleaned.split("<think>", 1)[1]


def _template_always_thinks(tokenizer, model_name: str) -> bool:
    """Check if model's template injects <think> even when enable_thinking=False.

    For templates that ignore the flag, we keep think_in_template=True so the
    streaming parser correctly classifies reasoning (suppress_reasoning hides it).
    Results are cached per model name since the template doesn't change at runtime.
    """
    if model_name in _template_always_thinks_cache:
        return _template_always_thinks_cache[model_name]

    # Qwen text models honor enable_thinking=False via the tokenizer template.
    # But Qwen VL models use a processor that ignores enable_thinking — the
    # MLLM _apply_chat_template strips <think> tags after rendering, so from
    # the server's perspective they DON'T always-think (the strip handles it).
    if "qwen" in model_name.lower():
        _template_always_thinks_cache[model_name] = False
        return False

    result = False
    try:
        test_msgs = [{"role": "user", "content": "__test__"}]
        try:
            rendered = tokenizer.apply_chat_template(
                test_msgs,
                enable_thinking=False,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            # Tokenizer doesn't accept enable_thinking — render without it
            # and check if <think> is always present (it's an always-thinking template)
            rendered = tokenizer.apply_chat_template(
                test_msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
        # Check if an UNCLOSED <think> appears after the user message.
        # Templates that output <think></think> (empty, closed) ARE honoring
        # enable_thinking=False — the model won't reason. Only flag templates
        # that inject an open <think> without immediate </think> (those truly
        # ignore the flag and the model will reason regardless).
        after_user = rendered.rsplit("__test__", 1)[-1]
        has_think = "<think>" in after_user
        # Strip empty closed think blocks such as <think></think> and
        # <think>\n</think>\n\n. Those are proper "no thinking" sentinels,
        # not evidence that the template ignores enable_thinking=False.
        cleaned = re.sub(r"<think>\s*</think>", "", after_user)
        result = "<think>" in cleaned  # still has an unclosed <think>?
        if result:
            logger.info(
                f"Template for {model_name} always injects <think> "
                "(ignores enable_thinking=False)"
            )
        elif has_think:
            logger.info(
                f"Template for {model_name} outputs <think></think> with "
                "enable_thinking=False (properly handled, not always-thinks)"
            )
    except Exception as e:
        logger.debug(f"_template_always_thinks check failed for {model_name}: {e}")

    _template_always_thinks_cache[model_name] = result
    return result


def _template_starts_reasoning(
    tokenizer,
    model_name: str,
    enable_thinking: bool,
) -> bool:
    """Check whether this request's rendered assistant prompt opens <think>."""
    key = (model_name, bool(enable_thinking))
    if key in _template_starts_reasoning_cache:
        return _template_starts_reasoning_cache[key]

    result = False
    try:
        test_msgs = [{"role": "user", "content": "__test__"}]
        try:
            rendered = tokenizer.apply_chat_template(
                test_msgs,
                enable_thinking=bool(enable_thinking),
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                test_msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
        result = _rendered_prompt_starts_in_reasoning(rendered)
    except Exception as exc:
        logger.debug(
            "_template_starts_reasoning check failed for %s: %s",
            model_name,
            exc,
        )

    _template_starts_reasoning_cache[key] = result
    return result


def _apply_tokenizer_thinking_vocab_fallback(
    think_in_template: bool,
    *,
    reasoning_parser,
    tokenizer,
    model_config,
) -> bool:
    """Use tokenizer thinking-token fallback only for unknown model configs.

    Several known families expose ``<think>`` tokens in the tokenizer vocabulary
    without leaving the rendered assistant prefix inside a reasoning block. For
    those families, the registry's explicit ``think_in_template=False`` verdict
    is authoritative; otherwise streaming can suppress plain visible text as
    hidden reasoning.
    """
    if think_in_template or not reasoning_parser:
        return bool(think_in_template)

    family_name = str(getattr(model_config, "family_name", "") or "")
    if family_name and family_name != "unknown":
        return False

    try:
        if getattr(tokenizer, "has_thinking", False):
            logger.info("Detected think_in_template from tokenizer vocabulary")
            return True
    except Exception:
        pass
    return False


def _engine_prompt_starts_in_reasoning(
    tokenizer,
    *,
    model_name: str,
    enable_thinking: bool,
    family_name: str | None = None,
    tools_present: bool = False,
) -> bool:
    """Check the parser seed against the engine's final prompt contract.

    Some tokenizers, notably MiniMax M2.x, render an open ``<think>`` even when
    ``enable_thinking=False``. The engine then closes that block for no-tool
    thinking-off requests before generation. Streaming parsers must seed from
    that post-render contract; otherwise visible tokens like ``Paris`` are
    misclassified as hidden reasoning and suppressed.
    """
    test_msgs = [{"role": "user", "content": "__test__"}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    try:
        rendered = tokenizer.apply_chat_template(
            test_msgs,
            enable_thinking=bool(enable_thinking),
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            test_msgs,
            add_generation_prompt=True,
            tokenize=False,
        )

    if enable_thinking is False:
        try:
            from .utils.chat_template_kwargs import ensure_thinking_off_sentinel

            rendered = ensure_thinking_off_sentinel(
                str(rendered),
                family_name=family_name,
                model_name=model_name,
                tools_present=tools_present,
            )
        except Exception as exc:
            logger.debug(
                "thinking-off prompt contract check failed for %s: %s",
                model_name,
                exc,
            )

    return _rendered_prompt_starts_in_reasoning(str(rendered))


# Cache: does a model's template complete thinking in the generation prompt?
# Some templates (e.g., Nemotron CRACK) use an "S5 seed" approach:
# generation prompt = <think>\nOK.\n</think>\n — the model output is plain text.
_template_completes_thinking_cache: dict[str, bool] = {}


def _template_completes_thinking(tokenizer, model_name: str) -> bool:
    """Check if template completes the thinking block inside the generation prompt.

    Some models use an "S5 seed" approach where the generation prompt includes
    a complete <think>...</think> block (e.g., <think>\\nOK.\\n</think>\\n).
    The model output starts AFTER the closed thinking block and is plain text.

    When detected, think_in_template should be False because the model output
    won't contain any <think>/</think> tags — they're all in the prompt.
    """
    if model_name in _template_completes_thinking_cache:
        return _template_completes_thinking_cache[model_name]

    result = False
    try:
        import re

        test_msgs = [{"role": "user", "content": "__test__"}]
        try:
            rendered = tokenizer.apply_chat_template(
                test_msgs,
                enable_thinking=True,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            # Tokenizer doesn't accept enable_thinking — render without it
            rendered = tokenizer.apply_chat_template(
                test_msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
        # Check if generation prompt ends with </think> (+ optional whitespace)
        # This means thinking is completed in the prompt, model outputs plain text
        if re.search(r"</think>\s*$", rendered):
            result = True
            logger.info(
                f"Template for {model_name} completes thinking in prompt "
                "(S5 seed — model output is plain text, disabling think_in_template)"
            )
    except Exception as e:
        logger.debug(f"_template_completes_thinking check failed for {model_name}: {e}")

    _template_completes_thinking_cache[model_name] = result
    return result


# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_call_parser_disabled_explicitly: bool = False

# reasoning_effort → token budget mapping (mirrors OpenAI o-series behavior).
# DSV4-Flash adds a "max" effort tier beyond OpenAI's low/medium/high
# (research/DSV4-RUNTIME-ARCHITECTURE.md §4) — allow extended budgets so
# the token ceiling doesn't truncate deep reasoning chains mid-thought.
_EFFORT_THINKING_BUDGET = {"low": 1024, "medium": 8192, "high": 32768, "max": 131072}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager

    caffeinate_process = None
    if platform.system() == "Darwin":
        try:
            # -d: Prevent display sleep
            # -s: Prevent system sleep
            # -i: Prevent idle sleep
            # -w: Wait for process to exit
            caffeinate_process = subprocess.Popen(
                ["caffeinate", "-s", "-i", "-w", str(os.getpid())]
            )
            logger.info(f"Started caffeinate system wake lock for PID {os.getpid()}")
        except Exception as e:
            logger.warning(f"Failed to start caffeinate process: {e}")

    # Startup: Start engine if loaded (needed for BatchedEngine in uvicorn's event loop)
    if _engine is not None and hasattr(_engine, "_loaded") and not _engine._loaded:
        await _engine.start()

        # Apply chat template override for BatchedEngine (SimpleEngine does it in load_model)
        try:
            from .model_config_registry import get_model_config_registry

            _mc = get_model_config_registry().lookup(_model_path or _model_name or "")
            if _mc.chat_template_custom and _engine.tokenizer:
                _engine.tokenizer.chat_template = _mc.chat_template_custom
                logger.info(
                    f"Applied custom chat template for {_mc.family_name} model (batched)"
                )
        except Exception as e:
            logger.warning(f"Failed to apply custom chat template (batched): {e}")

    # Apply Flash MoE for BatchedEngine (model just loaded via start() above).
    # SimpleEngine Flash MoE is applied in load_model() where it starts synchronously.
    if _flash_moe_enabled and _engine is not None:
        _apply_flash_moe_patching()

    # Apply JIT compilation for BatchedEngine (which just started above).
    # SimpleEngine JIT is applied in load_model() where it starts synchronously.
    if _enable_jit and _engine is not None:
        _apply_jit_compilation()

    # Initialize MCP from explicit env/CLI config or the standard mcp.json/yaml
    # discovery paths. Failure should not crash model serving.
    try:
        mcp_config = _discover_mcp_config_for_startup()
        if mcp_config:
            await init_mcp(mcp_config)
    except Exception as e:
        logger.error(
            f"Failed to initialize MCP — continuing without tool support: {e}"
        )

    yield

    # Shutdown: Close MCP connections, stop engine, and kill caffeinate
    # Wrap each in timeout to prevent hanging on quit
    if _mcp_manager is not None:
        try:
            await asyncio.wait_for(_mcp_manager.stop(), timeout=10)
            logger.info("MCP manager stopped")
        except Exception as e:
            logger.warning(f"MCP shutdown error (continuing): {e}")
    if _engine is not None:
        try:
            await asyncio.wait_for(_engine.stop(), timeout=10)
            logger.info("Engine stopped")
        except Exception as e:
            logger.warning(f"Engine shutdown error (continuing): {e}")
    if _image_gen is not None and _image_gen.is_loaded:
        try:
            await asyncio.wait_for(_run_image_gen_call(_image_gen.unload), timeout=10)
            logger.info("Image engine unloaded")
        except Exception as e:
            logger.warning(f"Image engine shutdown error (continuing): {e}")
    _shutdown_image_gen_executor()
    if caffeinate_process is not None:
        try:
            caffeinate_process.terminate()
            caffeinate_process.wait(timeout=5)
            logger.info("Caffeinate process terminated")
        except Exception:
            caffeinate_process.kill()


app = FastAPI(
    title="vmlx-engine API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version=__import__("vmlx_engine").__version__,
    lifespan=lifespan,
)

security = HTTPBearer(auto_error=False)


@app.middleware("http")
async def track_request_time(request: Request, call_next):
    """Track last request time, JIT wake from sleep, gate text endpoints on image servers."""
    global _last_request_time, _standby_state, _inference_endpoints, _wake_timeout
    path = request.url.path
    # Identify actual inference endpoints (not metadata like /v1/models, /v1/cache, /v1/mcp/tools)
    # This list drives BOTH idle timer reset AND JIT wake
    # Cancel endpoints should NOT trigger JIT wake or reset idle timer —
    # they are lightweight checks, not inference requests.
    is_cancel = "/cancel" in path
    is_inference = not is_cancel and any(
        path.startswith(p) for p in _inference_endpoints
    )
    # Update last request time for inference only — metadata queries shouldn't keep model awake
    if is_inference:
        _last_request_time = time.time()

    # JIT wake: if in standby and an inference request comes in, auto-wake first
    if _standby_state is not None and is_inference:
        global _wake_lock
        if _wake_lock is None:
            _wake_lock = asyncio.Lock()

        async with _wake_lock:
            # Re-check after acquiring lock (another request may have woken us)
            if _standby_state is None:
                pass  # Already awake — proceed
            else:
                logger.info(
                    f"JIT wake: request to {path} while in {_standby_state} sleep"
                )
                try:
                    from starlette.responses import JSONResponse

                    wake_result = await admin_wake()
                    if isinstance(wake_result, dict) and wake_result.get("error"):
                        return JSONResponse(
                            status_code=503,
                            content={
                                "error": {
                                    "message": f"Failed to wake model: {wake_result['error']}",
                                    "type": "server_error",
                                }
                            },
                        )
                    # For deep sleep, model takes time to reload — wait up to `_wake_timeout` seconds
                    # (large models like 60GB+ JANG MoE can take 30-60s to mmap load)
                    max_iterations = _wake_timeout * 2
                    for i in range(max_iterations):
                        if _standby_state is None:
                            break
                        if i > 0 and i % 20 == 0:
                            logger.info(
                                f"JIT wake: still loading after {i * 0.5:.0f}s..."
                            )
                        await asyncio.sleep(0.5)
                    if _standby_state is not None:
                        return JSONResponse(
                            status_code=503,
                            content={
                                "error": {
                                    "message": f"Model still loading after {_wake_timeout}s wake timeout",
                                    "type": "server_error",
                                }
                            },
                        )
                except Exception as e:
                    logger.error(f"JIT wake failed: {e}")
                    from starlette.responses import JSONResponse

                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": {
                                "message": f"JIT wake failed: {e}",
                                "type": "server_error",
                            }
                        },
                    )

    # Gate: image servers only serve /health, /v1/models, /v1/images/*
    if _model_type == "image" and path.startswith("/v1/"):
        allowed_image_paths = ["/v1/images/", "/v1/models"]
        if not any(path.startswith(p) for p in allowed_image_paths):
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "This is an image server. "
                        "Text endpoints are not available. "
                        "Use /v1/images/generations or /v1/images/edits.",
                        "type": "invalid_request_error",
                    }
                },
            )

    response = await call_next(request)
    return response


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0  # 1 minute window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for client.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            # Clean old requests outside window
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > window_start
            ]

            # Check rate limit
            if len(self._requests[client_id]) >= self.requests_per_minute:
                # Calculate retry-after
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_size - current_time) + 1
                return False, max(1, retry_after)

            # Record this request
            self._requests[client_id].append(current_time)
            return True, 0


# Global rate limiter (disabled by default)
_rate_limiter = RateLimiter(requests_per_minute=60, enabled=False)

# Settings configured via CLI (set in cli.py serve_command)
_log_level: str = "INFO"
_allowed_origins: str = "*"
_enable_jit: bool = False
_jang_metadata: dict | None = None  # Cached at model load time for /health
_model_type: str = "text"  # "text" or "image" — auto-detected from model directory
_image_quantize: int | None = None  # Image model quantization bits (set by cli.py)
_image_lora_paths: list[str] = []
_image_lora_scales: list[float] = []


class _CompiledModuleProxy:
    """Callable mx.compile proxy that preserves original module attributes."""

    def __init__(self, original, compiled):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_compiled", compiled)

    def __call__(self, *args, **kwargs):
        return self._compiled(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __setattr__(self, name, value):
        if name in {"_original", "_compiled"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._original, name, value)


_JIT_UNTRACEABLE_CACHE_CLASS_NAMES = frozenset(
    {
        "ArraysCache",
        "BatchMambaCache",
        "MambaCache",
    }
)


def _jit_untraceable_cache_class_names(cache_obj):
    """Return known Python cache containers that mx.compile cannot trace."""
    found: set[str] = set()
    seen: set[int] = set()

    def visit(obj):
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        class_name = type(obj).__name__
        if class_name in _JIT_UNTRACEABLE_CACHE_CLASS_NAMES:
            found.add(class_name)

        if isinstance(obj, dict):
            children = obj.values()
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = obj
        else:
            children = []

        for child in children:
            visit(child)

        try:
            concrete_attrs = vars(obj)
        except TypeError:
            concrete_attrs = {}

        for attr in ("caches", "cache", "kv_cache", "ssm_cache", "mamba_cache"):
            if attr not in concrete_attrs:
                continue
            child = concrete_attrs[attr]
            if child is not obj:
                visit(child)

    visit(cache_obj)
    return tuple(sorted(found))


def _jit_probe_untraceable_cache_names(*owners):
    for owner in owners:
        if owner is None:
            continue
        make_cache = getattr(owner, "make_cache", None)
        if make_cache is None:
            continue
        try:
            names = _jit_untraceable_cache_class_names(make_cache())
        except Exception as cache_probe_err:
            logger.debug(
                f"JIT: cache-shape probe failed for {type(owner).__name__} "
                f"({cache_probe_err}) - proceeding"
            )
            continue
        if names:
            return names
    return ()


def _apply_jit_compilation():
    """Apply mx.compile to the model forward pass for JIT-optimized inference.

    Wraps the model's __call__ with mx.compile for Metal kernel fusion.
    Falls back gracefully if compilation fails (some models have dynamic
    shapes or operations that mx.compile cannot handle).
    """
    global _engine
    if _engine is None:
        return

    # Flash MoE patches MoE blocks with Python loops + disk I/O that mx.compile
    # cannot trace. Warmup pass would bake specific experts into the compiled
    # graph, and subsequent requests with different experts would produce wrong
    # outputs. Skip JIT when Flash MoE is active.
    if _flash_moe_enabled and _flash_moe_loader is not None:
        logger.info(
            "JIT: Skipping mx.compile — Flash MoE is active. "
            "Flash MoE's on-demand expert loading is incompatible with JIT tracing."
        )
        return

    # Distributed pipeline parallelism splits model layers across worker nodes;
    # the coordinator only ever sees a subset (or none) of the layer objects
    # that JIT would trace. Compiling on the coordinator bakes a graph that
    # doesn't match what any worker actually executes. CLI guard already blocks
    # this combination — defensive skip here covers programmatic callers too.
    if _distributed_enabled and _distributed_coordinator is not None:
        logger.info(
            "JIT: Skipping mx.compile — distributed pipeline parallelism is active. "
            "Each worker owns a distinct layer range, so coordinator-side tracing "
            "is unsafe."
        )
        return

    try:
        _jit_scheduler = _get_scheduler()
        if getattr(_jit_scheduler, "_uses_dsv4_cache", False) is True:
            logger.info(
                "JIT: Skipping mx.compile — DeepSeek-V4 native composite cache "
                "(SWA+CSA/HCA) is active. The cache state is path-dependent and "
                "must stay on the uncompiled scheduler path."
            )
            return
    except Exception as _dsv4_jit_check_err:
        logger.debug(f"JIT: DSV4-cache presence check failed ({_dsv4_jit_check_err}) — proceeding")

    # GH issue #66: TurboQuantKVCache is a custom non-MLX class. mx.compile()
    # rejects any function arg that isn't an mx.array / int / float / str / None,
    # so the very first prefill on a JIT-compiled JANG model raises:
    #   ValueError: [compile] Function arguments must be trees of arrays or
    #   constants ... received type jang_tools.turboquant.cache.TurboQuantKVCache
    # Detect TQ via the patched make_cache name and skip JIT for those models.
    # This restores the JANG model load path without dropping JIT for non-TQ
    # models that legitimately benefit from it.
    try:
        _eng_model = getattr(_engine, "_model", None) or getattr(_engine, "model", None)
        if _eng_model is not None:
            _lm_for_tq = getattr(_eng_model, "language_model", None) or _eng_model
            _make_cache = getattr(_lm_for_tq, "make_cache", None)
            _make_cache_name = getattr(_make_cache, "__name__", "") if _make_cache else ""
            if _make_cache_name in ("_turboquant_make_cache", "_tq_make_cache"):
                logger.info(
                    "JIT: Skipping mx.compile — TurboQuantKVCache is active. "
                    "mx.compile() cannot trace custom cache objects (issue #66). "
                    "Use --kv-cache-quantization none if you need JIT for this model."
                )
                return
    except Exception as _tq_check_err:
        logger.debug(f"JIT: TQ-presence check failed ({_tq_check_err}) — proceeding")

    try:
        import mlx.core as mx

        # Get the inner model object — SimpleEngine and BatchedEngine both store it
        model_obj = getattr(_engine, "_model", None)
        if model_obj is None:
            model_obj = getattr(_engine, "model", None)
        if model_obj is None:
            logger.warning("JIT: Could not find model object on engine — skipping")
            return

        # Issue #56 Bug 2 — JIT + MLLM: mlx_vlm's generate() path accesses
        # `model.config.model_type`, `model.config.image_token_index`, and
        # `model.config.eos_token_id` directly on the top-level VLM wrapper.
        # If we mx.compile the VLM wrapper itself, it becomes an mlx.gc_func
        # and those attribute reads raise `AttributeError: 'mlx.gc_func' object
        # has no attribute 'config'`. For VLM engines we deliberately SKIP
        # top-level compilation and only compile the inner language_model.model
        # (the pure transformer), which mlx_vlm never introspects.
        is_mllm_engine = bool(getattr(_engine, "is_mllm", False) or getattr(_engine, "_is_mllm", False))
        if is_mllm_engine:
            # For VLM: compile `language_model.model` (inner transformer) if
            # available, leave the wrapper alone so .config survives.
            vlm_model = model_obj if hasattr(model_obj, "language_model") else getattr(model_obj, "model", None)
            language_model = getattr(vlm_model, "language_model", None) if vlm_model is not None else None
            inner_transformer = getattr(language_model, "model", None) if language_model is not None else None
            if inner_transformer is None or not callable(inner_transformer):
                logger.info("JIT: MLLM model has no inner language_model.model to compile — skipping JIT safely")
                return
            untraceable_cache_names = _jit_probe_untraceable_cache_names(
                language_model,
                vlm_model,
                model_obj,
            )
            if untraceable_cache_names:
                logger.info(
                    "JIT: Skipping mx.compile - MLLM hybrid cache contains "
                    f"{', '.join(untraceable_cache_names)}. mx.compile() cannot "
                    "trace these Python cache objects."
                )
                return
            logger.info("JIT: Applying mx.compile to language_model.model (VLM wrapper preserved for .config access)")
            # vmlx#83: mx.compile() is LAZY — it returns a wrapper that only
            # triggers tracing on the first forward pass. If that forward
            # pass raises (e.g. dynamic shape, TurboQuantKVCache despite the
            # upstream guard, unsupported op), we must RESTORE the original
            # uncompiled model so subsequent requests still work.
            _pre_compile_backup_vlm = inner_transformer  # for rollback
            try:
                compiled = mx.compile(inner_transformer)
                compiled_proxy = _CompiledModuleProxy(inner_transformer, compiled)
                language_model.model = compiled_proxy
                replaced = language_model.model is compiled_proxy
            except Exception as _vlm_jit_err:
                logger.warning(f"JIT: VLM language_model compile failed, running without JIT: {_vlm_jit_err}")
                return
        else:
            # LLM engine — compile the full inner module as before
            inner = getattr(model_obj, "model", model_obj)
            if inner is None or not callable(inner):
                logger.warning("JIT: Model object is not callable — skipping")
                return
            untraceable_cache_names = _jit_probe_untraceable_cache_names(
                inner,
                model_obj,
            )
            if untraceable_cache_names:
                logger.info(
                    "JIT: Skipping mx.compile - text hybrid cache contains "
                    f"{', '.join(untraceable_cache_names)}. mx.compile() cannot "
                    "trace these Python cache objects."
                )
                return

            logger.info("JIT: Applying mx.compile to model forward pass...")
            # vmlx#83: stash the pre-compile references so the warmup-failure
            # branch below can restore them. Without rollback, a lazy-compile
            # error during warmup leaves the model in a broken state where
            # every subsequent request fails silently (reporter's symptom).
            _pre_compile_inner = inner
            _pre_compile_backup_vlm = None
            try:
                compiled = mx.compile(inner)
            except Exception as _llm_jit_err:
                logger.warning(
                    f"JIT: mx.compile(inner) raised, running without JIT: {_llm_jit_err}"
                )
                return

            # Replace in-place on the wrapper and verify
            replaced = False
            if hasattr(model_obj, "model") and model_obj.model is inner:
                model_obj.model = compiled
                replaced = model_obj.model is compiled
            elif hasattr(_engine, "_model"):
                _engine._model = compiled
                replaced = _engine._model is compiled
            elif hasattr(_engine, "model"):
                _engine.model = compiled
                replaced = _engine.model is compiled

        if replaced:
            logger.info("JIT: mx.compile applied successfully — running warmup pass")
            # Warmup: 1-token forward pass to page in model weights and compile
            # Metal shaders. Without this, the first real request simultaneously
            # pages ~30GB+ of mmap weights AND allocates cache/activations,
            # causing Metal OOM on memory-constrained machines (e.g. 122B on 48GB).
            # This is ALSO where a lazy-compile error first surfaces (vmlx#83).
            try:
                lm = getattr(model_obj, "language_model", model_obj)
                # VLM wrappers commonly expose cache construction on the
                # language model, not the top-level multimodal wrapper. The
                # warmup must use the same cache shape real generation will
                # pass, otherwise mx.compile can appear healthy at startup
                # and then fail every request on ArraysCache/MambaCache.
                cache_owner = (
                    model_obj
                    if hasattr(model_obj, "make_cache")
                    else lm
                    if hasattr(lm, "make_cache")
                    else None
                )
                warmup_cache = cache_owner.make_cache() if cache_owner is not None else None
                warmup_input = mx.array([[0]])  # Single dummy token
                if warmup_cache is not None:
                    lm(warmup_input, cache=warmup_cache)
                else:
                    lm(warmup_input)
                mx.synchronize()  # Force Metal to finish
                del warmup_cache
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                logger.info(
                    "JIT: Warmup complete — model weights paged in, Metal shaders compiled"
                )
            except Exception as warmup_err:
                # vmlx#83 fix: before this, a warmup failure silently left
                # the model set to the broken compiled fn — every real
                # request then failed with no fallback. Now we roll back
                # the mutation so the model reverts to its pre-compile
                # uncompiled form, same state as "JIT disabled".
                logger.warning(
                    f"JIT: warmup raised ({warmup_err}) — rolling back "
                    f"mx.compile (vmlx#83 fallback)"
                )
                try:
                    if is_mllm_engine:
                        language_model.model = _pre_compile_backup_vlm
                    else:
                        if hasattr(model_obj, "model"):
                            model_obj.model = _pre_compile_inner
                        elif hasattr(_engine, "_model"):
                            _engine._model = _pre_compile_inner
                        elif hasattr(_engine, "model"):
                            _engine.model = _pre_compile_inner
                    logger.info(
                        "JIT: rollback complete — running without JIT "
                        "(first request will NOT be slower because JIT is off)"
                    )
                except Exception as rollback_err:
                    logger.error(
                        f"JIT: rollback itself failed ({rollback_err}) — "
                        f"model may be in inconsistent state. Restart server "
                        f"if generation stops responding."
                    )
        else:
            logger.warning(
                "JIT: mx.compile created but could not verify replacement — model structure may have changed"
            )
    except Exception as e:
        logger.warning(f"JIT: mx.compile failed, running without JIT: {e}")


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    # Use client IP for per-client rate limiting.
    # X-Forwarded-For for reverse proxy setups, then direct IP, then "unknown".
    client_id = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else None)
        or "unknown"
    )

    allowed, retry_after = _rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


# mlxstudio#63: kernel panic on heavy VS Code OAICopilot traffic.
# On macOS, continuous-batching under extreme unified-memory pressure can
# trip an IOKit command-buffer failure that escalates to a whole-system
# kernel panic. We reject new inference requests with 503 BEFORE Metal
# allocates new KV blocks when system memory is critically low. This is
# recoverable (503 + Retry-After) whereas a kernel panic is not.
#
# Threshold: default 97% RAM used (very high — only triggers when the
# machine is already in swap thrash territory). Overridable via
# VMLX_MEMORY_PRESSURE_REJECT_PCT env var. Disable entirely with
# VMLX_MEMORY_PRESSURE_GUARD=0.
_last_pressure_log: float = 0.0


async def check_memory_pressure(request: Request):
    """Reject new inference requests when unified memory is critically low.

    Wired onto all inference entry points (chat/completions, completions,
    responses, messages, api/chat, api/generate). Returns 503 with
    Retry-After=5 so well-behaved clients back off and retry.

    Skipped entirely when VMLX_MEMORY_PRESSURE_GUARD=0. The threshold
    defaults to 97% used — catastrophic territory, not a normal load.
    """
    if os.environ.get("VMLX_MEMORY_PRESSURE_GUARD", "1") == "0":
        return
    try:
        import psutil
    except ImportError:
        return
    try:
        threshold_pct = float(
            os.environ.get("VMLX_MEMORY_PRESSURE_REJECT_PCT", "97")
        )
    except (TypeError, ValueError):
        threshold_pct = 97.0
    try:
        mem = psutil.virtual_memory()
    except Exception:
        return
    if mem.percent < threshold_pct:
        return

    # Before rejecting, purge MLX's reusable graph/allocation cache once and
    # re-check. Large DSV4/MLA models can retain tens of GB in MLX's cache
    # after a completed request while the actual model working set is fine.
    # Prefix/L2 cache clearing is handled separately; this only releases
    # purgeable MLX allocator cache so we don't turn recoverable pressure into
    # a sticky 503 state.
    try:
        import gc as _gc

        _gc.collect()
        clear_mlx_memory_cache(log=logger)
        mem = psutil.virtual_memory()
        if mem.percent < threshold_pct:
            logger.info(
                "Memory pressure guard: purged MLX cache and continued "
                f"({mem.percent:.1f}% used, threshold {threshold_pct:.1f}%)."
            )
            return
    except Exception:
        pass

    # Log pressure rejections, but at most once per 5 s so we don't spam
    # the log under sustained overload.
    global _last_pressure_log
    now = time.monotonic()
    if now - _last_pressure_log > 5.0:
        _last_pressure_log = now
        logger.warning(
            f"mlxstudio#63 memory-pressure reject: {mem.percent:.1f}% used "
            f"(threshold {threshold_pct:.1f}%); rejecting inference "
            f"request with 503 to prevent Metal OOM / kernel panic. "
            f"available={mem.available / (1024**3):.1f}GB"
        )
    raise HTTPException(
        status_code=503,
        detail=(
            f"Server memory pressure too high "
            f"({mem.percent:.0f}%) — rejecting to prevent Metal OOM. "
            f"Retry in a few seconds, or reduce concurrent load. "
            f"Set VMLX_MEMORY_PRESSURE_REJECT_PCT to tune (default 97), "
            f"or VMLX_MEMORY_PRESSURE_GUARD=0 to disable."
        ),
        headers={"Retry-After": "5"},
    )


# mlxstudio#78: Metal command-buffer OOM on big-model + small-Mac configs.
# E.g. Gemma-4-31B-JANG_4M on M4 Max 64GB: weights load to ~41GB (64% of
# 64GB), Metal working-set cap is ~48GB (75% of RAM on Apple Silicon).
# First request's prefill attention tensors + KV allocations exceed the
# remaining ~7GB working-set headroom and the Metal command buffer dies
# with `Insufficient Memory` (IOGPUCommandBufferCallback error 8). The
# mlxstudio#63 system-RAM guard doesn't catch this because total RAM is
# only 64% used — the bottleneck is GPU working set, not system RAM.
#
# This guard reads `mx.get_active_memory()` (with mx.metal fallback per
# vmlx#94) and rejects when `active / max_working_set` exceeds threshold
# (default 98%, raised from 85% per Eric directive 2026-05-11). 98%
# lets users effectively fill their unified memory with the model + cache,
# while the 2% headroom still catches the genuine Metal command-buffer
# OOM signature before MLX raises [METAL] Insufficient Memory and the
# engine process dies.
#
# Knobs:
# - VMLX_METAL_WS_GUARD=0       → disable entirely (accept hard-crash risk)
# - VMLX_METAL_WS_REJECT_PCT=N  → tune threshold (default 98)
_last_metal_ws_log: float = 0.0


async def check_metal_working_set_pressure(request: Request):
    """Reject when Metal active_memory / max_recommended_working_set_size
    exceeds threshold — catches the big-model-on-small-Mac OOM that the
    system-RAM guard (mlxstudio#63) misses because RAM isn't exhausted
    but Metal's command-buffer cap is.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return
    try:
        from vmlx_engine.utils.memory_limits import (
            get_effective_metal_working_set_bytes,
            get_metal_ws_guard_threshold,
            is_metal_ws_guard_enabled,
        )
        if not is_metal_ws_guard_enabled():
            return
        threshold_pct = get_metal_ws_guard_threshold(98.0)
        active, max_ws = get_effective_metal_working_set_bytes(mx)
        if max_ws <= 0:
            return  # no limit exposed — skip
    except Exception:
        return

    pct = (active / max_ws) * 100.0
    if pct < threshold_pct:
        return

    # Log throttle (shared state with mlxstudio#63 is overkill — use own).
    global _last_metal_ws_log
    now = time.monotonic()
    if now - _last_metal_ws_log > 5.0:
        _last_metal_ws_log = now
        logger.warning(
            f"mlxstudio#78 Metal working-set pressure reject: "
            f"{pct:.1f}% of {max_ws / (1024**3):.1f}GB "
            f"(threshold {threshold_pct:.1f}%); rejecting to avoid "
            f"command-buffer OOM. active={active / (1024**3):.1f}GB"
        )
    raise HTTPException(
        status_code=503,
        detail=(
            f"Metal GPU working set too full "
            f"({pct:.0f}% of {max_ws / (1024**3):.1f}GB cap) — "
            f"rejecting to prevent command-buffer OOM. This model may "
            f"be too large for this Mac's unified memory budget. "
            f"This percentage is Metal memory occupancy, not a model-card "
            f"accuracy or MMLU score. "
            f"Retry after the GPU catches up, reduce concurrent load, "
            f"or try a smaller model / smaller quant. Tune via "
            f"VMLX_METAL_WS_REJECT_PCT (default 98), "
            f"set VMLX_METAL_WS_MAX_GB to raise/lower the working-set ceiling, "
            f"disable via VMLX_METAL_WS_GUARD=0."
        ),
        headers={"Retry-After": "5"},
    )


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if authentication is enabled."""
    global _auth_warning_logged

    if _api_key is None:
        # Log warning once about running without authentication
        if not _auth_warning_logged:
            logger.warning(
                "SECURITY WARNING: Server running without API key authentication. "
                "Anyone can access the API. Use --api-key to enable authentication."
            )
            _auth_warning_logged = True
        return True  # No auth required

    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_engine() -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


def _parse_tool_calls_with_parser(
    output_text: str, request: ChatCompletionRequest | ResponsesRequest | None = None
) -> tuple[str, list | None]:
    """
    Parse tool calls from model output using the configured parser.

    If --enable-auto-tool-choice is set with --tool-call-parser, uses the
    selected parser. Otherwise falls back to the generic parse_tool_calls.

    Args:
        output_text: The model output text
        request: The original request (for context)

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    def _allowed_tool_names() -> set[str]:
        effective_tools = _effective_tools_for_tool_parsing(request)
        if not request or not effective_tools:
            return set()
        names: set[str] = set()
        try:
            for tool in convert_tools_for_template(effective_tools) or []:
                fn = tool.get("function") if isinstance(tool, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(name, str) and name:
                    names.add(name)
        except Exception:
            pass
        return names

    def _filter_to_request_tools(tool_calls: list | None) -> list | None:
        if not tool_calls:
            return tool_calls
        allowed = _allowed_tool_names()
        if not allowed:
            return tool_calls

        def _function_payload(tc: Any) -> tuple[str | None, str, str]:
            try:
                return tc.function.name, tc.function.arguments, getattr(tc, "id", "")
            except Exception:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict):
                    return fn.get("name"), fn.get("arguments") or "{}", tc.get("id", "")
                if isinstance(tc, dict):
                    return tc.get("name"), tc.get("arguments") or "{}", tc.get("id", "")
                return None, "{}", ""

        def _coerce_json_args(raw_args: Any) -> dict[str, Any]:
            if isinstance(raw_args, dict):
                return dict(raw_args)
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return {}
            return {}

        def _rewrite_tool_alias(tc: Any) -> Any | None:
            name, raw_args, call_id = _function_payload(tc)
            if name != "create_file" or "write_file" not in allowed:
                return None
            args = _coerce_json_args(raw_args)
            path = (
                args.get("path")
                or args.get("file_path")
                or args.get("filename")
                or args.get("file")
                or args.get("name")
            )
            content = (
                args.get("content")
                or args.get("text")
                or args.get("value")
                or args.get("body")
            )
            if not isinstance(path, str) or not isinstance(content, str):
                return None
            return ToolCall(
                id=call_id or f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name="write_file",
                    arguments=json.dumps(
                        {"path": path, "content": content},
                        ensure_ascii=False,
                    ),
                ),
            )

        filtered = []
        for tc in tool_calls:
            name, _, _ = _function_payload(tc)
            if name in allowed:
                filtered.append(tc)
            else:
                rewritten = _rewrite_tool_alias(tc)
                if rewritten is not None:
                    filtered.append(rewritten)
                    continue
                logger.warning(
                    "Dropping parsed tool call for unavailable tool %r; "
                    "available tools=%s",
                    name,
                    sorted(allowed),
                )
        return filtered or None

    def _repair_instruction_echo_tool_call(text: str) -> tuple[str, list | None]:
        """Repair model output that echoes an explicit tool-use instruction.

        Live DSV4 JANGTQ2 can answer an auto-tool request with:

            <Use the list_directory tool for path '.' and do not answer in prose.

        That is not canonical DSML, but it is not normal prose either: it names
        a request-available tool and carries a schema-visible argument. Keep the
        repair schema-gated so arbitrary text cannot become a function call.
        """
        allowed = _allowed_tool_names()
        if not allowed:
            return text, None
        def _request_text() -> str:
            def _content_text(value: Any) -> str:
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    parts = []
                    for item in value:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text") or item.get("content") or ""))
                        elif isinstance(item, str):
                            parts.append(item)
                    return "\n".join(p for p in parts if p)
                if isinstance(value, dict):
                    return str(value.get("text") or value.get("content") or "")
                return ""

            chunks: list[str] = []
            raw_input = getattr(request, "input", None)
            if isinstance(raw_input, str):
                chunks.append(raw_input)
            elif isinstance(raw_input, list):
                for item in raw_input:
                    if isinstance(item, dict):
                        chunks.append(_content_text(item.get("content") or item.get("text")))
            for msg in getattr(request, "messages", None) or []:
                if isinstance(msg, dict):
                    chunks.append(_content_text(msg.get("content")))
            return "\n".join(c for c in chunks if c)

        stripped = text.strip()
        request_text = _request_text()
        combined_text = f"{stripped}\n{request_text}"

        m = re.match(
            r"^<\s*use\s+the\s+([A-Za-z_][\w.-]*)\s+tool\b",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        name = m.group(1) if m else None
        if not name:
            named_hits = [n for n in allowed if re.search(rf"\b{re.escape(n)}\b", stripped)]
            if len(named_hits) == 1:
                name = named_hits[0]
        if not name and len(allowed) == 1:
            tool_intent = re.search(
                r"\b(?:use|call|invoke|tool_call|function_call)\b|"
                r"<｜DSML｜tool_c|<\s*tool_call",
                stripped,
                flags=re.IGNORECASE,
            )
            if tool_intent:
                name = next(iter(allowed))
        if not name or name not in allowed:
            return text, None

        schema = None
        try:
            effective_tools = _effective_tools_for_tool_parsing(request)
            for tool in convert_tools_for_template(effective_tools) or []:
                fn = tool.get("function") if isinstance(tool, dict) else None
                if isinstance(fn, dict) and fn.get("name") == name:
                    schema = fn
                    break
        except Exception:
            schema = None
        params_schema = schema.get("parameters", {}) if isinstance(schema, dict) else {}
        props = (
            params_schema.get("properties", {})
            if isinstance(params_schema, dict)
            else {}
        )
        required = (
            params_schema.get("required", [])
            if isinstance(params_schema, dict)
            else []
        )

        args: dict[str, str] = {}
        for p_name in props:
            pat = re.compile(
                rf"\b{re.escape(p_name)}\b\s*(?:=|:|to|is)?\s*['\"]([^'\"]+)['\"]",
                flags=re.IGNORECASE,
            )
            pm = pat.search(combined_text)
            if pm:
                args[p_name] = pm.group(1)
        if not args and len(props) == 1 and ("'.'" in combined_text or '"."' in combined_text):
            args[next(iter(props))] = "."
        if any(p not in args for p in required):
            return text, None

        return "", [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            )
        ]

    def _generic_parse_filtered(text: str) -> tuple[str, list | None]:
        cleaned, calls = parse_tool_calls(text)
        if calls:
            calls = _filter_to_request_tools(calls)
            if calls:
                return cleaned, calls
            return text, None
        repaired_cleaned, repaired_calls = _repair_instruction_echo_tool_call(text)
        if repaired_calls:
            return repaired_cleaned, repaired_calls
        return text, None

    # Determine which parser to use.
    # If --enable-auto-tool-choice was set, use the configured parser.
    # Otherwise, if the request has tools, auto-detect from model config
    # so clients like Cherry Studio get proper parsing without CLI flags.
    active_parser = _tool_call_parser
    if not active_parser and request and getattr(request, "tools", None):
        try:
            from .model_config_registry import get_model_config_registry

            registry = get_model_config_registry()
            detected = registry.get_tool_parser(
                _model_path or _model_name or getattr(request, "model", "") or ""
            )
            if detected:
                active_parser = detected
        except Exception:
            pass

    if not active_parser:
        return _generic_parse_filtered(output_text)

    # Create a fresh parser instance per call for thread-safety.
    # Non-streaming requests can be concurrent — sharing a global instance
    # risks interleaved state even with reset().
    try:
        parser_cls = ToolParserManager.get_tool_parser(active_parser)
        tokenizer = None
        if _engine is not None and hasattr(_engine, "_tokenizer"):
            tokenizer = _engine._tokenizer
        parser_instance = parser_cls(tokenizer)
    except Exception as e:
        logger.warning(f"Failed to initialize tool parser '{active_parser}': {e}")
        return parse_tool_calls(output_text)

    # Use the configured parser, fall back to generic if it finds nothing
    try:
        # Convert request to dict format for parsers that need schema info (e.g., Step3p5 type coercion)
        parser_request = None
        effective_tools = _effective_tools_for_tool_parsing(request)
        if request and effective_tools:
            parser_request = {
                "tools": convert_tools_for_template(effective_tools),
                "model_path": (
                    _model_path or _model_name or getattr(request, "model", None)
                ),
            }
        result = parser_instance.extract_tool_calls(output_text, request=parser_request)
        if result.tools_called:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in result.tool_calls
            ]
            filtered_tool_calls = _filter_to_request_tools(tool_calls)
            if filtered_tool_calls:
                return result.content or "", filtered_tool_calls
            # Parser consumed only unavailable tool names. Treat as plain text
            # so clients do not receive hallucinated function calls like
            # README.md()/src()/tests() when the request only exposed
            # list_directory().
            return output_text, None
        else:
            # Specific parser found nothing — try generic parser as fallback
            # (handles Nemotron, Llama, raw JSON, etc.)
            return _generic_parse_filtered(output_text)
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return _generic_parse_filtered(output_text)


def _clean_suppressed_tool_markup_for_display(
    output_text: str, request: ChatCompletionRequest | ResponsesRequest | None = None
) -> str:
    """Strip native tool-call markup when a request explicitly suppresses tools."""
    if not output_text:
        return output_text
    try:
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(output_text, request)
    except Exception as exc:
        logger.debug("Suppressed tool-markup cleanup failed: %s", exc)
        return output_text
    if tool_calls:
        return _strip_tool_markup_residue_for_display(cleaned_text or "")
    return output_text


def _suppressed_tool_display_delta(
    accumulated_text: str,
    streamed_display_text: str,
    request: ChatCompletionRequest | ResponsesRequest | None = None,
) -> str | None:
    """Return the next visible delta while hiding suppressed native tool markup."""
    cleaned = _clean_suppressed_tool_markup_for_display(accumulated_text, request)
    if cleaned == accumulated_text:
        marker_positions = [
            pos
            for marker in _TOOL_CALL_MARKERS
            if (pos := accumulated_text.find(marker)) >= 0
        ]
        if marker_positions:
            cleaned = accumulated_text[: min(marker_positions)].rstrip()
        else:
            # Avoid streaming a partial marker split across token boundaries.
            partial_len = 0
            for marker in _TOOL_CALL_MARKERS:
                max_prefix = min(len(marker) - 1, len(accumulated_text))
                for n in range(max_prefix, 0, -1):
                    if accumulated_text.endswith(marker[:n]):
                        partial_len = max(partial_len, n)
                        break
            cleaned = accumulated_text[:-partial_len] if partial_len else accumulated_text
    cleaned = _strip_tool_markup_residue_for_display(cleaned)
    if not cleaned:
        return None
    if streamed_display_text and cleaned.startswith(streamed_display_text):
        delta = cleaned[len(streamed_display_text) :]
    elif streamed_display_text == cleaned:
        delta = ""
    else:
        delta = cleaned
    return delta if delta else None


def _suppress_tool_parsing_when_no_tools(
    all_tools: list,
    tool_choice,
    endpoint: str,
) -> bool:
    """Return True when tool-call parsing should be disabled for a request.

    Starting the server with ``--tool-call-parser auto`` or
    ``--enable-auto-tool-choice`` only selects a parser; it must not make a
    model invent API tool calls when the current request did not provide any
    tools (or MCP did not contribute any). Several reasoning models mention
    tool-call syntax in normal text after a tool-history continuation; parsing
    that text without an actual tool schema can turn a valid answer into a
    function_call-only response with no visible content.
    """
    if all_tools:
        return False
    if tool_choice == "required" or isinstance(tool_choice, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{endpoint}: tool_choice requires at least one available tool "
                "(request tools or MCP tools)."
            ),
        )
    return True


async def _await_chat_with_disconnect_abort(
    engine: Any,
    *,
    messages: list[dict[str, Any]],
    chat_kwargs: dict[str, Any],
    timeout: float,
    fastapi_request: Request | None,
    request_id: str,
    endpoint: str,
    poll_interval: float = 0.25,
):
    """Await non-streaming engine.chat while aborting scheduler work on disconnect.

    Streaming handlers poll `Request.is_disconnected()` inside their token loop.
    Non-streaming handlers used to await `engine.chat()` directly, so a closed
    HTTP client could leave a long prompt/generation running in the scheduler
    with no consumer. Pass the public response id through as the scheduler
    request id and abort it if the client disappears or the timeout fires.
    """
    chat_kwargs = dict(chat_kwargs)
    chat_kwargs.pop("request_id", None)
    task = asyncio.create_task(
        engine.chat(messages=messages, request_id=request_id, **chat_kwargs)
    )
    started = time.perf_counter()

    # Active receive-channel drain. `Request.is_disconnected()` is a lazy
    # zero-timeout poll of receive — it can miss `http.disconnect` events
    # in non-streaming flows where nothing else reads from receive (live
    # observed by codex 2026-05-09: client TCP close + read-timeout left
    # scheduler num_running=1 for >19s). Spawn a parallel task that
    # actively awaits receive() so the disconnect event is delivered to us
    # promptly, and signal it to the polling loop.
    disconnect_event = asyncio.Event()
    drain_task: asyncio.Task | None = None
    if fastapi_request is not None and hasattr(fastapi_request, "receive"):
        async def _drain_receive() -> None:
            try:
                while not disconnect_event.is_set():
                    msg = await fastapi_request.receive()
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "http.disconnect":
                        disconnect_event.set()
                        return
            except (asyncio.CancelledError, RuntimeError, Exception):
                # Receive may raise after the channel is closed; treat the
                # exception as a disconnect signal too (the connection is
                # gone either way).
                disconnect_event.set()
        drain_task = asyncio.create_task(_drain_receive())

    async def _abort(reason: str) -> None:
        logger.info("%s: aborting request %s (%s)", endpoint, request_id, reason)
        try:
            await engine.abort_request(request_id)
        except Exception as exc:
            logger.warning(
                "%s: abort_request(%s) failed after %s: %s",
                endpoint,
                request_id,
                reason,
                exc,
            )

    def _stop_drain() -> None:
        if drain_task is not None and not drain_task.done():
            drain_task.cancel()

    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=max(poll_interval, 0.001))
            if task in done:
                _stop_drain()
                try:
                    return task.result()
                except RuntimeError as exc:
                    if f"No collector for request {request_id}" in str(exc):
                        raise HTTPException(
                            status_code=499,
                            detail="Request cancelled",
                        ) from exc
                    raise

            # Active drain caught http.disconnect — abort immediately.
            if disconnect_event.is_set():
                _stop_drain()
                await _abort("client disconnected (receive-drain)")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise HTTPException(status_code=499, detail="Client disconnected")

            # If the active receive-drain is available, it owns the ASGI
            # receive channel. Starlette's is_disconnected() also calls
            # receive(), so polling both paths can create two concurrent
            # readers on servers that do not guarantee receive reentrancy.
            if fastapi_request is not None and drain_task is None:
                try:
                    disconnected = await fastapi_request.is_disconnected()
                except Exception:
                    disconnected = False
                if disconnected:
                    _stop_drain()
                    await _abort("client disconnected")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise HTTPException(status_code=499, detail="Client disconnected")

            if timeout is not None and (time.perf_counter() - started) >= timeout:
                _stop_drain()
                await _abort(f"timeout after {timeout:.1f}s")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.TimeoutError
    except asyncio.CancelledError:
        _stop_drain()
        await _abort("handler cancelled")
        task.cancel()
        raise


def _detect_native_tool_support() -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Checks two sources:
    1. The parser class's ``supports_native_format()`` method.
    2. The model config registry ``preserve_native_tool_format`` flag
       (fallback for parsers that don't declare native support).

    Returns:
        True if native format should be preserved
    """
    if _tool_call_parser_disabled_explicitly:
        return False

    # Determine active parser: CLI-configured or auto-detected from model config
    active_parser = _tool_call_parser
    if not active_parser:
        try:
            from .model_config_registry import get_model_config_registry

            registry = get_model_config_registry()
            active_parser = registry.get_tool_parser(_model_path or _model_name or "")
        except Exception:
            pass
    if not active_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(active_parser)
        if parser_cls.supports_native_format():
            return True
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{active_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(f"Failed to detect native tool support: {e}")

    # Fallback: check model config registry for preserve_native_tool_format
    try:
        from .model_config_registry import get_model_config_registry

        _mc = get_model_config_registry().lookup(_model_path or _model_name or "")
        if _mc.preserve_native_tool_format:
            return True
    except Exception:
        logger.debug(
            "Model config registry lookup failed, skipping native format detection"
        )

    return False


def load_embedding_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the embedding model engine when configured."""
    global _embedding_engine, _embedding_model_locked

    if not model_name:
        return

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(model_name)
    _embedding_engine.load()


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name: extract 'org/model' from local paths.

    Examples:
        "~/.mlxstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit"
        → "mlx-community/Llama-3.2-3B-Instruct-4bit"

        HF cache paths (walk up past snapshots/<hash>):
        "~/.cache/huggingface/hub/models--JANGQ-AI--Gemma-4-26B-A4B-it-JANG_2L/snapshots/695690b3…"
        → "JANGQ-AI/Gemma-4-26B-A4B-it-JANG_2L"

        "mlx-community/Llama-3.2-3B-Instruct-4bit"
        → "mlx-community/Llama-3.2-3B-Instruct-4bit" (unchanged)
    """
    if os.path.sep in model_name or model_name.startswith("/"):
        parts = model_name.rstrip("/").split("/")
        # HuggingFace hub cache layout: .../hub/models--<org>--<repo>/snapshots/<hash>[/...]
        # When loaded from there, raw path tail is "snapshots/<hash>" which is
        # useless as a display name. Walk up to find the "models--..." marker
        # and decode it back to "org/repo".
        for part in parts:
            if part.startswith("models--") and "--" in part[len("models--"):]:
                marker = part[len("models--"):]
                # HF uses double-dash as separator between org and repo segments.
                seg = marker.split("--", 1)
                if len(seg) == 2 and seg[0] and seg[1]:
                    return f"{seg[0]}/{seg[1]}"
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return parts[-1]
    return model_name


def _resolve_model_name() -> str:
    """Return the model name to expose via the API.

    Priority: _served_model_name > _model_name > 'default'
    """
    return _served_model_name or _model_name or "default"


def _registry_model_key(requested_model: str | None = None) -> str:
    """Resolve the internal model key used for registry/family lookups.

    ``--served-model-name`` is a public API alias. Registry lookups need the
    loaded artifact path/name so they can read config.json/jang_config.json and
    avoid false warnings like ``alias/config.json`` missing.
    """
    return _model_path or _model_name or requested_model or ""


def _get_raw_model_from_engine():
    """Extract the raw MLX nn.Module from the current engine.

    Engine types store models differently:
      - SimpleEngine: _model is MLXLanguageModel/MLXMultimodalLM, raw at _model.model
      - BatchedEngine LLM: _model is MLLMModelWrapper, raw at _model._model
      - BatchedEngine MLLM: _model is raw model from _mllm_instance.model

    Returns None if engine or model not loaded.
    """
    if _engine is None:
        return None
    model = getattr(_engine, "_model", None)
    if model is None:
        return None
    # MLLMModelWrapper wraps the raw model in ._model
    raw = getattr(model, "_model", None)
    if raw is not None:
        return raw
    # MLXLanguageModel/MLXMultimodalLM wraps in .model
    raw = getattr(model, "model", None)
    if raw is not None:
        return raw
    # Direct model (shouldn't happen but be defensive)
    return model


def _apply_flash_moe_patching():
    """Apply Flash MoE SSD streaming to the loaded model.

    Must be called AFTER the engine loads the model. Works for both
    SimpleEngine and BatchedEngine.
    """
    global _flash_moe_loader

    if not _flash_moe_enabled:
        return

    # Defensive: should be blocked by CLI guard, but check again here in case
    # load_model is called directly (programmatic use, wake-from-sleep, etc.)
    if _smelt_enabled:
        logger.warning(
            "Flash MoE: refusing to patch — smelt is also enabled (mutually exclusive)"
        )
        return
    if _distributed_enabled:
        logger.warning(
            "Flash MoE: refusing to patch — distributed mode is active (mutually exclusive)"
        )
        return

    raw_model = _get_raw_model_from_engine()
    if raw_model is None:
        logger.warning("Flash MoE: could not find raw model in engine")
        return

    try:
        from .utils.flash_moe_loader import FlashMoEExpertLoader, SlotBankCache
        from .utils.smelt_loader import ExpertIndex
        from .models.flash_moe_integration import apply_flash_moe, free_expert_weights
        from .api.utils import resolve_to_local_path

        resolved_path = resolve_to_local_path(_model_path or _model_name)

        # vmlx#81: if this is a JANGTQ (weight_format=mxtq) model, the
        # ExpertIndex scan will find 0 MoE layers because JANGTQ stores
        # weights as tq_packed / tq_norms / tq_bits, not the standard
        # .weight / .scales / .biases suffixes the flash-moe loader
        # relies on. Previously this just silently skipped with "no MoE
        # layers, skipping", which was misleading. Detect + log clearly.
        try:
            import json as _json
            from pathlib import Path as _Path
            _jc = _json.loads((_Path(resolved_path) / "jang_config.json").read_text())
            if _jc.get("weight_format") == "mxtq" or _jc.get("format") == "mxtq":
                logger.warning(
                    "vmlx#81: --flash-moe is not supported on JANGTQ "
                    "(weight_format=mxtq) models. JANGTQ stores expert weights "
                    "as codebook-packed tq_packed tensors; flash-moe's "
                    "on-demand loader expects standard .weight/.scales/.biases "
                    "safetensor keys. Skipping flash-moe setup. "
                    "Use the standard JANG_* variant of this model for "
                    "flash-moe, or drop --flash-moe (JANGTQ is already "
                    "sub-2-bit effective)."
                )
                _flash_moe_loader = None
                return
        except (FileNotFoundError, ValueError, KeyError):
            pass  # not JANG or not resolvable — let downstream handle

        ei = ExpertIndex.build(resolved_path)

        if ei.num_moe_layers == 0:
            logger.info("Flash MoE: model has no MoE layers, skipping")
            return

        cli = _cli_args or {}
        slot_bank = cli.get("flash_moe_slot_bank", 64)
        io_split = cli.get("flash_moe_io_split", 4)

        cache = SlotBankCache(max_slots=slot_bank)
        _flash_moe_loader = FlashMoEExpertLoader(
            expert_index=ei,
            cache=cache,
            io_workers=io_split,
        )
        patched = apply_flash_moe(raw_model, _flash_moe_loader)
        if patched > 0:
            freed = free_expert_weights(raw_model)
            logger.info(
                "Flash MoE enabled: %d layers patched, %.2f GB freed, "
                "slot bank=%d, io_workers=%d",
                patched, freed / 1e9, slot_bank, io_split,
            )
        else:
            logger.warning("Flash MoE: no MoE layers found to patch")
            _flash_moe_loader = None
    except Exception as e:
        logger.error("Flash MoE setup failed: %s", e)
        _flash_moe_loader = None


def _reset_mllm_generation_streams() -> None:
    """Clear MLLM thread-local MLX streams after engine teardown/reload."""
    try:
        from .mllm_batch_generator import reset_generation_streams

        reset_generation_streams()
    except Exception as e:
        logger.debug("MLLM generation stream reset skipped: %s", e)
    try:
        from .models.mllm import reset_vlm_stream

        reset_vlm_stream()
    except Exception as e:
        logger.debug("Direct VLM stream reset skipped: %s", e)


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = _FALLBACK_MAX_OUTPUT_TOKENS,
    max_tokens_explicit: bool = False,
    max_prompt_tokens: int | None = None,
    force_mllm: bool = False,
    served_model_name: str | None = None,
    smelt: bool = False,
    smelt_experts: int = 50,
    distributed: bool = False,
    distributed_mode: str = "pipeline",
    cluster_secret: str = "",
    worker_nodes: str | None = None,
    flash_moe: bool = False,
    flash_moe_slot_bank: int = 64,
    flash_moe_prefetch: str = "none",
    flash_moe_io_split: int = 4,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        max_prompt_tokens: Explicit prompt/context token cap before prefill.
            If unset, vMLX uses the memory-safe estimate.
        force_mllm: Force loading as MLLM even if not auto-detected
        distributed: Enable distributed inference (coordinator mode)
        distributed_mode: 'pipeline' or 'tensor'
        cluster_secret: Shared secret for worker authentication
        worker_nodes: Comma-separated ip:port list (overrides Bonjour discovery)
    """
    global \
        _engine, \
        _model_name, \
        _model_path, \
        _default_max_tokens, \
        _default_max_tokens_explicit, \
        _served_model_name, \
        _model_load_error, \
        _jang_metadata, \
        _cli_args, \
        _smelt_enabled, \
        _smelt_experts, \
        _flash_moe_enabled, \
        _flash_moe_loader, \
        _distributed_enabled, \
        _distributed_mode, \
        _distributed_coordinator

    _smelt_enabled = smelt
    _smelt_experts = smelt_experts
    _flash_moe_enabled = flash_moe
    _distributed_enabled = distributed
    _distributed_mode = distributed_mode

    # Save CLI args for model reload on wake from deep sleep
    _cli_args = {
        "use_batching": use_batching,
        "scheduler_config": scheduler_config,
        "stream_interval": stream_interval,
        "max_tokens": max_tokens,
        "max_tokens_explicit": max_tokens_explicit,
        "max_prompt_tokens": max_prompt_tokens,
        "force_mllm": force_mllm,
        "served_model_name": served_model_name,
        "smelt": smelt,
        "smelt_experts": smelt_experts,
        "flash_moe": flash_moe,
        "flash_moe_slot_bank": flash_moe_slot_bank,
        "flash_moe_prefetch": flash_moe_prefetch,
        "flash_moe_io_split": flash_moe_io_split,
        "distributed": distributed,
        "distributed_mode": distributed_mode,
        "cluster_secret": cluster_secret,
        "worker_nodes": worker_nodes,
    }

    # Stop previous engine before loading new model — frees GPU memory, disk cache threads, etc.
    # Note: load_model() is called from CLI startup and also from admin_wake() during live serving.
    if _engine is not None:
        try:
            if hasattr(_engine, "_scheduler") and hasattr(
                _engine._scheduler, "deep_reset"
            ):
                _engine._scheduler.deep_reset()
        except Exception as e:
            logger.warning(f"Failed to stop previous engine: {e}")
        _reset_mllm_generation_streams()
        _engine = None

    _model_load_error = None  # Clear any previous error
    _jang_metadata = None  # Clear any previous JANG metadata

    _default_max_tokens = max_tokens
    _default_max_tokens_explicit = bool(max_tokens_explicit)
    _model_path = model_name  # Full path for config.json lookups
    # Normalize model name: extract "org/model" from full local paths
    # e.g. "~/.mlxstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit"
    #    → "mlx-community/Llama-3.2-3B-Instruct-4bit"
    _model_name = _normalize_model_name(model_name)
    _served_model_name = served_model_name  # Custom name override (may be None)
    if _served_model_name:
        logger.info(f"Serving model as: {_served_model_name} (actual: {_model_name})")

    # Log system memory before model load for diagnostics
    try:
        import psutil

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        logger.info(
            f"System memory before load: {available_gb:.1f}GB available / "
            f"{total_gb:.1f}GB total ({mem.percent}% used)"
        )
        if mem.percent > 90:
            logger.warning(
                f"HIGH MEMORY PRESSURE: {mem.percent}% RAM used. "
                f"Only {available_gb:.1f}GB available. Model load may cause system instability."
            )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Memory check failed: {e}")

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    # Distributed inference: set up mesh coordinator instead of local loading.
    # Image models are single-node only — mflux has no concept of layer splits,
    # so binding mesh TCP ports would just be noise.
    if distributed and _model_type == "image":
        logger.info(
            "Distributed flag ignored for image model — image generation is "
            "single-node only."
        )
        _distributed_enabled = False
        distributed = False

    # VL / multimodal models are not yet supported in distributed mode.
    # DistributedEngine hardcodes is_mllm=False, so an image request would
    # silently route through the text-only coordinator path and embed image
    # tokens as text garbage. Fail loudly instead of producing wrong output.
    # Phase 2 work item: vision encoder on coordinator + language tower on
    # the pipeline, with image tokens flowing as ordinary embeddings.
    if distributed:
        try:
            from .api.utils import resolve_to_local_path
            from .model_config_registry import is_mllm_model

            _resolved_for_mllm = resolve_to_local_path(model_name)
            _is_vl = False
            try:
                _is_vl = is_mllm_model(_resolved_for_mllm)
            except Exception:
                # Fallback: check config.json directly for vision_config
                try:
                    import json as _json
                    with open(f"{_resolved_for_mllm}/config.json") as _f:
                        _cfg = _json.load(_f)
                    if (
                        "vision_config" in _cfg
                        or _cfg.get("model_type", "").endswith("_vl")
                        or "vlm" in _cfg.get("model_type", "").lower()
                    ):
                        _is_vl = True
                except Exception:
                    pass
            # force_mllm overrides auto-detection — user explicitly asked for MLLM
            if force_mllm:
                _is_vl = True

            if _is_vl:
                logger.error(
                    "Refusing to start distributed mode for VL/multimodal model. "
                    "DistributedEngine is text-only today (is_mllm=False). A VL "
                    "request would silently route through the text-only coordinator "
                    "path and embed image tokens as text garbage. Run this model "
                    "single-node for now. Phase 2 will add VL support once the "
                    "vision encoder + language tower pipeline split is implemented."
                )
                _distributed_enabled = False
                _distributed_coordinator = None
                distributed = False
        except Exception as e:
            logger.debug(
                "VL detection failed (non-fatal): %s — proceeding with distributed",
                e,
            )

    if distributed:
        logger.info("Distributed mode enabled (%s parallelism)", distributed_mode)
        _cluster_secret_resolved = (
            cluster_secret or os.environ.get("VMLX_CLUSTER_SECRET", "")
        )
        if not _cluster_secret_resolved:
            logger.error(
                "Refusing to start distributed mesh without a cluster secret. "
                "Set --cluster-secret or VMLX_CLUSTER_SECRET. Generate one with: "
                "python3 -c 'import secrets; print(secrets.token_urlsafe(24))'"
            )
            _distributed_enabled = False
            _distributed_coordinator = None
            distributed = False

    if distributed:
        try:
            from .distributed.mesh_manager import MeshManager
            _distributed_coordinator = MeshManager(
                cluster_secret=_cluster_secret_resolved,
                model_path=model_name,
                mode=distributed_mode,
                smelt_percent=smelt_experts if smelt else 100,
                worker_nodes=worker_nodes,
            )
            # Run mesh setup — discovery, election, layer assignment, loading.
            # Store the task handle + done-callback so startup exceptions land
            # in logs instead of silently hanging the first request for 60s.
            try:
                loop = asyncio.get_running_loop()
                _mesh_start_task = loop.create_task(
                    _distributed_coordinator.start()
                )

                def _on_mesh_start_done(task: asyncio.Task) -> None:
                    if task.cancelled():
                        return
                    exc = task.exception()
                    if exc is not None:
                        logger.error(
                            "Distributed mesh startup failed: %s", exc,
                            exc_info=exc,
                        )

                _mesh_start_task.add_done_callback(_on_mesh_start_done)
                _distributed_coordinator._start_task = _mesh_start_task
                logger.info("Distributed mesh setup deferred to event loop")
            except RuntimeError:
                asyncio.run(_distributed_coordinator.start())
                logger.info(
                    "Distributed mesh ready: %d nodes",
                    len(_distributed_coordinator.topology.active_nodes),
                )

            # If mesh has >1 node, use DistributedEngine instead of local engine
            num_nodes = len(_distributed_coordinator.topology.active_nodes)
            if num_nodes > 1 and _distributed_coordinator._is_coordinator:
                from .distributed.engine import DistributedEngine
                _engine = DistributedEngine(_distributed_coordinator)
                try:
                    asyncio.get_running_loop()
                    _engine._needs_async_start = True
                    logger.info("DistributedEngine created (deferred start — %d nodes)", num_nodes)
                except RuntimeError:
                    asyncio.run(_engine.start())
                logger.info("Distributed inference active: %d nodes, skipping local engine", num_nodes)
                return  # Skip local SimpleEngine/BatchedEngine creation
            else:
                # Fall through to local engine. Clear the distributed globals
                # so /v1/cluster/* endpoints + /health don't lie about an
                # active mesh that doesn't exist.
                logger.info(
                    "Single-node or worker mode — falling through to local "
                    "engine and clearing distributed state"
                )
                try:
                    shutdown = getattr(_distributed_coordinator, "shutdown", None) \
                        or getattr(_distributed_coordinator, "stop", None)
                    if shutdown is not None:
                        result = shutdown()
                        if asyncio.iscoroutine(result):
                            try:
                                asyncio.get_running_loop().create_task(result)
                            except RuntimeError:
                                asyncio.run(result)
                except Exception as e:
                    logger.debug("Mesh cleanup on fallback failed: %s", e)
                _distributed_coordinator = None
                _distributed_enabled = False
        except Exception as e:
            logger.error("Failed to start distributed mesh: %s", e)
            logger.info("Falling back to single-node loading...")
            _distributed_coordinator = None
            _distributed_enabled = False
            # Fall through to normal loading below

    if use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
    else:
        logger.info(f"Loading model with SimpleEngine: {model_name}")
        _engine = SimpleEngine(model_name=model_name, force_mllm=force_mllm)
        # Start SimpleEngine — asyncio.run() crashes inside a running event loop
        # (e.g., when called from admin_wake during deep sleep recovery).
        # Detect and skip; the engine's lazy start will handle it on first request.
        try:
            asyncio.get_running_loop()
            # Inside running loop — can't use asyncio.run(). Mark for deferred async start.
            _engine._needs_async_start = True
            logger.info("SimpleEngine created (deferred start — inside event loop)")
        except RuntimeError:
            # No running loop (CLI startup) — safe to use asyncio.run
            asyncio.run(_engine.start())
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

    # Apply Flash MoE for SimpleEngine (model already loaded above).
    # BatchedEngine defers model loading to lifespan() — Flash MoE applied there.
    if flash_moe and _engine is not None and hasattr(_engine, "_loaded") and _engine._loaded:
        _apply_flash_moe_patching()

    # Apply JIT compilation if enabled — only for SimpleEngine (already started above).
    # BatchedEngine starts in lifespan(), so JIT is applied there instead.
    if (
        _enable_jit
        and _engine is not None
        and hasattr(_engine, "_loaded")
        and _engine._loaded
    ):
        _apply_jit_compilation()

    # Apply chat template override from model config registry (e.g. Harmony for GPT-OSS)
    try:
        from .model_config_registry import get_model_config_registry

        _mc = get_model_config_registry().lookup(model_name)
        if _mc.chat_template_custom and _engine and _engine.tokenizer:
            _engine.tokenizer.chat_template = _mc.chat_template_custom
            logger.info(f"Applied custom chat template for {_mc.family_name} model")
    except Exception as e:
        logger.warning(f"Failed to apply custom chat template: {e}")

    # Apply user's custom chat template (highest priority — overrides both model and registry templates)
    if _custom_chat_template and _engine and _engine.tokenizer:
        _engine.tokenizer.chat_template = _custom_chat_template
        logger.info(
            f"Applied custom chat template from --chat-template ({len(_custom_chat_template)} chars)"
        )

    # Compute max safe prompt token limit based on available GPU memory
    global _max_prompt_tokens
    _auto_max_prompt_tokens = _estimate_max_prompt_tokens()
    _max_prompt_tokens = _resolve_max_prompt_tokens(
        _auto_max_prompt_tokens,
        max_prompt_tokens,
    )
    if _max_prompt_tokens > 0:
        if max_prompt_tokens is not None and max_prompt_tokens > 0:
            logger.info(
                f"Max prompt/context length: ~{_max_prompt_tokens:,} tokens "
                "(explicit --max-prompt-tokens)"
            )
        else:
            logger.info(
                f"Max safe prompt length: ~{_max_prompt_tokens:,} tokens "
                f"(based on available GPU memory)"
            )
        # vmlx#85 @Benjamin-Wegener: advise user when their requested
        # max_tokens exceeds the memory-safe limit. This is the most
        # common question — "can I run <big model> at <large context>
        # on <low RAM Mac>?" — and the answer is usually "yes, but
        # only up to N tokens before Metal OOM."
        if _default_max_tokens > _max_prompt_tokens:
            logger.warning(
                f"CONTEXT ADVISORY (vmlx#85): --max-tokens="
                f"{_default_max_tokens} exceeds the memory-safe limit "
                f"(~{_max_prompt_tokens:,} tokens) for this model on "
                f"this Mac. Prompts longer than ~{_max_prompt_tokens:,} "
                f"tokens will be REJECTED with a 413 prompt_too_long error to avoid "
                f"Metal OOM. Use a Mac with more RAM, a smaller model, "
                f"or set --max-tokens={_max_prompt_tokens} to silence this warning. "
                f"Use --max-prompt-tokens={_max_prompt_tokens} only when you want "
                f"to set the prompt/context cap explicitly."
            )

    # Cache JANG metadata at load time (avoids sync file IO in async /health handler)
    try:
        from .utils.jang_loader import is_jang_model, JANG_CONFIG_FILENAMES
        from .api.utils import resolve_to_local_path

        _resolved_model = resolve_to_local_path(model_name)
        if is_jang_model(_resolved_model):
            from pathlib import Path

            for cfg_name in JANG_CONFIG_FILENAMES:
                cfg_path = Path(_resolved_model) / cfg_name
                if cfg_path.exists():
                    jang_meta = json.loads(cfg_path.read_text())
                    q = jang_meta.get("quantization", {})
                    _jang_metadata = {
                        "type": "jang",
                        "target_bits": q.get("target_bits"),
                        "actual_bits": q.get("actual_bits"),
                        "block_size": q.get("block_size", 64),
                    }
                    break
    except Exception:
        pass

    # Log Metal GPU memory after model load
    try:
        import mlx.core as mx

        active_gb = peak_gb = None
        if hasattr(mx, "get_active_memory"):
            active_gb = mx.get_active_memory() / (1024**3)
            peak_gb = mx.get_peak_memory() / (1024**3)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            active_gb = mx.metal.get_active_memory() / (1024**3)
            peak_gb = mx.metal.get_peak_memory() / (1024**3)
        if active_gb is not None:
            logger.info(
                f"Metal GPU memory after load: {active_gb:.2f}GB active, {peak_gb:.2f}GB peak"
            )
    except Exception:
        pass

    # Drain any async dispatch queues left behind by the loader.
    try:
        import mlx.core as _mx
        _mx.synchronize()
    except Exception:
        pass


    # Log system memory after model load
    try:
        import psutil

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        logger.info(
            f"System memory after load: {available_gb:.1f}GB available ({mem.percent}% used)"
        )
        if mem.percent > 95:
            logger.warning(
                f"CRITICAL: System memory at {mem.percent}%. "
                f"Only {available_gb:.1f}GB free. Risk of OOM under load. "
                "Consider a smaller model or quantized variant."
            )
    except Exception:
        pass

    # Set native tool format support on the engine (thread-safe via instance property)
    _engine.preserve_native_tool_format = _detect_native_tool_support()
    if _engine.preserve_native_tool_format:
        logger.info(f"Native tool format enabled for parser: {_tool_call_parser}")

    if _default_max_tokens_explicit:
        logger.info(f"Default max tokens override: {_default_max_tokens}")
    else:
        logger.info(
            "Default max tokens fallback: %s (bundle max_new_tokens wins when present)",
            _FALLBACK_MAX_OUTPUT_TOKENS,
        )


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    cached = getattr(output, "cached_tokens", 0)
    detail = getattr(output, "cache_detail", "") or None
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=cached, cache_detail=detail
        )
        if cached > 0 or detail
        else None,
    )


def _get_responses_usage(output: GenerationOutput) -> "ResponsesUsage":
    """Extract usage metrics in Responses API format."""
    _pt = getattr(output, "prompt_tokens", 0)
    _ct = getattr(output, "completion_tokens", 0)
    cached = getattr(output, "cached_tokens", 0)
    detail = getattr(output, "cache_detail", "") or None
    return ResponsesUsage(
        input_tokens=_pt,
        output_tokens=_ct,
        total_tokens=_pt + _ct,
        input_tokens_details=InputTokensDetails(
            cached_tokens=cached, cache_detail=detail
        )
        if cached > 0 or detail
        else None,
    )


def _reject_unsupported_logprobs_request(
    *,
    endpoint: str,
    requested: bool,
    stream: bool = False,
    is_mllm: bool = False,
) -> None:
    """Reject logprobs only on surfaces that cannot return truthful token data."""
    if not requested:
        return
    if is_mllm:
        mode = "streaming " if stream else ""
        raise HTTPException(
            status_code=400,
            detail=(
                f"{endpoint} {mode}logprobs currently require text-only LLM "
                "requests; multimodal/VLM logprobs are not implemented."
            ),
        )


@app.get("/")
async def ollama_root():
    """Ollama-compatible root probe.

    Several clients verify an Ollama provider with GET/HEAD / before
    probing /api/version. Real Ollama returns a plain text liveness string;
    returning 404 here makes those clients report a generic version failure.
    """
    return Response(content="Ollama is running\n", media_type="text/plain")


@app.head("/")
async def ollama_root_head():
    """HEAD variant of the Ollama root liveness probe."""
    return Response(status_code=200)


def _diagnostic_attr(obj, name, default=None):
    """getattr that avoids manufacturing unittest.mock child objects."""
    if obj is None:
        return default
    if type(obj).__module__ == "unittest.mock" and name not in vars(obj):
        return default
    return getattr(obj, name, default)


def _read_bundle_json(bundle_path: str | None, filename: str) -> dict:
    if not bundle_path:
        return {}
    try:
        from pathlib import Path

        path = Path(bundle_path) / filename
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bundle_has_prestacked_jangtq(bundle_path: str | None) -> bool:
    """Detect bundles that already store routed JANGTQ experts pre-stacked."""
    if not bundle_path:
        return False
    try:
        from pathlib import Path

        index_path = Path(bundle_path) / "model.safetensors.index.json"
        if not index_path.is_file():
            return False
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            return False
        return any(
            ".switch_mlp." in str(key) and ".tq_packed" in str(key)
            for key in weight_map
        )
    except Exception:
        return False


def _weight_index_role_for_key(key: str) -> str:
    """Best-effort role classifier for safetensors index telemetry.

    This is diagnostic metadata only. Runtime loading still owns the actual
    tensor interpretation.
    """
    lowered = key.lower()
    if ".switch_mlp." in lowered or ".experts." in lowered:
        return "routed_expert"
    if "shared_expert" in lowered or ".shared_mlp." in lowered:
        return "shared_expert"
    if (
        ".self_attn." in lowered
        or ".attention." in lowered
        or ".attn." in lowered
    ):
        return "attention"
    if "embed_tokens" in lowered or ".embed." in lowered or "embeddings" in lowered:
        return "embed_tokens"
    if "lm_head" in lowered:
        return "lm_head"
    if "mamba" in lowered or "conv1d" in lowered or "ssm" in lowered:
        return "hybrid_ssm"
    return "other"


def _bundle_weight_index_status(bundle_path: str | None) -> dict | None:
    """Summarize safetensors index codec targets without reading weights."""
    if not bundle_path:
        return None
    try:
        from pathlib import Path

        index_path = Path(bundle_path) / "model.safetensors.index.json"
        if not index_path.is_file():
            return None
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            return {"error": "model.safetensors.index.json has no weight_map object"}

        suffix_counts: dict[str, int] = defaultdict(int)
        tq_target_counts: dict[str, int] = defaultdict(int)
        routed_layout_counts: dict[str, int] = defaultdict(int)
        sample_tq_packed_targets: list[str] = []
        mtp_tensor_count = 0
        suffixes = (
            "tq_packed",
            "tq_norms",
            "tq_bits",
            "scales",
            "biases",
            "weight",
        )

        for raw_key in weight_map:
            key = str(raw_key)
            lowered = key.lower()
            suffix = next(
                (candidate for candidate in suffixes if key.endswith(f".{candidate}")),
                None,
            )
            if suffix is not None:
                suffix_counts[suffix] += 1
            if suffix in ("tq_packed", "tq_norms", "tq_bits"):
                tq_target_counts[_weight_index_role_for_key(key)] += 1
            if suffix == "tq_packed" and len(sample_tq_packed_targets) < 12:
                sample_tq_packed_targets.append(key[: -len(".tq_packed")])
            if key.startswith("mtp.") or ".mtp." in lowered:
                mtp_tensor_count += 1
            if ".switch_mlp." in lowered:
                routed_layout_counts["prestacked_switch"] += 1
            if re.search(r"(?:^|\.)layers\.\d+\.(?:mlp|ffn)\.experts\.\d+\.", lowered):
                routed_layout_counts["split_expert"] += 1

        result = {
            "index_file": str(index_path),
            "total_tensors": len(weight_map),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "tq_target_counts": dict(sorted(tq_target_counts.items())),
            "routed_layout_counts": {
                "prestacked_switch": routed_layout_counts.get("prestacked_switch", 0),
                "split_expert": routed_layout_counts.get("split_expert", 0),
            },
            "mtp_tensor_count": mtp_tensor_count,
            "sample_tq_packed_targets": sample_tq_packed_targets or None,
        }
        return {k: v for k, v in result.items() if v is not None}
    except Exception as exc:
        return {"error": f"model.safetensors.index.json read failed: {type(exc).__name__}: {exc}"}


def _coerce_routed_layer_bits(raw) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    routed_layer_bits: dict[str, int] = {}
    for key, value in raw.items():
        try:
            routed_layer_bits[str(int(key))] = int(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(routed_layer_bits.items(), key=lambda item: int(item[0]))) or None


def _find_routed_layer_bit_plan(cfg: dict, jang_cfg: dict, q_jang: dict) -> tuple[dict[str, int] | None, str | None]:
    routed_experts = q_jang.get("routed_experts")
    bit_plan = routed_experts.get("bit_plan") if isinstance(routed_experts, dict) else None
    candidates = (
        (
            "jang_config.quantization.routed_experts.bit_plan.routed_layer_bits",
            bit_plan.get("routed_layer_bits") if isinstance(bit_plan, dict) else None,
        ),
        (
            "jang_config.quantization.routed_layer_bits",
            q_jang.get("routed_layer_bits"),
        ),
        (
            "jang_config.routed_layer_bits",
            jang_cfg.get("routed_layer_bits"),
        ),
        (
            "config.routed_layer_bits",
            cfg.get("routed_layer_bits"),
        ),
    )
    for source, raw in candidates:
        routed_layer_bits = _coerce_routed_layer_bits(raw)
        if routed_layer_bits:
            return routed_layer_bits, source
    return None, None


def _coerce_routed_projection_bits(raw) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    projection_bits: dict[str, int] = {}
    for key, value in raw.items():
        try:
            projection_bits[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return projection_bits or None


def _find_routed_projection_bit_plan(
    cfg: dict, jang_cfg: dict, q_jang: dict
) -> tuple[dict[str, int] | None, str | None]:
    routed_experts = q_jang.get("routed_experts")
    bit_plan = routed_experts.get("bit_plan") if isinstance(routed_experts, dict) else None
    q_cfg = cfg.get("quantization") if isinstance(cfg.get("quantization"), dict) else {}
    cfg_bit_plan = (
        q_cfg.get("routed_expert_bit_plan")
        if isinstance(q_cfg.get("routed_expert_bit_plan"), dict)
        else {}
    )
    candidates = (
        (
            "jang_config.quantization.routed_experts.bit_plan.routed_projection_bits",
            bit_plan.get("routed_projection_bits") if isinstance(bit_plan, dict) else None,
        ),
        (
            "jang_config.quantization.routed_projection_bits",
            q_jang.get("routed_projection_bits"),
        ),
        (
            "jang_config.routed_projection_bits",
            jang_cfg.get("routed_projection_bits"),
        ),
        (
            "config.quantization.routed_expert_bit_plan.routed_projection_bits",
            cfg_bit_plan.get("routed_projection_bits"),
        ),
        (
            "config.routed_projection_bits",
            cfg.get("routed_projection_bits"),
        ),
    )
    for source, raw in candidates:
        projection_bits = _coerce_routed_projection_bits(raw)
        if projection_bits:
            return projection_bits, source
    return None, None


def _find_routed_down_layer_bit_plan(
    cfg: dict, jang_cfg: dict, q_jang: dict
) -> tuple[dict[str, int] | None, str | None]:
    routed_experts = q_jang.get("routed_experts")
    bit_plan = routed_experts.get("bit_plan") if isinstance(routed_experts, dict) else None
    q_cfg = cfg.get("quantization") if isinstance(cfg.get("quantization"), dict) else {}
    cfg_bit_plan = (
        q_cfg.get("routed_expert_bit_plan")
        if isinstance(q_cfg.get("routed_expert_bit_plan"), dict)
        else {}
    )
    candidates = (
        (
            "jang_config.quantization.routed_experts.bit_plan.routed_down_layer_bits",
            bit_plan.get("routed_down_layer_bits") if isinstance(bit_plan, dict) else None,
        ),
        (
            "jang_config.quantization.routed_down_layer_bits",
            q_jang.get("routed_down_layer_bits"),
        ),
        (
            "jang_config.routed_down_layer_bits",
            jang_cfg.get("routed_down_layer_bits"),
        ),
        (
            "config.quantization.routed_expert_bit_plan.routed_down_layer_bits",
            cfg_bit_plan.get("routed_down_layer_bits"),
        ),
        (
            "config.routed_down_layer_bits",
            cfg.get("routed_down_layer_bits"),
        ),
    )
    for source, raw in candidates:
        down_layer_bits = _coerce_routed_layer_bits(raw)
        if down_layer_bits:
            return down_layer_bits, source
    return None, None


def _weight_matmul_dispatch_status(codec: str) -> dict:
    if codec == "turboquant_codebook":
        return {
            "primary": "jang_tools_turboquant_custom_kernels",
            "uses_mlx_quantized_matmul": False,
            "metal_na_eligible": False,
            "reason": "turboquant_codebook_uses_custom_tq_kernels",
        }
    if codec == "affine_quantized_matmul":
        return {
            "primary": "mlx_affine_quantized_matmul",
            "uses_mlx_quantized_matmul": True,
            "metal_na_eligible": True,
            "reason": None,
        }
    return {
        "primary": "full_precision_or_unknown",
        "uses_mlx_quantized_matmul": False,
        "metal_na_eligible": False,
        "reason": "not_quantized_matmul",
    }


def _normalize_jangtq_mpp_nax_mode(raw: str | None = None) -> tuple[str, str | None]:
    """Normalize the internal JANGTQ acceleration env."""
    value = os.environ.get("JANGTQ_MPP_NAX", "auto") if raw is None else raw
    text = str(value or "").strip().lower()
    if text in ("", "0", "false", "no", "off"):
        return "off", None
    if text == "auto":
        return "auto", None
    if text in ("1", "true", "yes", "on"):
        return "on", None
    return "auto", f"invalid JANGTQ_MPP_NAX={value!r}; expected off/auto/on"


@functools.lru_cache(maxsize=1)
def _jangtq_mpp_nax_available() -> tuple[bool, str | None]:
    try:
        from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

        # This must stay a non-executing capability query. The implementation in
        # jang-tools intentionally avoids a smoke Metal dispatch because /health
        # can run after a large mmap-backed model has claimed most of the MLX
        # working set. Real generation kernels still compile/dispatch on demand
        # and fall back unless strict mode is enabled.
        return bool(mpp_nax_tensorops_available()), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _jangtq_mpp_nax_runtime_status(host: dict | None = None) -> dict:
    mode, mode_issue = _normalize_jangtq_mpp_nax_mode()
    requested = mode in ("auto", "on")
    host = host or _host_supports_metal_na()
    host_supported = bool(host.get("supported"))
    available, availability_error = (
        _jangtq_mpp_nax_available() if requested else (False, None)
    )

    active = bool(requested and host_supported and available)
    reason = None
    if mode_issue:
        reason = mode_issue
    elif not requested:
        reason = "disabled"
    elif not host_supported:
        reason = host.get("reason") or "host_not_known_na_capable"
    elif not available:
        reason = availability_error or "mpp_nax_tensorops_unavailable"

    return {
        "mode": mode,
        "requested": requested,
        "available": bool(available),
        "active": active,
        "uses_mlx_quantized_matmul": False,
        "reason": reason,
    }


def _bundle_index_tensor_prefix_status(
    bundle_path: str | None,
    prefix: str,
) -> tuple[bool, str | None]:
    if not bundle_path:
        return False, None
    try:
        from pathlib import Path

        index_path = Path(bundle_path) / "model.safetensors.index.json"
        if not index_path.is_file():
            return False, None
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            return False, "model.safetensors.index.json has no weight_map object"
        return any(str(key).startswith(prefix) for key in weight_map), None
    except Exception as exc:
        return False, f"model.safetensors.index.json read failed: {type(exc).__name__}: {exc}"


def _bundle_index_has_tensor_prefix(bundle_path: str | None, prefix: str) -> bool:
    return _bundle_index_tensor_prefix_status(bundle_path, prefix)[0]


def _bundle_index_mtp_layer_count(bundle_path: str | None) -> int | None:
    """Count distinct ``mtp.N.*`` layer indexes in the bundle's weight map.

    Returns the highest-N + 1 (i.e. layer count) when at least one mtp.N tensor
    is present, else None. Used to cross-check
    ``config.num_nextn_predict_layers`` so a bundle that ships partial MTP
    weights doesn't silently report ``artifact_available=True`` with a wrong
    layer count.
    """
    try:
        from .native_mtp import bundle_index_mtp_layer_count

        return bundle_index_mtp_layer_count(bundle_path)
    except Exception:
        pass
    if not bundle_path:
        return None
    try:
        from pathlib import Path
        import re as _re

        index_path = Path(bundle_path) / "model.safetensors.index.json"
        if not index_path.is_file():
            return None
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            return None
        layer_re = _re.compile(r"^mtp\.(\d+)(?:\.|$)")
        indexes: set[int] = set()
        for key in weight_map:
            m = layer_re.match(str(key))
            if m:
                indexes.add(int(m.group(1)))
        if not indexes:
            return None
        return len(indexes)
    except Exception:
        return None


_JANGMTP_FAMILY_ALIAS = {
    "qwen3_5_moe_text": "qwen3_5_moe",
    "hy3": "hy_v3",
    "zaya_vl": "zaya1_vl",
    "mininax": "minimax",
}

_JANGMTP_SUPPORTED_FAMILIES: set[str] = {
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_next",
    "gemma4",
    "nemotron",
    "nemotron_h",
    "hy_v3",
    "minimax",
    "zaya",
    "zaya1_vl",
}


def _normalize_jangmtp_family(name: str | None) -> str | None:
    if not name:
        return None
    normalized = str(name).strip().lower()
    return _JANGMTP_FAMILY_ALIAS.get(normalized, normalized)


def _bundle_mtp_family(bundle_path: str | None) -> str | None:
    if not bundle_path:
        return None
    try:
        from .model_config_registry import get_model_config_registry

        cfg = get_model_config_registry().lookup(bundle_path)
        family_name = getattr(cfg, "family_name", None)
        if family_name:
            normalized = _normalize_jangmtp_family(family_name)
            if normalized and normalized != "unknown":
                return normalized
    except Exception:
        # Keep telemetry functional even if registry is unavailable.
        pass

    cfg = _read_bundle_json(bundle_path, "config.json")
    return _normalize_jangmtp_family(cfg.get("model_type"))


def _bundle_mtp_runtime_supported(family: str | None) -> bool:
    if not family:
        return False
    return _normalize_jangmtp_family(family) in _JANGMTP_SUPPORTED_FAMILIES


def _model_mtp_status(bundle_path: str | None) -> dict:
    """Describe whether a bundle contains usable multi-token prediction heads."""
    try:
        from .native_mtp import inspect_native_mtp_bundle

        return inspect_native_mtp_bundle(bundle_path)
    except Exception:
        pass
    cfg = _read_bundle_json(bundle_path, "config.json")
    jang_cfg = _read_bundle_json(bundle_path, "jang_config.json")
    family = _bundle_mtp_family(bundle_path)
    raw_layers = cfg.get("num_nextn_predict_layers")
    invalid_config_layers = False
    try:
        config_layers = int(raw_layers) if raw_layers is not None else None
    except (TypeError, ValueError):
        config_layers = None
        invalid_config_layers = True
    drop_mtp_raw = jang_cfg.get("drop_mtp")
    # Strict boolean check: only literal True is treated as "drop". Truthy
    # non-bool values (e.g. "yes", 1) historically slipped past the
    # `drop_mtp is True` identity check and were silently treated as "don't
    # drop", which then claimed weights_present_runtime_unwired even when the
    # bundle author tried to disable MTP. Flag invalid types loudly.
    drop_mtp_invalid_type = (
        drop_mtp_raw is not None and not isinstance(drop_mtp_raw, bool)
    )
    drop_mtp = drop_mtp_raw if isinstance(drop_mtp_raw, bool) else None
    has_mtp_tensors, index_error = _bundle_index_tensor_prefix_status(
        bundle_path,
        "mtp.",
    )
    indexed_mtp_layer_count = _bundle_index_mtp_layer_count(bundle_path)

    issues: list[str] = []
    if index_error:
        issues.append(index_error)
    if invalid_config_layers:
        issues.append(
            "config.num_nextn_predict_layers is invalid; expected an integer"
        )
    if config_layers is not None and config_layers < 0:
        # Negative layer count is corrupt metadata. Pin loudly so callers
        # don't silently treat it as "configured without runtime".
        issues.append(
            "config.num_nextn_predict_layers is negative "
            f"({config_layers}); expected a non-negative integer"
        )
    if drop_mtp_invalid_type:
        issues.append(
            "jang_config.drop_mtp must be a boolean; got "
            f"{type(drop_mtp_raw).__name__}: {drop_mtp_raw!r}"
        )
    if config_layers and config_layers > 0 and drop_mtp is not True and not has_mtp_tensors:
        issues.append(
            "config expects MTP next-token prediction layers, but the bundle "
            "index has no mtp.* tensors"
        )
    if config_layers in (None, 0) and drop_mtp is not True and has_mtp_tensors:
        issues.append(
            "bundle indexes mtp.* tensors but config disables MTP runtime"
        )
    if drop_mtp is True and config_layers not in (None, 0):
        issues.append(
            "jang_config.drop_mtp=true but config.num_nextn_predict_layers="
            f"{config_layers}"
        )
    if (
        config_layers is not None
        and config_layers > 0
        and indexed_mtp_layer_count is not None
        and indexed_mtp_layer_count != config_layers
        and drop_mtp is not True
    ):
        # Bundle indexes a different number of distinct mtp.N layers than the
        # config claims. Either the converter wrote partial MTP weights
        # (artifact corruption) or num_nextn_predict_layers was edited
        # without re-running the converter. Either way, runtime acceleration
        # would silently use the wrong layer count.
        issues.append(
            f"config.num_nextn_predict_layers={config_layers} but bundle "
            f"index has {indexed_mtp_layer_count} distinct mtp.N layer(s)"
        )
    if drop_mtp is True and has_mtp_tensors:
        issues.append("jang_config.drop_mtp=true but bundle still indexes mtp.* tensors")

    artifact_available = bool(
        config_layers
        and config_layers > 0
        and drop_mtp is not True
        and has_mtp_tensors
        and not issues
    )
    runtime_supported = _bundle_mtp_runtime_supported(family)
    # vMLX does not yet have a verifier / accept-reject MTP decode path wired
    # through the scheduler. Presence of mtp.* weights is an artifact fact, not
    # proof that runtime acceleration is active.
    runtime_available = False
    runtime_reason = None
    if issues:
        status = "metadata_inconsistent"
        runtime_reason = "metadata_inconsistent"
    elif drop_mtp is True:
        status = "dropped"
        runtime_reason = "jang_config.drop_mtp=true"
    elif artifact_available:
        status = "weights_present_runtime_unwired"
        if runtime_supported:
            runtime_reason = (
                "MTP metadata is present and family is in the JangMTP "
                "candidate set; runtime decode wiring is not implemented in vMLX yet"
            )
        else:
            runtime_reason = (
                f"MTP metadata is present for family '{family or 'unknown'}', but "
                "this family is not currently on the JangMTP support map"
            )
    elif config_layers:
        status = "configured_without_runtime"
        runtime_reason = "config requests MTP but runtime requirements are incomplete"
    else:
        status = "not_configured"
        runtime_reason = "config does not request MTP"

    return {
        "config_num_nextn_predict_layers": config_layers,
        "jang_drop_mtp": drop_mtp,
        "index_has_mtp_tensors": has_mtp_tensors,
        "artifact_available": artifact_available,
        "family": family,
        "runtime_supported": runtime_supported,
        "runtime_available": runtime_available,
        "runtime_reason": runtime_reason,
        "status": status,
        "issues": issues,
    }


def _model_mtp_status_with_loaded_runtime(bundle_path: str | None) -> dict:
    """MTP artifact status plus the currently loaded runtime activation bit."""
    status = dict(_model_mtp_status(bundle_path))
    native_disabled = os.environ.get("VMLINUX_NATIVE_MTP", "1") in (
        "0",
        "false",
        "FALSE",
        "no",
        "NO",
        "off",
        "OFF",
    )
    status["request_policy"] = (
        "disabled" if native_disabled else os.environ.get(
            "VMLINUX_NATIVE_MTP_SAMPLING_POLICY",
            _native_mtp_sampling_policy,
        )
    )
    status["request_gate"] = "temperature=0,repetition_penalty=1.0"
    try:
        from .native_mtp import model_has_native_mtp_runtime

        model = _get_raw_model_from_engine()
        active = bool(model is not None and model_has_native_mtp_runtime(model))
        status["runtime_active"] = False if native_disabled else active
        if native_disabled and status.get("runtime_available"):
            status["runtime_reason"] = "native MTP disabled by --disable-native-mtp/VMLINUX_NATIVE_MTP=0"
        if (not native_disabled) and active and status.get("runtime_available"):
            status["status"] = "native_runtime_active"
            scope = status.get("runtime_scope") or "text"
            status["runtime_reason"] = f"native MTP runtime is active for {scope}"
            if status.get("effective_depth") is None:
                try:
                    from .native_mtp import native_mtp_effective_depth

                    depth, source = native_mtp_effective_depth()
                    status["effective_depth"] = depth
                    status["effective_depth_source"] = source
                except Exception:
                    pass
    except Exception:
        pass
    return status


def _model_quantization_status(bundle_path: str | None) -> dict:
    """Describe the loaded weight codec from bundle metadata without guessing.

    This is intentionally separate from live KV-cache quantization. Weight
    quantization decides which matmul kernel family is used; TQ-KV decides how
    prefix/cache tensors are stored.
    """
    cfg = _read_bundle_json(bundle_path, "config.json")
    jang_cfg = _read_bundle_json(bundle_path, "jang_config.json")
    q_cfg = cfg.get("quantization") if isinstance(cfg.get("quantization"), dict) else {}
    q_jang = (
        jang_cfg.get("quantization")
        if isinstance(jang_cfg.get("quantization"), dict)
        else {}
    )

    weight_format = (
        cfg.get("weight_format")
        or jang_cfg.get("weight_format")
        or q_jang.get("format")
        or q_jang.get("weight_format")
        or q_cfg.get("format")
    )
    backend = q_jang.get("quantization_backend") or q_jang.get("backend")
    profile = q_jang.get("profile") or jang_cfg.get("profile") or cfg.get("profile")
    mxtq_bits = (
        cfg.get("mxtq_bits")
        or jang_cfg.get("mxtq_bits")
        or q_jang.get("mxtq_bits")
        or q_jang.get("bits_default")
    )
    affine_bits = (
        cfg.get("affine_bits")
        or jang_cfg.get("affine_bits")
        or q_jang.get("affine_bits")
    )
    profile_bits = _jangtq_bits_from_profile(profile)
    normalized_format = str(weight_format or "").lower()
    is_affine_jang_metadata = normalized_format in ("affine", "jang", "jjqf", "mxq")
    bits_by_role = affine_bits if is_affine_jang_metadata and affine_bits else mxtq_bits
    routed_expert_bits_raw = (
        bits_by_role.get("routed_expert")
        if isinstance(bits_by_role, dict)
        else bits_by_role
    )
    if routed_expert_bits_raw is None:
        routed_expert_bits_raw = profile_bits
    routed_expert_bits = None
    routed_expert_bits_by_projection = None
    routed_expert_bits_label = None
    if isinstance(routed_expert_bits_raw, dict):
        order = ("gate_proj", "up_proj", "down_proj")
        routed_expert_bits_by_projection = {
            key: int(routed_expert_bits_raw[key])
            for key in order
            if key in routed_expert_bits_raw
        }
        for key, value in routed_expert_bits_raw.items():
            if key not in routed_expert_bits_by_projection:
                routed_expert_bits_by_projection[str(key)] = int(value)
        short = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }
        routed_expert_bits_label = (
            "/".join(
                f"{short.get(key, key)}={value}"
                for key, value in routed_expert_bits_by_projection.items()
            )
            + "-bit"
        )
    elif routed_expert_bits_raw is not None:
        routed_expert_bits = int(routed_expert_bits_raw)
    target_bits = (
        q_jang.get("target_bits")
        or q_jang.get("bits")
        or routed_expert_bits
        or (
            min(routed_expert_bits_by_projection.values())
            if routed_expert_bits_by_projection
            else None
        )
        or profile_bits
        or q_cfg.get("bits")
    )
    actual_bits = (
        q_jang.get("actual_bits")
        or q_jang.get("actual_bits_per_weight")
        or jang_cfg.get("actual_bits")
    )
    routed_layer_bits, routed_layer_bits_source = _find_routed_layer_bit_plan(
        cfg, jang_cfg, q_jang
    )
    routed_projection_bits, routed_projection_bits_source = (
        _find_routed_projection_bit_plan(cfg, jang_cfg, q_jang)
    )
    routed_down_layer_bits, routed_down_layer_bits_source = (
        _find_routed_down_layer_bit_plan(cfg, jang_cfg, q_jang)
    )
    weight_index = _bundle_weight_index_status(bundle_path)

    try:
        from pathlib import Path

        root = Path(bundle_path) if bundle_path else None
        has_jangtq_sidecar = bool(root and (root / "jangtq_runtime.safetensors").is_file())
        has_jang_config = bool(root and (root / "jang_config.json").is_file())
        has_prestacked_bundle = _bundle_has_prestacked_jangtq(bundle_path)
    except Exception:
        has_jangtq_sidecar = False
        has_jang_config = False
        has_prestacked_bundle = False

    normalized_backend = str(backend or "").lower()
    if normalized_format in ("mxtq", "jangtq") or has_jangtq_sidecar:
        codec = "turboquant_codebook"
    elif normalized_format in ("mxfp4", "mxfp8", "nvfp4", "fp4", "fp8"):
        codec = "affine_quantized_matmul"
    elif normalized_format in ("jang", "jjqf", "mxq"):
        codec = "affine_quantized_matmul"
    elif normalized_backend in ("mlx", "mlx_uniform", "affine"):
        codec = "affine_quantized_matmul"
    elif q_cfg or target_bits or actual_bits:
        codec = "affine_quantized_matmul"
    else:
        codec = "full_precision_or_unknown"

    model_types = {
        str(cfg.get("model_type") or "").lower(),
        str(
            cfg.get("text_config", {}).get("model_type")
            if isinstance(cfg.get("text_config"), dict)
            else ""
        ).lower(),
    }
    hybrid_conv_families = {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "nemotron_h",
        "nemotron_h_v2",
        "mamba",
        "mamba2",
        "codestral_mamba",
        "bailing_hybrid",
        "bailing_moe_v2_5",
        "granitemoehybrid",
        "lfm2_moe",
    }
    compat_warnings = []
    if codec == "turboquant_codebook" and (
        profile_bits == 1 or routed_expert_bits == 1
    ):
        compat_warnings.append(
            "JANGTQ1 is recognized for metadata compatibility only and is not "
            "production-supported; use JANGTQ2/JANGTQ4 or affine JANG for "
            "release-quality generation."
        )
    if (
        has_jang_config
        and model_types.intersection(hybrid_conv_families)
        and q_jang.get("passthrough_tensor_count") is None
    ):
        compat_warnings.append(
            "Hybrid SSM/Mamba bundle does not report fp16 state passthrough "
            "metadata; vMLX applies a runtime grouped-Conv1d layout backstop, "
            "but direct mlx_lm users should re-convert with a newer jang build."
        )
    routed_experts_cfg = (
        q_jang.get("routed_experts")
        if isinstance(q_jang.get("routed_experts"), dict)
        else {}
    )
    runtime_cfg = (
        jang_cfg.get("runtime")
        if isinstance(jang_cfg.get("runtime"), dict)
        else {}
    )
    affine_dsv4_routed = (
        "deepseek_v4" in model_types
        and codec == "affine_quantized_matmul"
        and (
            normalized_format == "affine"
            or str(q_jang.get("method") or "").lower() == "affine"
            or str(routed_experts_cfg.get("codec") or "").lower() == "affine"
        )
        and routed_experts_cfg
    )
    dsv4_affine_verified = bool(
        jang_cfg.get("coherency_verified")
        or jang_cfg.get("runtime_verified")
        or runtime_cfg.get("coherency_verified")
        or runtime_cfg.get("ar_coherency_verified")
    )
    if affine_dsv4_routed and not dsv4_affine_verified:
        compat_warnings.append(
            "DSV4 affine routed JANG runtime is loaded but not coherency-verified; "
            "require a live DeepSeek-V4 AR logits/decode gate before production use."
        )

    result = {
        "codec": codec,
        "weight_format": weight_format,
        "backend": backend,
        "profile": profile,
        "mxtq_bits": mxtq_bits if not isinstance(mxtq_bits, dict) else None,
        "mxtq_bits_by_role": mxtq_bits if isinstance(mxtq_bits, dict) else None,
        "affine_bits": affine_bits if not isinstance(affine_bits, dict) else None,
        "affine_bits_by_role": affine_bits if isinstance(affine_bits, dict) else None,
        "routed_expert_bits": routed_expert_bits,
        "routed_expert_bits_by_projection": routed_expert_bits_by_projection,
        "routed_expert_bits_label": routed_expert_bits_label,
        "target_bits": target_bits,
        "actual_bits": actual_bits,
        "config_bits": q_cfg.get("bits"),
        "group_size": q_cfg.get("group_size") or q_jang.get("group_size") or q_jang.get("block_size"),
        "routed_layer_bits": routed_layer_bits,
        "routed_layer_bits_source": routed_layer_bits_source,
        "routed_projection_bits": routed_projection_bits,
        "routed_projection_bits_source": routed_projection_bits_source,
        "routed_down_layer_bits": routed_down_layer_bits,
        "routed_down_layer_bits_source": routed_down_layer_bits_source,
        "passthrough_bit_widths_used": q_jang.get("passthrough_bit_widths_used"),
        "passthrough_tensor_count": q_jang.get("passthrough_tensor_count"),
        "weight_index": weight_index,
        "weight_matmul_dispatch": _weight_matmul_dispatch_status(codec),
        "compat_warnings": compat_warnings or None,
        "sidecar": {
            "jang_config": has_jang_config,
            "jangtq_runtime": has_jangtq_sidecar,
            "prestacked_bundle": has_prestacked_bundle,
        },
    }
    return {k: v for k, v in result.items() if v is not None}


def _model_routing_status(bundle_path: str | None) -> dict:
    """Expose trained MoE active-expert K and any explicit JANGTQ override."""
    cfg = _read_bundle_json(bundle_path, "config.json")

    def _config_sections():
        yield "config", cfg
        for key in ("text_config", "language_config", "llm_config"):
            sub = cfg.get(key)
            if isinstance(sub, dict):
                yield f"config.{key}", sub

    def _first_int(keys: tuple[str, ...]) -> tuple[int | None, str | None]:
        for prefix, section in _config_sections():
            for key in keys:
                raw = section.get(key)
                if raw is None:
                    continue
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value, f"{prefix}.{key}"
        return None, None

    trained_k, trained_source = _first_int(
        (
            "num_experts_per_tok",
            "top_k_experts",
            "moe_router_topk",
            "num_activated_experts",
            "top_k",
        )
    )
    routed_experts, _routed_source = _first_int(
        ("n_routed_experts", "num_local_experts", "num_experts")
    )

    if trained_k is None and routed_experts is None:
        return {}

    effective_k = trained_k
    effective_source = "trained_default" if trained_k is not None else None

    result = {
        "trained_active_experts": trained_k,
        "trained_active_experts_source": trained_source,
        "n_routed_experts": routed_experts,
        "override_env": None,
        "effective_active_experts": effective_k,
        "effective_active_experts_source": effective_source,
    }
    return {k: v for k, v in result.items() if v is not None or k == "override_env"}


def _mlx_metal_na_status() -> dict:
    """Return cached MLX metallib NA symbol availability for this process."""
    try:
        import importlib
        import mlx
        from pathlib import Path

        mlx_file = getattr(mlx, "__file__", None)
        if mlx_file:
            mlx_dir = Path(mlx_file).resolve().parent
        else:
            # `mlx` can be a namespace package with `__file__ == None`; the
            # compiled extension path lives on `mlx.core.__file__`.
            mlx_core = importlib.import_module("mlx.core")
            mlx_dir = Path(getattr(mlx_core, "__file__")).resolve().parent

        metallib = mlx_dir / "lib" / "mlx.metallib"
        key = str(metallib)
        if key in _metal_na_status_cache:
            return dict(_metal_na_status_cache[key])
        if not metallib.is_file():
            status = {
                "metallib": key,
                "available": False,
                "nax_symbols": 0,
                "naxtile_symbols": 0,
                "affine_qmm_symbols": 0,
                "affine_gather_qmm_symbols": 0,
            }
        else:
            data = metallib.read_bytes()
            nax = data.count(b"_nax_")
            naxtile = data.count(b"NAXTile")
            affine_qmm = data.count(b"affine_qmm")
            affine_gather_qmm = data.count(b"affine_gather_qmm")
            status = {
                "metallib": key,
                "available": bool(nax or naxtile or affine_qmm or affine_gather_qmm),
                "nax_symbols": nax,
                "naxtile_symbols": naxtile,
                "affine_qmm_symbols": affine_qmm,
                "affine_gather_qmm_symbols": affine_gather_qmm,
            }
        _metal_na_status_cache[key] = dict(status)
        return status
    except Exception as exc:
        return {
            "available": False,
            "nax_symbols": 0,
            "naxtile_symbols": 0,
            "affine_qmm_symbols": 0,
            "affine_gather_qmm_symbols": 0,
            "error": type(exc).__name__,
        }


def _host_supports_metal_na() -> dict:
    """Best-effort host check for Apple NA-capable GPU generations."""
    if platform.system() != "Darwin":
        return {"supported": False, "reason": "not_darwin"}
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        brand = (proc.stdout or "").strip()
    except Exception:
        brand = ""
    # Be conservative: report active only on known M5+ Apple Silicon names.
    supported = bool(re.search(r"\bApple M([5-9]|\d{2,})\b", brand))
    return {
        "supported": supported,
        "brand": brand or None,
        "reason": None if supported else "host_not_known_na_capable",
    }


def _model_acceleration_status(bundle_path: str | None = None) -> dict:
    quantization = _model_quantization_status(bundle_path or _model_path)
    codec = quantization.get("codec")
    na_status = _mlx_metal_na_status()
    host = _host_supports_metal_na()

    if codec == "turboquant_codebook":
        mpp_nax = _jangtq_mpp_nax_runtime_status(host)
        active = bool(mpp_nax.get("active"))
        kernel_type = (
            "turboquant_codebook_mpp_nax" if active else "turboquant_codebook"
        )
        reason = None
        if not active:
            reason = (
                "turboquant_custom_kernels_do_not_use_mlx_na"
                if mpp_nax.get("mode") == "off"
                else (
                    mpp_nax.get("reason")
                    or "turboquant_custom_kernels_do_not_use_mlx_na"
                )
            )
        metal_na_capable = bool(
            mpp_nax.get("requested") and host.get("supported") and mpp_nax.get("available")
        )
    elif codec == "affine_quantized_matmul":
        kernel_type = "affine_quantized_matmul"
        active = bool(na_status.get("available") and host.get("supported"))
        reason = None if active else "mlx_na_symbols_or_host_support_unavailable"
        metal_na_capable = True
    else:
        kernel_type = "full_precision_or_unknown"
        active = False
        reason = "not_affine_quantized_matmul"
        metal_na_capable = False

    result = {
        "kernel_type": kernel_type,
        "metal_na_capable": metal_na_capable,
        "metal_na_active_on_host": active,
        "reason": reason,
        "metal_na_symbols": {
            "available": bool(na_status.get("available")),
            "nax_symbols": na_status.get("nax_symbols", 0),
            "naxtile_symbols": na_status.get("naxtile_symbols", 0),
            "affine_qmm_symbols": na_status.get("affine_qmm_symbols", 0),
            "affine_gather_qmm_symbols": na_status.get("affine_gather_qmm_symbols", 0),
        },
        "host": host,
    }
    if codec == "turboquant_codebook":
        result["jangtq_acceleration"] = mpp_nax
    return result


def _turboquant_kv_cache_status(engine=None, scheduler=None) -> dict:
    """Detect live TurboQuant KV cache patching across LLM and MLLM engines."""
    tq_active = bool(getattr(scheduler, "_tq_active", False)) if scheduler else False

    def _check_make_cache(obj):
        mc = getattr(obj, "make_cache", None)
        return mc is not None and getattr(mc, "__name__", "") in (
            "_tq_make_cache",
            "_turboquant_make_cache",
        )

    if not tq_active and engine is not None:
        # Walk the candidate chain and probe each level + .language_model.
        _seen = set()
        _stack = []
        for _attr in ("_model", "model"):
            _v = _diagnostic_attr(engine, _attr)
            if _v is not None:
                _stack.append(_v)
        while _stack:
            _v = _stack.pop()
            if id(_v) in _seen:
                continue
            _seen.add(id(_v))
            if _check_make_cache(_v):
                tq_active = True
                break
            for _next_attr in ("model", "language_model"):
                _nxt = _diagnostic_attr(_v, _next_attr)
                if _nxt is not None and id(_nxt) not in _seen:
                    _stack.append(_nxt)

    if tq_active:
        status = {
            "enabled": True,
            "default_bits": 3,
            "critical_bits": 4,
            "critical_layers": "first 3 + last 3",
        }
        cfg = getattr(scheduler, "config", None) if scheduler is not None else None
        if cfg is not None:
            batch_api = bool(getattr(scheduler, "_tq_batch_api", False))
            single_sequence_only = not batch_api
            runtime = {
                "batch_api": "turboquant_kv_v1" if batch_api else None,
                "single_sequence_only": single_sequence_only,
                "effective_max_num_seqs": getattr(cfg, "max_num_seqs", None),
                "effective_prefill_batch_size": getattr(
                    cfg, "prefill_batch_size", None
                ),
                "effective_completion_batch_size": getattr(
                    cfg, "completion_batch_size", None
                ),
            }
            if single_sequence_only:
                runtime["single_sequence_reason"] = "cache_extend_not_supported"
            status.update(
                {k: v for k, v in runtime.items() if v is not None}
            )
        return status
    return {"enabled": False}


def _current_model_config():
    """Return the loaded model registry config for status endpoints."""
    model_key = _model_path or _model_name or ""
    if not model_key:
        return None
    try:
        from .model_config_registry import get_model_config_registry

        return get_model_config_registry().lookup(model_key)
    except Exception:
        return None


def _native_cache_status(scheduler=None, *, family: str | None = None, cfg=None) -> dict:
    """Report architecture-native cache contracts separately from generic TQ-KV."""
    if scheduler is None:
        return {}

    scheduler_family = str(getattr(scheduler, "_model_type_for_runtime", "") or "")
    family_name = family or scheduler_family
    cache_subtype = getattr(cfg, "cache_subtype", None) if cfg is not None else None
    block_aware_cache = getattr(scheduler, "block_aware_cache", None)
    paged_cache_manager = getattr(scheduler, "paged_cache_manager", None)
    block_disk_store = (
        getattr(paged_cache_manager, "_disk_store", None)
        if paged_cache_manager is not None
        else None
    )

    if getattr(scheduler, "_uses_dsv4_cache", False) or family_name == "deepseek_v4":
        pool_quant_enabled = str(os.environ.get("DSV4_POOL_QUANT", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        layout: dict[str, Any] = {}
        try:
            model = getattr(scheduler, "model", None)
            cache_layers = model.make_cache() if model is not None else None
            if cache_layers:
                ratios: list[int | None] = []
                windows: list[int] = []
                for layer_cache in cache_layers:
                    ratio = getattr(layer_cache, "compress_ratio", None)
                    try:
                        ratios.append(int(ratio) if ratio is not None else None)
                    except (TypeError, ValueError):
                        ratios.append(None)
                    local_cache = getattr(layer_cache, "local", None)
                    window = getattr(local_cache, "max_size", None)
                    if window is not None:
                        try:
                            windows.append(int(window))
                        except (TypeError, ValueError):
                            pass
                ratio_counts: dict[str, int] = {}
                for ratio in ratios:
                    key = "unknown" if ratio is None else str(ratio)
                    ratio_counts[key] = ratio_counts.get(key, 0) + 1
                layout = {
                    "layers": len(cache_layers),
                    "compress_ratios": [
                        "unknown" if ratio is None else ratio for ratio in ratios
                    ],
                    "compress_ratio_counts": ratio_counts,
                }
                if windows:
                    layout["sliding_window"] = windows[0]
                    if len(set(windows)) > 1:
                        layout["sliding_windows"] = windows
        except Exception:
            layout = {}
        status = {
            "family": "deepseek_v4",
            "schema": "deepseek_v4_v7",
            "cache_type": "native_composite",
            "components": [
                "swa_local",
                "csa_compressed_pool",
                "hca_compressed_pool",
                "incomplete_tail_state",
            ],
            "generic_turboquant_kv": {"enabled": False, "reason": "native_dsv4_composite"},
            "pool_quant": {
                "enabled": pool_quant_enabled,
                "env": os.environ.get("DSV4_POOL_QUANT", "0"),
            },
            "layer_cache_roles": {
                "ratio_0": "swa_local_only",
                "ratio_4": "csa_overlap_compressed_pool_plus_indexer",
                "ratio_128": "hca_compressed_pool",
            },
            "cache_store_policy": {
                "prompt_boundary_snapshot": "preferred",
                "post_generation_trim": "disabled_unless_explicit_unsafe_override",
                "generic_kv_quantization": "forced_off",
            },
            "decode_activation_order": [
                "make_cache_builds_per_layer_swa_or_composite_cache",
                "prefill_updates_local_swa_and_compressor_indexer_pools",
                "prompt_boundary_snapshot_or_clean_reprefill_is_stored_for_prefix",
                "decode_appends_new_tokens_to_live_composite_cache",
                "cache_hit_restores_full_nested_state_and_refeeds_last_prompt_token",
            ],
            "prefix": bool(block_aware_cache is not None),
            "paged": bool(block_aware_cache is not None and paged_cache_manager is not None),
            "block_disk_l2": bool(block_disk_store is not None),
        }
        status.update(layout)
        return status

    if (
        family_name == "zaya"
        or cache_subtype == "zaya_cca"
        or getattr(scheduler, "_uses_zaya_cache", False)
    ):
        return {
            "family": "zaya",
            "schema": "zaya_cca_v1",
            "cache_type": "typed_cca",
            "components": [
                "standard_kv",
                "cca_conv_state",
                "cca_prev_hidden",
                "moe_no_state_slots",
            ],
            "generic_turboquant_kv": {"enabled": False, "reason": "cca_state_is_path_dependent"},
            "prefix": bool(block_aware_cache is not None),
            "paged": bool(block_aware_cache is not None and paged_cache_manager is not None),
            "block_disk_l2": bool(block_disk_store is not None),
        }

    if (
        getattr(scheduler, "_is_hybrid", False)
        and not getattr(scheduler, "_uses_dsv4_cache", False)
        and not getattr(scheduler, "_uses_zaya_cache", False)
    ):
        ssm_cache = getattr(scheduler, "_ssm_state_cache", None)
        ssm_entries = None
        try:
            ssm_entries = len(getattr(ssm_cache, "_store", {}) or {})
        except Exception:
            pass
        hybrid_live_tq_policy = getattr(
            scheduler, "_hybrid_live_tq_policy", None
        )
        hybrid_tq_enabled = bool(
            getattr(scheduler, "_tq_active", False)
            and hybrid_live_tq_policy == "attention_kv_only"
        )
        hybrid_tq_attention_layers = list(
            getattr(scheduler, "_hybrid_live_tq_attention_layers", []) or []
        )
        hybrid_tq_companion_layers = list(
            getattr(scheduler, "_hybrid_live_tq_companion_layers", []) or []
        )
        try:
            stored_kv_bits = int(getattr(scheduler, "_kv_cache_bits", 0) or 0)
        except (TypeError, ValueError):
            stored_kv_bits = 0
        try:
            stored_kv_group = int(getattr(scheduler, "_kv_cache_group_size", 64) or 64)
        except (TypeError, ValueError):
            stored_kv_group = 64
        return {
            "family": family_name or scheduler_family or "hybrid",
            "schema": "hybrid_ssm_v1",
            "cache_type": "hybrid_ssm_typed",
            "components": [
                "attention_kv",
                "ssm_companion_state",
                "async_rederive",
            ],
            "generic_turboquant_kv": {
                "enabled": hybrid_tq_enabled,
                "reason": (
                    "hybrid_attention_kv_only"
                    if hybrid_tq_enabled
                    else "hybrid_ssm_state"
                ),
            },
            "live_attention_tq_kv": {
                "enabled": hybrid_tq_enabled,
                "mode": "live_decode" if hybrid_tq_enabled else None,
                "applies_to": "attention_kv_layers_only",
                "ssm_policy": "native_full_precision_companion_state",
                "attention_layers": hybrid_tq_attention_layers,
                "companion_layers": hybrid_tq_companion_layers,
            },
            "attention_kv_storage_quantization": {
                "enabled": stored_kv_bits > 0,
                "mode": "storage_boundary",
                "bits": stored_kv_bits if stored_kv_bits > 0 else None,
                "group_size": stored_kv_group if stored_kv_bits > 0 else None,
                "applies_to": "attention_kv_layers_only",
                "ssm_policy": "native_companion_state",
                "rederive": "async_clean_prefill_on_miss_or_warm_pass",
            },
            "prefix": bool(block_aware_cache is not None),
            "paged": bool(block_aware_cache is not None and paged_cache_manager is not None),
            "block_disk_l2": bool(block_disk_store is not None),
            "ssm_entries": ssm_entries,
            "kv_layer_indices": list(getattr(scheduler, "_hybrid_kv_positions", []) or []),
        }

    if (
        getattr(scheduler, "_mixed_attention_cache_model", False)
        or cache_subtype in {"mixed_swa_kv", "step3p7_full_sliding_kv"}
    ):
        tq_enabled = bool(getattr(scheduler, "_tq_active", False))
        try:
            stored_kv_bits = int(getattr(scheduler, "_kv_cache_bits", 0) or 0)
        except (TypeError, ValueError):
            stored_kv_bits = 0
        try:
            stored_kv_group = int(getattr(scheduler, "_kv_cache_group_size", 64) or 64)
        except (TypeError, ValueError):
            stored_kv_group = 64
        return {
            "family": family_name or scheduler_family or "mixed_attention",
            "schema": "mixed_swa_kv_v1",
            "cache_type": "mixed_swa_kv",
            "cache_subtype": cache_subtype or "mixed_swa_kv",
            "components": [
                "full_attention_kv",
                "sliding_window_kv",
                "rotating_window_metadata",
            ],
            "generic_turboquant_kv": {
                "enabled": tq_enabled,
                "reason": "live_tq_kv" if tq_enabled else "not_active",
            },
            "storage_quantization": {
                "enabled": stored_kv_bits > 0,
                "mode": "storage_boundary",
                "bits": stored_kv_bits if stored_kv_bits > 0 else None,
                "group_size": stored_kv_group if stored_kv_bits > 0 else None,
                "applies_to": "full_and_sliding_attention_kv",
                "metadata_policy": "preserve_rotating_window_metadata",
            },
            "prefix": bool(block_aware_cache is not None),
            "paged": bool(block_aware_cache is not None and paged_cache_manager is not None),
            "block_disk_l2": bool(block_disk_store is not None),
        }

    tq_enabled = bool(getattr(scheduler, "_tq_active", False))
    try:
        stored_kv_bits = int(getattr(scheduler, "_kv_cache_bits", 0) or 0)
    except (TypeError, ValueError):
        stored_kv_bits = 0
    try:
        stored_kv_group = int(getattr(scheduler, "_kv_cache_group_size", 64) or 64)
    except (TypeError, ValueError):
        stored_kv_group = 64
    return {
        "family": family_name or scheduler_family or "plain_attention",
        "schema": "plain_kv_v1",
        "cache_type": "paged_kv",
        "components": ["attention_kv"],
        "generic_turboquant_kv": {
            "enabled": tq_enabled,
            "reason": "plain_attention_kv" if tq_enabled else "not_active",
        },
        "storage_quantization": {
            "enabled": stored_kv_bits > 0,
            "bits": stored_kv_bits if stored_kv_bits > 0 else None,
            "group_size": stored_kv_group if stored_kv_bits > 0 else None,
        },
        "prefix": bool(block_aware_cache is not None),
        "paged": bool(block_aware_cache is not None and paged_cache_manager is not None),
        "block_disk_l2": bool(block_disk_store is not None),
    }


def _stat_int(stats: dict[str, Any] | None, *keys: str) -> int:
    """Return the first integer-like stat value for any key."""
    if not isinstance(stats, dict):
        return 0
    for key in keys:
        value = stats.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _kv_cache_quantization_status(scheduler: Any | None) -> dict[str, Any] | None:
    """Return a truthful stored-KV quantization status block.

    The scheduler keeps ``_kv_cache_bits=0`` when stored-cache quantization is
    inactive. Exposing that raw integer makes the UI render a bogus "0-bit"
    cache codec, so both /health and /v1/cache/stats normalize through this
    helper.
    """
    if scheduler is None or not hasattr(scheduler, "_kv_cache_bits"):
        return None
    try:
        bits = int(getattr(scheduler, "_kv_cache_bits", 0) or 0)
    except (TypeError, ValueError):
        bits = 0
    if bits <= 0:
        return {"enabled": False}
    return {
        "enabled": True,
        "bits": bits,
        "group_size": getattr(scheduler, "_kv_cache_group_size", 64),
    }


def _ssm_companion_snapshot(scheduler: Any) -> dict[str, Any] | None:
    """Collect SSM companion L1/L2 stats without assuming scheduler flavor."""
    if scheduler is None:
        return None
    ssm_cache = getattr(scheduler, "_ssm_state_cache", None)
    if ssm_cache is None:
        batch_generator = getattr(scheduler, "batch_generator", None)
        if batch_generator is not None:
            ssm_cache = getattr(batch_generator, "_ssm_state_cache", None)
    if ssm_cache is None:
        # MLLM creates HybridSSMStateCache lazily with the batch generator, but
        # its scheduler owns the L2 disk store at startup. Surface that
        # configured L2 tier before the first request so /health and
        # /v1/cache/stats don't falsely imply SSM persistence is absent.
        disk = getattr(scheduler, "_ssm_companion_disk_store", None)
        if disk is None:
            return None
        cfg = getattr(scheduler, "config", None)
        max_entries = int(getattr(cfg, "ssm_state_cache_size", 0) or 0)
        max_mb = getattr(cfg, "ssm_state_cache_max_mb", None)
        snapshot: dict[str, Any] = {
            "entries": 0,
            "max_entries": max_entries,
            "nbytes": 0,
            "nbytes_mb": 0.0,
            "max_bytes": (
                int(max_mb) * 1024 * 1024
                if max_mb is not None
                else None
            ),
            "max_bytes_mb": (
                round(float(max_mb), 2)
                if max_mb is not None
                else None
            ),
            "disk_enabled": True,
            "disk_directory": str(getattr(disk, "directory", None) or ""),
        }
        if hasattr(disk, "stats"):
            try:
                snapshot["disk"] = disk.stats()
            except Exception as exc:
                snapshot["disk"] = {"enabled": True, "error": str(exc)}
        return snapshot

    entries = getattr(ssm_cache, "size", None)
    if entries is None:
        entries = len(getattr(ssm_cache, "_store", {}) or {})
    max_entries = getattr(ssm_cache, "max_entries", None)
    if max_entries is None:
        max_entries = getattr(ssm_cache, "_max_entries", 0)
    nbytes = int(getattr(ssm_cache, "total_nbytes", 0) or 0)
    snapshot: dict[str, Any] = {
        "entries": int(entries or 0),
        "max_entries": int(max_entries or 0),
        "nbytes": nbytes,
        "nbytes_mb": round(nbytes / (1024 * 1024), 2),
        "disk_enabled": bool(getattr(ssm_cache, "disk_enabled", False)),
        "disk_directory": getattr(ssm_cache, "disk_directory", None),
    }
    disk = getattr(ssm_cache, "_disk", None)
    if disk is not None and hasattr(disk, "stats"):
        try:
            snapshot["disk"] = disk.stats()
        except Exception as exc:
            snapshot["disk"] = {"enabled": True, "error": str(exc)}
    return snapshot


def _cache_telemetry_snapshot(scheduler: Any | None = None) -> dict[str, Any]:
    """Summarize effective cache state for /health and /v1/cache/stats."""
    scheduler = scheduler if scheduler is not None else (_get_scheduler() if _engine else None)
    result: dict[str, Any] = {}

    scheduler_cache = None
    if _engine is not None and hasattr(_engine, "get_cache_stats"):
        try:
            scheduler_cache = _engine.get_cache_stats()
        except Exception:
            scheduler_cache = None
    if scheduler_cache:
        result["scheduler_cache"] = scheduler_cache

    disk_cache = getattr(scheduler, "disk_cache", None) if scheduler else None
    if disk_cache is not None and hasattr(disk_cache, "stats"):
        try:
            result["disk_cache"] = disk_cache.stats()
        except Exception as exc:
            result["disk_cache"] = {"enabled": True, "error": str(exc)}

    paged_mgr = getattr(scheduler, "paged_cache_manager", None) if scheduler else None
    block_store = getattr(paged_mgr, "_disk_store", None) if paged_mgr else None
    if block_store is not None and hasattr(block_store, "get_stats"):
        try:
            result["block_disk_cache"] = block_store.get_stats()
        except Exception as exc:
            result["block_disk_cache"] = {"enabled": True, "error": str(exc)}

    ssm = _ssm_companion_snapshot(scheduler)
    if ssm is not None:
        result["ssm_companion"] = ssm

    if result:
        disk_tokens = _stat_int(result.get("disk_cache"), "total_tokens_on_disk", "total_cached_tokens")
        block_tokens = _stat_int(result.get("block_disk_cache"), "total_tokens_on_disk", "total_cached_tokens")
        ssm_disk = result.get("ssm_companion", {}).get("disk") if isinstance(result.get("ssm_companion"), dict) else None
        ssm_tokens = _stat_int(ssm_disk, "total_tokens_on_disk", "total_cached_tokens")
        l2_store_sum = disk_tokens + block_tokens + ssm_tokens
        result["totals"] = {
            "ram_tokens_cached": _stat_int(
                scheduler_cache,
                "total_tokens_cached",
                "total_cached_tokens",
                "tokens_cached",
            ),
            "l2_prompt_tokens_on_disk": disk_tokens,
            "l2_block_tokens_on_disk": block_tokens,
            "l2_ssm_tokens_on_disk": ssm_tokens,
            "ssm_tokens_on_disk": ssm_tokens,
            "l2_tokens_on_disk": l2_store_sum,
            "l2_tokens_on_disk_store_sum": l2_store_sum,
            "l2_tokens_on_disk_note": "store_sum_across_prompt_block_ssm_l2; tiers_may_overlap",
        }
    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    mcp_info = None
    if _mcp_manager is not None:
        connected = sum(
            1 for s in _mcp_manager.get_server_status() if s.state.value == "connected"
        )
        total = len(_mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_mcp_manager.get_all_tools(policy=_mcp_policy)),
        }

    engine_stats = _engine.get_stats() if _engine else {}
    scheduler = _get_scheduler() if _engine else None

    # Differentiate status: "healthy" when model is loaded, "no_model" otherwise
    # Image models don't use _engine — they use _image_gen
    if _standby_state:
        status = f"standby_{_standby_state}"  # "standby_soft" or "standby_deep"
    elif _model_type == "image":
        status = (
            "healthy"
            if (_image_gen is not None and _image_gen.is_loaded)
            else "no_model"
        )
    else:
        status = "healthy" if _engine is not None else "no_model"

    # Include Metal GPU memory info when available
    memory_info = None
    try:
        import mlx.core as mx

        active = peak = cache = None
        if hasattr(mx, "get_active_memory"):
            active = mx.get_active_memory()
            peak = mx.get_peak_memory()
            cache = mx.get_cache_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            active = mx.metal.get_active_memory()
            peak = mx.metal.get_peak_memory()
            cache = mx.metal.get_cache_memory()
        if active is not None:
            memory_info = {
                "active_mb": round(active / (1024 * 1024), 1),
                "peak_mb": round(peak / (1024 * 1024), 1),
                "cache_mb": round(cache / (1024 * 1024), 1),
            }
    except Exception:
        pass

    # Include stored-KV cache quantization status for diagnostics.
    kv_quant_info = _kv_cache_quantization_status(scheduler)

    # Include speculative decoding status
    spec_info = None
    try:
        from .speculative import get_spec_stats

        spec_info = get_spec_stats()
    except ImportError:
        pass

    if _model_type == "image":
        result = {
            "status": status,
            "model_loaded": _image_gen is not None and _image_gen.is_loaded,
            "model_name": _model_name,
            "model_type": "image",
            "engine_type": "mflux",
            "last_request_time": _last_request_time if _last_request_time > 0 else None,
        }
    else:
        result = {
            "status": status,
            "model_loaded": _engine is not None,
            "model_name": _model_name,
            "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
            "engine_type": engine_stats.get("engine_type", "unknown"),
            "last_request_time": _last_request_time if _last_request_time > 0 else None,
            "mcp": mcp_info,
        }
    if _model_load_error:
        # Sanitize: strip absolute filesystem paths from error messages for security
        import re

        sanitized = re.sub(r"(/[^\s:]+)+", "<path>", _model_load_error)
        result["error"] = sanitized
    if memory_info:
        result["memory"] = memory_info
    if kv_quant_info:
        result["kv_cache_quantization"] = kv_quant_info
    if spec_info:
        result["speculative_decoding"] = spec_info.get(
            "speculative_decoding", spec_info
        )
    if scheduler:
        try:
            scheduler_stats = scheduler.get_stats()
        except Exception:
            scheduler_stats = {}
        result["scheduler"] = {
            "num_waiting": scheduler_stats.get("num_waiting", 0),
            "num_running": scheduler_stats.get("num_running", 0),
            "engine_path": scheduler_stats.get("engine_path"),
            "batch_generator": scheduler_stats.get("batch_generator", {}),
            "single_active_decode": (
                scheduler_stats.get("batch_generator", {}) or {}
            ).get("single_active_decode", False),
            "ewma_ttft_seconds": scheduler_stats.get("ewma_ttft_seconds", 0),
            "cache_hit_requests": scheduler_stats.get("cache_hit_requests", 0),
            "cache_hit_tokens": scheduler_stats.get("cache_hit_tokens", 0),
            "cache_hit_tokens_by_detail": scheduler_stats.get(
                "cache_hit_tokens_by_detail", {}
            ),
            "hybrid_kv_without_ssm_hits": (
                scheduler_stats.get("batch_generator", {}) or {}
            ).get("hybrid_kv_without_ssm_hits", 0),
            "hybrid_kv_without_ssm_tokens": (
                scheduler_stats.get("batch_generator", {}) or {}
            ).get("hybrid_kv_without_ssm_tokens", 0),
            "last_hybrid_kv_without_ssm": (
                scheduler_stats.get("batch_generator", {}) or {}
            ).get("last_hybrid_kv_without_ssm"),
            "cache_reuse_skips": scheduler_stats.get("cache_reuse_skips", 0),
            "cache_reuse_skip_tokens": scheduler_stats.get(
                "cache_reuse_skip_tokens", 0
            ),
            "last_cache_reuse_skip": scheduler_stats.get("last_cache_reuse_skip"),
            "cache_reuse_partial_downgrades": scheduler_stats.get(
                "cache_reuse_partial_downgrades", 0
            ),
            "cache_reuse_partial_tokens": scheduler_stats.get(
                "cache_reuse_partial_tokens", 0
            ),
            "last_cache_reuse_partial": scheduler_stats.get(
                "last_cache_reuse_partial"
            ),
            "last_cache_selection": scheduler_stats.get("last_cache_selection"),
            "last_cache_execution": scheduler_stats.get("last_cache_execution"),
        }
        cache_snapshot = _cache_telemetry_snapshot(scheduler)
        if cache_snapshot:
            result["cache"] = cache_snapshot

    # Smelt mode: report partial expert loading status
    if _smelt_enabled:
        result["smelt"] = {
            "enabled": True,
            "expert_percent": _smelt_experts,
        }

    # Distributed compute: report cluster status
    if _distributed_enabled:
        dist_info = {
            "enabled": True,
            "mode": _distributed_mode,
        }
        if _distributed_coordinator:
            dist_info.update(_distributed_coordinator.status)
            # Worker count and pipeline from coordinator (if we are coordinator)
            coord = _distributed_coordinator.coordinator
            if coord:
                dist_info["workers"] = len(coord.workers)
                dist_info["pipeline"] = [
                    a.to_dict() for a in coord.assignments
                ] if coord.assignments else []
        result["distributed"] = dist_info

    # Flash MoE: report SSD streaming status
    if _flash_moe_enabled and _flash_moe_loader is not None:
        result["flash_moe"] = _flash_moe_loader.stats()

    # JANG format: report cached quantization metadata (populated at load time)
    if _jang_metadata:
        result["quantization_format"] = _jang_metadata
    if _model_path or _model_name:
        bundle_key = _model_path or _model_name
        result["quantization"] = _model_quantization_status(bundle_key)
        result["acceleration"] = _model_acceleration_status(bundle_key)
        result["mtp"] = _model_mtp_status_with_loaded_runtime(bundle_key)
        routing_status = _model_routing_status(bundle_key)
        if routing_status:
            result["routing"] = routing_status

    if _max_prompt_tokens > 0:
        result["max_prompt_tokens"] = _max_prompt_tokens

    # Nemotron-3-Nano-Omni multimodal: surface bundle status + active backend.
    # Lets the UI show "Vision/Audio: Native MLX (fast)" vs "PyTorch bridge".
    try:
        from .omni_multimodal import (
            OmniMultimodalDispatcher,
            omni_multimodal_component_status,
        )
        _omni_path = _model_path or _model_name
        _omni_status = (
            omni_multimodal_component_status(_omni_path) if _omni_path else {}
        )
        if _omni_status.get("bundle_compatible"):
            result["omni_multimodal"] = {
                "bundle_compatible": True,
                "backend": OmniMultimodalDispatcher._pick_backend(),
                "modalities": _omni_status.get("modalities")
                or ["text", "audio", "image"],
                "components": {
                    "radio": bool(_omni_status.get("has_radio_weights")),
                    "parakeet": bool(_omni_status.get("has_parakeet_weights")),
                    "media_projector": bool(_omni_status.get("has_media_projector")),
                },
            }
    except Exception:
        pass

    if _engine is not None:
        scheduler = _get_scheduler()
        result["turboquant_kv_cache"] = _turboquant_kv_cache_status(
            _engine, scheduler
        )
        _mc = _current_model_config()
        native_cache = _native_cache_status(
            scheduler,
            family=getattr(_mc, "family_name", None),
            cfg=_mc,
        )
        if native_cache:
            result["native_cache"] = native_cache

    return result


@app.get("/health.mtp")
async def health_mtp():
    """Native MTP status endpoint used by the app and release gates."""
    return _model_mtp_status_with_loaded_runtime(_model_path or _model_name)


def _get_scheduler():
    """Get the scheduler from the engine, or None if not available."""
    if _engine is None:
        return None
    # MLLM path: scheduler is directly on the engine
    mllm_sched = getattr(_engine, "_mllm_scheduler", None)
    if mllm_sched is not None:
        return mllm_sched
    # LLM path: BatchedEngine._engine is AsyncEngineCore, which has .engine (EngineCore)
    # EngineCore has .scheduler
    async_core = getattr(_engine, "_engine", None)
    if async_core is not None:
        engine_core = getattr(async_core, "engine", None)
        if engine_core is not None:
            return getattr(engine_core, "scheduler", None)
    return None


# ── Admin: Sleep / Wake ──


@app.post("/admin/soft-sleep", dependencies=[Depends(verify_api_key)])
async def admin_soft_sleep():
    """Enter soft sleep: clear all caches, reduce Metal cache limit. Model stays loaded."""
    global _standby_state, _pre_sleep_cache_limit, _wake_lock
    from starlette.responses import JSONResponse

    if _wake_lock is None:
        _wake_lock = asyncio.Lock()

    async with _wake_lock:
        if _standby_state == "deep":
            return JSONResponse(
                status_code=409, content={"error": "Already in deep sleep"}
            )
        if _standby_state == "soft":
            return {"status": "already_soft"}

        try:
            import mlx.core as mx

            scheduler = _get_scheduler()
            if scheduler is not None:
                if hasattr(scheduler, "deep_reset"):
                    scheduler.deep_reset()
                elif hasattr(scheduler, "_prefix_cache") and scheduler._prefix_cache:
                    scheduler._prefix_cache.clear()

            _set_cache = getattr(mx, "set_cache_limit", None) or getattr(
                mx.metal, "set_cache_limit", None
            )
            if _set_cache:
                _pre_sleep_cache_limit = _set_cache(512 * 1024 * 1024)

            clear_mlx_memory_cache(log=logger)

            _standby_state = "soft"
            logger.info("Entered soft sleep — caches cleared, model loaded")
            return {"status": "soft_sleep"}

        except Exception as e:
            logger.error(f"Failed to enter soft sleep: {e}")
            return {"error": str(e)}


@app.post("/admin/deep-sleep", dependencies=[Depends(verify_api_key)])
async def admin_deep_sleep():
    """Enter deep sleep: unload model entirely. Process stays alive, port stays allocated."""
    global _engine, _standby_state, _pre_sleep_cache_limit, _wake_lock
    global _flash_moe_loader, _distributed_coordinator
    from starlette.responses import JSONResponse

    if _wake_lock is None:
        _wake_lock = asyncio.Lock()

    async with _wake_lock:
        if _standby_state == "deep":
            return JSONResponse(
                status_code=409, content={"error": "Already in deep sleep"}
            )

        try:
            import mlx.core as mx
            import gc

            # Save Metal cache limit before deep sleep (if not already saved by soft sleep)
            if _pre_sleep_cache_limit is None:
                _set_cache = getattr(mx, "set_cache_limit", None) or getattr(
                    mx.metal, "set_cache_limit", None
                )
                if _set_cache:
                    _pre_sleep_cache_limit = _set_cache(512 * 1024 * 1024)

            scheduler = _get_scheduler()
            if scheduler is not None:
                if hasattr(scheduler, "deep_reset"):
                    scheduler.deep_reset()

            if _model_type == "image" and _image_gen is not None:
                await _run_image_gen_call(_image_gen.unload)
            elif _engine is not None:
                if hasattr(_engine, "stop"):
                    try:
                        await _engine.stop()
                    except Exception:
                        pass
                _engine = None
                _reset_mllm_generation_streams()

            # Clear stale Flash MoE loader — the FlashMoEExpertLoader holds
            # references to the now-freed model's ExpertIndex and thread pool.
            # Leaving this set would cause /health to report stale stats and
            # would leak the I/O worker threads across the sleep cycle.
            if _flash_moe_loader is not None:
                try:
                    _flash_moe_loader.shutdown()
                except Exception as e:
                    logger.debug(f"Flash MoE loader shutdown during deep sleep: {e}")
                _flash_moe_loader = None

            # Tear down the distributed mesh so worker TCP connections close
            # cleanly and the coordinator doesn't keep heartbeating a model
            # that no longer exists. admin_wake() rebuilds from _cli_args.
            if _distributed_coordinator is not None:
                try:
                    shutdown = getattr(_distributed_coordinator, "shutdown", None) \
                        or getattr(_distributed_coordinator, "stop", None)
                    if shutdown is not None:
                        result = shutdown()
                        if asyncio.iscoroutine(result):
                            await result
                    logger.info("Distributed mesh coordinator shut down for deep sleep")
                except Exception as e:
                    logger.warning(f"Distributed shutdown during deep sleep failed: {e}")
                _distributed_coordinator = None

            # Unload speculative draft model to free GPU memory
            try:
                from .speculative import unload_draft_model

                unload_draft_model()
            except Exception as e:
                logger.debug(f"Draft model unload during deep sleep: {e}")

            gc.collect()
            clear_mlx_memory_cache(log=logger)

            _standby_state = "deep"
            logger.info("Entered deep sleep — model unloaded, process alive")
            return {"status": "deep_sleep"}

        except Exception as e:
            logger.error(f"Failed to enter deep sleep: {e}")
            return {"error": str(e)}


@app.post("/admin/wake", dependencies=[Depends(verify_api_key)])
async def admin_wake():
    """Wake from sleep: reload model. Triggered by JIT or manual wake."""
    global _engine, _standby_state, _pre_sleep_cache_limit, _model_load_error

    if _standby_state is None:
        return {"status": "already_active"}

    try:
        import mlx.core as mx

        if _standby_state == "soft":
            # Soft sleep: model is still loaded, just restore cache limit
            _set_cache = getattr(mx, "set_cache_limit", None) or getattr(
                mx.metal, "set_cache_limit", None
            )
            if _set_cache and _pre_sleep_cache_limit is not None:
                _set_cache(_pre_sleep_cache_limit)
            _standby_state = None
            _pre_sleep_cache_limit = None
            logger.info("Woke from soft sleep — cache limit restored")
            return {"status": "active"}

        elif _standby_state == "deep":
            # Deep sleep: need to reload model
            # Restore cache limit first
            _set_cache = getattr(mx, "set_cache_limit", None) or getattr(
                mx.metal, "set_cache_limit", None
            )
            if _set_cache and _pre_sleep_cache_limit is not None:
                _set_cache(_pre_sleep_cache_limit)
            _pre_sleep_cache_limit = None

            if _model_type == "image" and _image_gen is not None:
                # Reload image model on the same dedicated worker used for
                # generation. MLX streams are thread-local; reloading on the
                # default executor and generating elsewhere reopens the
                # Stream(gpu,N) crash class.
                await _run_image_gen_call(
                    _image_gen.load,
                    _model_name,
                    quantize=getattr(_image_gen, "_quantize", None),
                    model_path=getattr(_image_gen, "_model_path", None),
                    mflux_class=getattr(_image_gen, "_mflux_class", None),
                    lora_paths=_image_lora_paths,
                    lora_scales=_image_lora_scales,
                )
                _standby_state = None
                logger.info(
                    f"Woke from deep sleep — image model {_model_name} reloaded"
                )
                return {"status": "active"}
            elif _model_path or _model_name:
                # Reload text model — run in thread to avoid blocking event loop
                # (loading large models takes 10-60s; _wake_lock prevents concurrent
                # access to the globals that load_model modifies)
                await asyncio.to_thread(
                    load_model,
                    _model_path or _model_name,
                    use_batching=_cli_args.get("use_batching", False),
                    scheduler_config=_cli_args.get("scheduler_config"),
                    stream_interval=_cli_args.get("stream_interval", 1),
                    max_tokens=_cli_args.get("max_tokens", _FALLBACK_MAX_OUTPUT_TOKENS),
                    max_tokens_explicit=_cli_args.get("max_tokens_explicit", False),
                    max_prompt_tokens=_cli_args.get("max_prompt_tokens"),
                    force_mllm=_cli_args.get("force_mllm", False),
                    served_model_name=_cli_args.get("served_model_name"),
                    smelt=_cli_args.get("smelt", False),
                    smelt_experts=_cli_args.get("smelt_experts", 50),
                    flash_moe=_cli_args.get("flash_moe", False),
                    flash_moe_slot_bank=_cli_args.get("flash_moe_slot_bank", 64),
                    flash_moe_prefetch=_cli_args.get("flash_moe_prefetch", "none"),
                    flash_moe_io_split=_cli_args.get("flash_moe_io_split", 4),
                    distributed=_cli_args.get("distributed", False),
                    distributed_mode=_cli_args.get("distributed_mode", "pipeline"),
                    cluster_secret=_cli_args.get("cluster_secret", ""),
                    worker_nodes=_cli_args.get("worker_nodes"),
                )
                # Start engine after wake. Both SimpleEngine and BatchedEngine
                # expose `_loaded` — start() is idempotent for SimpleEngine and
                # required for BatchedEngine (lifespan usually handles this,
                # but wake bypasses lifespan).
                if (
                    _engine is not None
                    and hasattr(_engine, "_loaded")
                    and not _engine._loaded
                ):
                    await _engine.start()
                    if hasattr(_engine, "_needs_async_start"):
                        _engine._needs_async_start = False
                # Re-apply Flash MoE after wake (load_model skipped it because
                # _engine._loaded was False when called inside event loop)
                if _flash_moe_enabled and _engine is not None:
                    _apply_flash_moe_patching()
                # Re-apply JIT compilation after deep wake (load_model skips it
                # because _engine._loaded is False at check time inside event loop)
                if _enable_jit and _engine is not None:
                    _apply_jit_compilation()
                # Reload speculative draft model if configured
                _spec_model = _cli_args.get("speculative_model")
                if _spec_model:
                    try:
                        from .speculative import SpeculativeConfig, load_draft_model

                        _spec_cfg = SpeculativeConfig(
                            model=_spec_model,
                            num_tokens=_cli_args.get("num_draft_tokens", 3),
                        )
                        await asyncio.to_thread(load_draft_model, _spec_cfg)
                        logger.info(f"Draft model reloaded: {_spec_model}")
                    except Exception as e:
                        logger.warning(f"Failed to reload draft model on wake: {e}")
                _standby_state = None
                logger.info(f"Woke from deep sleep — model {_model_name} reloaded")
                return {"status": "active"}
            else:
                return {"error": "No model name saved — cannot reload"}

    except Exception as e:
        logger.error(f"Failed to wake from sleep: {e}")
        # Clear standby state to prevent infinite JIT wake retry loop.
        # Server stays alive but reports error via /health.
        _standby_state = None
        _model_load_error = f"Wake failed: {e}"
        return {"error": str(e)}


@app.get("/v1/cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Get cache statistics for debugging and monitoring."""
    result = {}

    # Scheduler-level prefix cache stats
    if _engine:
        scheduler_cache = _engine.get_cache_stats()
        if scheduler_cache:
            result["scheduler_cache"] = scheduler_cache

    # Scheduler-level overall stats (includes cache stats + request stats)
    scheduler = _get_scheduler()
    if scheduler:
        stats = scheduler.get_stats()
        result["scheduler_stats"] = {
            "num_waiting": stats.get("num_waiting", 0),
            "num_running": stats.get("num_running", 0),
            "engine_path": stats.get("engine_path"),
            "batch_generator": stats.get("batch_generator", {}),
            "single_active_decode": (
                stats.get("batch_generator", {}) or {}
            ).get("single_active_decode", False),
            "num_requests_processed": stats.get("num_requests_processed", 0),
            "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
            "total_completion_tokens": stats.get("total_completion_tokens", 0),
            "ewma_ttft_seconds": stats.get("ewma_ttft_seconds", 0),
            "cache_hit_requests": stats.get("cache_hit_requests", 0),
            "cache_hit_tokens": stats.get("cache_hit_tokens", 0),
            "cache_hit_tokens_by_detail": stats.get("cache_hit_tokens_by_detail", {}),
            "hybrid_kv_without_ssm_hits": (
                stats.get("batch_generator", {}) or {}
            ).get("hybrid_kv_without_ssm_hits", 0),
            "hybrid_kv_without_ssm_tokens": (
                stats.get("batch_generator", {}) or {}
            ).get("hybrid_kv_without_ssm_tokens", 0),
            "last_hybrid_kv_without_ssm": (
                stats.get("batch_generator", {}) or {}
            ).get("last_hybrid_kv_without_ssm"),
            "cache_reuse_skips": stats.get("cache_reuse_skips", 0),
            "cache_reuse_skip_tokens": stats.get("cache_reuse_skip_tokens", 0),
            "last_cache_reuse_skip": stats.get("last_cache_reuse_skip"),
            "cache_reuse_partial_downgrades": stats.get(
                "cache_reuse_partial_downgrades", 0
            ),
            "cache_reuse_partial_tokens": stats.get(
                "cache_reuse_partial_tokens", 0
            ),
            "last_cache_reuse_partial": stats.get("last_cache_reuse_partial"),
            "last_cache_selection": stats.get("last_cache_selection"),
            "last_cache_execution": stats.get("last_cache_execution"),
        }
        # Stored-KV cache quantization info. Normalize disabled state so the
        # panel never renders raw bits=0 as a fake "0-bit" codec.
        kv_quant = _kv_cache_quantization_status(scheduler)
        if kv_quant:
            result["kv_cache_quantization"] = kv_quant

        # TurboQuant KV cache status (separate path from generic
        # kv_cache_quantization). Use the same recursive detector as /health
        # so source and packaged app expose the actual patched make_cache path.
        result["turboquant_kv_cache"] = _turboquant_kv_cache_status(
            _engine, scheduler
        )
        _mc = _current_model_config()
        native_cache = _native_cache_status(
            scheduler,
            family=getattr(_mc, "family_name", None),
            cfg=_mc,
        )
        if native_cache:
            result["native_cache"] = native_cache

    # Disk cache (L2) stats — prompt-level
    if scheduler and getattr(scheduler, "disk_cache", None) is not None:
        result["disk_cache"] = scheduler.disk_cache.stats()

    # Block disk store (L2) stats — paged cache blocks
    paged_mgr = getattr(scheduler, "paged_cache_manager", None) if scheduler else None
    block_store = getattr(paged_mgr, "_disk_store", None) if paged_mgr else None
    if block_store is not None and hasattr(block_store, "get_stats"):
        result["block_disk_cache"] = block_store.get_stats()

    cache_snapshot = _cache_telemetry_snapshot(scheduler)
    if cache_snapshot.get("totals"):
        result["cache_totals"] = cache_snapshot["totals"]

    # MLLM-specific cache stats
    try:
        from mlx_vlm.utils import (
            get_multimodal_kv_cache_stats,
            get_pil_cache_stats,
            get_pixel_values_cache_stats,
        )

        result["multimodal_kv_cache"] = get_multimodal_kv_cache_stats()
        result["pixel_values_cache"] = get_pixel_values_cache_stats()
        result["pil_image_cache"] = get_pil_cache_stats()
    except ImportError:
        pass

    # TurboQuant status
    if _engine:
        model = getattr(_engine, "_model", None) or getattr(_engine, "model", None)
        if model and hasattr(model, "make_cache"):
            _tq_active = getattr(model.make_cache, "__name__", "") in (
                "_tq_make_cache",
                "_turboquant_make_cache",
            )
            if _tq_active:
                result["turbo_quant"] = {"enabled": True}

    # SSM companion cache stats (hybrid models only). Use the same helper as
    # /health so the panel sees disk token totals and IO counters on both
    # polling surfaces.
    ssm_snapshot = cache_snapshot.get("ssm_companion")
    if ssm_snapshot:
        result["ssm_companion"] = ssm_snapshot

    # Metal GPU memory info
    try:
        import mlx.core as mx

        active = peak = cache_mem = None
        if hasattr(mx, "get_active_memory"):
            active = mx.get_active_memory()
            peak = mx.get_peak_memory()
            cache_mem = mx.get_cache_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            active = mx.metal.get_active_memory()
            peak = mx.metal.get_peak_memory()
            cache_mem = mx.metal.get_cache_memory()
        if active is not None:
            result["memory"] = {
                "active_mb": round(active / (1024 * 1024), 1),
                "peak_mb": round(peak / (1024 * 1024), 1),
                "cache_mb": round(cache_mem / (1024 * 1024), 1),
            }
    except Exception:
        pass

    if not result:
        return {"error": "No cache stats available"}
    return result


@app.get("/v1/cache/entries", dependencies=[Depends(verify_api_key)])
async def cache_entries():
    """List cached prefix entries with metadata."""
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"error": "Scheduler not available (SimpleEngine mode)"}

    entries = []
    cache_type = "none"

    if getattr(scheduler, "memory_aware_cache", None) is not None:
        cache_type = "memory_aware"
        from .memory_cache import _BYTES_PER_MB

        for tokens_key, entry in scheduler.memory_aware_cache._entries.items():
            entries.append(
                {
                    "tokens_count": len(tokens_key),
                    "memory_bytes": entry.memory_bytes,
                    "memory_mb": round(entry.memory_bytes / _BYTES_PER_MB, 2),
                    "cache_type": "memory_aware",
                }
            )
    elif getattr(scheduler, "block_aware_cache", None) is not None:
        cache_type = "paged"
        for (
            block_id,
            block,
        ) in scheduler.block_aware_cache.paged_cache.allocated_blocks.items():
            entries.append(
                {
                    "block_id": block_id,
                    "tokens_count": block.token_count,
                    "ref_count": block.ref_count,
                    "has_data": block.cache_data is not None,
                    "cache_type": "paged",
                }
            )
    elif getattr(scheduler, "prefix_cache", None) is not None:
        cache_type = "legacy"
        for _, tokens_tuple in scheduler.prefix_cache._lru:
            entries.append(
                {
                    "tokens_count": len(tokens_tuple),
                    "cache_type": "legacy",
                }
            )

    return {
        "cache_type": cache_type,
        "count": len(entries),
        "entries": entries,
    }


@app.post("/v1/cache/warm", dependencies=[Depends(verify_api_key)])
async def cache_warm(request: dict):
    """
    Warm the prefix cache by pre-computing KV states for given prompts.

    Request body: {"prompts": ["system prompt text", ...]}
    """
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"error": "Scheduler not available (SimpleEngine mode)"}

    prompts = request.get("prompts", [])
    if not prompts:
        return {"error": "No prompts provided"}

    def _encode_prompt_for_warm(_scheduler, _prompt: str):
        tokenizer = getattr(_scheduler, "_actual_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(_scheduler, "tokenizer", None)
        if tokenizer is None:
            processor = getattr(_scheduler, "processor", None)
            tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise AttributeError("scheduler has no tokenizer/processor tokenizer")
        return tokenizer.encode(_prompt)

    def _do_warm():
        """Run prefill warming in a thread to avoid blocking the event loop."""
        warmed = 0
        token_counts = []
        errors = []

        for i, prompt in enumerate(prompts):
            try:
                tokens = _encode_prompt_for_warm(scheduler, prompt)

                if not tokens:
                    errors.append(f"Prompt {i}: empty after tokenization")
                    continue

                cache_tokens = tokens[:-1]
                if not cache_tokens:
                    errors.append(f"Prompt {i}: too short (1 token)")
                    continue

                cache = scheduler._prefill_for_prompt_only_cache(cache_tokens)
                if cache is None:
                    errors.append(f"Prompt {i}: prefill failed")
                    continue

                stored = False
                if getattr(scheduler, "memory_aware_cache", None) is not None:
                    stored = scheduler.memory_aware_cache.store(tokens, cache)
                elif getattr(scheduler, "block_aware_cache", None) is not None:
                    extracted = scheduler._extract_cache_states(cache)
                    if extracted:
                        result = scheduler.block_aware_cache.store_cache(
                            f"warm-{i}-{uuid.uuid4().hex[:8]}", tokens, extracted
                        )
                        stored = result is not None
                elif getattr(scheduler, "prefix_cache", None) is not None:
                    scheduler.prefix_cache.store_cache(tokens, cache)
                    stored = True
                else:
                    errors.append(f"Prompt {i}: no cache backend enabled")
                    continue

                if stored:
                    warmed += 1
                    token_counts.append(len(tokens))
                else:
                    errors.append(
                        f"Prompt {i}: cache store rejected (too large or full)"
                    )
            except Exception as e:
                errors.append(f"Prompt {i}: {type(e).__name__}: {e}")

        return {
            "warmed": warmed,
            "token_counts": token_counts,
            "errors": errors if errors else None,
        }

    return await asyncio.to_thread(_do_warm)


@app.delete("/v1/cache", dependencies=[Depends(verify_api_key)])
async def clear_cache(cache_type: str = Query("all", alias="type")):
    """
    Clear caches.

    Query params:
        type: "prefix" | "multimodal" | "all" (default: "all")
    """
    cleared = []

    # Clear prefix cache
    if cache_type in ("prefix", "all"):
        scheduler = _get_scheduler()
        if scheduler is not None:
            if getattr(scheduler, "memory_aware_cache", None) is not None:
                scheduler.memory_aware_cache.clear()
                cleared.append("memory_aware_prefix")
            if getattr(scheduler, "block_aware_cache", None) is not None:
                scheduler.block_aware_cache.clear()
                cleared.append("paged_prefix")
            if getattr(scheduler, "prefix_cache", None) is not None:
                scheduler.prefix_cache.clear()
                cleared.append("legacy_prefix")
            if getattr(scheduler, "disk_cache", None) is not None:
                scheduler.disk_cache.clear()
                cleared.append("disk_cache")
            # Also clear L2 paged block disk store (separate from legacy disk_cache)
            paged_mgr = getattr(scheduler, "paged_cache_manager", None)
            if paged_mgr is not None:
                disk_store = getattr(paged_mgr, "_disk_store", None)
                if disk_store is not None:
                    disk_store.clear()
                    cleared.append("block_disk_store")
            try:
                import gc as _gc

                _gc.collect()
                if clear_mlx_memory_cache(log=logger):
                    cleared.append("mlx_cache")
            except Exception:
                pass

    # Clear multimodal caches
    if cache_type in ("multimodal", "all"):
        try:
            from mlx_vlm.utils import (
                clear_multimodal_kv_cache,
                clear_pixel_values_cache,
            )

            clear_multimodal_kv_cache()
            clear_pixel_values_cache()
            cleared.append("multimodal_kv")
            cleared.append("pixel_values")
        except ImportError:
            pass

    if not cleared:
        return {"status": "no_caches_found", "cache_type": cache_type}
    return {"status": "cleared", "caches": cleared, "cache_type": cache_type}


# ── Distributed Cluster Endpoints ──


@app.get("/v1/cluster/status", dependencies=[Depends(verify_api_key)])
async def cluster_status():
    """Get distributed cluster status."""
    if not _distributed_enabled or not _distributed_coordinator:
        return {"enabled": False}
    return {
        "enabled": True,
        **_distributed_coordinator.status,
    }


@app.get("/v1/cluster/nodes", dependencies=[Depends(verify_api_key)])
async def cluster_nodes():
    """Get list of nodes in the distributed mesh."""
    if not _distributed_enabled or not _distributed_coordinator:
        return {"nodes": []}

    nodes = []
    for peer in _distributed_coordinator.topology.peers.values():
        node_info = {
            "node_id": peer.node_id,
            "hostname": peer.hostname,
            "address": peer.address,
            "port": peer.port,
            "chip": peer.chip,
            "ram_gb": peer.ram_gb,
            "gpu_cores": peer.gpu_cores,
            "available_gb": peer.available_gb,
            "capability_score": peer.capability_score,
            "state": peer.state,
            "is_coordinator": peer.is_coordinator,
            "assigned_layers": peer.assigned_layers,
            "is_alive": peer.is_alive,
        }
        # Add link info if available
        local_id = _distributed_coordinator.node.node_id
        link = _distributed_coordinator.topology.get_link(local_id, peer.node_id)
        if link:
            node_info["link_type"] = link.link_type.value
            node_info["bandwidth_mbps"] = link.bandwidth_mbps
            node_info["latency_ms"] = link.latency_ms
        nodes.append(node_info)

    return {"nodes": nodes}


@app.post("/v1/cluster/nodes", dependencies=[Depends(verify_api_key)])
async def cluster_add_node(request: dict):
    """Add a worker node manually by IP:port."""
    if not _distributed_enabled or not _distributed_coordinator:
        raise HTTPException(status_code=400, detail="Distributed mode not enabled")

    address = request.get("address")
    port = request.get("port", 9100)
    if not address:
        raise HTTPException(status_code=400, detail="address is required")

    coord = _distributed_coordinator.coordinator
    if not coord:
        raise HTTPException(status_code=400, detail="This node is not the coordinator")

    node = await coord.add_node_manual(address, port)
    if node:
        return {"success": True, "node": node.to_dict()}
    raise HTTPException(status_code=502, detail=f"Failed to connect to worker at {address}:{port}")


@app.delete("/v1/cluster/nodes/{node_id}", dependencies=[Depends(verify_api_key)])
async def cluster_remove_node(node_id: str):
    """Remove a node from the mesh."""
    if not _distributed_enabled or not _distributed_coordinator:
        raise HTTPException(status_code=400, detail="Distributed mode not enabled")

    coord = _distributed_coordinator.coordinator
    if coord and node_id in coord.workers:
        wc = coord.workers[node_id]
        await wc.disconnect()
        del coord.workers[node_id]
    _distributed_coordinator.topology.remove_peer(node_id)
    return {"success": True}


@app.post("/v1/cluster/scan", dependencies=[Depends(verify_api_key)])
async def cluster_scan():
    """Trigger node discovery scan."""
    if not _distributed_enabled or not _distributed_coordinator:
        raise HTTPException(status_code=400, detail="Distributed mode not enabled")

    coord = _distributed_coordinator.coordinator
    if not coord:
        raise HTTPException(status_code=400, detail="This node is not the coordinator")

    discovered = await coord.discover_nodes(timeout=5.0)
    nodes = [n.to_dict() for n in discovered]
    return {"nodes": nodes, "count": len(nodes)}


@app.post(
    "/v1/chat/completions/{request_id}/cancel", dependencies=[Depends(verify_api_key)]
)
async def cancel_chat_completion(request_id: str):
    """
    Cancel an ongoing chat completion request.

    Stops token generation immediately and marks the request as aborted.
    Any tokens generated so far are preserved in the partial response.

    Args:
        request_id: The request ID from the response (e.g., chatcmpl-abc123)

    Returns:
        Success message or 404 if request not found

    Example:
        ```bash
        curl -X POST http://localhost:8092/v1/chat/completions/chatcmpl-abc123/cancel \\
          -H "Authorization: Bearer your-api-key"
        ```
    """
    if _engine is None:
        raise HTTPException(
            status_code=404, detail="No model loaded — no active requests to cancel"
        )

    # Abort the request (returns True if found, False if not found/already finished)
    success = await _engine.abort_request(request_id)

    if success:
        logger.info(f"Request {request_id} cancelled via API")
        return {"success": True, "message": f"Request {request_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request_id} not found or already finished",
        )


@app.post("/v1/responses/{response_id}/cancel", dependencies=[Depends(verify_api_key)])
async def cancel_response(response_id: str):
    """
    Cancel an ongoing Responses API request.

    Args:
        response_id: The response ID (e.g., resp_abc123def456)

    Returns:
        Success message or 404 if request not found
    """
    if _engine is None:
        raise HTTPException(
            status_code=404, detail="No model loaded — no active requests to cancel"
        )
    success = await _engine.abort_request(response_id)

    if success:
        logger.info(f"Response {response_id} cancelled via API")
        return {"success": True, "message": f"Response {response_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Response {response_id} not found or already finished",
        )


@app.post("/v1/completions/{request_id}/cancel", dependencies=[Depends(verify_api_key)])
async def cancel_completion(request_id: str):
    """
    Cancel an ongoing text completion request.

    Same as chat completions cancel, but for /v1/completions endpoint.

    Args:
        request_id: The request ID from the response (e.g., cmpl-abc123)

    Returns:
        Success message or 404 if request not found
    """
    if _engine is None:
        raise HTTPException(
            status_code=404, detail="No model loaded — no active requests to cancel"
        )
    success = await _engine.abort_request(request_id)

    if success:
        logger.info(f"Request {request_id} cancelled via API")
        return {"success": True, "message": f"Request {request_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request_id} not found or already finished",
        )


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models — chat + embedding if both loaded."""
    models = []
    seen: set[str] = set()
    resolved = _resolve_model_name()
    if resolved and resolved != "default":
        models.append(ModelInfo(id=resolved))
        seen.add(resolved)
        # If served_model_name differs from actual, also list the actual name
        # so clients using either name can find the model
        if _served_model_name and _model_name and _served_model_name != _model_name:
            if _model_name not in seen:
                models.append(ModelInfo(id=_model_name))
                seen.add(_model_name)
    # Advertise the loaded embedding model so clients probing /v1/models
    # can discover it for /v1/embeddings without having to know the
    # full path. Previously only the chat model appeared → embedding
    # clients had to guess the model name.
    if _embedding_engine is not None and _embedding_model_locked:
        if _embedding_model_locked not in seen:
            models.append(ModelInfo(id=_embedding_model_locked))
            seen.add(_embedding_model_locked)
    return ModelsResponse(data=models)


@app.get("/v1/models/{model_id:path}/capabilities", dependencies=[Depends(verify_api_key)])
async def model_capabilities(model_id: str) -> dict:
    """Return vMLX runtime capabilities for the loaded model.

    This is intentionally separate from Ollama's flat `capabilities` list:
    the panel needs structured mode data so it can show Instruct / Reasoning /
    Max Thinking only when the loaded family actually supports those modes.
    """
    model_key = _model_path or _model_name or model_id
    try:
        from .model_config_registry import get_model_config_registry
        cfg = get_model_config_registry().lookup(model_key)
    except Exception:
        cfg = None

    family = getattr(cfg, "family_name", "unknown") if cfg is not None else "unknown"
    reasoning_parser = getattr(cfg, "reasoning_parser", None) if cfg is not None else None
    tool_parser = getattr(cfg, "tool_parser", None) if cfg is not None else None
    think_in_template = bool(getattr(cfg, "think_in_template", False)) if cfg is not None else False
    registry_is_mllm = bool(getattr(cfg, "is_mllm", False)) if cfg is not None else False
    engine_is_mllm = (
        bool(getattr(_engine, "is_mllm", False))
        if _engine is not None
        else registry_is_mllm
    )
    modalities = _loaded_runtime_modalities()
    if modalities == ["text"] and engine_is_mllm:
        modalities = ["text", "vision"]
    supports_thinking_explicit = getattr(cfg, "supports_thinking", None) if cfg is not None else None
    supports_thinking = (
        bool(supports_thinking_explicit)
        if supports_thinking_explicit is not None
        else bool(reasoning_parser or think_in_template)
    )
    if family == "deepseek_v4":
        supports_thinking = True
        supported_modes = ["instruct", "reasoning"]
        reasoning_efforts = ["high", "max"]
    elif supports_thinking:
        supported_modes = ["instruct", "reasoning"]
        if family == "hy_v3":
            reasoning_efforts = ["low", "high"]
        elif reasoning_parser in ("openai_gptoss", "mistral"):
            reasoning_efforts = ["low", "medium", "high"]
        else:
            reasoning_efforts = []
    else:
        supported_modes = ["instruct"]
        reasoning_efforts = []

    bundle_path = _model_path or model_key
    sampling_defaults = {
        key: _bundle_sampling_default(bundle_path, key)
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_new_tokens",
            "repetition_penalty",
            "repetition_penalty_chat",
            "repetition_penalty_thinking",
        )
    }
    sampling_defaults = {k: v for k, v in sampling_defaults.items() if v is not None}

    scheduler = _get_scheduler()
    scheduler_config = getattr(scheduler, "config", None)
    block_aware_cache = getattr(scheduler, "block_aware_cache", None)
    paged_cache_manager = getattr(scheduler, "paged_cache_manager", None)
    memory_aware_cache = getattr(scheduler, "memory_aware_cache", None)
    trie_prefix_cache = getattr(scheduler, "prefix_cache", None)
    if block_aware_cache is not None:
        cache_type = "paged"
    elif memory_aware_cache is not None:
        cache_type = "memory_aware"
    elif trie_prefix_cache is not None:
        cache_type = "trie"
    else:
        cache_type = "disabled"
    prefix_cache_enabled = bool(
        getattr(scheduler_config, "enable_prefix_cache", False)
        or block_aware_cache is not None
        or memory_aware_cache is not None
        or trie_prefix_cache is not None
    )
    paged_cache_enabled = bool(block_aware_cache is not None and paged_cache_manager is not None)
    block_disk_store = getattr(paged_cache_manager, "_disk_store", None)
    dsv4_composite_state = bool(
        family == "deepseek_v4"
        and paged_cache_enabled
        and getattr(scheduler, "_uses_dsv4_cache", False)
    )
    native_cache = _native_cache_status(scheduler, family=family, cfg=cfg)
    quantization_status = _model_quantization_status(bundle_path)
    acceleration_status = _model_acceleration_status(bundle_path)
    mtp_status = _model_mtp_status_with_loaded_runtime(bundle_path)
    routing_status = _model_routing_status(bundle_path)

    _capability_compat_warnings = list(quantization_status.get("compat_warnings", []))

    return {
        "id": model_id,
        "loaded_model": _resolve_model_name(),
        "model_path": _model_path,
        "family": family,
        "compat_warnings": _capability_compat_warnings,
        "supports_tools": bool(tool_parser),
        "tool_parser": tool_parser,
        "supports_thinking": supports_thinking,
        "reasoning_parser": reasoning_parser,
        "think_in_template": think_in_template,
        "supported_modes": supported_modes,
        "experimental_modes": [],
        "reasoning_efforts": reasoning_efforts,
        "modalities": modalities,
        "cache": {
            "prefix": prefix_cache_enabled,
            "type": cache_type,
            "paged": paged_cache_enabled,
            "block_disk_l2": block_disk_store is not None,
            "dsv4_composite_state": dsv4_composite_state,
            "native": native_cache or None,
        },
        "quantization": quantization_status,
        "acceleration": acceleration_status,
        "mtp": mtp_status,
        "routing": routing_status or None,
        "sampling_defaults": sampling_defaults,
    }


# =============================================================================
# Anthropic Messages API
# =============================================================================


@app.post(
    "/v1/messages",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def create_anthropic_message(
    fastapi_request: Request,
):
    """
    Anthropic Messages API endpoint.

    Enables Claude Code and other Anthropic SDK clients to use vMLX as a
    local inference backend. Converts Anthropic wire format to internal
    Chat Completions pipeline.

    Both streaming and non-streaming paths consume stream_chat_completion(
    so Anthropic inherits the same reasoning/tool wiring as Chat Completions.
    """
    from .api.anthropic_adapter import (
        AnthropicRequest,
        AnthropicStreamAdapter,
        to_chat_completion,
    )

    from starlette.responses import JSONResponse

    try:
        body = await fastapi_request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid JSON in request body",
                },
            },
        )
    try:
        anthropic_req = AnthropicRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e)},
            },
        )

    # Convert to chat completion request
    chat_req = to_chat_completion(anthropic_req)

    # Resolve model name
    resolved_name = _resolve_model_name()
    chat_req.model = resolved_name
    _log_multimodal_request_shape(
        "/v1/messages",
        _model_path or _model_name or chat_req.model,
        _messages_multimodal_summary(chat_req.messages),
    )

    _msg_max_prompt_tokens = _effective_max_prompt_tokens(chat_req)
    prompt_limit_response = _reject_if_prompt_too_long_for_messages(
        chat_req.messages,
        max_prompt_tokens=_msg_max_prompt_tokens,
    )
    if prompt_limit_response is not None:
        return prompt_limit_response

    # Nemotron-Omni multimodal dispatch (Anthropic /v1/messages parity
    # with /v1/chat/completions; live test 2026-04-30 surfaced that
    # image content blocks were converted to image_url correctly by the
    # adapter at api/anthropic_adapter.py:338-354 but the converted
    # request never reached `dispatch_omni_chat_completion` because the
    # /v1/messages handler skipped that hook. Result: model received
    # text-only context and emitted a generic color list instead of
    # actually seeing the image. Mirrors the dispatch in
    # `create_chat_completion` at server.py:5295-5311.
    try:
        from .omni_multimodal import (
            is_omni_multimodal_bundle,
            request_has_multimodal,
            dispatch_omni_chat_completion,
        )
        _omni_path = _model_path or _model_name
        if (
            _omni_path
            and is_omni_multimodal_bundle(_omni_path)
            and request_has_multimodal(
                [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                 for m in (chat_req.messages or [])]
            )
        ):
            # Reuse the same dispatcher; its output is OpenAI chat-completion
            # shape, which the Anthropic adapter's `to_anthropic_response`
            # path can then re-wrap into Anthropic's content_block format.
            from .api.anthropic_adapter import to_anthropic_response
            cc = await dispatch_omni_chat_completion(chat_req, _omni_path)
            # `dispatch_omni_chat_completion` returns either a JSONResponse
            # (non-streaming) or a StreamingResponse. Anthropic streaming
            # has its own SSE shape, so OpenAI chat-completion chunks must
            # be adapted before returning them to Anthropic SDK clients.
            from starlette.responses import JSONResponse as _JR
            if isinstance(cc, _JR):
                cc_dict = json.loads(cc.body.decode("utf-8")) if cc.body else {}
                return _JR(
                    content=to_anthropic_response(cc_dict, resolved_name)
                )
            if isinstance(cc, dict):
                return _JR(
                    content=to_anthropic_response(cc, resolved_name)
                )
            from starlette.responses import StreamingResponse as _SR
            if isinstance(cc, _SR) and anthropic_req.stream:
                adapter = AnthropicStreamAdapter(model=resolved_name)

                async def _adapt_omni_stream():
                    try:
                        async for raw in cc.body_iterator:
                            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                            for event_text in text.split("\n\n"):
                                event_text = event_text.strip()
                                if not event_text:
                                    continue
                                if event_text.startswith(":"):
                                    yield event_text + "\n\n"
                                    continue
                                for line in event_text.splitlines():
                                    if line.startswith("data: "):
                                        for event in adapter.process_chunk(line):
                                            yield event
                    finally:
                        for event in adapter.finalize():
                            yield event

                return _SR(_adapt_omni_stream(), media_type="text/event-stream")
            return cc
    except HTTPException:
        raise
    except Exception as _omni_route_err:  # pragma: no cover
        logger.warning(
            "Omni multimodal dispatch failed in /v1/messages (%s); "
            "falling back to standard path",
            _omni_route_err,
        )

    engine = get_engine()
    _msg_requested_modalities = _messages_requested_modalities(chat_req.messages)
    if _msg_requested_modalities:
        _reject_unsupported_multimodal(
            "/v1/messages",
            _msg_requested_modalities,
        )
    if _messages_have_multimodal(chat_req.messages) and not engine.is_mllm:
        _reject_unsupported_multimodal("/v1/messages")

    # Build generation kwargs from the converted chat request (shared by streaming + non-streaming)
    _msg_kwargs: dict = {
        "temperature": _resolve_temperature(chat_req.temperature, chat_req.model),
        "top_p": _resolve_top_p(chat_req.top_p, chat_req.model),
        "max_tokens": _resolve_max_tokens(chat_req.max_tokens, chat_req.model),
        "max_prompt_tokens": _msg_max_prompt_tokens,
    }
    _set_resolved_top_k(_msg_kwargs, chat_req.top_k, chat_req.model)
    _set_resolved_min_p(_msg_kwargs, chat_req.min_p, chat_req.model)
    _rp = _resolve_repetition_penalty(chat_req.repetition_penalty, chat_req.model)
    if _rp is not None:
        _msg_kwargs["repetition_penalty"] = _rp
    if _compute_bypass_prefix_cache(chat_req):
        _msg_kwargs["_bypass_prefix_cache"] = True
    if chat_req.stop:
        _msg_kwargs["stop"] = chat_req.stop
    # Merge server-wide --chat-template-kwargs defaults with any adapter-populated
    # kwargs (e.g., thinking_budget from Anthropic thinking.budget_tokens)
    _ct_kwargs = _merge_ct_kwargs(chat_req.chat_template_kwargs)

    # Forward enable_thinking to engine — without this, the model always thinks
    # internally even when the client sends thinking: {type: "disabled"}
    _et = _resolve_enable_thinking(
        request_value=chat_req.enable_thinking,
        ct_kwargs=_ct_kwargs,
        tools_present=bool(chat_req.tools),
        model_key=_model_path or _model_name or chat_req.model,
        engine=engine,
        auto_detect=True,
    )
    if _et is not None:
        _msg_kwargs["enable_thinking"] = _et
        chat_req.enable_thinking = _et

    # Auto-map enable_thinking → reasoning_effort for Mistral 4 (same as OpenAI path)
    if (
        _reasoning_parser
        and type(_reasoning_parser).__name__ == "MistralReasoningParser"
    ):
        if chat_req.enable_thinking is True and "reasoning_effort" not in _ct_kwargs:
            _ct_kwargs["reasoning_effort"] = "high"
            _msg_kwargs["reasoning_effort"] = "high"
        elif chat_req.enable_thinking is False and "reasoning_effort" not in _ct_kwargs:
            _ct_kwargs["reasoning_effort"] = "none"

    _apply_hy3_reasoning_policy(
        _msg_kwargs,
        _ct_kwargs,
        model_key=_model_path or _model_name or chat_req.model,
        enable_thinking=chat_req.enable_thinking,
    )

    # DeepSeek V4 three-mode encoder (mirror OpenAI chat-completions path).
    # See research/DSV4-RUNTIME-ARCHITECTURE.md §4 + dsv4_chat_encoder.py.
    try:
        from .model_config_registry import get_model_config_registry
        _mc_for_dsv4 = get_model_config_registry().lookup(
            _model_path or _model_name or getattr(chat_req, "model", "") or ""
        )
        _is_dsv4 = getattr(_mc_for_dsv4, "family_name", "") == "deepseek_v4"
    except Exception:
        _is_dsv4 = False
    if _is_dsv4:
        # See create_chat_completion `_is_dsv4` block for full rationale.
        # The DSV4 Jinja template only branches on `reasoning_effort=='max'`.
        _cur_effort = (
            getattr(chat_req, "reasoning_effort", None)
            or _ct_kwargs.get("reasoning_effort")
        )
        _ct_kwargs.pop("thinking_mode", None)

        _dsv4_thinking = _resolve_dsv4_thinking_policy(
            requested_enable_thinking=chat_req.enable_thinking,
            effort_requested=bool(_cur_effort),
            tools_present=bool(getattr(chat_req, "tools", None)),
            tool_choice=getattr(chat_req, "tool_choice", None),
            default_mode=_jang_chat_default_mode(
                _model_path or _model_name or getattr(chat_req, "model", "") or ""
            ),
        )
        if (
            not _dsv4_thinking.enable_thinking
            and (chat_req.enable_thinking is True or _cur_effort)
        ):
            logger.info(
                "DSV4: using direct rail because %s.",
                _dsv4_thinking.reason,
            )
        chat_req.enable_thinking = _dsv4_thinking.enable_thinking
        _msg_kwargs["enable_thinking"] = _dsv4_thinking.enable_thinking
        _ct_kwargs["enable_thinking"] = _dsv4_thinking.enable_thinking
        _set_resolved_repetition_penalty(
            _msg_kwargs,
            chat_req.repetition_penalty,
            chat_req.model,
            enable_thinking=_msg_kwargs.get("enable_thinking"),
        )
        _stable_effort = (
            _normalize_dsv4_reasoning_effort(_cur_effort)
            if _dsv4_thinking.reasoning_effort_allowed
            else None
        )
        if _stable_effort:
            _ct_kwargs["reasoning_effort"] = _stable_effort
        else:
            _ct_kwargs.pop("reasoning_effort", None)

    # Pass tools to engine so batched.py knows not to inject <think></think>
    if chat_req.tools:
        from .api.tool_calling import convert_tools_for_template
        _msg_kwargs["tools"] = convert_tools_for_template(chat_req.tools)

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            _msg_kwargs["chat_template_kwargs"] = extra_ct

    messages_dump = [m.model_dump(exclude_none=True) for m in chat_req.messages]

    # Strip <think> blocks from prior assistant messages when thinking is disabled.
    # Same logic as the OpenAI path — prevents model from mimicking prior reasoning.
    _explicit_thinking_off = chat_req.enable_thinking is False or (
        _ct_kwargs.get("enable_thinking") is False
    )
    if _explicit_thinking_off and messages_dump:
        cleaned = []
        for msg in messages_dump:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                stripped = _THINK_STRIP_RE.sub("", msg["content"]).strip()
                if stripped != msg["content"]:
                    if not stripped and not msg.get("tool_calls"):
                        continue  # Only drop if no tool_calls (GH mlxstudio #62)
                    msg = {**msg, "content": stripped or None}
            cleaned.append(msg)
        messages_dump = cleaned

    if anthropic_req.stream:
        # Streaming: adapt Chat Completions SSE to Anthropic SSE
        adapter = AnthropicStreamAdapter(model=resolved_name)

        async def generate():
            try:
                async for chunk_str in stream_chat_completion(
                    engine=engine,
                    messages=messages_dump,
                    request=chat_req,
                    fastapi_request=fastapi_request,
                    **_msg_kwargs,
                ):
                    # Pass through SSE comments (keep-alive) to prevent client timeout
                    if chunk_str.startswith(":"):
                        yield chunk_str
                        continue
                    for event in adapter.process_chunk(chunk_str):
                        yield event
            finally:
                for event in adapter.finalize():
                    yield event

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Non-streaming: collect full response then convert
        full_text = ""
        reasoning_text = ""
        tool_calls = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        async for chunk_str in stream_chat_completion(
            engine=engine,
            messages=messages_dump,
            request=chat_req,
            fastapi_request=fastapi_request,
            **_msg_kwargs,
        ):
            if not chunk_str.startswith("data: "):
                continue
            data_str = chunk_str[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:
                err = chunk.get("error") or {}
                status_code = 413 if err.get("code") == "prompt_too_long" else 500
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "type": "error",
                        "error": {
                            "type": err.get("type", "server_error"),
                            "message": err.get("message", "Generation failed"),
                        },
                    },
                )

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    full_text += delta["content"]
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_text += reasoning
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        tc_idx = tc.get("index", 0)
                        if tc.get("id"):
                            # Ensure tool_calls list is big enough for this index
                            while len(tool_calls) <= tc_idx:
                                tool_calls.append(
                                    {
                                        "id": "",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )
                            tool_calls[tc_idx] = {
                                "id": tc["id"],
                                "function": tc.get("function", {}),
                            }
                        elif tc_idx < len(tool_calls):
                            # Append arguments to the correct tool call by index
                            args = tc.get("function", {}).get("arguments", "")
                            if args:
                                fn = tool_calls[tc_idx].get("function", {})
                                fn["arguments"] = fn.get("arguments", "") + args
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

        # Build Anthropic response
        content = []
        stop_reason = "end_turn"

        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

        if reasoning_text:
            content.append({"type": "thinking", "thinking": reasoning_text})
        if full_text:
            content.append({"type": "text", "text": full_text})
        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_data = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                    "name": func.get("name", ""),
                    "input": input_data,
                }
            )

        if not content:
            content = [{"type": "text", "text": ""}]

        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": resolved_name,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        }


# =============================================================================
# Embeddings Endpoint
# =============================================================================


@app.post(
    "/v1/embeddings",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),
        Depends(check_metal_working_set_pressure),
    ],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    OpenAI-compatible embeddings API supporting single or batch inputs.

    Single text:
    ```json
    {
      "model": "mlx-community/all-MiniLM-L6-v2-4bit",
      "input": "The quick brown fox jumps over the lazy dog"
    }
    ```

    Batch of texts:
    ```json
    {
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "input": [
        "I love machine learning",
        "Deep learning is fascinating",
        "Neural networks are powerful"
      ]
    }
    ```

    Response:
    ```json
    {
      "object": "list",
      "data": [
        {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
        {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]},
        {"object": "embedding", "index": 2, "embedding": [0.876, 0.221, ...]}
      ],
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "usage": {"prompt_tokens": 24, "total_tokens": 24}
    }
    ```

    Supported models:
    - mlx-community/all-MiniLM-L6-v2-4bit (fast, compact)
    - mlx-community/embeddinggemma-300m-6bit (high quality)
    - mlx-community/bge-large-en-v1.5-4bit (best for English)
    - Any BERT/XLM-RoBERTa/ModernBERT model from HuggingFace
    """
    global _embedding_engine

    try:
        # Resolve model name
        model_name = request.model

        # If an embedding model was pre-configured at startup, only allow that model
        if (
            _embedding_model_locked is not None
            and model_name != _embedding_model_locked
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding model '{model_name}' is not available. "
                    f"This server was started with --embedding-model {_embedding_model_locked}. "
                    f"Only '{_embedding_model_locked}' can be used for embeddings. "
                    f"Restart the server with a different --embedding-model to use '{model_name}'."
                ),
            )

        # Normalize input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        # Reject empty strings — they produce zero-vector embeddings that break
        # distance metrics (cosine similarity, L2 norm). OpenAI API also rejects them.
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty_indices:
            raise HTTPException(
                status_code=400,
                detail=f"Input contains empty string(s) at index {empty_indices}. All texts must be non-empty.",
            )

        # Batch size limit to prevent OOM on large requests
        if len(texts) > 2048:
            raise HTTPException(
                status_code=400,
                detail=f"Too many inputs ({len(texts)}). Maximum 2048 texts per request.",
            )

        # Lock protects the load-and-use sequence so a concurrent request
        # cannot hot-swap _embedding_engine between load and embed calls.
        global _embedding_lock
        if _embedding_lock is None:
            _embedding_lock = asyncio.Lock()
        async with _embedding_lock:
            # Lazy-load or swap embedding engine
            load_embedding_model(model_name, lock=False, reuse_existing=True)

            start_time = time.perf_counter()

            # Count tokens for usage reporting
            prompt_tokens = _embedding_engine.count_tokens(texts)

            # Generate embeddings (batch)
            embeddings = _embedding_engine.embed(texts)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        # Build OpenAI-compatible response with ordered indices
        data = [
            EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "mlx-embeddings not installed. Install with: pip install mlx-embeddings"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Rerank Endpoint
# =============================================================================

_reranker = None  # Lazy-loaded reranker instance
_reranker_lock: asyncio.Lock | None = (
    None  # Lazy-init to avoid binding to wrong event loop
)


@app.post(
    "/v1/rerank",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),
        Depends(check_metal_working_set_pressure),
    ],
)
async def create_rerank(request: Request):
    """
    Rerank documents by relevance to a query.

    Compatible with Cohere/Jina rerank API format.
    """
    global _reranker

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    query = body.get("query", "")
    documents = body.get("documents", [])
    top_n = body.get("top_n")
    return_documents = body.get("return_documents", False)
    model = body.get("model", "")

    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if not documents:
        raise HTTPException(
            status_code=400, detail="documents list is required and must be non-empty"
        )
    if len(documents) > 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents ({len(documents)}). Maximum is 1000.",
        )

    # Normalize documents to strings
    doc_texts = []
    for doc in documents:
        if isinstance(doc, str):
            doc_texts.append(doc)
        elif isinstance(doc, dict):
            doc_texts.append(doc.get("text", str(doc)))
        else:
            doc_texts.append(str(doc))

    # Serialize model load/swap — prevents two concurrent requests from
    # racing on unload+create when different models are requested
    global _reranker_lock
    if _reranker_lock is None:
        _reranker_lock = asyncio.Lock()
    async with _reranker_lock:
        # Load reranker if needed (or if model changed)
        if _reranker is None or (model and _reranker.model_path != model):
            if not model:
                raise HTTPException(
                    status_code=400, detail="model is required for first rerank request"
                )
            from .reranker import Reranker

            if _reranker is not None:
                _reranker.unload()
            try:
                _reranker = Reranker(model)
            except Exception as _load_err:
                # Bad model path / HF repo / network issue → 400, not 500
                # (the caller can fix their request). Log the full error
                # server-side for diagnosis; return a concise message.
                logger.error(f"Reranker load failed for '{model}': {_load_err}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load reranker model '{model}': {_load_err}",
                )
        # Capture local ref inside lock to prevent concurrent unload during rerank
        local_reranker = _reranker

    try:
        results = local_reranker.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_n,
            return_documents=return_documents,
        )
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        # Distinguish bad-model errors (→ 400) from genuine runtime
        # errors (→ 500). Reranker load is lazy inside rerank(), so
        # "Model not found" / "Could not load reranker" bubble up here
        # and are really user-fixable input errors.
        _msg = str(e)
        _is_load_err = (
            "Model not found" in _msg
            or "Could not load reranker" in _msg
            or "Repository not found" in _msg
        )
        raise HTTPException(
            status_code=400 if _is_load_err else 500,
            detail=_msg,
        )

    return {
        "id": f"rerank-{uuid.uuid4().hex[:8]}",
        "results": [
            {
                "index": r.index,
                "relevance_score": r.relevance_score,
                **({"document": {"text": r.document}} if r.document else {}),
            }
            for r in results
        ],
        "meta": {
            "model": model or local_reranker.model_path,
        },
    }


# =============================================================================
# Ollama API Compatibility (/api/chat, /api/generate, /api/tags, /api/show)
# =============================================================================


@app.get("/api/tags", dependencies=[Depends(verify_api_key)])
async def ollama_tags():
    """Ollama-compatible model list — advertises chat model + loaded
    embedding model so Ollama clients can discover both."""
    from .api.ollama_adapter import build_tags_response

    name = _resolve_model_name()
    extras: list[str] = []
    if _embedding_engine is not None and _embedding_model_locked:
        extras.append(_embedding_model_locked)
    return build_tags_response(name, _model_name or name, extra_models=extras)


@app.get("/api/ps", dependencies=[Depends(verify_api_key)])
async def ollama_ps():
    """Ollama-compatible running model list. Advertises chat + embedding
    models (both are 'running' in the sense that they're loaded in RAM)."""
    name = _resolve_model_name()
    entries = [
        {"name": name, "model": _model_name or name, "size": 0, "digest": ""}
    ]
    seen = {name, _model_name or name}
    if _embedding_engine is not None and _embedding_model_locked:
        if _embedding_model_locked not in seen:
            entries.append({
                "name": _embedding_model_locked,
                "model": _embedding_model_locked,
                "size": 0,
                "digest": "",
            })
    return {"models": entries}


@app.head("/api/version")
async def ollama_version_head():
    """HEAD variant for Ollama client version probes."""
    return Response(status_code=200)


@app.get("/api/version")
async def ollama_version():
    """Ollama version shim for client compat checks.

    GitHub Copilot in VS Code gates on ``version >= 0.6.4`` (mlxstudio#72).
    Real Ollama is on 0.12.x; we report a plausible recent real version so
    other version-gated clients don't refuse to connect. Kept in sync with
    panel/src/main/api-gateway.ts.

    This endpoint intentionally does not require vMLX API-key auth. Ollama
    clients generally perform unauthenticated provider verification before
    sending chat requests, and refusing this harmless metadata probe surfaces
    as "Unable to verify Ollama server version".
    """
    return {"version": "0.12.6"}


@app.post("/api/show", dependencies=[Depends(verify_api_key)])
async def ollama_show(fastapi_request: Request):
    """Ollama-compatible model info.

    Ollama spec v0.20.x adds a ``capabilities`` list that GitHub Copilot
    gates on to decide whether to surface a model (mlxstudio#72 follow-up).
    Without it, Copilot drops all vMLX models from its picker.

    Populates `details` and `model_info` from the loaded bundle's
    `config.json` + `jang_config.json` so Ollama clients (Open WebUI,
    Continue, Copilot) can render quant level, parameter count, and
    family. The previous all-empty-stub response made vMLX look like a
    stripped-down or broken Ollama backend in those UIs.
    """
    name = _resolve_model_name()
    capabilities = ["completion"]
    _tp = globals().get("_tool_parser")
    capabilities.append("tools")  # permissive default — most models support tools

    # Vision — on when engine is MLLM mode AND the loader did not explicitly
    # fall back to a text-only runtime (for example Mistral 4 before mlx_vlm
    # ships a matching VLM class).
    try:
        _eng = globals().get("_engine")
        if _eng is not None and getattr(_eng, "is_mllm", False):
            _vision_disabled = False
            try:
                from .utils import jang_loader as _jl
                _vision_disabled = bool(getattr(_jl, "_LAST_LOAD_VLM_FALLBACK", False))
            except Exception:
                pass
            if not _vision_disabled:
                capabilities.append("vision")
    except Exception:
        pass
    if globals().get("_reasoning_parser") is not None:
        capabilities.append("thinking")
    capabilities.append("insert")

    # Build details + model_info from bundle metadata. Falls back to the
    # all-empty stub on any read failure so we never 500 the endpoint.
    family = "mlx"
    families = ["mlx"]
    parameter_size = ""
    quantization_level = ""
    template_text = ""
    try:
        from pathlib import Path
        _path = Path(_model_path) if _model_path else None
        if _path is not None and _path.is_dir():
            _cfg_p = _path / "config.json"
            if _cfg_p.exists():
                try:
                    _cfg = json.loads(_cfg_p.read_text())
                    _mt = _cfg.get("model_type") or (_cfg.get("text_config") or {}).get("model_type")
                    if _mt:
                        family = str(_mt)
                        families = [family]
                except Exception:
                    pass
            _jang_p = _path / "jang_config.json"
            if _jang_p.exists():
                try:
                    _jc = json.loads(_jang_p.read_text())
                    _q = _jc.get("quantization") or {}
                    _src = _jc.get("source_model") or {}
                    _bits = _q.get("actual_bits") or _q.get("target_bits")
                    if _bits is not None:
                        quantization_level = f"Q{int(round(float(_bits)))}_K_{_q.get('profile') or 'JANG'}"
                    if _src.get("parameters"):
                        parameter_size = str(_src["parameters"])
                except Exception:
                    pass
            _tpl_p = _path / "chat_template.jinja"
            if _tpl_p.exists():
                try:
                    template_text = _tpl_p.read_text()
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "modelfile": "",
        "parameters": "",
        "template": template_text,
        "details": {
            "parent_model": "",
            "format": "mlx",
            "family": family,
            "families": families,
            "parameter_size": parameter_size,
            "quantization_level": quantization_level,
        },
        "model_info": {"name": name},
        "capabilities": capabilities,
    }


@app.post(
    "/api/chat",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def ollama_chat(fastapi_request: Request):
    """Ollama-compatible chat — translates to /v1/chat/completions internally."""
    from .api.ollama_adapter import (
        ollama_chat_to_openai,
        openai_chat_response_to_ollama,
        openai_chat_chunk_to_ollama_ndjson,
    )

    body = await fastapi_request.json()
    model_name = body.get("model", _resolve_model_name())
    is_streaming = body.get("stream", True)

    openai_req = ollama_chat_to_openai(body)

    from .api.models import ChatCompletionRequest

    chat_req = ChatCompletionRequest(**openai_req)
    _log_multimodal_request_shape(
        "/api/chat",
        _model_path or _model_name or chat_req.model,
        _messages_multimodal_summary(chat_req.messages),
    )
    _ollama_max_prompt_tokens = _effective_max_prompt_tokens(chat_req)
    prompt_limit_response = _reject_if_prompt_too_long_for_messages(
        chat_req.messages,
        max_prompt_tokens=_ollama_max_prompt_tokens,
    )
    if prompt_limit_response is not None:
        return prompt_limit_response

    if not is_streaming:
        result = await create_chat_completion(chat_req, fastapi_request)
        # Convert Pydantic response to dict
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump(exclude_none=True)
        elif hasattr(result, "body"):
            # JSONResponse (e.g., tool call path returns manual JSON)
            try:
                result_dict = json.loads(result.body)
            except (json.JSONDecodeError, TypeError):
                result_dict = {"choices": []}
        else:
            result_dict = dict(result)
        return openai_chat_response_to_ollama(result_dict, model_name)

    # Streaming: wrap SSE generator → NDJSON
    from starlette.responses import StreamingResponse as _SR

    engine = get_engine()
    _ollama_requested_modalities = _messages_requested_modalities(chat_req.messages)
    if _ollama_requested_modalities:
        _reject_unsupported_multimodal(
            "/api/chat",
            _ollama_requested_modalities,
        )
    if _messages_have_multimodal(chat_req.messages) and not engine.is_mllm:
        _reject_unsupported_multimodal("/api/chat")
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(chat_req.max_tokens, chat_req.model),
        "temperature": _resolve_temperature(chat_req.temperature, chat_req.model),
        "top_p": _resolve_top_p(chat_req.top_p, chat_req.model),
        "max_prompt_tokens": _ollama_max_prompt_tokens,
    }
    if chat_req.stop:
        chat_kwargs["stop"] = chat_req.stop
    _set_resolved_top_k(chat_kwargs, chat_req.top_k, chat_req.model)
    _set_resolved_min_p(chat_kwargs, chat_req.min_p, chat_req.model)
    _rp = _resolve_repetition_penalty(chat_req.repetition_penalty, chat_req.model)
    if _rp is not None:
        chat_kwargs["repetition_penalty"] = _rp
    if _compute_bypass_prefix_cache(chat_req):
        chat_kwargs["_bypass_prefix_cache"] = True
    # enable_thinking precedence: per-request > chat_template_kwargs > server default.
    # Mirrors the OpenAI path at create_chat_completion so clients get identical
    # behavior whether they speak the OpenAI or Ollama wire format.
    _ollama_ct_kwargs = _merge_ct_kwargs(chat_req.chat_template_kwargs)
    _et = _resolve_enable_thinking(
        request_value=chat_req.enable_thinking,
        ct_kwargs=_ollama_ct_kwargs,
        tools_present=bool(chat_req.tools),
        model_key=_model_path or _model_name or chat_req.model,
        engine=engine,
        auto_detect=True,
    )
    if _et is not None:
        chat_kwargs["enable_thinking"] = _et
        chat_req.enable_thinking = _et

    # Keep Ollama streaming parity with /v1/chat/completions. The non-streaming
    # Ollama path delegates to create_chat_completion(), but streaming builds
    # chat_kwargs locally. Without this, fields accepted by the adapter
    # (reasoning_effort/chat_template_kwargs) reach ChatCompletionRequest but
    # are silently dropped before the engine sees the template kwargs.
    if chat_req.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = chat_req.reasoning_effort
        _ollama_ct_kwargs.setdefault("reasoning_effort", chat_req.reasoning_effort)

    # DSV4 three-mode mapping mirrored onto the Ollama adapter path so
    # clients that speak the Ollama wire format land on the right
    # thinking_mode/reasoning_effort pair for DSV4 bundles too.
    try:
        from .model_config_registry import get_model_config_registry
        _mc_for_dsv4_o = get_model_config_registry().lookup(
            _model_path or _model_name or chat_req.model
        )
        _is_dsv4_o = getattr(_mc_for_dsv4_o, "family_name", "") == "deepseek_v4"
    except Exception:
        _is_dsv4_o = False
    if _is_dsv4_o:
        # See `_is_dsv4` block in create_chat_completion for the full
        # rationale. The DSV4 Jinja template only branches on
        # `reasoning_effort == 'max'` — every other value (including
        # 'high') falls through to the standard-thinking branch. So:
        #   chat              → enable_thinking=False, no reasoning_effort
        #   thinking standard → enable_thinking=True,  no reasoning_effort
        #   thinking max      → enable_thinking=True,  reasoning_effort='max'
        _cur_effort_o = (
            getattr(chat_req, "reasoning_effort", None)
            or _ollama_ct_kwargs.get("reasoning_effort")
        )
        _ollama_ct_kwargs.pop("thinking_mode", None)
        _dsv4_thinking_o = _resolve_dsv4_thinking_policy(
            requested_enable_thinking=chat_req.enable_thinking,
            effort_requested=bool(_cur_effort_o),
            tools_present=bool(getattr(chat_req, "tools", None)),
            tool_choice=getattr(chat_req, "tool_choice", None),
            default_mode=_jang_chat_default_mode(
                _model_path or _model_name or getattr(chat_req, "model", "") or ""
            ),
        )
        if (
            not _dsv4_thinking_o.enable_thinking
            and (chat_req.enable_thinking is True or _cur_effort_o)
        ):
            logger.info(
                "DSV4 (Ollama): using direct rail because %s.",
                _dsv4_thinking_o.reason,
            )
        chat_req.enable_thinking = _dsv4_thinking_o.enable_thinking
        chat_kwargs["enable_thinking"] = _dsv4_thinking_o.enable_thinking
        _ollama_ct_kwargs["enable_thinking"] = _dsv4_thinking_o.enable_thinking
        _set_resolved_repetition_penalty(
            chat_kwargs,
            chat_req.repetition_penalty,
            chat_req.model,
            enable_thinking=chat_kwargs.get("enable_thinking"),
        )
        _stable_effort_o = (
            _normalize_dsv4_reasoning_effort(_cur_effort_o)
            if _dsv4_thinking_o.reasoning_effort_allowed
            else None
        )
        if _stable_effort_o:
            _ollama_ct_kwargs["reasoning_effort"] = _stable_effort_o
        else:
            _ollama_ct_kwargs.pop("reasoning_effort", None)
    # Forward resolved template kwargs for every family, not only DSV4. This
    # covers Mistral/GPT-OSS reasoning_effort, Qwen thinking_budget, and any
    # model-specific kwargs accepted by the tokenizer template.
    _extra_o = {k: v for k, v in _ollama_ct_kwargs.items() if k != "enable_thinking"}
    if _extra_o:
        chat_kwargs["chat_template_kwargs"] = _extra_o

    _log_resolved_sampling_kwargs(
        "/api/chat",
        _model_path or _model_name or chat_req.model,
        chat_kwargs,
    )

    # Pass tools to engine so batched.py knows not to inject <think></think>
    # when tool calling is active (model needs to think to decide on tools)
    if chat_req.tools:
        from .api.tool_calling import convert_tools_for_template
        chat_kwargs["tools"] = convert_tools_for_template(chat_req.tools)

    # Extract messages (same logic as create_chat_completion)
    if engine.is_mllm:
        messages = []
        for msg in chat_req.messages:
            if hasattr(msg, "model_dump"):
                messages.append(msg.model_dump(exclude_none=True))
            else:
                messages.append(dict(msg))
    else:
        from .api.utils import extract_multimodal_content

        messages, _, _ = extract_multimodal_content(chat_req.messages)

    # mlxstudio#72: stateful Ollama NDJSON translation. vMLX's
    # stream_chat_completion emits tool calls across two SSE chunks:
    #   (1) delta with complete tool_calls + finish_reason=null
    #   (2) empty delta + finish_reason="tool_calls"
    # Real Ollama places tool_calls on the final `done:true` NDJSON line.
    # The old stateless adapter put them on chunk (1) (done:false) and
    # left chunk (2) with no tool_calls — GitHub Copilot, Continue.dev,
    # and Cline all silently ignore the call in that shape.
    #
    # We buffer tool_calls from chunk (1) locally, skip the adapter's
    # tool_calls emission entirely, and splice the buffered calls into
    # the adapter's final done-chunk output before yielding it.
    async def ndjson_stream():
        buffered_tcs: list[dict] = []  # ollama-shape {function:{name,arguments}}
        done_sent = False
        async for sse_line in stream_chat_completion(
            engine, messages, chat_req, fastapi_request=fastapi_request, **chat_kwargs
        ):
            # Peek at the SSE line to harvest tool_calls and rewrite the
            # chunk so the stateless adapter never sees them. `[DONE]` and
            # non-data lines pass through untouched.
            rewritten = sse_line
            if sse_line.startswith("data: "):
                _payload = sse_line[6:].strip()
                if _payload and _payload != "[DONE]":
                    try:
                        _chunk = json.loads(_payload)
                    except json.JSONDecodeError:
                        _chunk = None
                    if _chunk is not None:
                        _choices = _chunk.get("choices") or []
                        if _choices:
                            _delta = _choices[0].get("delta", {}) or {}
                            _dtcs = _delta.get("tool_calls")
                            if _dtcs:
                                for _tc in _dtcs:
                                    _fn = _tc.get("function", {}) or {}
                                    _name = _fn.get("name") or ""
                                    _args = _fn.get("arguments", "")
                                    if isinstance(_args, str):
                                        try:
                                            _args_obj = json.loads(_args) if _args else {}
                                        except json.JSONDecodeError:
                                            _args_obj = {"_raw": _args}
                                    elif _args is None:
                                        _args_obj = {}
                                    else:
                                        _args_obj = _args
                                    if _name:
                                        buffered_tcs.append(
                                            {
                                                "function": {
                                                    "name": _name,
                                                    "arguments": _args_obj,
                                                }
                                            }
                                        )
                                # Strip tool_calls from the chunk before it hits
                                # the stateless adapter so we don't double-emit.
                                _delta.pop("tool_calls", None)
                                _choices[0]["delta"] = _delta
                                _chunk["choices"] = _choices
                                rewritten = f"data: {json.dumps(_chunk)}\n\n"
                                # If this chunk has nothing useful left, skip it.
                                if not _delta.get("content") and not _choices[0].get("finish_reason"):
                                    continue

            ndjson = openai_chat_chunk_to_ollama_ndjson(rewritten, model_name)
            if not ndjson:
                continue

            # On the final done=true line, splice buffered tool_calls in.
            if buffered_tcs:
                try:
                    _ndjson_obj = json.loads(ndjson)
                    if _ndjson_obj.get("done") is True:
                        _msg = _ndjson_obj.setdefault(
                            "message", {"role": "assistant", "content": ""}
                        )
                        _msg["tool_calls"] = buffered_tcs
                        _ndjson_obj["done_reason"] = "tool_calls"
                        ndjson = json.dumps(_ndjson_obj) + "\n"
                        buffered_tcs = []
                except (json.JSONDecodeError, TypeError):
                    pass
            try:
                _ndjson_done_obj = json.loads(ndjson)
                if _ndjson_done_obj.get("done") is True:
                    if done_sent:
                        continue
                    done_sent = True
            except (json.JSONDecodeError, TypeError):
                pass
            yield ndjson

    return _SR(ndjson_stream(), media_type="application/x-ndjson")


@app.post(
    "/api/generate",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def ollama_generate(fastapi_request: Request):
    """Ollama-compatible generate.

    Ollama applies the model template by default and uses raw completion only
    when `raw: true` is requested. Keep that distinction so instruct/chat
    bundles do not receive untemplated prompts.
    """
    body = await fastapi_request.json()
    model_name = body.get("model", _resolve_model_name())
    is_streaming = body.get("stream", True)
    raw_mode = bool(body.get("raw"))

    from .api.ollama_adapter import (
        ollama_generate_to_openai,
        ollama_generate_to_openai_chat,
        openai_chat_response_to_ollama_generate,
        openai_chat_chunk_to_ollama_generate_ndjson,
    )

    if not raw_mode:
        openai_chat_req = ollama_generate_to_openai_chat(body)
        from .api.models import ChatCompletionRequest

        chat_req = ChatCompletionRequest(**openai_chat_req)
        _ollama_gen_max_prompt_tokens = _effective_max_prompt_tokens(chat_req)
        prompt_limit_response = _reject_if_prompt_too_long_for_messages(
            chat_req.messages,
            max_prompt_tokens=_ollama_gen_max_prompt_tokens,
        )
        if prompt_limit_response is not None:
            return prompt_limit_response
        if not is_streaming:
            result = await create_chat_completion(chat_req, fastapi_request)
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump(exclude_none=True)
            elif hasattr(result, "body"):
                try:
                    result_dict = json.loads(result.body)
                except (json.JSONDecodeError, TypeError):
                    result_dict = {"choices": []}
            else:
                result_dict = dict(result)
            return openai_chat_response_to_ollama_generate(result_dict, model_name)

        from starlette.responses import StreamingResponse as _SR

        streaming_response = await create_chat_completion(chat_req, fastapi_request)

        async def templated_ndjson_stream():
            done_sent = False
            async for sse_line in streaming_response.body_iterator:
                if isinstance(sse_line, bytes):
                    sse_line = sse_line.decode("utf-8", "replace")
                ndjson = openai_chat_chunk_to_ollama_generate_ndjson(
                    sse_line, model_name
                )
                if ndjson:
                    try:
                        _ndjson_obj = json.loads(ndjson)
                        if _ndjson_obj.get("done") is True:
                            if done_sent:
                                continue
                            done_sent = True
                    except (json.JSONDecodeError, TypeError):
                        pass
                    yield ndjson

        return _SR(templated_ndjson_stream(), media_type="application/x-ndjson")

    openai_req = ollama_generate_to_openai(body)

    from .api.models import CompletionRequest

    if not is_streaming:
        openai_req["stream"] = False
        comp_req = CompletionRequest(**openai_req)
        prompts = [comp_req.prompt] if isinstance(comp_req.prompt, str) else comp_req.prompt
        _ollama_raw_max_prompt_tokens = _effective_max_prompt_tokens(comp_req)
        prompt_limit_response = _reject_if_prompt_too_long_for_prompts(
            prompts,
            max_prompt_tokens=_ollama_raw_max_prompt_tokens,
        )
        if prompt_limit_response is not None:
            return prompt_limit_response
        # create_completion takes only the CompletionRequest — no
        # fastapi_request arg (unlike create_chat_completion which uses
        # it for disconnect detection). Pre-session regression from
        # v1.3.12 where the call passed 2 args causing TypeError.
        result = await create_completion(comp_req)
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump(exclude_none=True)
        else:
            result_dict = dict(result)

        text = ""
        choices = result_dict.get("choices", [])
        if choices:
            text = choices[0].get("text", "")

        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "response": text,
            "done": True,
            "done_reason": choices[0].get("finish_reason", "stop")
            if choices
            else "stop",
        }

    # Streaming: wrap completions SSE → NDJSON
    from starlette.responses import StreamingResponse as _SR
    from .api.ollama_adapter import openai_completion_chunk_to_ollama_ndjson

    openai_req["stream"] = True
    comp_req = CompletionRequest(**openai_req)

    engine = get_engine()
    prompts = [comp_req.prompt] if isinstance(comp_req.prompt, str) else comp_req.prompt
    _ollama_raw_max_prompt_tokens = _effective_max_prompt_tokens(comp_req)
    prompt_limit_response = _reject_if_prompt_too_long_for_prompts(
        prompts,
        max_prompt_tokens=_ollama_raw_max_prompt_tokens,
    )
    if prompt_limit_response is not None:
        return prompt_limit_response

    async def ndjson_stream():
        done_sent = False
        async for sse_line in stream_completions_multi(engine, prompts, comp_req):
            ndjson = openai_completion_chunk_to_ollama_ndjson(sse_line, model_name)
            if ndjson:
                try:
                    _ndjson_obj = json.loads(ndjson)
                    if _ndjson_obj.get("done") is True:
                        if done_sent:
                            continue
                        done_sent = True
                except (json.JSONDecodeError, TypeError):
                    pass
                yield ndjson

    return _SR(ndjson_stream(), media_type="application/x-ndjson")


@app.post(
    "/api/embeddings",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_memory_pressure),
        Depends(check_metal_working_set_pressure),
    ],
)
@app.post(
    "/api/embed",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_memory_pressure),
        Depends(check_metal_working_set_pressure),
    ],
)
async def ollama_embed(fastapi_request: Request):
    """Ollama-compatible embeddings."""
    body = await fastapi_request.json()
    # Forward to OpenAI embeddings endpoint
    from .api.models import EmbeddingRequest

    openai_req = EmbeddingRequest(
        model=body.get("model", _resolve_model_name()),
        input=body.get("input") or body.get("prompt", ""),
    )
    result = await create_embeddings(openai_req)
    if hasattr(result, "model_dump"):
        result_dict = result.model_dump(exclude_none=True)
    else:
        result_dict = dict(result)
    embeddings = [d["embedding"] for d in result_dict.get("data", [])]
    return {"model": body.get("model", ""), "embeddings": embeddings}


# Ollama no-op endpoints (prevent client errors)
@app.post("/api/pull", dependencies=[Depends(verify_api_key)])
async def ollama_pull():
    return {"status": "success"}


@app.post("/api/delete", dependencies=[Depends(verify_api_key)])
async def ollama_delete(request: Request):
    """vmlx#57 — actually remove the model directory; returns {"status": "success"}.

    Ollama-compatible body: `{"name": "<model>"}` or `{"model": "<path>"}`.
    Successful and already-absent deletes return `{"status": "success"}` for
    Ollama client compatibility.
    Resolves to a local path via the same logic the loader uses, then walks
    upward to confirm we're inside the configured model_directories before
    deleting. Refuses any path outside those roots.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    target = body.get("name") or body.get("model") or ""
    if not target:
        raise HTTPException(status_code=400, detail="Provide 'name' or 'model' in body")

    from pathlib import Path
    import shutil
    from .api.utils import resolve_to_local_path

    local = resolve_to_local_path(target)
    p = Path(local).resolve()

    # Enforce the path is inside one of the configured model directories.
    allowed_roots: list[Path] = []
    md = getattr(_app_state, "model_directories", None) if "_app_state" in dir() else None
    if md is None:
        # Fallbacks: HF cache + ~/.mlxstudio
        allowed_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        allowed_roots.append(Path.home() / ".mlxstudio")
    else:
        allowed_roots = [Path(d).resolve() for d in (md or [])]

    inside = any(str(p).startswith(str(root.resolve()) + "/") for root in allowed_roots)
    if not inside:
        raise HTTPException(
            status_code=403,
            detail=f"Refusing to delete path outside allowed model directories: {p}",
        )
    if not p.exists():
        return {"status": "success", "deleted": str(p), "note": "already absent"}
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"status": "success", "deleted": str(p)}


@app.post("/api/copy", dependencies=[Depends(verify_api_key)])
async def ollama_copy():
    return {"status": "success"}


@app.post("/v1/images/cancel", dependencies=[Depends(verify_api_key)])
async def image_generation_cancel():
    """Cancel the in-flight or next image generation request.

    Cooperative — interrupts the engine at the next boundary it owns
    (between images in a batch, or before the next generate() call).
    The currently-running denoise step inside mflux still completes;
    this prevents subsequent steps and aborts the request flow.

    Tracks vmlx#100 / mlxstudio#100. Without this, cancelled jobs in
    the panel kept running and could OOM the machine.

    Returns 200 with `cancelled: true` even if no job is active —
    setting the flag is idempotent and harmless on idle.
    """
    global _image_gen
    if _image_gen is not None:
        try:
            _image_gen.cancel()
        except Exception:
            pass
    return {"cancelled": True}


@app.post("/api/create", dependencies=[Depends(verify_api_key)])
async def ollama_create():
    return {"status": "success"}


# =============================================================================
# Image Generation Endpoint (OpenAI-compatible /v1/images/generations)
# =============================================================================

_image_gen = None  # Lazy-loaded ImageGenEngine
_image_gen_lock: asyncio.Lock | None = (
    None  # Lazy-init to avoid binding to wrong event loop
)
_image_gen_executor: ThreadPoolExecutor | None = None


def _get_image_gen_executor() -> ThreadPoolExecutor:
    """Return the dedicated single-thread executor for mflux/MLX image work.

    MLX streams are thread-local. Image model load, wake reload, generation,
    editing, and unload must stay on the same worker thread; using
    `asyncio.to_thread()` lets Python pick arbitrary default-executor threads
    and can crash real mflux requests with `There is no Stream(gpu,N) in
    current thread`.
    """
    global _image_gen_executor
    if _image_gen_executor is None:
        _image_gen_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="image-worker"
        )
    return _image_gen_executor


async def _run_image_gen_call(fn, /, *args, **kwargs):
    """Run an ImageGenEngine method on the dedicated image MLX thread."""
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_get_image_gen_executor(), call)


def _run_image_gen_call_sync(fn, /, *args, **kwargs):
    """Synchronous form for CLI startup image preload before uvicorn exists."""
    call = functools.partial(fn, *args, **kwargs)
    return _get_image_gen_executor().submit(call).result()


def _shutdown_image_gen_executor() -> None:
    """Stop the dedicated image executor at process shutdown/test teardown."""
    global _image_gen_executor
    if _image_gen_executor is not None:
        _image_gen_executor.shutdown(wait=False, cancel_futures=True)
        _image_gen_executor = None


async def _cancel_image_disconnect_watch(task: asyncio.Task | None) -> None:
    """Cancel and drain an image disconnect watcher."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@app.post(
    "/v1/images/generations",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def create_image(request: Request):
    """
    Generate images from text prompts using Flux models.

    OpenAI-compatible format. Supports: schnell, dev, z-image-turbo, flux2-klein.

    ```json
    {
      "model": "schnell",
      "prompt": "A cat astronaut floating in space",
      "n": 1,
      "size": "1024x1024",
      "quality": "standard",
      "response_format": "b64_json"
    }
    ```
    """
    global _image_gen

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # vmlx#100 / mlxstudio#100 — disconnect watchdog. If the client closes
    # the HTTP connection (panel "Stop" button, browser reload, network
    # cut), cancel the in-flight generation cooperatively. Mflux's denoise
    # step is uncancellable mid-step, so this aborts at the next boundary
    # we own (between images in a batch, or before the next generate()).
    async def _watch_disconnect():
        try:
            while True:
                await asyncio.sleep(0.5)
                if await request.is_disconnected():
                    if _image_gen is not None:
                        try:
                            _image_gen.cancel()
                        except Exception:
                            pass
                    return
        except asyncio.CancelledError:
            return

    prompt = body.get("prompt", "")
    model = body.get("model", "schnell")
    n = min(body.get("n", 1), 4)  # Cap at 4 images per request
    size = body.get("size", "1024x1024")
    quality = body.get("quality", "standard")
    response_format = body.get("response_format", "b64_json")
    seed = body.get("seed")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if response_format == "url":
        raise HTTPException(
            status_code=400,
            detail="response_format 'url' is not supported. Use 'b64_json'.",
        )

    # Parse size
    try:
        width, height = [int(x) for x in size.split("x")]
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size format: {size}. Use WxH (e.g., 1024x1024)",
        )

    # Enforce dimension limits (prevent OOM from absurd sizes)
    MAX_DIM = 4096
    if width < 64 or height < 64:
        raise HTTPException(
            status_code=400, detail=f"Minimum dimension is 64. Got {width}x{height}"
        )
    if width > MAX_DIM or height > MAX_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum dimension is {MAX_DIM}. Got {width}x{height}",
        )

    # Map quality to steps
    steps = body.get("steps")  # Allow explicit steps override
    if steps is None:
        if quality == "hd":
            steps = 30
        # else: use model default (handled by engine)

    # Map quantize from body or use default
    quantize = body.get("quantize")
    negative_prompt = body.get("negative_prompt")
    model_path = body.get("model_path")  # Custom local model path

    # Guidance scale
    guidance = body.get("guidance", 3.5)

    # img2img: optional source image + strength for iterative generation
    source_image_b64 = body.get("image")
    image_strength = body.get("strength")

    # Hold the lock for both load AND generate to prevent model-swap races.
    # On a single-GPU Mac, image generation must be serialized anyway.
    global _image_gen_lock
    if _image_gen_lock is None:
        _image_gen_lock = asyncio.Lock()
    async with _image_gen_lock:
        # If in standby (deep/soft sleep), wake first to avoid split-brain
        # where _standby_state='deep' but we're about to load a model
        if _standby_state is not None:
            await admin_wake()

        # Load engine if needed (or if model changed)
        if _image_gen is None:
            try:
                from .image_gen import ImageGenEngine

                _image_gen = ImageGenEngine()
            except ImportError:
                raise HTTPException(
                    status_code=501,
                    detail="mflux not installed. Install with: pip install mflux",
                )

        # Resolve aliases before comparison (e.g., "flux-schnell" → "schnell")
        from .image_gen import EDIT_MODELS as _EDIT_MODELS
        from .image_gen import SUPPORTED_MODELS as _IMG_MODELS

        # Reject edit-only models on the gen endpoint. Calling generate()
        # with an edit model (e.g. qwen-image-edit) raises TypeError deep
        # in mflux and surfaces as a confusing 500. Return a clear 400
        # pointing the caller at /v1/images/edits.
        if model and model.lower() in _EDIT_MODELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model}' is an editing model. Use "
                    f"POST /v1/images/edits, not /v1/images/generations."
                ),
            )

        resolved_model = _IMG_MODELS.get(model.lower(), model) if model else model

        # If pre-loaded (from serve command), skip re-loading unless explicitly different model.
        already_loaded = _image_gen.is_loaded and (
            not model
            or resolved_model == _image_gen.model_name
            or model == _model_name
            or model == _served_model_name
            or model == _model_path
        )
        if not already_loaded:
            try:
                if _image_gen.is_loaded:
                    await _run_image_gen_call(_image_gen.unload)
                # mlxstudio#98: frontend doesn't send model_path on /v1/images/generations,
                # so fall back to the global _model_path set by `serve`. /v1/images/edits
                # already does this — keep both paths symmetric so standby-wake reloads
                # don't fail with "No local model files found".
                _preserved_mflux_class = getattr(_image_gen, "_mflux_class", None)
                _preserve_mflux_class_for_same_model = bool(
                    _preserved_mflux_class
                    and (
                        not model
                        or resolved_model == getattr(_image_gen, "model_name", None)
                        or model == _model_name
                        or model == _served_model_name
                        or model == _model_path
                    )
                )
                await _run_image_gen_call(
                    _image_gen.load,
                    model,
                    quantize=quantize,
                    model_path=model_path or _model_path,
                    mflux_class=(
                        _preserved_mflux_class
                        if _preserve_mflux_class_for_same_model
                        else None
                    ),
                    lora_paths=_image_lora_paths,
                    lora_scales=_image_lora_scales,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load image model '{model}': {e}"
                )

        # If source image provided (img2img), save to temp file for mflux
        source_image_path = None
        images = []
        _disconnect_watch = asyncio.create_task(_watch_disconnect())
        try:
            if source_image_b64 and image_strength is not None:
                import tempfile
                from pathlib import Path

                raw_b64 = re.sub(r"^data:image/[^;]+;base64,", "", source_image_b64)
                img_bytes = base64.b64decode(raw_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                source_image_path = tmp.name

            # Generate images (inside lock to prevent concurrent model swap)
            for i in range(n):
                # Check for client disconnect between images
                if await request.is_disconnected():
                    logger.info("Image generation cancelled: client disconnected")
                    break
                img_seed = (seed + i) if seed is not None else None
                try:
                    result = await _run_image_gen_call(
                        _image_gen.generate,
                        prompt=prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance=guidance,
                        seed=img_seed,
                        negative_prompt=negative_prompt,
                        image_path=source_image_path,
                        image_strength=image_strength if source_image_path else None,
                    )
                except Exception as e:
                    logger.error(f"Image generation failed: {e}")
                    raise HTTPException(
                        status_code=500, detail=f"Image generation failed: {e}"
                    )

                images.append(
                    {
                        "b64_json": result.b64_json,
                        "revised_prompt": prompt,
                        "seed": result.seed,
                    }
                )
        finally:
            # Clean up temp source image
            if source_image_path:
                try:
                    os.unlink(source_image_path)
                except OSError:
                    pass
            await _cancel_image_disconnect_watch(_disconnect_watch)

    return {
        "created": int(time.time()),
        "data": images,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "images_generated": len(images),
        },
    }


# =============================================================================
# Image Editing Endpoint
# =============================================================================


@app.post(
    "/v1/images/edits",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def create_image_edit(request: Request):
    """Edit an image using an instruction and a source image.

    OpenAI-compatible format. Supports: Qwen-Image-Edit.

    Request body:
        model: Edit model name (e.g., "qwen-image-edit")
        prompt: Text instruction for the edit
        image: Base64-encoded source image
        mask: Base64-encoded mask image (optional, for inpainting)
        size: Output size (e.g., "1024x1024")
        n: Number of images to generate (default 1)
        strength: Edit strength 0.0-1.0 (default 0.75)
        guidance: Guidance scale (default 3.5)
        steps: Inference steps (optional)
        seed: Random seed (optional)
    """
    global _image_gen

    body = await _parse_image_edit_request_body(request)

    from pathlib import Path
    import shutil

    model = body.get("model", "qwen-image-edit")
    prompt = body.get("prompt", "")
    image_b64 = body.get("image", "")
    mask_b64 = body.get("mask")
    size = body.get("size", "1024x1024")
    n = min(body.get("n", 1), 4)
    strength = body.get("strength", 0.75)
    guidance = body.get("guidance", 3.5)
    steps = body.get("steps")
    seed = body.get("seed")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image (base64) is required")

    # Parse size
    try:
        width, height = map(int, size.lower().split("x"))
    except (ValueError, AttributeError):
        width, height = 1024, 1024
    # Validate dimensions (same limits as image generation)
    if width < 64 or width > 4096:
        raise HTTPException(
            status_code=400, detail=f"Width must be between 64 and 4096, got {width}"
        )
    if height < 64 or height > 4096:
        raise HTTPException(
            status_code=400, detail=f"Height must be between 64 and 4096, got {height}"
        )

    # Save base64 image to temp file (mflux needs file paths)
    tmp_dir = tempfile.mkdtemp(prefix="vmlx_edit_")
    image_path = Path(tmp_dir) / "input.png"
    mask_path = None
    images = []

    try:
        # Decode and save input image — convert to proper RGB PNG via PIL
        # Handles: JPEG, PNG (with alpha), MPO (iPhone Portrait), WebP, GIF,
        # BMP, TIFF, AVIF, CMYK, 16-bit, EXIF-rotated images.
        # HEIC requires pillow-heif plugin (not bundled).
        image_b64 = re.sub(r"^data:image/[^;]+;base64,", "", image_b64)
        image_data = base64.b64decode(image_b64)
        try:
            from PIL import Image as _PILImage, ImageOps as _ImageOps
            import io as _io

            src_img = _PILImage.open(_io.BytesIO(image_data))
            src_format = src_img.format
            src_original_mode = src_img.mode

            # For animated images (GIF, APNG, WebP), use only the first frame
            if getattr(src_img, "is_animated", False):
                src_img.seek(0)
                logger.info(
                    f"Edit: animated {src_format} detected, using first frame only"
                )

            # Apply EXIF orientation (iPhone/DSLR photos may be rotated)
            try:
                src_img = _ImageOps.exif_transpose(src_img)
            except Exception:
                pass  # No EXIF data or unsupported — continue without rotation

            # Convert to 8-bit RGB (handles RGBA, CMYK, P, L, I, I;16, F, LA, PA modes)
            if src_img.mode in ("RGBA", "LA", "PA"):
                # Composite alpha onto white background before converting
                background = _PILImage.new("RGB", src_img.size, (255, 255, 255))
                background.paste(src_img, mask=src_img.split()[-1])
                src_img = background
            elif src_img.mode != "RGB":
                src_img = src_img.convert("RGB")

            src_img.save(str(image_path), format="PNG")
            logger.info(
                f"Edit: received image {len(image_b64)} b64 chars -> {len(image_data)} bytes "
                f"(format={src_format}, mode={src_original_mode}, size={src_img.size}), "
                f"converted to RGB PNG at {image_path}"
            )
        except Exception as conv_err:
            # Provide actionable error instead of silently saving raw bytes
            err_msg = str(conv_err)
            if (
                "HEIF" in err_msg.upper()
                or "HEIC" in err_msg.upper()
                or "cannot identify" in err_msg.lower()
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format: {conv_err}. "
                    f"HEIC/HEIF images require the pillow-heif package. "
                    f"Please convert to JPEG or PNG before uploading.",
                )
            raise HTTPException(
                status_code=400, detail=f"Failed to process input image: {conv_err}"
            )

        # Decode and save mask if provided — also convert via PIL.
        # vmlx#97: paint-tool masks arriving as RGBA with the painted
        # region only in the alpha channel previously lost all signal
        # on `.convert("L")` (grayscale drops alpha). Now we detect RGBA
        # and use max(RGB, alpha) so both channel forms work. Also
        # reject all-zero masks with a clear error so users know the
        # mask never registered, instead of feeding an empty mask into
        # Flux Fill and getting confusing downstream output.
        if mask_b64 and isinstance(mask_b64, str) and mask_b64.strip():
            mask_path = Path(tmp_dir) / "mask.png"
            mask_b64_clean = re.sub(r"^data:image/[^;]+;base64,", "", mask_b64)
            mask_data = base64.b64decode(mask_b64_clean)
            try:
                from PIL import Image as _PILImage, ImageOps as _ImageOps
                import io as _io

                mask_img = _PILImage.open(_io.BytesIO(mask_data))
                # Apply EXIF orientation to match source image
                try:
                    mask_img = _ImageOps.exif_transpose(mask_img)
                except Exception:
                    pass

                # vmlx#97: paint-tool output is typically RGBA with the
                # painted strokes in the alpha channel (transparent bg +
                # opaque strokes). Simple .convert("L") drops alpha,
                # which collapses the mask to the underlying RGB —
                # usually solid black or the canvas fill. Blend alpha
                # into the grayscale so either encoding works:
                #   - RGBA → L: pixel = max(RGB→L, alpha)
                #   - RGB  → L: pixel = standard L conversion
                #   - LA   → L: pixel = max(L, alpha)
                if mask_img.mode == "RGBA":
                    rgb_l = mask_img.convert("RGB").convert("L")
                    alpha = mask_img.split()[-1]
                    # Element-wise max(rgb_l, alpha) via ImageChops
                    from PIL import ImageChops as _ImageChops
                    mask_img = _ImageChops.lighter(rgb_l, alpha)
                elif mask_img.mode == "LA":
                    l = mask_img.convert("L")
                    alpha = mask_img.split()[-1]
                    from PIL import ImageChops as _ImageChops
                    mask_img = _ImageChops.lighter(l, alpha)
                elif mask_img.mode != "L":
                    mask_img = mask_img.convert("L")

                # Reject all-zero masks — far more actionable than the
                # downstream "requires mask_path for inpainting" error.
                extrema = mask_img.getextrema()
                # PIL returns (min, max) for L mode
                if isinstance(extrema, tuple) and len(extrema) == 2 and extrema[1] == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Mask is all black (no painted region). Paint some "
                            "pixels white/opaque on the mask before submitting. "
                            "If you used the brush tool and the strokes look "
                            "painted but this error fires, re-open the mask "
                            "painter and use the Rectangle tool as a workaround. "
                            "(vmlx#97)"
                        ),
                    )

                mask_img.save(str(mask_path), format="PNG")
                logger.info(
                    f"Edit: mask converted to grayscale PNG at {mask_path} "
                    f"(min={extrema[0] if isinstance(extrema, tuple) else '?'}, "
                    f"max={extrema[1] if isinstance(extrema, tuple) else '?'})"
                )
            except HTTPException:
                raise
            except Exception as mask_err:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process mask image: {mask_err}"
                )

        global _image_gen_lock
        if _image_gen_lock is None:
            _image_gen_lock = asyncio.Lock()
        async with _image_gen_lock:
            # Wake from standby if needed (prevents split-brain state)
            if _standby_state is not None:
                await admin_wake()

            # Load edit engine if needed
            if _image_gen is None:
                try:
                    from .image_gen import ImageGenEngine

                    _image_gen = ImageGenEngine()
                except ImportError:
                    raise HTTPException(
                        status_code=501,
                        detail="mflux not installed. Install with: pip install mflux",
                    )

            from .image_gen import EDIT_MODELS, SUPPORTED_MODELS

            # Reject gen-only models on the edit endpoint. Without this
            # check, calling edit() with a gen model (e.g. "schnell")
            # silently falls through to img2img, ignoring the user's
            # explicit instruction-based-edit intent.
            if model and model.lower() in SUPPORTED_MODELS and model.lower() not in EDIT_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model '{model}' is a generation model. Use "
                        f"POST /v1/images/generations, not /v1/images/edits. "
                        f"For instruction-based editing, load qwen-image-edit, "
                        f"flux-kontext, or flux-fill."
                    ),
                )

            resolved = EDIT_MODELS.get(model.lower(), model)

            # Check if we need to load a different model
            already_loaded = _image_gen.is_loaded and (
                resolved == _image_gen.model_name
            )
            if not already_loaded:
                try:
                    if _image_gen.is_loaded:
                        await _run_image_gen_call(_image_gen.unload)
                    edit_model_path = body.get("model_path") or _model_path
                    await _run_image_gen_call(
                        _image_gen.load,
                        model,
                        model_path=edit_model_path,
                        quantize=_image_quantize,
                        lora_paths=_image_lora_paths,
                        lora_scales=_image_lora_scales,
                    )
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load edit model '{model}': {e}",
                    )

            # Generate edited images (inside lock)
            # Verify the input image can be opened
            try:
                from PIL import Image as _PILImage

                _src = _PILImage.open(str(image_path))
                logger.info(
                    f"Edit: source image verified: {_src.size} mode={_src.mode} format={_src.format}"
                )
            except Exception as _e:
                logger.error(f"Edit: source image INVALID: {_e}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid source image: {_e}"
                )

            logger.info(
                f"Edit: model={model} resolved={resolved} prompt={prompt[:50]!r} "
                f"size={width}x{height} steps={steps} guidance={guidance} "
                f"strength={strength} seed={seed}"
            )

            for i in range(n):
                if await request.is_disconnected():
                    logger.info("Image edit cancelled: client disconnected")
                    break
                img_seed = (seed + i) if seed is not None else None
                try:
                    result = await _run_image_gen_call(
                        _image_gen.edit,
                        prompt=prompt,
                        image_path=str(image_path),
                        width=width,
                        height=height,
                        steps=steps,
                        guidance=guidance,
                        seed=img_seed,
                        strength=strength,
                        negative_prompt=body.get("negative_prompt"),
                        mask_path=str(mask_path) if mask_path else None,
                    )
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Image editing failed: {e}")
                    raise HTTPException(
                        status_code=500, detail=f"Image editing failed: {e}"
                    )

                images.append(
                    {
                        "b64_json": result.b64_json,
                        "revised_prompt": prompt,
                        "seed": result.seed,
                    }
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image edit endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image editing failed: {e}")
    finally:
        # Clean up temp files
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return {
        "created": int(time.time()),
        "data": images,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "images_generated": len(images),
        },
    }


async def _parse_image_edit_request_body(request: Request) -> dict[str, Any]:
    """Parse OpenAI-compatible image edit bodies.

    The OpenAI edit API is multipart/form-data for image and mask uploads, while
    older vMLX clients may still send JSON with base64 image fields. Keep both
    forms, but never ask Starlette to decode a multipart PNG body as UTF-8 JSON.
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid request body: object required")
        return body

    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid multipart body: {e}")

    parsed: dict[str, Any] = {}
    int_fields = {"n", "steps", "seed"}
    float_fields = {"strength", "guidance"}

    async def _file_to_b64(value: Any) -> str:
        data = await value.read()
        if not data:
            return ""
        return base64.b64encode(data).decode("ascii")

    for key, value in form.multi_items():
        if key in {"image", "mask"} and hasattr(value, "read"):
            parsed[key] = await _file_to_b64(value)
            continue

        if hasattr(value, "read"):
            # Unknown file field: ignore for OpenAI compatibility.
            continue

        text = str(value)
        if key in int_fields and text.strip():
            try:
                parsed[key] = int(text)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{key} must be an integer")
        elif key in float_fields and text.strip():
            try:
                parsed[key] = float(text)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{key} must be a number")
        else:
            parsed[key] = text

    return parsed


# =============================================================================
# MCP Endpoints
# =============================================================================


@app.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    if _mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    status = _mcp_manager.get_policy_status(policy=_mcp_policy)
    for tool in status["tools"]:
        tools.append(
            MCPToolInfo(
                name=tool["name"],
                description=tool["description"],
                server=tool["server"],
                parameters=tool["parameters"],
                enabled=tool["enabled"],
                effective=tool["effective"],
                transport=tool["transport"],
                server_state=tool["server_state"],
                error=tool["error"],
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@app.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    if _mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    status = _mcp_manager.get_policy_status(policy=_mcp_policy)
    for server in status["servers"]:
        servers.append(
            MCPServerInfo(
                name=server["name"],
                state=server["state"],
                transport=server["transport"],
                tools_count=server["tools_count"],
                error=server["error"],
                enabled=server["enabled"],
                configured=server["configured"],
                command_redacted=server["command_redacted"],
                url_redacted=server["url_redacted"],
                last_connected=server["last_connected"],
                env_keys=server["env_keys"],
                header_keys=server["header_keys"],
            )
        )

    return MCPServersResponse(servers=servers)


@app.post(
    "/v1/mcp/execute", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    if _mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    result = await _mcp_manager.execute_tool(
        request.tool_name,
        request.arguments,
        policy=_mcp_policy,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )


# =============================================================================
# Audio Endpoints
# =============================================================================

# Global audio engines (lazy loaded)
_stt_engine = None
_tts_engine = None
_stt_lock: asyncio.Lock | None = None  # Lazy-init to avoid binding to wrong event loop
_tts_lock: asyncio.Lock | None = None  # Lazy-init to avoid binding to wrong event loop


@app.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_transcription(
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """
    Transcribe audio to text (OpenAI Whisper API compatible).

    Supported models:
    - whisper-large-v3 (multilingual, best quality)
    - whisper-large-v3-turbo (faster)
    - whisper-medium, whisper-small (lighter)
    - parakeet-tdt-0.6b-v2 (English, fastest)
    """
    global _stt_engine, _stt_lock

    try:
        from .audio.stt import STTEngine  # Lazy import - optional feature

        # Map model aliases to full names
        model_map = {
            "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
            "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "whisper-medium": "mlx-community/whisper-medium-mlx",
            "whisper-small": "mlx-community/whisper-small-mlx",
            "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
            "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
        }
        model_name = model_map.get(model, model)

        # Serialize model load/swap — prevents concurrent requests from
        # racing on unload+create when different models are requested
        if _stt_lock is None:
            _stt_lock = asyncio.Lock()
        async with _stt_lock:
            if _stt_engine is None or _stt_engine.model_name != model_name:
                if _stt_engine is not None:
                    _stt_engine.unload()
                new_engine = STTEngine(model_name)
                new_engine.load()
                _stt_engine = new_engine
            # Capture local ref inside lock to prevent concurrent unload during transcription
            local_stt = _stt_engine

        # Save uploaded file temporarily
        ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = local_stt.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            return Response(content=result.text, media_type="text/plain")

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError as e:
        msg = str(e)
        if "mlx_audio" in msg or "mlx-audio" in msg:
            detail = "mlx-audio not installed. Install with: pip install mlx-audio"
        else:
            detail = f"STT dependency missing: {msg}"
        raise HTTPException(status_code=503, detail=detail)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/v1/audio/speech",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_speech(request: AudioSpeechRequest):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supported models:
    - kokoro (fast, lightweight)
    - chatterbox (multilingual, expressive)
    - vibevoice (realtime)
    - voxcpm (Chinese/English)
    """
    global _tts_engine, _tts_lock

    try:
        from .audio.tts import TTSEngine  # Lazy import - optional feature

        # Map model aliases to full names
        model_map = {
            "kokoro": "mlx-community/Kokoro-82M-bf16",
            "kokoro-4bit": "mlx-community/Kokoro-82M-4bit",
            "chatterbox": "mlx-community/chatterbox-turbo-fp16",
            "chatterbox-4bit": "mlx-community/chatterbox-turbo-4bit",
            "vibevoice": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            "voxcpm": "mlx-community/VoxCPM1.5",
        }
        model_name = model_map.get(request.model, request.model)

        # Serialize model load/swap — prevents concurrent requests from
        # racing on unload+create when different models are requested
        if _tts_lock is None:
            _tts_lock = asyncio.Lock()
        async with _tts_lock:
            if _tts_engine is None or _tts_engine.model_name != model_name:
                if _tts_engine is not None:
                    _tts_engine.unload()
                new_engine = TTSEngine(model_name)
                new_engine.load()
                _tts_engine = new_engine
            # Capture local ref inside lock to prevent concurrent unload during generation
            local_tts = _tts_engine

        audio = local_tts.generate(
            request.input, voice=request.voice, speed=request.speed
        )
        audio_bytes = local_tts.to_bytes(audio, format=request.response_format)

        content_type = (
            "audio/wav"
            if request.response_format == "wav"
            else f"audio/{request.response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError as e:
        msg = str(e)
        if "mlx_audio" in msg or "mlx-audio" in msg:
            detail = "mlx-audio not installed. Install with: pip install mlx-audio"
        else:
            detail = f"TTS dependency missing: {msg}"
        raise HTTPException(status_code=503, detail=detail)
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        # Distinguish bad-model/bad-input errors (→ 400) from real runtime
        # errors (→ 500). HF "Repository Not Found" and mlx-audio's
        # "Unsupported model" are user-fixable input errors.
        _msg = str(e)
        _is_user_err = (
            "Repository Not Found" in _msg
            or "Repository not found" in _msg
            or "RepositoryNotFoundError" in _msg
            or "Unsupported model" in _msg
            or "No such file" in _msg
        )
        raise HTTPException(
            status_code=400 if _is_user_err else 500,
            detail=_msg,
        )


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from .audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}


# =============================================================================
# Completion Endpoints
# =============================================================================


@app.post(
    "/v1/completions",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    engine = get_engine()
    _reject_unsupported_logprobs_request(
        endpoint="Completions",
        requested=request.logprobs is not None,
        stream=bool(request.stream),
        is_mllm=bool(getattr(engine, "is_mllm", False)),
    )

    # Normalize model name (consistent with chat/responses endpoints)
    resolved_name = _resolve_model_name()
    request.model = resolved_name

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    _completion_max_prompt_tokens = _effective_max_prompt_tokens(request)
    prompt_limit_response = _reject_if_prompt_too_long_for_prompts(
        prompts,
        max_prompt_tokens=_completion_max_prompt_tokens,
    )
    if prompt_limit_response is not None:
        return prompt_limit_response

    if request.stream:
        return StreamingResponse(
            stream_completions_multi(engine, prompts, request),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout if request.timeout is not None else _default_timeout
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    for i, prompt in enumerate(prompts):
        try:
            gen_kwargs = {
                "prompt": prompt,
                "max_tokens": _resolve_max_tokens(request.max_tokens, getattr(request, "model", "")),
                "temperature": _resolve_temperature(request.temperature, request.model),
                "top_p": _resolve_top_p(request.top_p, request.model),
                "stop": request.stop,
                "max_prompt_tokens": _completion_max_prompt_tokens,
            }
            _set_resolved_top_k(gen_kwargs, request.top_k, request.model)
            _set_resolved_min_p(gen_kwargs, request.min_p, request.model)
            _rp = _resolve_repetition_penalty(request.repetition_penalty, request.model)
            if _rp is not None:
                gen_kwargs["repetition_penalty"] = _rp
            if _compute_bypass_prefix_cache(request):
                gen_kwargs["_bypass_prefix_cache"] = True
            if request.logprobs is not None:
                gen_kwargs["logprobs"] = True
                gen_kwargs["top_logprobs"] = request.logprobs
            if _is_loaded_dsv4_model(request.model):
                chat_prompt = _dsv4_completion_prompt_for_chat_rail(prompt, request)
                chat_kwargs = {
                    key: value
                    for key, value in gen_kwargs.items()
                    if key != "prompt"
                }
                chat_kwargs["max_tokens"] = _dsv4_completion_internal_max_tokens(
                    chat_kwargs["max_tokens"],
                    request,
                )
                decision = _resolve_dsv4_thinking_policy(
                    requested_enable_thinking=True,
                    effort_requested=False,
                    tools_present=False,
                    tool_choice=None,
                    default_mode=_jang_chat_default_mode(
                        _model_path or _model_name or request.model or ""
                    ),
                )
                chat_kwargs["enable_thinking"] = decision.enable_thinking
                _set_resolved_repetition_penalty(
                    chat_kwargs,
                    request.repetition_penalty,
                    request.model,
                    enable_thinking=decision.enable_thinking,
                )
                output = await asyncio.wait_for(
                    engine.chat(
                        messages=[{"role": "user", "content": chat_prompt}],
                        **chat_kwargs,
                    ),
                    timeout=timeout,
                )
            else:
                output = await asyncio.wait_for(
                    engine.generate(**gen_kwargs),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504, detail=f"Request timed out after {timeout:.1f} seconds"
            )
        except PromptTooLongError as e:
            return _prompt_too_long_response_from_error(e)
        except VLMImagePrefillBudgetError as e:
            return _vlm_image_prefill_budget_response_from_error(e)
        except Exception as e:
            logger.error(f"Completion failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {type(e).__name__}: {e}",
            )

        completion_text = (
            _visible_text_for_dsv4_completion(output, engine, request)
            if _is_loaded_dsv4_model(request.model)
            else output.text
        )

        choices.append(
            CompletionChoice(
                index=i,
                text=completion_text,
                logprobs=_format_completion_logprobs(
                    getattr(output, "logprobs", None),
                    engine.tokenizer,
                )
                if request.logprobs is not None
                else None,
                finish_reason=output.finish_reason,
            )
        )
        total_completion_tokens += output.completion_tokens
        total_prompt_tokens += (
            output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
        )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    return CompletionResponse(
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


@app.post(
    "/v1/chat/completions",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
    response_model_exclude_none=True,
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    fastapi_request: Request,
):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    # Model name validation: accept served name, actual name, or any name (permissive)
    resolved_name = _resolve_model_name()
    if (
        request.model
        and request.model != resolved_name
        and request.model != _model_name
    ):
        # v1.3.67 mlxstudio#79: log INFO once per distinct pair (not
        # DEBUG-after-first). Claude Code sends model="claude-3-5-sonnet"
        # or similar; operators need to see the substitution in logs to
        # diagnose "why did my claude-3-5-sonnet request return MiniMax
        # output". Silently demoting to DEBUG after one warning hid the
        # bug from operators.
        _pair = (str(request.model), str(resolved_name))
        if _pair not in _model_name_mismatch_seen:
            _model_name_mismatch_seen.add(_pair)
            logger.info(
                f"Request model '{request.model}' differs from served model "
                f"'{resolved_name}' — using loaded model (single-model server). "
                f"This is expected when routing tools like Claude Code, opencode, "
                f"or CCSwitch at vMLX — they send their own model alias."
            )
    # Normalize response model field to the resolved name
    request.model = resolved_name
    _log_multimodal_request_shape(
        "/v1/chat/completions",
        _model_path or _model_name or request.model,
        _messages_multimodal_summary(request.messages),
    )

    _chat_max_prompt_tokens = _effective_max_prompt_tokens(request)
    prompt_limit_response = _reject_if_prompt_too_long_for_messages(
        request.messages,
        max_prompt_tokens=_chat_max_prompt_tokens,
    )
    if prompt_limit_response is not None:
        return prompt_limit_response

    if request.logprobs:
        _logprobs_engine = get_engine()
        _reject_unsupported_logprobs_request(
            endpoint="Chat Completions",
            requested=True,
            stream=request.stream,
            is_mllm=bool(getattr(_logprobs_engine, "is_mllm", False)),
        )

    # Nemotron-3-Nano-Omni multimodal dispatch (Stage-1 PyTorch+MLX bridge).
    # If the loaded bundle is a nemotron_h Omni bundle (has config_omni.json
    # + vision_model.* weights) AND the request includes any image/audio/
    # video content parts, route through `OmniMultimodalDispatcher` which
    # wraps `jang_tools.nemotron_omni_session.OmniSession`. Text-only
    # requests fall through to the standard chat path (already verified
    # working at full LLM speed). See research/NEMOTRON-OMNI-MULTIMODAL-2026-04-28.md.
    try:
        from .omni_multimodal import (
            is_omni_multimodal_bundle,
            request_has_multimodal,
            dispatch_omni_chat_completion,
        )

        _omni_path = _model_path or _model_name
        if (
            _omni_path
            and is_omni_multimodal_bundle(_omni_path)
            and request_has_multimodal(
                [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                 for m in (request.messages or [])]
            )
        ):
            return await dispatch_omni_chat_completion(request, _omni_path)
    except HTTPException:
        raise
    except Exception as _omni_route_err:  # pragma: no cover
        logger.warning(
            "Omni multimodal dispatch failed (%s); falling back to standard path",
            _omni_route_err,
        )

    # Reject empty prompts with a clear 400 instead of letting mlx-vlm
    # crash inside stream_generate with ValueError:
    # "[reshape] Cannot infer the shape of an empty array".
    # A message with content="" contributes zero tokens — the model has
    # nothing to decode. Covers str content, content parts list with
    # no non-empty text, and messages arrays that are empty after role
    # filtering. iter 26 added: caught live on Qwen3.6-JANGTQ2.
    def _has_non_empty_content(msg) -> bool:
        c = getattr(msg, "content", None)
        if c is None:
            # assistant with tool_calls only is valid even if content is None
            if getattr(msg, "tool_calls", None):
                return True
            return False
        if isinstance(c, str):
            return c.strip() != ""
        if isinstance(c, list):
            # content parts — at least one text/image/video part with a value
            for p in c:
                t = getattr(p, "type", None) or (p.get("type") if isinstance(p, dict) else None)
                if t == "text":
                    txt = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                    if txt and str(txt).strip():
                        return True
                elif t in _MULTIMODAL_CONTENT_TYPES:
                    return True
            return False
        return bool(c)

    if request.messages:
        _user_msgs = [m for m in request.messages if getattr(m, "role", None) == "user"]
        # Two bugs gated by this check:
        # (iter 26) user message with empty content → 500 reshape crash
        # (iter 27) NO user messages at all (only system) → same 500 crash
        # Both resolve to "the engine will have 0 prompt tokens to work with".
        if not _user_msgs:
            raise HTTPException(
                status_code=400,
                detail="No user message — at least one message with role='user' "
                       "and non-empty content is required. (System-only prompts "
                       "cannot generate output.)"
            )
        if not any(_has_non_empty_content(m) for m in _user_msgs):
            raise HTTPException(
                status_code=400,
                detail="Empty user content — at least one user message must have "
                       "non-empty text, image, video, or audio content."
            )

    # Warn about unsupported penalty parameters
    if request.frequency_penalty and request.frequency_penalty != 0:
        logger.warning(
            "frequency_penalty=%.2f is not implemented and will be ignored",
            request.frequency_penalty,
        )
    if request.presence_penalty and request.presence_penalty != 0:
        logger.warning(
            "presence_penalty=%.2f is not implemented and will be ignored",
            request.presence_penalty,
        )

    engine = get_engine()
    _chat_requested_modalities = _messages_requested_modalities(request.messages)
    if _chat_requested_modalities:
        _reject_unsupported_multimodal(
            "/v1/chat/completions",
            _chat_requested_modalities,
        )

    # For MLLM models, keep original messages with embedded images
    # (MLLM.chat() extracts images from message content internally)
    if engine.is_mllm:
        # Convert Pydantic messages to dicts, excluding None fields.
        # CRITICAL: model_dump() without exclude_none includes ALL optional fields
        # (e.g. image_url=None on text items). The Jinja2 chat template checks
        # 'image_url' in item (key existence), so None-valued fields cause text items
        # to be misdetected as images, producing duplicate <|image_pad|> tokens.
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                msg_dict = dict(msg)
            # Map "developer" role to "system" (OpenAI API compatibility)
            if msg_dict.get("role") == "developer":
                msg_dict["role"] = "system"
            messages.append(msg_dict)
        images, videos = [], []  # MLLM extracts these from messages
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        # For LLM, extract text, images, and videos separately
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if has_media and not engine.is_mllm:
        _reject_unsupported_multimodal("/v1/chat/completions")

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            # Inject JSON instruction into messages
            messages = _inject_json_instruction(messages, json_instruction)
    if engine.is_mllm and _should_coerce_zaya_vl_tool_history(request.model):
        messages = _coerce_zaya_vl_tool_history_for_template(messages)
    messages = _normalize_leading_system_messages(messages)

    # When thinking is explicitly disabled, strip <think> blocks from prior assistant
    # messages in the conversation history. Without this, the model sees prior thinking
    # in context and mimics the pattern, producing reasoning even when the generation
    # prompt doesn't inject <think>. This is the root cause of "thinking OFF but model
    # still thinks on 2nd message" bugs.
    _ct_kwargs = _merge_ct_kwargs(request.chat_template_kwargs)
    _explicit_thinking_off = request.enable_thinking is False or (
        _ct_kwargs.get("enable_thinking") is False
    )
    if _explicit_thinking_off and messages:
        cleaned = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                # Skip list content (assistant messages with images from PR #29)
                if isinstance(content, str):
                    stripped = _THINK_STRIP_RE.sub("", content).strip()
                    if stripped != content:
                        if not stripped and not msg.get("tool_calls"):
                            continue  # Drop assistant messages that were ONLY thinking (no tool calls)
                        # Preserve message if it has tool_calls — orphaned tool results
                        # after dropping the assistant message causes template breakage
                        # and repetition loops (GH mlxstudio #62).
                        msg = {**msg, "content": stripped or None}
            cleaned.append(msg)
        messages = cleaned

    # Prepare kwargs. Preserve explicit zero for validation/error paths:
    # request.max_tokens if request.max_tokens is not None else None
    # feeds `_resolve_max_tokens()` without using a falsy `or` fallback.
    _request_max_tokens = request.max_tokens if request.max_tokens is not None else None
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(_request_max_tokens, request.model),
        "temperature": _resolve_temperature(request.temperature, request.model),
        "top_p": _resolve_top_p(request.top_p, request.model),
        "max_prompt_tokens": _chat_max_prompt_tokens,
    }
    # Forward stop sequences from request to engine
    if request.stop:
        chat_kwargs["stop"] = request.stop
    # Extended sampling params (only pass if explicitly set)
    _set_resolved_top_k(chat_kwargs, request.top_k, request.model)
    _set_resolved_min_p(chat_kwargs, request.min_p, request.model)
    _rp = _resolve_repetition_penalty(request.repetition_penalty, request.model)
    if _rp is not None:
        chat_kwargs["repetition_penalty"] = _rp
    if _compute_bypass_prefix_cache(request):
        chat_kwargs["_bypass_prefix_cache"] = True
    if request.logprobs:
        chat_kwargs["logprobs"] = True
        chat_kwargs["top_logprobs"] = request.top_logprobs or 0

    # Pass enable_thinking to engine
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    _et = _resolve_enable_thinking(
        request_value=request.enable_thinking,
        ct_kwargs=_ct_kwargs,
        tools_present=bool(request.tools),
        model_key=_model_path or _model_name or request.model,
        engine=engine,
        auto_detect=True,
    )
    if _et is not None:
        chat_kwargs["enable_thinking"] = _et
        request.enable_thinking = _et

    # Pass reasoning_effort if provided (for GPT-OSS and models that support thinking levels).
    # Map it to template-side thinking_budget only; output length remains owned
    # by explicit max_tokens/max_output_tokens, then bundle max_new_tokens.
    if request.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = request.reasoning_effort
        # Also inject into chat_template_kwargs so Jinja templates can access it
        # (e.g., Mistral 4's [MODEL_SETTINGS]{"reasoning_effort":"high"} block).
        _ct_kwargs.setdefault("reasoning_effort", request.reasoning_effort)
        _effort_lower = request.reasoning_effort.lower()
        _budget = _EFFORT_THINKING_BUDGET.get(_effort_lower)
        if _budget:
            _ct_kwargs.setdefault("thinking_budget", _budget)
    if getattr(request, "max_thinking_tokens", None) is not None and request.enable_thinking is not False:
        _ct_kwargs["thinking_budget"] = int(request.max_thinking_tokens)

    # Auto-map enable_thinking → reasoning_effort for Mistral 4.
    # Mistral 4's template uses reasoning_effort ("none"/"high"), not enable_thinking.
    # When the user toggles thinking ON in the UI, set reasoning_effort="high" if not already set.
    # Original Mistral 4 reasoning integration by Jinho Jang (eric@jangq.ai) — vMLX.
    # This approach (auto-mapping enable_thinking to reasoning_effort for models that
    # use [MODEL_SETTINGS] blocks) was developed through extensive trial and error.
    # If you adapt this pattern, please credit the original author.
    if (
        _reasoning_parser
        and type(_reasoning_parser).__name__ == "MistralReasoningParser"
    ):
        if request.enable_thinking is True and "reasoning_effort" not in _ct_kwargs:
            _ct_kwargs["reasoning_effort"] = "high"
            chat_kwargs["reasoning_effort"] = "high"
        elif request.enable_thinking is False and "reasoning_effort" not in _ct_kwargs:
            _ct_kwargs["reasoning_effort"] = "none"

    _apply_hy3_reasoning_policy(
        chat_kwargs,
        _ct_kwargs,
        model_key=_model_path or _model_name or request.model,
        enable_thinking=request.enable_thinking,
    )

    # DeepSeek V4 three-mode encoder: thinking_mode=chat|thinking + optional
    # reasoning_effort=high|max. Auto-map enable_thinking toggle to the right
    # pair so the UI's simple on/off switch lands on the correct prompt
    # suffix (</think> for chat mode, no suffix for thinking, extra system
    # hint for max). See research/DSV4-RUNTIME-ARCHITECTURE.md §4 and
    # vmlx_engine/loaders/dsv4_chat_encoder.py::_resolve_mode_and_effort.
    try:
        from .model_config_registry import get_model_config_registry
        _model_family_key = _model_path or _model_name or getattr(request, "model", "") or ""
        _mc_for_dsv4 = get_model_config_registry().lookup(_model_family_key)
        _is_dsv4 = getattr(_mc_for_dsv4, "family_name", "") == "deepseek_v4"
    except Exception:
        _is_dsv4 = False
    if _is_dsv4:
        # The DSV4 Jinja template (baked into tokenizer_config.json) accepts
        # exactly two kwargs:
        #     enable_thinking: bool
        #     reasoning_effort: 'max' or None
        # The template branches ONLY on `reasoning_effort == 'max'`. Any
        # other string value (including 'high', 'medium', 'low') silently
        # falls through to the default branch — i.e. behaves identical to
        # `reasoning_effort=None`. So:
        #
        #   chat mode             → enable_thinking=False, reasoning_effort=None
        #   thinking (standard)   → enable_thinking=True,  reasoning_effort=None
        #   thinking (max)        → enable_thinking=True,  reasoning_effort='max'
        #
        # Setting `thinking_mode` here is a no-op for the template (it's only
        # used by the legacy `dsv4_chat_encoder.py` Python adapter, not the
        # Jinja template applied via tokenizer.apply_chat_template). We keep
        # it off the kwargs to avoid Jinja "undefined kwarg" warnings on
        # strict templates.
        _cur_effort = (
            request.reasoning_effort
            or _ct_kwargs.get("reasoning_effort")
        )
        # Strip stale fields from any prior path so the final kwargs are clean.
        _ct_kwargs.pop("thinking_mode", None)

        _dsv4_thinking = _resolve_dsv4_thinking_policy(
            requested_enable_thinking=request.enable_thinking,
            effort_requested=bool(_cur_effort),
            tools_present=bool(getattr(request, "tools", None)),
            tool_choice=getattr(request, "tool_choice", None),
            default_mode=_jang_chat_default_mode(
                _model_path or _model_name or request.model or ""
            ),
        )
        if (
            not _dsv4_thinking.enable_thinking
            and (request.enable_thinking is True or _cur_effort)
        ):
            logger.info(
                "DSV4: using direct rail because %s.",
                _dsv4_thinking.reason,
            )
        request.enable_thinking = _dsv4_thinking.enable_thinking
        chat_kwargs["enable_thinking"] = _dsv4_thinking.enable_thinking
        _ct_kwargs["enable_thinking"] = _dsv4_thinking.enable_thinking
        _set_resolved_repetition_penalty(
            chat_kwargs,
            request.repetition_penalty,
            request.model,
            enable_thinking=chat_kwargs.get("enable_thinking"),
        )
        _stable_effort = (
            _normalize_dsv4_reasoning_effort(_cur_effort)
            if _dsv4_thinking.reasoning_effort_allowed
            else None
        )
        if _stable_effort:
            _ct_kwargs["reasoning_effort"] = _stable_effort
        else:
            _ct_kwargs.pop("reasoning_effort", None)

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            chat_kwargs["chat_template_kwargs"] = extra_ct

    _log_resolved_sampling_kwargs(
        "/v1/chat/completions",
        _model_path or _model_name or request.model,
        chat_kwargs,
    )

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
    # Video controls — passed regardless of has_media since MLLM models
    # extract media from messages internally (has_media is always False for them)
    if request.video_fps:
        chat_kwargs["video_fps"] = request.video_fps
    if request.video_max_frames:
        chat_kwargs["video_max_frames"] = request.video_max_frames

    # Handle tool_choice: "none" suppresses tools entirely, "auto"/None is default,
    # "required" or specific tool dict are handled post-generation.
    _tool_choice = request.tool_choice
    _suppress_tools = _tool_choice == "none"

    # response_format = json_object / json_schema implies the output IS the
    # structured data — not a tool call. The generic tool-call parser runs
    # on any JSON-ish output and is happy to interpret `{"name":"alice",
    # "age":30}` as a tool call `alice()`. Suppress the tool parser when
    # the client asked for JSON output and didn't pass tools themselves.
    # If both tools AND response_format are present, tool parsing still
    # runs (structured tool-call arguments are allowed).
    if not request.tools and not _suppress_tools:
        _rf = request.response_format
        _rf_type = None
        if isinstance(_rf, dict):
            _rf_type = _rf.get("type")
        elif _rf is not None and hasattr(_rf, "type"):
            _rf_type = getattr(_rf, "type", None)
        if _rf_type in ("json_object", "json_schema"):
            _suppress_tools = True

    # Merge MCP tools with user-provided tools
    all_tools = []

    if not _suppress_tools:
        # Add MCP tools if available
        if _mcp_manager is not None:
            mcp_tools = _mcp_manager.get_all_tools_openai(policy=_mcp_policy)
            mcp_tools = _drop_colliding_mcp_tools(mcp_tools, request.tools)
            mcp_tools = _filter_tools_for_specific_choice(mcp_tools, _tool_choice)
            all_tools.extend(mcp_tools)
            if mcp_tools:
                logger.debug(f"Added {len(mcp_tools)} MCP tools")

        # Add user-provided tools
        if request.tools:
            # If tool_choice is a specific tool dict, filter to only that tool
            if isinstance(_tool_choice, dict):
                target_name = _tool_choice.get("function", {}).get(
                    "name"
                ) or _tool_choice.get("name")
                if target_name:
                    # request.tools are ToolDefinition Pydantic models (type + function dict)
                    filtered = [
                        t
                        for t in request.tools
                        if t.function.get("name") == target_name
                    ]
                    all_tools.extend(filtered if filtered else request.tools)
                else:
                    all_tools.extend(request.tools)
            else:
                all_tools.extend(request.tools)
            logger.debug(f"Added {len(all_tools)} tools (tool_choice={_tool_choice})")

    if not _suppress_tools:
        _suppress_tools = _suppress_tool_parsing_when_no_tools(
            all_tools, _tool_choice, "Chat Completions"
        )
    _attach_effective_tools_for_tool_parsing(request, all_tools)

    # Pass merged tools to engine (normalize all to template format)
    if all_tools:
        chat_kwargs["tools"] = convert_tools_for_template(all_tools)

    # Inject Harmony analysis prefix for GPT-OSS models when thinking is enabled.
    # The suffix replaces the template's generation prompt (<|start|>assistant<|message|>)
    # with the analysis channel prefix to guide the model into reasoning mode.
    if isinstance(_reasoning_parser, GptOssReasoningParser):
        _think_val = chat_kwargs.get("enable_thinking")
        if _think_val is True:
            _analysis_hint = "<|start|>assistant<|channel|>analysis<|message|>"
            chat_kwargs["prompt_suffix"] = _analysis_hint
            # Skip template's generation prompt — suffix provides the full assistant prefix
            chat_kwargs["skip_generation_prompt"] = True
            _effort = chat_kwargs.get("reasoning_effort", "").lower()
            logger.info(
                f"[chat] Injecting Harmony analysis prefix for GPT-OSS model (effort={_effort or 'default'})"
            )
        # GPT-OSS/GLM Flash models sometimes generate <|im_end|> (ChatML format)
        # which is NOT a single token in the Harmony vocabulary — it shatters into
        # sub-tokens and leaks into output. Add it as a string stop sequence.
        _stop = chat_kwargs.get("stop") or []
        if "<|im_end|>" not in _stop:
            _stop = list(_stop) + ["<|im_end|>"]
            chat_kwargs["stop"] = _stop

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(
                engine,
                messages,
                request,
                fastapi_request=fastapi_request,
                **chat_kwargs,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout if request.timeout is not None else _default_timeout
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    try:
        output = await _await_chat_with_disconnect_abort(
            engine,
            messages=messages,
            chat_kwargs=chat_kwargs,
            timeout=timeout,
            fastapi_request=fastapi_request,
            request_id=response_id,
            endpoint="Chat Completions",
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"Request timed out after {timeout:.1f} seconds"
        )
    except PromptTooLongError as e:
        return _prompt_too_long_response_from_error(e)
    except VLMImagePrefillBudgetError as e:
        return _vlm_image_prefill_budget_response_from_error(e)
    except ValueError as e:
        conv1d_detail = _grouped_conv1d_layout_error_detail(e)
        if conv1d_detail:
            logger.error("Chat completion failed with grouped Conv1d layout error: %s", e)
            raise HTTPException(status_code=503, detail=conv1d_detail)
        logger.warning(f"Chat completion rejected: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_generation_error_detail(e),
        )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Extract reasoning content FIRST from raw output (before stripping tags).
    # Must happen before tool call parsing since that strips <think> blocks.
    reasoning_text = None
    content_for_parsing = output.text
    if _reasoning_parser:
        # Clone parser per-request to avoid shared state across concurrent requests
        request_parser = _reasoning_parser.__class__()
        # Initialize parser state (harmony_active, etc.) — same as streaming path
        _harmony_prefix_active = "prompt_suffix" in chat_kwargs and chat_kwargs.get(
            "prompt_suffix", ""
        ).startswith("<|start|>assistant<|channel|>analysis")

        # Compute think_in_prompt the same way the streaming path does at
        # ~line 5840. MiniMax and other "always-thinks" templates inject
        # <think>\n in the generation prompt, so model output starts in
        # reasoning mode. Without think_in_prompt=True, `extract_reasoning`
        # falls through to "no tags → pure content" and the reasoning prose
        # shows up raw in content (§15 regression).
        try:
            from .model_config_registry import get_model_config_registry as _mcr
            _mc_nonstream = _mcr().lookup(_model_path or _model_name or request.model)
            _think_in_prompt_ns = _mc_nonstream.think_in_template
            _think_in_prompt_ns = _apply_tokenizer_thinking_vocab_fallback(
                _think_in_prompt_ns,
                reasoning_parser=_reasoning_parser,
                tokenizer=engine.tokenizer,
                model_config=_mc_nonstream,
            )
            if _think_in_prompt_ns and _template_completes_thinking(
                engine.tokenizer, _model_name or request.model
            ):
                _think_in_prompt_ns = False
            _eff_thinking_ns = chat_kwargs.get("enable_thinking")
            _prompt_enable_ns = (
                True if _eff_thinking_ns is None else bool(_eff_thinking_ns)
            )
            _tools_present_ns = bool(getattr(request, "tools", None) or chat_kwargs.get("tools"))
            if _engine_prompt_starts_in_reasoning(
                engine.tokenizer,
                model_name=_model_name or request.model,
                enable_thinking=_prompt_enable_ns,
                family_name=getattr(_mc_nonstream, "family_name", None),
                tools_present=_tools_present_ns,
            ):
                _think_in_prompt_ns = True
            else:
                _think_in_prompt_ns = False
            if _hy3_prompt_starts_in_reasoning(
                model_key=_model_path or _model_name or request.model,
                enable_thinking=_eff_thinking_ns,
                ct_kwargs=chat_kwargs.get("chat_template_kwargs") or _ct_kwargs,
            ):
                _think_in_prompt_ns = True
            if _synthetic_mllm_think_prompt_starts(
                model_key=_model_path or _model_name or request.model,
                enable_thinking=_eff_thinking_ns,
                engine=engine,
            ):
                _think_in_prompt_ns = True
        except Exception as _tpe:
            logger.debug(f"think_in_prompt derivation failed non-stream: {_tpe}")
            _think_in_prompt_ns = False

        request_parser.reset_state(
            think_in_prompt=_think_in_prompt_ns,
            harmony_active=_harmony_prefix_active,
        )
        # Prefer raw_text (pre-clean) when the engine tracked it — reasoning
        # parsers (esp. Gemma 4 channel markers, Qwen 3.6 think tags) need the
        # special tokens that clean_output_text strips for display.
        _raw_for_parse = getattr(output, "raw_text", "") or output.text
        reasoning_text, remaining_text = request_parser.extract_reasoning(_raw_for_parse)
        if _is_dsv4:
            token_reasoning, token_content = _dsv4_split_reasoning_from_token_ids(
                _output_token_ids_for_reasoning(output),
                engine.tokenizer,
            )
            if token_content:
                reasoning_text = token_reasoning
                remaining_text = token_content
        if remaining_text is not None:
            content_for_parsing = remaining_text
        elif reasoning_text is not None:
            # Truncated reasoning (e.g., max_tokens hit during <think> phase):
            # reasoning extracted but no content after it — don't leak raw <think> tags
            content_for_parsing = ""
        # Suppress reasoning when thinking is disabled (check resolved value from chat_kwargs,
        # not just raw request — covers --default-enable-thinking and auto-detect paths)
        _resolved = chat_kwargs.get("enable_thinking")
        _suppress = _resolved is False or request.enable_thinking is False
        if not _suppress:
            _suppress = _ct_kwargs.get("enable_thinking") is False
        if _suppress:
            # Recover truncated-think text into content when thinking is off.
            # Without this, a model that ignores enable_thinking=False and emits
            # <think>... but hits max_tokens before </think> produces a response
            # with content=null and reasoning_content=null — the tokens are lost.
            if not content_for_parsing and reasoning_text:
                content_for_parsing = reasoning_text
            elif content_for_parsing:
                content_for_parsing = _strip_residual_think_markup_for_display(
                    content_for_parsing
                )
            reasoning_text = None

        # Post-parse cleaning: the parser operates on raw_text which carries
        # special markers (Gemma 4 `<|channel>`/`<channel|>`/`<turn|>`, Qwen
        # `<|im_end|>`, etc.). Markers that survive the parser's split (e.g. a
        # second `<channel|>` embedded in content, or `thought\n` degraded
        # form) must not leak into the user-visible content field. Apply the
        # same cleaner that the engine applies to `output.text` for display,
        # but AFTER the parser has done its structural extraction.
        if content_for_parsing:
            content_for_parsing = clean_output_text(content_for_parsing)
        if reasoning_text:
            reasoning_text = clean_output_text(reasoning_text)

    _cc_parse_text = _strip_think_for_tool_parse(content_for_parsing)

    # Parse tool calls from output using configured parser (skip when tool_choice="none")
    cleaned_text, tool_calls = (
        _parse_tool_calls_with_parser(_cc_parse_text, request)
        if not _suppress_tools
        else (
            _clean_suppressed_tool_markup_for_display(
                _cc_parse_text or content_for_parsing or "",
                request,
            ),
            None,
        )
    )

    # Process response_format if specified
    if response_format and not tool_calls:
        cleaned_text, parsed_json, is_valid, error = parse_json_output(
            cleaned_text or content_for_parsing, response_format
        )
        if parsed_json is not None:
            # Return JSON as string
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            # Check if strict mode is enabled — return 400 error instead of a warning
            _rf_strict = False
            if isinstance(response_format, dict):
                _rf_strict = response_format.get("json_schema", {}).get("strict", False)
            elif (
                hasattr(response_format, "json_schema") and response_format.json_schema
            ):
                _rf_strict = getattr(response_format.json_schema, "strict", False)
            if _rf_strict:
                raise HTTPException(
                    status_code=400, detail=f"response_format strict mode: {error}"
                )
            logger.warning(f"JSON validation failed: {error}")

    # Enforce tool_choice="required": model MUST produce at least one tool call
    if _tool_choice == "required" and not tool_calls:
        logger.warning(
            f"tool_choice='required' but model produced no tool calls. "
            f"Returning error to client."
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "tool_choice='required' was set but the model did not produce "
                "any tool calls. Try rephrasing your prompt or using a model "
                "with better tool-calling support."
            ),
        )

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason
    response_content = clean_output_text(cleaned_text) if cleaned_text else None
    if response_content:
        response_content = _finalize_visible_text_for_request(response_content, request)
    response_warnings = _chat_completion_warnings_for_reasoning_only(
        response_content,
        reasoning_text,
        tool_calls,
    )

    response = ChatCompletionResponse(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=response_content,
                    reasoning=reasoning_text,
                    tool_calls=tool_calls,
                ),
                logprobs=_format_chat_logprobs(
                    getattr(output, "logprobs", None),
                    engine.tokenizer,
                )
                if request.logprobs
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=get_usage(output),
        warnings=response_warnings,
    )

    # When tool_calls are present, serialize manually to ensure content:null is
    # included (response_model_exclude_none=True would drop it, but OpenAI
    # always includes content:null with tool_calls — required by many clients).
    if tool_calls:
        from starlette.responses import JSONResponse

        resp_dict = response.model_dump(exclude_none=True)
        # Force content:null back into the message
        for choice in resp_dict.get("choices", []):
            msg = choice.get("message", {})
            if "tool_calls" in msg and "content" not in msg:
                msg["content"] = None
        return JSONResponse(content=resp_dict)

    return response


# =============================================================================
# Responses API (OpenAI /v1/responses)
# =============================================================================


_RESPONSES_HISTORY_MAX = int(os.environ.get("VMLX_RESPONSES_HISTORY_MAX", "512"))
_responses_history: "OrderedDict[str, list[dict]]" = OrderedDict()
_responses_history_lock = threading.Lock()
# Marker set: response_ids whose stored output was reasoning-only (no visible
# text, no tool calls). Used to surface a non-fatal `warnings` field on chained
# responses. Evicted in lockstep with `_responses_history` capacity enforcement.
_responses_was_reasoning_only: set[str] = set()


def _clone_response_messages(messages: list[dict]) -> list[dict]:
    """JSON-deep-copy response history so later request mutation cannot leak."""
    try:
        return json.loads(json.dumps(messages))
    except Exception:
        return [dict(m) for m in messages if isinstance(m, dict)]


_RESPONSES_VISIBLE_PART_TYPES = {"output_text", "text"}
_RESPONSES_REASONING_ITEM_TYPES = {"reasoning", "reasoning_summary"}
_RESPONSES_REASONING_PART_TYPES = {
    "reasoning",
    "reasoning_text",
    "reasoning_summary_text",
    "summary_text",
}


def _responses_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _responses_part_text(part) -> str:
    text = _responses_field(part, "text")
    if text is None:
        text = _responses_field(part, "content")
    return text if isinstance(text, str) else ""


def _responses_output_has_reasoning(output_items: list) -> bool:
    for item in output_items or []:
        item_type = _responses_field(item, "type")
        if item_type in _RESPONSES_REASONING_ITEM_TYPES:
            return True
        for part in _responses_field(item, "content", []) or []:
            if _responses_field(part, "type") in _RESPONSES_REASONING_PART_TYPES:
                return True
    return False


def _responses_output_has_visible_or_tool(output_items: list) -> bool:
    for item in output_items or []:
        item_type = _responses_field(item, "type")
        if item_type == "function_call":
            return True
        if item_type in _RESPONSES_VISIBLE_PART_TYPES and _responses_part_text(item):
            return True
        for part in _responses_field(item, "content", []) or []:
            if (
                _responses_field(part, "type") in _RESPONSES_VISIBLE_PART_TYPES
                and _responses_part_text(part)
            ):
                return True
    return False


def _responses_output_is_reasoning_only(output_items: list) -> bool:
    return _responses_output_has_reasoning(
        output_items
    ) and not _responses_output_has_visible_or_tool(output_items)


def _responses_output_to_assistant_messages(output_items: list) -> list[dict]:
    """Convert a Responses output array back to chat-history assistant turns."""
    assistant_messages: list[dict] = []
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    saw_reasoning = False

    for item in output_items or []:
        item_type = _responses_field(item, "type")
        if item_type in _RESPONSES_REASONING_ITEM_TYPES:
            # Reasoning is display metadata, not replayable chat content.
            saw_reasoning = True
            continue
        if item_type == "function_call":
            name = _responses_field(item, "name", "")
            arguments = _responses_field(item, "arguments", "{}")
            call_id = _responses_field(item, "call_id") or f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue
        content = _responses_field(item, "content") or []
        for part in content:
            ptype = _responses_field(part, "type")
            text = _responses_part_text(part)
            if ptype in _RESPONSES_VISIBLE_PART_TYPES and text:
                content_parts.append(text)
            elif ptype in _RESPONSES_REASONING_PART_TYPES:
                saw_reasoning = True
    if tool_calls:
        # A Responses turn can contain visible assistant text and function_call
        # items. Store them on the same assistant turn so the next
        # function_call_output remains adjacent to assistant.tool_calls for
        # strict native chat templates.
        assistant_messages.append(
            {
                "role": "assistant",
                "content": "\n".join(content_parts) if content_parts else "",
                "tool_calls": tool_calls,
            }
        )
    elif content_parts:
        assistant_messages.append(
            {"role": "assistant", "content": "\n".join(content_parts)}
        )
    elif saw_reasoning and not assistant_messages:
        # Reasoning-only output (no visible text, no tool calls): inject an
        # empty-content assistant placeholder so `previous_response_id` chains
        # preserve the user→assistant→user template anchor and cache prefix
        # alignment. Without this, chained turns lose all record that turn N
        # happened and cache reuse drops because the next turn's prefill prompt
        # has no overlap with prior assistant tokens.
        assistant_messages.append({"role": "assistant", "content": ""})
    return assistant_messages


def _responses_store_history(
    response_id: str,
    messages: list[dict],
    *,
    reasoning_only: bool = False,
) -> None:
    """Persist a Responses history slot keyed by response_id.

    Args:
        response_id: identifier returned in `ResponsesObject.id`.
        messages: the chat-history messages to store (input + replayed
            assistant turns from `_responses_output_to_assistant_messages`).
        reasoning_only: True when the response produced ONLY reasoning items
            (no visible text, no tool calls). Tracked in
            `_responses_was_reasoning_only` so chained turns can surface a
            warning to the client.
    """
    if not response_id:
        return
    with _responses_history_lock:
        _responses_history[response_id] = _clone_response_messages(messages)
        _responses_history.move_to_end(response_id)
        if reasoning_only:
            _responses_was_reasoning_only.add(response_id)
        else:
            _responses_was_reasoning_only.discard(response_id)
        # Capacity enforcement — also evict the matching reasoning-only
        # marker so the marker set never references a freed slot (would
        # otherwise give stale-warning false positives if a future
        # response_id collides via uuid prefix).
        while len(_responses_history) > _RESPONSES_HISTORY_MAX:
            evicted_id, _ = _responses_history.popitem(last=False)
            _responses_was_reasoning_only.discard(evicted_id)


def _chain_warnings_for_previous_response_id(
    previous_response_id: str | None,
) -> list[str] | None:
    """Return chain-warnings list when `previous_response_id` references a
    reasoning-only predecessor, else None.

    Used when constructing a new ResponsesObject for a chained turn so the
    client can surface coherence/cache-prefix risk without breaking the API
    contract. Always returns None for missing/unknown ids.
    """
    if not previous_response_id:
        return None
    with _responses_history_lock:
        if previous_response_id not in _responses_history:
            # The normal store path evicts marker+history in lockstep. Keep
            # the read side defensive too, so tests/admin cleanup paths that
            # clear history directly cannot leave a stale warning behind.
            _responses_was_reasoning_only.discard(previous_response_id)
            return None
        if previous_response_id not in _responses_was_reasoning_only:
            return None
    return [
        "previous_response_id chained a response that produced reasoning only "
        "(no visible message, no tool calls). Chat continuity may be impaired "
        "and prefix-cache reuse may be lower than expected. Consider raising "
        "max_output_tokens or sending enable_thinking=false on the prior turn."
    ]


def _current_response_warnings_for_reasoning_only(
    reasoning_only: bool,
) -> list[str] | None:
    if not reasoning_only:
        return None
    return [
        "This response produced reasoning only (no visible message, no tool "
        "calls). The reasoning was preserved separately, but the visible answer "
        "is empty. Consider raising max_output_tokens or sending "
        "enable_thinking=false for the final synthesis turn."
    ]


def _chat_completion_warnings_for_reasoning_only(
    content: str | None,
    reasoning: str | None,
    tool_calls: list | None,
) -> list[str] | None:
    has_visible_content = bool((content or "").strip())
    has_reasoning = bool((reasoning or "").strip())
    has_tool_calls = bool(tool_calls)
    return _current_response_warnings_for_reasoning_only(
        has_reasoning and not has_visible_content and not has_tool_calls
    )


def _merge_responses_warnings(*warning_lists: list[str] | None) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for warnings in warning_lists:
        for warning in warnings or []:
            if not isinstance(warning, str):
                continue
            text = warning.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged or None


def _responses_get_history(response_id: str | None) -> list[dict]:
    if not response_id:
        return []
    with _responses_history_lock:
        history = _responses_history.get(response_id)
        if history is None:
            return []
        _responses_history.move_to_end(response_id)
        return _clone_response_messages(history)


def _extract_text_from_content(content) -> str:
    """Extract plain text from a content field that may be a string or list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Content parts: [{"type": "input_text", "text": "..."}, ...]
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", part.get("content", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content) if content else ""


def _normalize_leading_system_messages(messages: list[dict]) -> list[dict]:
    """Merge all system/developer messages into one leading system message.

    Some tokenizer templates, including Qwen-family templates, reject a system
    message anywhere after the first user/assistant/tool turn. Responses
    follow-ups can create that shape when previous_response_id history is
    prepended before a new request with instructions. Treat system/developer
    content as global instructions and keep non-system turns in order.
    """
    if not messages:
        return messages

    system_template: dict | None = None
    system_parts: list[str] = []
    seen_system_parts: set[str] = set()
    non_system: list[dict] = []

    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role in ("system", "developer"):
            if system_template is None:
                system_template = dict(msg) if isinstance(msg, dict) else {
                    "role": "system",
                    "content": getattr(msg, "content", ""),
                }
            text = _extract_text_from_content(
                msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            ).strip()
            if text and text not in seen_system_parts:
                system_parts.append(text)
                seen_system_parts.add(text)
            continue
        non_system.append(msg)

    if system_template is None:
        return list(messages)

    merged_system = dict(system_template)
    merged_system["role"] = "system"
    merged_system["content"] = "\n\n".join(system_parts)
    return [merged_system] + non_system


def _responses_input_to_messages(
    input_data: str | list,
    instructions: str | None = None,
    preserve_multimodal: bool = False,
) -> list[dict]:
    """Convert Responses API input to chat messages format.

    Handles all Responses API input item types:
    - {"role": "user/assistant/system", "content": "..."} → standard messages
    - {"type": "message", "role": "...", "content": [...]} → message with content parts
    - {"type": "function_call", "name": "...", "call_id": "...", "arguments": "..."} → tool call
    - {"type": "function_call_output", "call_id": "...", "output": "..."} → tool result

    Args:
        preserve_multimodal: When True (MLLM models), preserve content arrays with
            image_url/video_url parts instead of extracting text only. MLLM engines
            extract images from message content internally.
    """

    def _resolve_content(raw_content):
        """Resolve content: preserve arrays for MLLM, extract text for LLM."""
        if preserve_multimodal and isinstance(raw_content, list):
            # Convert Pydantic content parts to clean dicts (exclude None fields)
            # to avoid template issues where 'image_url' in item returns True
            # for text items that have image_url=None as a model field.
            clean_parts = []
            for p in raw_content:
                if hasattr(p, "model_dump"):
                    clean_parts.append(p.model_dump(exclude_none=True))
                elif isinstance(p, dict):
                    clean_parts.append({k: v for k, v in p.items() if v is not None})
                else:
                    clean_parts.append(p)
            # OpenAI Responses API uses `input_image` / `input_video` /
            # `input_audio` as the canonical content-part types (with
            # `image_url` / `audio` etc. as sub-fields), distinct from
            # the chat-completions API's `image_url` / `video_url` types.
            # Both shapes must be detected as media-bearing AND normalized
            # so downstream multimodal extraction (`extract_media_from_messages`,
            # OmniMultimodalDispatcher) sees a uniform shape.
            # Live test 2026-04-30 with /v1/responses + input_image → empty
            # response (model never saw the image) was traced to this gap.
            _MEDIA_TYPES = {
                "image_url", "image", "video_url", "video", "input_audio",
                "input_image", "input_video",
            }
            has_media = any(
                isinstance(p, dict) and p.get("type") in _MEDIA_TYPES
                for p in clean_parts
            )
            if has_media:
                # Normalize Responses-API shapes → chat-completions shapes
                # so downstream code paths (which expect image_url +
                # video_url + input_audio) get a consistent envelope.
                normalized = []
                for p in clean_parts:
                    if not isinstance(p, dict):
                        normalized.append(p); continue
                    t = p.get("type")
                    if t == "input_image":
                        # `input_image: {image_url: "data:image/...;base64,..."}`
                        # OR `input_image: {file_id: "..."}` (not supported,
                        # left as-is for clear downstream error).
                        url = p.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if url:
                            normalized.append({"type": "image_url",
                                               "image_url": {"url": url}})
                        else:
                            normalized.append(p)
                    elif t == "input_video":
                        url = p.get("video_url") or p.get("file_id")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if url:
                            normalized.append({"type": "video_url",
                                               "video_url": {"url": url}})
                        else:
                            normalized.append(p)
                    elif t == "input_text":
                        # OpenAI Responses uses input_text; chat completions
                        # uses just `text`. Re-tag for consistency.
                        normalized.append({"type": "text",
                                           "text": p.get("text", "")})
                    else:
                        normalized.append(p)
                return normalized
        return _extract_text_from_content(raw_content)

    messages = []

    # Add system/instructions message
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Handle string input (simple prompt)
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    # Handle list of input items
    # Each function_call becomes its own assistant message with a single tool_call
    # (many models only support single tool-calls per message)
    # function_call_output becomes a tool message

    def _normalize_role(role: str) -> str:
        """Map 'developer' role to 'system' (OpenAI API compatibility)."""
        return "system" if role == "developer" else role

    for item in input_data:
        if not isinstance(item, dict):
            if hasattr(item, "role"):
                role = _normalize_role(item.role)
                raw = item.content if hasattr(item, "content") else ""
                msg: dict = {"role": role, "content": _resolve_content(raw)}
                if hasattr(item, "tool_calls") and item.tool_calls:
                    msg["tool_calls"] = (
                        item.tool_calls
                        if isinstance(item.tool_calls, list)
                        else [item.tool_calls]
                    )
                    if msg.get("content") is None:
                        msg["content"] = ""
                if hasattr(item, "tool_call_id") and item.tool_call_id:
                    msg["tool_call_id"] = item.tool_call_id
                messages.append(msg)
            continue

        item_type = item.get("type", "")

        # function_call → assistant message with single tool_call
        if item_type == "function_call":
            call_id = item.get("call_id", f"call_{uuid.uuid4().hex[:8]}")
            # Parse arguments to dict — chat templates (Qwen3, Llama, etc.)
            # call .items() on arguments, so they must be a mapping, not a string
            args_raw = item.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args_parsed = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    args_parsed = {}
            else:
                args_parsed = args_raw if args_raw else {}
            messages.append(
                {
                    "role": "assistant",
                    # Tool-call-only assistant turns are valid Responses/OpenAI
                    # history anchors. Use an empty string instead of None:
                    # strict templates such as Mistral 4 accept the tool_calls
                    # but later evaluate `message['content'] | length`, which
                    # crashes on None.
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": args_parsed,
                            },
                        }
                    ],
                }
            )
            continue

        # function_call_output → tool message with result
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                }
            )
            continue

        # output_text is how the panel preserves assistant text between
        # Responses API follow-up calls (including tool auto-continue). Treat
        # it as an assistant message when converting back to Chat Completions
        # history; skipping it makes continuation forget the model's previous
        # answer/context after a tool round.
        if item_type == "output_text":
            text = item.get("text", "")
            if text:
                messages.append({"role": "assistant", "content": text})
            continue

        # message type with content parts
        if item_type == "message":
            role = _normalize_role(item.get("role", "user"))
            content = _resolve_content(item.get("content", ""))
            messages.append({"role": role, "content": content})
        # Standard role-based message (no type field, or type is not a special one)
        elif "role" in item:
            role = _normalize_role(item.get("role", "user"))
            # Preserve tool messages with tool_call_id (Chat Completions format)
            if role == "tool" and "tool_call_id" in item:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["tool_call_id"],
                        "content": _extract_text_from_content(item.get("content", "")),
                    }
                )
            # Preserve assistant messages with tool_calls (Chat Completions format)
            elif role == "assistant" and "tool_calls" in item:
                _assistant_content = item.get("content")
                if _assistant_content is None:
                    _assistant_content = ""
                messages.append(
                    {
                        "role": "assistant",
                        "content": _assistant_content,
                        "tool_calls": item["tool_calls"],
                    }
                )
            else:
                content = _resolve_content(item.get("content", ""))
                messages.append({"role": role, "content": content})
        # Skip unknown item types (e.g. reasoning, web_search_call, etc.)

    # Responses clients may send a previous response as function_call /
    # function_call_output items without also replaying the original user
    # prompt that caused the call. Native chat templates still need a valid
    # user→assistant ordering anchor before assistant tool_calls. DSV4 is
    # especially sensitive here: a leading assistant DSML block after system
    # prompt can produce only "<think></think>" on continuation. Insert a
    # neutral user anchor before a leading assistant tool-call turn; models
    # still see the actual tool result and the real follow-up user message.
    first_non_system = next(
        (
            idx
            for idx, msg in enumerate(messages)
            if msg.get("role") not in ("system", "developer")
        ),
        None,
    )
    if first_non_system is not None:
        first_msg = messages[first_non_system]
        if first_msg.get("role") == "assistant" and first_msg.get("tool_calls"):
            messages.insert(
                first_non_system,
                {
                    "role": "user",
                    "content": "Previous assistant tool-call history follows.",
                },
            )

    return messages


def _zaya_vl_text_part(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _render_zaya_tool_call_history(tool_calls: list) -> str:
    blocks: list[str] = []
    for tc in tool_calls or []:
        if hasattr(tc, "model_dump"):
            tc = tc.model_dump()
        elif hasattr(tc, "dict"):
            tc = tc.dict()
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or ""
        if not name:
            continue
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        lines = ["<zyphra_tool_call>", f"<function={name}>"]
        if isinstance(args, dict):
            for key, value in args.items():
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                lines.extend([f"<parameter={key}>", value, "</parameter>"])
        lines.extend(["</function>", "</zyphra_tool_call>"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _coerce_zaya_vl_tool_history_for_template(messages: list[dict]) -> list[dict]:
    """Render ZAYA-VL tool history as text parts instead of role=tool turns.

    ZAYA-VL's template is multimodal/list-content aware, but it does not have a
    native role="tool" contract. Preserving OpenAI tool messages renders
    ``<|im_start|>tool`` and live runs showed the model continuing in visual
    grounding tokens (``<|point_start|>tool``). Keep zaya_xml output parsing
    native, but serialize prior calls/results as ordinary text parts.
    """
    coerced: list[dict] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            coerced.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            rendered = _render_zaya_tool_call_history(msg.get("tool_calls") or [])
            if rendered:
                coerced.append({**msg, "content": _zaya_vl_text_part(rendered)})
            else:
                coerced.append(msg)
            continue
        if role == "tool":
            content = _extract_text_from_content(msg.get("content", ""))
            coerced.append(
                {
                    "role": "user",
                    "content": _zaya_vl_text_part(
                        "<zyphra_tool_response>"
                        + content
                        + "</zyphra_tool_response>"
                    ),
                }
            )
            continue
        if role in {"user", "assistant", "system"} and isinstance(msg.get("content"), str):
            coerced.append({**msg, "content": _zaya_vl_text_part(msg.get("content", ""))})
            continue
        coerced.append(msg)
    return coerced


def _should_coerce_zaya_vl_tool_history(model_hint: str | None = None) -> bool:
    try:
        from .model_config_registry import get_model_config_registry

        cfg = get_model_config_registry().lookup(
            _model_path or _model_name or model_hint or ""
        )
        return cfg.family_name == "zaya1_vl"
    except Exception:
        return False


def _synthesize_tools_from_message_tool_calls(messages: list[dict]) -> list[dict]:
    """Build minimal tool schemas from assistant tool-call history.

    Responses follow-up requests can replay prior ``function_call`` and
    ``function_call_output`` items without including the original ``tools``
    array. Some native templates (notably DSV4 DSML) need tool definitions on
    the system prompt to render tool-call conversations correctly, even when
    the current request is not allowed to call tools. Infer a conservative
    schema from historical arguments so the prompt has the missing context.
    """
    by_name: dict[str, dict] = {}
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            props = {}
            required = []
            if isinstance(args, dict):
                for key, value in args.items():
                    if not isinstance(key, str) or not key:
                        continue
                    if isinstance(value, bool):
                        typ = "boolean"
                    elif isinstance(value, int):
                        typ = "integer"
                    elif isinstance(value, float):
                        typ = "number"
                    elif isinstance(value, list):
                        typ = "array"
                    elif isinstance(value, dict):
                        typ = "object"
                    else:
                        typ = "string"
                    props[key] = {"type": typ}
                    required.append(key)
            by_name.setdefault(
                name,
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (
                            "Previously available tool from conversation history."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": required,
                        },
                    },
                },
            )
    return list(by_name.values())


@app.post(
    "/v1/responses",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
        Depends(check_memory_pressure),  # mlxstudio#63
        Depends(check_metal_working_set_pressure),  # mlxstudio#78
    ],
)
async def create_response(
    request: ResponsesRequest,
    fastapi_request: Request,
):
    """
    Create a response using the OpenAI Responses API format.

    This endpoint provides compatibility with the newer OpenAI Responses API
    (POST /v1/responses) while using the same underlying engine as
    /v1/chat/completions.

    Request format:
    ```json
    {
      "model": "model-name",
      "input": "What is the meaning of life?",
      "instructions": "You are a helpful assistant.",
      "stream": false
    }
    ```

    Or with message list:
    ```json
    {
      "model": "model-name",
      "input": [
        {"role": "user", "content": "Hello!"}
      ]
    }
    ```
    """
    # Model name validation (same as chat completions)
    resolved_name = _resolve_model_name()
    if (
        request.model
        and request.model != resolved_name
        and request.model != _model_name
    ):
        # v1.3.67 mlxstudio#79: INFO log once per distinct pair (see
        # create_chat_completion for full rationale).
        _pair = (str(request.model), str(resolved_name))
        if _pair not in _model_name_mismatch_seen:
            _model_name_mismatch_seen.add(_pair)
            logger.info(
                f"Request model '{request.model}' differs from served model "
                f"'{resolved_name}' — using loaded model (single-model server). "
                f"This is expected when routing tools like Claude Code, opencode, "
                f"or CCSwitch at vMLX — they send their own model alias."
            )
    request.model = resolved_name

    # Warn about unsupported penalty parameters
    if request.frequency_penalty and request.frequency_penalty != 0:
        logger.warning(
            "frequency_penalty=%.2f is not implemented and will be ignored",
            request.frequency_penalty,
        )
    if request.presence_penalty and request.presence_penalty != 0:
        logger.warning(
            "presence_penalty=%.2f is not implemented and will be ignored",
            request.presence_penalty,
        )

    engine = get_engine()
    _responses_has_media = _responses_input_has_multimodal(request.input)
    _responses_requested_modalities = _responses_input_requested_modalities(request.input)
    _log_multimodal_request_shape(
        "/v1/responses",
        _model_path or _model_name or request.model,
        _responses_input_multimodal_summary(request.input),
    )
    if _responses_requested_modalities:
        _reject_unsupported_multimodal(
            "/v1/responses",
            _responses_requested_modalities,
        )
    if (
        _responses_has_media
        and not engine.is_mllm
        and _loaded_omni_modalities() is None
    ):
        _reject_unsupported_multimodal("/v1/responses")

    # Convert Responses API input to chat messages
    # For MLLM models, preserve content arrays with image/video parts — MLLM engines
    # extract images from message content internally (same as Chat Completions path).
    # Nemotron-Omni is loaded as an LLM (engine.is_mllm=False) but the omni dispatcher
    # picks up multimodal content at request time — so we must preserve the array
    # for omni bundles too, otherwise input_image gets collapsed to text and the
    # encoder never sees the image.
    _preserve_mm = bool(engine.is_mllm)
    if not _preserve_mm:
        try:
            from .omni_multimodal import is_omni_multimodal_bundle
            _omni_path_resp = _model_path or _model_name
            if _omni_path_resp and is_omni_multimodal_bundle(_omni_path_resp):
                _preserve_mm = True
        except Exception:
            pass
    messages = _responses_input_to_messages(
        request.input,
        request.instructions,
        preserve_multimodal=_preserve_mm,
    )
    if request.previous_response_id:
        previous_messages = _responses_get_history(request.previous_response_id)
        if previous_messages:
            messages = previous_messages + messages
            logger.debug(
                "Responses API restored %d history message(s) from %s",
                len(previous_messages),
                request.previous_response_id,
            )
        else:
            logger.info(
                "Responses API previous_response_id=%s not found in local history; "
                "continuing with request input only",
                request.previous_response_id,
            )
    if engine.is_mllm and _should_coerce_zaya_vl_tool_history(request.model):
        messages = _coerce_zaya_vl_tool_history_for_template(messages)

    # Re-detect DSV4 for the Responses rail policy below. This is cheap
    # (cached registry lookup) and does not mutate caller messages.
    try:
        from .model_config_registry import get_model_config_registry
        _mc_dsv4_resp_msgs = get_model_config_registry().lookup(
            _model_path or _model_name or getattr(request, "model", "") or ""
        )
        _is_dsv4_resp_msgs = getattr(_mc_dsv4_resp_msgs, "family_name", "") == "deepseek_v4"
    except Exception:
        _is_dsv4_resp_msgs = False

    # Handle text format (json_object / json_schema) — translate to response_format
    if (
        request.text
        and isinstance(request.text, dict)
        and request.text.get("type") != "text"
    ):
        # Convert Responses API text format to Chat Completions response_format
        json_instruction = build_json_system_prompt(request.text)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)
    elif hasattr(request.text, "type") and request.text.type != "text":
        text_dict = (
            request.text.model_dump(exclude_none=True)
            if hasattr(request.text, "model_dump")
            else {"type": request.text.type}
        )
        json_instruction = build_json_system_prompt(text_dict)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)
    messages = _normalize_leading_system_messages(messages)

    _responses_max_prompt_tokens = _effective_max_prompt_tokens(request)
    prompt_limit_response = _reject_if_prompt_too_long_for_messages(
        messages,
        max_prompt_tokens=_responses_max_prompt_tokens,
    )
    if prompt_limit_response is not None:
        return prompt_limit_response

    # Strip <think> blocks from history when thinking is OFF (same as Chat Completions path)
    _ct_kwargs = _merge_ct_kwargs(request.chat_template_kwargs)
    _explicit_thinking_off = request.enable_thinking is False or (
        _ct_kwargs.get("enable_thinking") is False
    )
    if _explicit_thinking_off and messages:
        cleaned = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                # Skip list content (assistant messages with images from PR #29)
                if isinstance(content, str):
                    stripped = _THINK_STRIP_RE.sub("", content).strip()
                    if stripped != content:
                        if not stripped and not msg.get("tool_calls"):
                            continue  # Drop assistant messages that were ONLY thinking (no tool calls)
                        # Preserve message if it has tool_calls — orphaned tool results
                        # after dropping the assistant message causes template breakage
                        # and repetition loops (GH mlxstudio #62).
                        msg = {**msg, "content": stripped or None}
            cleaned.append(msg)
        messages = cleaned

    # Build kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_output_tokens, request.model),
        "temperature": _resolve_temperature(request.temperature, request.model),
        "top_p": _resolve_top_p(request.top_p, request.model),
        "max_prompt_tokens": _responses_max_prompt_tokens,
    }
    # Forward stop sequences from request to engine
    if request.stop:
        chat_kwargs["stop"] = request.stop
    # Extended sampling params (only pass if explicitly set)
    _set_resolved_top_k(chat_kwargs, request.top_k, request.model)
    _set_resolved_min_p(chat_kwargs, request.min_p, request.model)
    _rp = _resolve_repetition_penalty(request.repetition_penalty, request.model)
    if _rp is not None:
        chat_kwargs["repetition_penalty"] = _rp
    if _compute_bypass_prefix_cache(request):
        chat_kwargs["_bypass_prefix_cache"] = True

    # Pass enable_thinking to engine
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    _et = _resolve_enable_thinking(
        request_value=request.enable_thinking,
        ct_kwargs=_ct_kwargs,
        tools_present=bool(request.tools),
        model_key=_model_path or _model_name or request.model,
        engine=engine,
        auto_detect=True,
    )
    if _et is not None:
        chat_kwargs["enable_thinking"] = _et
        request.enable_thinking = _et

    # Pass reasoning_effort if provided (for GPT-OSS and models that support thinking levels).
    # Map it to template-side thinking_budget only; output length remains owned
    # by explicit max_tokens/max_output_tokens, then bundle max_new_tokens.
    if request.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = request.reasoning_effort
        # Also inject into chat_template_kwargs so Jinja templates can access it
        # (e.g., Mistral 4's [MODEL_SETTINGS]{"reasoning_effort":"high"} block).
        _ct_kwargs.setdefault("reasoning_effort", request.reasoning_effort)
        _effort_lower = request.reasoning_effort.lower()
        _budget = _EFFORT_THINKING_BUDGET.get(_effort_lower)
        if _budget:
            _ct_kwargs.setdefault("thinking_budget", _budget)
    if getattr(request, "max_thinking_tokens", None) is not None and request.enable_thinking is not False:
        _ct_kwargs["thinking_budget"] = int(request.max_thinking_tokens)

    # Auto-map enable_thinking → reasoning_effort for Mistral 4 (Responses API path).
    if (
        _reasoning_parser
        and type(_reasoning_parser).__name__ == "MistralReasoningParser"
    ):
        _resp_thinking = chat_kwargs.get("enable_thinking")
        if _resp_thinking is True and "reasoning_effort" not in _ct_kwargs:
            _ct_kwargs["reasoning_effort"] = "high"
            chat_kwargs["reasoning_effort"] = "high"
        elif _resp_thinking is False and "reasoning_effort" not in _ct_kwargs:
            _ct_kwargs["reasoning_effort"] = "none"

    _apply_hy3_reasoning_policy(
        chat_kwargs,
        _ct_kwargs,
        model_key=_model_path or _model_name or request.model,
        enable_thinking=request.enable_thinking,
    )

    # DSV4 rail-policy + reasoning_effort handling — REAL /v1/responses path.
    # This is the actual route the panel uses; the earlier DSV4 block in this
    # function only sets the default system prompt. Without this block,
    # /v1/responses would skip the same direct/reasoning policy used by chat
    # completions and could drift back into stale bare-thinking behavior.
    if _is_dsv4_resp_msgs:
        _dsv4_thinking_resp = _resolve_dsv4_thinking_policy(
            requested_enable_thinking=request.enable_thinking,
            effort_requested=bool(
                getattr(request, "reasoning_effort", None)
                or _ct_kwargs.get("reasoning_effort")
            ),
            tools_present=bool(getattr(request, "tools", None)),
            tool_choice=getattr(request, "tool_choice", None),
            default_mode=_jang_chat_default_mode(
                _model_path or _model_name or request.model or ""
            ),
        )
        if (
            not _dsv4_thinking_resp.enable_thinking
            and (
                request.enable_thinking is True
                or getattr(request, "reasoning_effort", None)
                or _ct_kwargs.get("reasoning_effort")
            )
        ):
            logger.info(
                "DSV4 (/v1/responses): using direct rail because %s.",
                _dsv4_thinking_resp.reason,
            )
        request.enable_thinking = _dsv4_thinking_resp.enable_thinking
        chat_kwargs["enable_thinking"] = _dsv4_thinking_resp.enable_thinking
        _ct_kwargs["enable_thinking"] = _dsv4_thinking_resp.enable_thinking
        _set_resolved_repetition_penalty(
            chat_kwargs,
            request.repetition_penalty,
            request.model,
            enable_thinking=chat_kwargs.get("enable_thinking"),
        )

        # (b) reasoning_effort handling — DSV4 encoder only injects the
        #     REASONING_EFFORT_MAX template at index 0 when effort=='max'.
        #     'high' is currently a render no-op in the bundle encoder
        #     but pass it through (forward-compat). 'low'/'medium' are
        #     normalized to 'high' so a future encoder version that
        #     handles non-max tiers picks them up.
        _cur_resp_effort = (
            getattr(request, "reasoning_effort", None)
            or _ct_kwargs.get("reasoning_effort")
        )
        _ct_kwargs.pop("thinking_mode", None)
        _stable_resp_effort = (
            _normalize_dsv4_reasoning_effort(_cur_resp_effort)
            if _dsv4_thinking_resp.reasoning_effort_allowed
            else None
        )
        if _stable_resp_effort:
            _ct_kwargs["reasoning_effort"] = _stable_resp_effort
        else:
            _ct_kwargs.pop("reasoning_effort", None)

        # (c) Diagnostic log so we can verify what actually goes to the
        #     encoder. Keep explicit logs of effective values.
        logger.info(
            f"DSV4 (/v1/responses) effective: enable_thinking="
            f"{chat_kwargs.get('enable_thinking')}, reasoning_effort="
            f"{_ct_kwargs.get('reasoning_effort')}, "
            f"messages={len(messages) if messages else 0}"
        )

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            chat_kwargs["chat_template_kwargs"] = extra_ct

    _log_resolved_sampling_kwargs(
        "/v1/responses",
        _model_path or _model_name or request.model,
        chat_kwargs,
    )

    # Video processing controls (MLLM models)
    if request.video_fps:
        chat_kwargs["video_fps"] = request.video_fps
    if request.video_max_frames:
        chat_kwargs["video_max_frames"] = request.video_max_frames

    # Handle tool_choice: "none" suppresses tools entirely, "auto"/None is default,
    # "required" or specific tool dict are handled post-generation.
    _tool_choice = request.tool_choice
    _suppress_tools = _tool_choice == "none"

    # response_format = json_object / json_schema implies the output IS the
    # structured data — not a tool call. The generic tool-call parser runs
    # on any JSON-ish output and is happy to interpret `{"name":"alice",
    # "age":30}` as a tool call `alice()`. Suppress the tool parser when
    # the client asked for JSON output and didn't pass tools themselves.
    # Responses API exposes the format signal on request.text.format
    # (ResponsesTextFormat); ResponsesRequest has no response_format field,
    # so getattr avoids AttributeError. Accept either shape.
    if not request.tools and not _suppress_tools:
        _rf = getattr(request, "response_format", None) or getattr(request, "text", None)
        _rf_type = None
        if isinstance(_rf, dict):
            _rf_type = _rf.get("type")
            if _rf_type is None:
                _fmt = _rf.get("format")
                if isinstance(_fmt, dict):
                    _rf_type = _fmt.get("type")
        elif _rf is not None:
            _rf_type = getattr(_rf, "type", None)
            if _rf_type is None:
                _fmt = getattr(_rf, "format", None)
                if isinstance(_fmt, dict):
                    _rf_type = _fmt.get("type")
                elif _fmt is not None:
                    _rf_type = getattr(_fmt, "type", None)
        if _rf_type in ("json_object", "json_schema"):
            _suppress_tools = True

    # Merge MCP tools with user-provided tools
    all_tools = []

    if not _suppress_tools:
        # Add MCP tools if available
        if _mcp_manager is not None:
            mcp_tools = _mcp_manager.get_all_tools_openai(policy=_mcp_policy)
            mcp_tools = _drop_colliding_mcp_tools(mcp_tools, request.tools)
            mcp_tools = _filter_tools_for_specific_choice(mcp_tools, _tool_choice)
            all_tools.extend(mcp_tools)
            if mcp_tools:
                logger.debug(f"Added {len(mcp_tools)} MCP tools")

        # Add user-provided tools — convert from Responses API flat format to Chat Completions nested format
        if request.tools:
            # If tool_choice is a specific tool dict, filter to only that tool
            if isinstance(_tool_choice, dict):
                target_name = _tool_choice.get("function", {}).get(
                    "name"
                ) or _tool_choice.get("name")
            else:
                target_name = None

            for tool in request.tools:
                tool_type = tool.get("type", "")
                # Skip built-in Responses API tools (web_search, code_interpreter, file_search, etc.)
                if (
                    tool_type != "function"
                    and "function" not in tool
                    and "name" not in tool
                ):
                    continue
                # Chat Completions nested format: {"type": "function", "function": {...}}
                if "function" in tool:
                    td = ToolDefinition(**tool)
                    # Filter to specific tool if tool_choice is a dict
                    if target_name and td.function.get("name") != target_name:
                        continue
                    all_tools.append(td)
                # Responses API flat format: {"type": "function", "name": "...", "parameters": {...}}
                elif "name" in tool:
                    # Filter to specific tool if tool_choice is a dict
                    if target_name and tool.get("name") != target_name:
                        continue
                    flat = ResponsesToolDefinition(**tool)
                    all_tools.append(
                        ToolDefinition(**flat.to_chat_completions_format())
                    )
            logger.debug(f"Added {len(all_tools)} tools (tool_choice={_tool_choice})")

    if not _suppress_tools:
        _suppress_tools = _suppress_tool_parsing_when_no_tools(
            all_tools, _tool_choice, "Responses API"
        )
    _attach_effective_tools_for_tool_parsing(request, all_tools)

    # Pass merged tools to engine
    if all_tools:
        chat_kwargs["tools"] = convert_tools_for_template(all_tools)
    elif not _suppress_tools and _is_dsv4_resp_msgs and any(
        isinstance(m, dict) and m.get("role") == "tool" for m in messages
    ):
        historical_tools = _synthesize_tools_from_message_tool_calls(messages)
        if historical_tools:
            chat_kwargs["tools"] = historical_tools
            logger.info(
                "DSV4 (Responses): synthesized %d historical tool schema(s) "
                "for tool-result continuation prompt context",
                len(historical_tools),
            )

    # Inject Harmony analysis prefix for GPT-OSS models (same as Chat Completions path)
    if isinstance(_reasoning_parser, GptOssReasoningParser):
        _think_val = chat_kwargs.get("enable_thinking")
        if _think_val is True:
            _analysis_hint = "<|start|>assistant<|channel|>analysis<|message|>"
            chat_kwargs["prompt_suffix"] = _analysis_hint
            chat_kwargs["skip_generation_prompt"] = True
            _effort = chat_kwargs.get("reasoning_effort", "").lower()
            logger.info(
                f"[responses] Injecting Harmony analysis prefix for GPT-OSS model (effort={_effort or 'default'})"
            )
        # GPT-OSS/GLM Flash: add <|im_end|> as string stop sequence (see Chat Completions path)
        _stop = chat_kwargs.get("stop") or []
        if "<|im_end|>" not in _stop:
            _stop = list(_stop) + ["<|im_end|>"]
            chat_kwargs["stop"] = _stop

    # Nemotron-Omni multimodal dispatch (Responses API parity with
    # /v1/chat/completions + /v1/messages). Without this hook, input_image /
    # input_video / input_audio reach the engine but engine.chat() is the
    # text-only path — image+audio encoders never run. Live test 2026-04-30
    # T6 surfaced this gap: T6 returned "Could you clarify what you'd like a
    # brief description of?" because the model never saw the image. Mirrors
    # /v1/messages dispatch at server.py:3189-3224.
    try:
        from .omni_multimodal import (
            is_omni_multimodal_bundle as _is_omni_resp,
            request_has_multimodal as _has_mm_resp,
            dispatch_omni_chat_completion as _omni_dispatch_resp,
        )
        _omni_path_dispatch = _model_path or _model_name
        if (
            _omni_path_dispatch
            and _is_omni_resp(_omni_path_dispatch)
            and _has_mm_resp(messages)
            and not request.stream
        ):
            from .api.models import ChatCompletionRequest as _CCR
            from starlette.responses import JSONResponse as _JR2
            _cc_req = _CCR(
                model=resolved_name,
                messages=messages,  # already preserved-multimodal above
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_output_tokens,
                stream=False,
                enable_thinking=request.enable_thinking,
                reasoning_effort=request.reasoning_effort,
            )
            cc = await _omni_dispatch_resp(_cc_req, _omni_path_dispatch)
            cc_dict = None
            if isinstance(cc, _JR2):
                cc_dict = json.loads(cc.body.decode("utf-8")) if cc.body else {}
            elif isinstance(cc, dict):
                cc_dict = cc
            if cc_dict is not None:
                _msg = (cc_dict.get("choices") or [{}])[0].get("message") or {}
                _txt = _msg.get("content") or ""
                _u = cc_dict.get("usage") or {}
                _resp_payload = {
                    "id": cc_dict.get("id", "resp_omni"),
                    "object": "response",
                    "created_at": cc_dict.get("created", int(time.time())),
                    "model": resolved_name,
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "msg_omni",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _txt}],
                    }],
                    "usage": {
                        "input_tokens": _u.get("prompt_tokens", 0),
                        "output_tokens": _u.get("completion_tokens", 0),
                        "total_tokens": _u.get("total_tokens", 0),
                    },
                }
                return _JR2(content=_resp_payload)
            return cc
    except HTTPException:
        raise
    except Exception as _omni_resp_err:  # pragma: no cover
        logger.warning(
            "Omni dispatch failed in /v1/responses (%s); "
            "falling back to text-only LLM path.",
            _omni_resp_err,
        )

    if request.stream:
        return StreamingResponse(
            stream_responses_api(
                engine, messages, request, fastapi_request, **chat_kwargs
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    start_time = time.perf_counter()
    timeout = request.timeout if request.timeout is not None else _default_timeout
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    try:
        output = await _await_chat_with_disconnect_abort(
            engine,
            messages=messages,
            chat_kwargs=chat_kwargs,
            timeout=timeout,
            fastapi_request=fastapi_request,
            request_id=response_id,
            endpoint="Responses API",
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"Request timed out after {timeout:.1f} seconds"
        )
    except PromptTooLongError as e:
        return _prompt_too_long_response_from_error(e)
    except VLMImagePrefillBudgetError as e:
        return _vlm_image_prefill_budget_response_from_error(e)
    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(e).__name__}: {e}",
        )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Response: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Extract reasoning content FIRST from raw output (before stripping tags).
    # Must happen before tool call parsing since that strips <think> blocks.
    reasoning_text = None
    content_for_parsing = output.text
    if _reasoning_parser:
        # Clone parser per-request to avoid shared state across concurrent requests
        request_parser = _reasoning_parser.__class__()
        # Initialize parser state (harmony_active, etc.) — same as streaming path
        _harmony_prefix_active = "prompt_suffix" in chat_kwargs and chat_kwargs.get(
            "prompt_suffix", ""
        ).startswith("<|start|>assistant<|channel|>analysis")

        # Compute think_in_prompt the same way the streaming path does at
        # ~line 5840. MiniMax and other "always-thinks" templates inject
        # <think>\n in the generation prompt, so model output starts in
        # reasoning mode. Without think_in_prompt=True, `extract_reasoning`
        # falls through to "no tags → pure content" and the reasoning prose
        # shows up raw in content (§15 regression).
        try:
            from .model_config_registry import get_model_config_registry as _mcr
            _mc_nonstream = _mcr().lookup(_model_path or _model_name or request.model)
            _think_in_prompt_ns = _mc_nonstream.think_in_template
            _think_in_prompt_ns = _apply_tokenizer_thinking_vocab_fallback(
                _think_in_prompt_ns,
                reasoning_parser=_reasoning_parser,
                tokenizer=engine.tokenizer,
                model_config=_mc_nonstream,
            )
            if _think_in_prompt_ns and _template_completes_thinking(
                engine.tokenizer, _model_name or request.model
            ):
                _think_in_prompt_ns = False
            _eff_thinking_ns = chat_kwargs.get("enable_thinking")
            _prompt_enable_ns = (
                True if _eff_thinking_ns is None else bool(_eff_thinking_ns)
            )
            _tools_present_ns = bool(getattr(request, "tools", None) or chat_kwargs.get("tools"))
            if _engine_prompt_starts_in_reasoning(
                engine.tokenizer,
                model_name=_model_name or request.model,
                enable_thinking=_prompt_enable_ns,
                family_name=getattr(_mc_nonstream, "family_name", None),
                tools_present=_tools_present_ns,
            ):
                _think_in_prompt_ns = True
            else:
                _think_in_prompt_ns = False
            if _hy3_prompt_starts_in_reasoning(
                model_key=_model_path or _model_name or request.model,
                enable_thinking=_eff_thinking_ns,
                ct_kwargs=chat_kwargs.get("chat_template_kwargs") or _ct_kwargs,
            ):
                _think_in_prompt_ns = True
            if _synthetic_mllm_think_prompt_starts(
                model_key=_model_path or _model_name or request.model,
                enable_thinking=_eff_thinking_ns,
                engine=engine,
            ):
                _think_in_prompt_ns = True
        except Exception as _tpe:
            logger.debug(f"think_in_prompt derivation failed non-stream: {_tpe}")
            _think_in_prompt_ns = False

        request_parser.reset_state(
            think_in_prompt=_think_in_prompt_ns,
            harmony_active=_harmony_prefix_active,
        )
        # Prefer raw_text (pre-clean) when the engine tracked it — reasoning
        # parsers (esp. Gemma 4 channel markers, Qwen 3.6 think tags) need the
        # special tokens that clean_output_text strips for display.
        _raw_for_parse = getattr(output, "raw_text", "") or output.text
        # DSV4 /v1/responses debug instrumentation:
        # log raw model output to confirm whether parser is failing or
        # model emitted </think> immediately, etc.
        if _is_dsv4_resp_msgs:
            _raw_len = len(_raw_for_parse) if _raw_for_parse else 0
            _has_open = "<think>" in (_raw_for_parse or "")
            _has_close = "</think>" in (_raw_for_parse or "")
            logger.info(
                f"DSV4 (/v1/responses) raw_output: len={_raw_len}, "
                f"has_<think>={_has_open}, has_</think>={_has_close}, "
                f"think_in_prompt={_think_in_prompt_ns}, "
                f"head={(_raw_for_parse or '')[:200]!r}, "
                f"tail={(_raw_for_parse or '')[-200:]!r}"
            )
        reasoning_text, remaining_text = request_parser.extract_reasoning(_raw_for_parse)
        if _is_dsv4_resp_msgs:
            logger.info(
                f"DSV4 (/v1/responses) parser: reasoning_len="
                f"{len(reasoning_text or '')}, remaining_len="
                f"{len(remaining_text or '')}"
            )
        if _is_dsv4_resp_msgs:
            token_reasoning, token_content = _dsv4_split_reasoning_from_token_ids(
                _output_token_ids_for_reasoning(output),
                engine.tokenizer,
            )
            if token_content:
                reasoning_text = token_reasoning
                remaining_text = token_content
                logger.info(
                    "DSV4 (/v1/responses) token split: reasoning_len=%s, content_len=%s",
                    len(reasoning_text or ""),
                    len(remaining_text or ""),
                )
        if remaining_text is not None:
            content_for_parsing = remaining_text
        elif reasoning_text is not None:
            # Truncated reasoning (e.g., max_tokens hit during <think> phase):
            # reasoning extracted but no content after it — don't leak raw <think> tags
            content_for_parsing = ""
        # Suppress reasoning when thinking is disabled (check resolved value from chat_kwargs,
        # not just raw request — covers --default-enable-thinking and auto-detect paths)
        _resolved = chat_kwargs.get("enable_thinking")
        _suppress = _resolved is False or request.enable_thinking is False
        if not _suppress:
            _suppress = _ct_kwargs.get("enable_thinking") is False
        if _suppress:
            # Recover truncated-think text into content when thinking is off.
            # Without this, a model that ignores enable_thinking=False and emits
            # <think>... but hits max_tokens before </think> produces a response
            # with content=null and reasoning_content=null — the tokens are lost.
            if not content_for_parsing and reasoning_text:
                content_for_parsing = reasoning_text
            elif content_for_parsing:
                content_for_parsing = _strip_residual_think_markup_for_display(
                    content_for_parsing
                )
            reasoning_text = None

        # Post-parse cleaning — matches the chat_completions path so content
        # emitted by /v1/responses is stripped of residual `<channel|>`,
        # `thought\n`, `<turn|>` etc. that survive the parser's structural
        # split. Applied AFTER extract_reasoning so parser still sees the
        # special tokens it needs for detection.
        if content_for_parsing:
            content_for_parsing = clean_output_text(content_for_parsing)
        if reasoning_text:
            reasoning_text = clean_output_text(reasoning_text)

    parse_text = _strip_think_for_tool_parse(content_for_parsing)

    # Parse tool calls (skip when tool_choice="none")
    cleaned_text, tool_calls = (
        _parse_tool_calls_with_parser(parse_text, request)
        if not _suppress_tools
        else (
            _clean_suppressed_tool_markup_for_display(
                parse_text or content_for_parsing or "",
                request,
            ),
            None,
        )
    )

    if (
        not tool_calls
        and reasoning_text
        and not _suppress_tools
        and any(m in reasoning_text for m in _TOOL_CALL_MARKERS)
    ):
        logger.info(
            f"Request {response_id}: tool markers in reasoning — "
            "extracting before Responses API finalization"
        )
        _tc_cleaned, _tc_calls = _parse_tool_calls_with_parser(
            reasoning_text.strip(), request
        )
        if _tc_calls:
            tool_calls = _tc_calls
            cleaned_text = ""
            reasoning_text = clean_output_text(_tc_cleaned).strip() or None

    # Process text format (json_schema with strict) if specified — mirrors Chat Completions behavior
    _text_format = request.text if hasattr(request, "text") else None
    if _text_format and hasattr(_text_format, "model_dump"):
        _text_format = _text_format.model_dump(exclude_none=True)
    if (
        _text_format
        and isinstance(_text_format, dict)
        and _text_format.get("type") not in (None, "text")
        and not tool_calls
    ):
        # Build a response_format-compatible dict for parse_json_output
        _rf_type = _text_format.get("type", "")
        if _rf_type in ("json_schema", "json_object"):
            response_format = _text_format
            _out_text = cleaned_text or content_for_parsing
            _out_text, parsed_json, is_valid, error = parse_json_output(
                _out_text, response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                _rf_strict = _text_format.get("json_schema", {}).get("strict", False)
                if _rf_strict:
                    raise HTTPException(
                        status_code=400, detail=f"response_format strict mode: {error}"
                    )
                logger.warning(f"[responses] JSON validation failed: {error}")

    # Enforce tool_choice="required": model MUST produce at least one tool call
    _resp_tool_choice = getattr(request, "tool_choice", None)
    if _resp_tool_choice == "required" and not tool_calls:
        logger.warning(
            f"tool_choice='required' but model produced no tool calls. "
            f"Returning error to client."
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "tool_choice='required' was set but the model did not produce "
                "any tool calls. Try rephrasing your prompt or using a model "
                "with better tool-calling support."
            ),
        )

    # Build output array
    output_items = []

    # Add reasoning output if present
    if reasoning_text:
        output_items.append(
            ResponsesOutputMessage(
                type="reasoning",
                role="assistant",
                content=[ResponsesOutputText(type="reasoning", text=reasoning_text)],
            )
        )

    # Add main message (use cleaned text if tool calls were extracted, otherwise raw text)
    final_text = (
        clean_output_text(cleaned_text)
        if cleaned_text
        else ("" if tool_calls else (clean_output_text(content_for_parsing) if content_for_parsing else ""))
    )
    if final_text:
        final_text = _finalize_visible_text_for_request(final_text, request)
    if final_text:
        output_items.append(
            ResponsesOutputMessage(
                role="assistant",
                content=[ResponsesOutputText(text=final_text)],
            )
        )

    # Add tool calls as separate function_call output items (Responses API format)
    if tool_calls:
        for tc in tool_calls:
            func = tc.function if hasattr(tc, "function") else tc
            # Forward tc.id from parser so stateful tool-use loops can correlate results
            tc_call_id = tc.id if hasattr(tc, "id") and tc.id else None
            fc_kwargs = dict(
                name=func.name if hasattr(func, "name") else func.get("name", ""),
                arguments=func.arguments
                if hasattr(func, "arguments")
                else func.get("arguments", ""),
            )
            if tc_call_id:
                fc_kwargs["call_id"] = tc_call_id
            output_items.append(ResponsesFunctionCall(**fc_kwargs))

    # Detect reasoning-only output: any reasoning items present, no visible
    # message text, no tool/function calls. Empty streaming-style message
    # shells do not count as visible output. Tracked so chained turns surface
    # a warning.
    _reasoning_only = _responses_output_is_reasoning_only(output_items)

    response_obj = ResponsesObject(
        id=response_id,
        model=request.model,
        output=output_items,
        usage=_get_responses_usage(output),
        previous_response_id=request.previous_response_id,
        warnings=_merge_responses_warnings(
            _chain_warnings_for_previous_response_id(request.previous_response_id),
            _current_response_warnings_for_reasoning_only(_reasoning_only),
        ),
    )
    _responses_store_history(
        response_obj.id,
        messages + _responses_output_to_assistant_messages(output_items),
        reasoning_only=_reasoning_only,
    )
    return response_obj


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Find existing system message
    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_completions_multi(
    engine: BaseEngine,
    prompts: list[str],
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion responses for one or more prompts.

    Each prompt gets its own choice index in the streaming chunks.
    A single [DONE] sentinel is sent after all prompts complete.
    """
    response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Use per-request timeout if provided, otherwise server default
    _stream_timeout = (
        request.timeout if request.timeout is not None else _default_timeout
    )

    for prompt_index, prompt in enumerate(prompts):
        # Per-prompt request_id so abort targets the correct engine request
        prompt_request_id = f"{response_id}-{prompt_index}"
        try:
            gen_kwargs: dict = {
                "prompt": prompt,
                "max_tokens": _resolve_max_tokens(request.max_tokens, getattr(request, "model", "")),
                "temperature": _resolve_temperature(request.temperature, request.model),
                "top_p": _resolve_top_p(request.top_p, request.model),
                "stop": request.stop,
                "request_id": prompt_request_id,
                "max_prompt_tokens": _effective_max_prompt_tokens(request),
            }
            _set_resolved_top_k(gen_kwargs, request.top_k, request.model)
            _set_resolved_min_p(gen_kwargs, request.min_p, request.model)
            _rp = _resolve_repetition_penalty(request.repetition_penalty, request.model)
            if _rp is not None:
                gen_kwargs["repetition_penalty"] = _rp
            if _compute_bypass_prefix_cache(request):
                gen_kwargs["_bypass_prefix_cache"] = True
            if request.logprobs is not None:
                gen_kwargs["logprobs"] = True
                gen_kwargs["top_logprobs"] = request.logprobs
            stream_logprob_offset = 0
            async for output in _stream_with_keepalive(
                engine.stream_generate(**gen_kwargs), total_timeout=_stream_timeout
            ):
                if output is None:
                    yield ": keep-alive\n\n"
                    continue
                raw_logprobs = getattr(output, "logprobs", None) or []
                choice = {
                    "index": prompt_index,
                    "text": output.new_text,
                    "finish_reason": output.finish_reason
                    if output.finished
                    else None,
                }
                if request.logprobs is not None:
                    delta_logprobs = raw_logprobs[stream_logprob_offset:]
                    stream_logprob_offset = len(raw_logprobs)
                    choice["logprobs"] = _format_completion_logprobs(
                        delta_logprobs,
                        engine.tokenizer,
                    )
                data = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [choice],
                }
                if output.finished:
                    data["usage"] = get_usage(output).model_dump(exclude_none=True)
                yield f"data: {json.dumps(data, ensure_ascii=True)}\n\n"
        except PromptTooLongError as e:
            if hasattr(engine, "abort_request"):
                await engine.abort_request(prompt_request_id)
            error_data = {
                "id": response_id,
                "object": "text_completion",
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "prompt_too_long",
                },
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        except VLMImagePrefillBudgetError as e:
            if hasattr(engine, "abort_request"):
                await engine.abort_request(prompt_request_id)
            error_data = {
                "id": response_id,
                "object": "text_completion",
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": VLMImagePrefillBudgetError.code,
                },
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        except Exception as e:
            logger.error(f"Stream error for {response_id}: {e}", exc_info=True)
            if hasattr(engine, "abort_request"):
                await engine.abort_request(prompt_request_id)
            error_data = {
                "id": response_id,
                "object": "text_completion",
                "error": {
                    "message": f"Stream generation failed: {e}",
                    "type": "server_error",
                    "code": "internal_error",
                },
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"


def _dump_sse_json(obj) -> str:
    """Serialize Pydantic model to ASCII-safe JSON for SSE streaming.

    Uses ensure_ascii=True so emoji and other multi-byte UTF-8 characters
    are encoded as \\uXXXX escape sequences. This prevents corruption when
    HTTP chunk boundaries split raw UTF-8 bytes.
    Uses exclude_none=True so fields like reasoning_content don't appear
    when they have no value (cleaner streaming output).
    """
    return json.dumps(obj.model_dump(exclude_none=True), ensure_ascii=True)


# ─── SSE keep-alive helper ──────────────────────────────────────────────
# During long prefills (VLMs, large contexts), no tokens are emitted for
# many seconds. Clients, proxies, and load balancers may treat the silence
# as a stalled connection and drop it. This wrapper yields None sentinels
# when no item arrives within `interval` seconds, which the streaming
# generators convert to SSE comments (`: keep-alive\n\n`).

_SSE_KEEPALIVE_INTERVAL = 15.0  # seconds


async def _stream_with_keepalive(
    async_gen,
    interval: float = _SSE_KEEPALIVE_INTERVAL,
    total_timeout: float | None = None,
):
    """Yield items from async generator, inserting None sentinels on timeout.
    If total_timeout is set, raises TimeoutError after that many seconds of total streaming.

    Uses asyncio.wait() which does NOT cancel the pending future on timeout —
    critical because cancelling an async generator's __anext__() finalizes it.
    asyncio.wait_for() MUST NOT be used here: it cancels on timeout, which
    kills the stream during long prefills or tool call generation.
    """
    it = async_gen.__aiter__()
    pending = asyncio.ensure_future(it.__anext__())
    start = time.monotonic()
    try:
        while True:
            if total_timeout and (time.monotonic() - start) > total_timeout:
                raise TimeoutError(f"Streaming exceeded {total_timeout}s timeout")
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if done:
                try:
                    yield pending.result()
                except StopAsyncIteration:
                    return
                except Exception as e:
                    logger.error(f"Stream generator error: {e}")
                    raise
                pending = asyncio.ensure_future(it.__anext__())
            else:
                yield None  # keep-alive sentinel
    finally:
        if not pending.done():
            pending.cancel()


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    fastapi_request: Request | None = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion with auto-detection of closed connections."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Pass response_id as request_id to engine for unified tracking
    kwargs["request_id"] = response_id

    # Check if we should include usage in the final chunk
    include_usage = request.stream_options and request.stream_options.include_usage

    # Stable timestamp for all chunks in this stream (OpenAI spec compliance)
    _created_ts = int(time.time())

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        created=_created_ts,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {_dump_sse_json(first_chunk)}\n\n"

    # Resolve effective enable_thinking:
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    # (None = fall through to template/tokenizer auto-detect below — this site
    # drives SSE parser behavior, not engine kwargs, so no Gemma4+tools override.)
    _ct_kwargs = _merge_ct_kwargs(request.chat_template_kwargs)
    _effective_thinking = _resolve_enable_thinking(
        request_value=request.enable_thinking,
        ct_kwargs=_ct_kwargs,
        tools_present=bool(getattr(request, "tools", None)),
        model_key=_model_path or _model_name or request.model,
        engine=engine,
        auto_detect=True,
    )

    # Check if model's chat template injects <think> in the assistant prefix
    # Use _model_name (actual model path) not request.model (which may be "default")
    from .model_config_registry import get_model_config_registry

    _model_config = get_model_config_registry().lookup(
        _model_path or _model_name or request.model
    )
    think_in_template = _model_config.think_in_template

    # Fallback only for unknown configs. Known families may expose thinking
    # tokens in vocab without opening a reasoning rail in the rendered prompt.
    think_in_template = _apply_tokenizer_thinking_vocab_fallback(
        think_in_template,
        reasoning_parser=_reasoning_parser,
        tokenizer=engine.tokenizer,
        model_config=_model_config,
    )

    # S5 seed detection: some templates (e.g., Nemotron CRACK) complete the
    # thinking block in the generation prompt (<think>\nOK.\n</think>\n).
    # The model output is plain text — think_in_template must be False so
    # the parser doesn't misclassify all output as reasoning.
    if think_in_template:
        if _template_completes_thinking(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # When user explicitly disables thinking, check if template actually respects it.
    # This is only a raw-template precheck. The parser seed below uses the
    # engine's final prompt contract, including MiniMax's closed no-thinking
    # sentinel for no-tool direct-chat requests.
    if _effective_thinking is False and think_in_template:
        if not _template_always_thinks(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    if _reasoning_parser:
        _prompt_enable = True if _effective_thinking is None else bool(_effective_thinking)
        _tools_present_for_prompt = bool(getattr(request, "tools", None) or kwargs.get("tools"))
        if _engine_prompt_starts_in_reasoning(
            engine.tokenizer,
            model_name=_model_name or request.model,
            enable_thinking=_prompt_enable,
            family_name=getattr(_model_config, "family_name", None),
            tools_present=_tools_present_for_prompt,
        ):
            think_in_template = True
        else:
            think_in_template = False

    if _hy3_prompt_starts_in_reasoning(
        model_key=_model_path or _model_name or request.model,
        enable_thinking=_effective_thinking,
        ct_kwargs=kwargs.get("chat_template_kwargs") or _ct_kwargs,
    ):
        think_in_template = True

    if _synthetic_mllm_think_prompt_starts(
        model_key=_model_path or _model_name or request.model,
        enable_thinking=_effective_thinking,
        engine=engine,
    ):
        think_in_template = True

    # The parser seed must match the final prompt the engine renders. Tool
    # requests may intentionally leave reasoning open; no-tool thinking-off
    # requests may close it before decode.
    effective_think_in_template = think_in_template

    # Track if we need to add <think> prefix for thinking models (when no reasoning parser)
    # The template adds <think> to the prompt, so the model output starts inside the think block
    is_thinking_model = effective_think_in_template and not _reasoning_parser
    think_prefix_sent = False

    # Suppress reasoning output when user explicitly disabled thinking.
    # The parser still runs (to strip <think> tags from content), but reasoning
    # chunks are dropped so the user never sees them.
    suppress_reasoning = _effective_thinking is False

    # Create a per-request parser instance to avoid mutable-state conflicts
    # when multiple requests stream concurrently.
    # Check if Harmony analysis prefix was injected (set by chat endpoint above)
    _harmony_prefix_active = "prompt_suffix" in kwargs and kwargs.get(
        "prompt_suffix", ""
    ).startswith("<|start|>assistant<|channel|>analysis")
    request_parser = None
    if _reasoning_parser:
        request_parser = _reasoning_parser.__class__()
        request_parser.reset_state(
            think_in_prompt=effective_think_in_template,
            harmony_active=_harmony_prefix_active,
        )
        logger.debug(
            f"[chat] Reasoning parser: {type(request_parser).__name__} (think_in_template={effective_think_in_template}, suppress={suppress_reasoning}, harmony_active={_harmony_prefix_active})"
        )
    else:
        logger.debug("[chat] No reasoning parser active for this request")

    # Track accumulated text for reasoning parser and tool call detection
    accumulated_text = ""
    accumulated_reasoning = ""  # Track reasoning text for fallback
    accumulated_content = ""  # Track content-only text for tool call marker detection
    streamed_content = (
        ""  # Track content actually yielded to client (for post-stream dedup)
    )
    content_was_emitted = False  # Whether any content chunk was actually sent
    reasoning_was_streamed = False  # Whether any reasoning_content chunk was sent

    # Tool call buffering: when we detect a tool call marker in the stream,
    # we stop emitting content and buffer the rest. At end of stream, we parse
    # the buffer for tool calls and emit them as proper tool_calls chunks.
    tool_call_buffering = False  # Are we currently buffering for tool calls?
    tool_call_buffering_notified = False  # Have we sent the buffering signal?
    tool_calls_emitted = False  # Were actual tool calls parsed and emitted?
    _suppress_tools = getattr(request, "tool_choice", None) == "none"
    # Auto-enable tool call detection when client sends tools in request (#46),
    # even if server wasn't started with --enable-auto-tool-choice
    _request_has_tools = bool(getattr(request, "tools", None))
    _stream_tools_available = bool(kwargs.get("tools")) or _request_has_tools
    tool_call_active = _stream_tools_available and not _suppress_tools

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    cache_detail: str | None = None
    last_output = None
    stream_logprob_offset = 0

    # Use per-request timeout if provided, otherwise server default
    _stream_timeout = (
        request.timeout if request.timeout is not None else _default_timeout
    )

    try:
        # Stream content (with SSE keep-alive during long prefills)
        async for output in _stream_with_keepalive(
            engine.stream_chat(messages=messages, **kwargs),
            total_timeout=_stream_timeout,
        ):
            # Keep-alive sentinel — emit SSE comment to prevent connection timeout
            if output is None:
                yield ": keep-alive\n\n"
                continue

            # Check if client disconnected
            if fastapi_request and await fastapi_request.is_disconnected():
                logger.info(f"Client disconnected, aborting request {response_id}")
                if hasattr(engine, "abort_request"):
                    await engine.abort_request(response_id)
                break

            delta_text = output.new_text
            last_output = output

            # Track token counts from output (updated each chunk)
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            if hasattr(output, "cached_tokens") and output.cached_tokens:
                cached_tokens = output.cached_tokens
            _detail = getattr(output, "cache_detail", "") or None
            if _detail is not None:
                cache_detail = _detail
            chunk_logprobs = None
            if request.logprobs:
                raw_logprobs = getattr(output, "logprobs", None) or []
                delta_logprobs = raw_logprobs[stream_logprob_offset:]
                stream_logprob_offset = len(raw_logprobs)
                chunk_logprobs = _format_chat_logprobs(
                    delta_logprobs,
                    engine.tokenizer,
                )

            # Always accumulate full text (needed for both reasoning and tool call parsing)
            accumulated_text += delta_text if delta_text else ""

            # Use reasoning parser if enabled
            if request_parser and delta_text:
                previous_text = (
                    accumulated_text[: -len(delta_text)]
                    if delta_text
                    else accumulated_text
                )
                delta_msg = request_parser.extract_reasoning_streaming(
                    previous_text, accumulated_text, delta_text
                )

                if delta_msg is None:
                    # Skip this chunk (e.g., <think> token itself)
                    if suppress_reasoning:
                        logger.debug(
                            f"[reasoning-debug] parser returned None for delta={repr(delta_text[:50])}, suppress={suppress_reasoning}, think_in_prompt={effective_think_in_template}"
                        )
                    continue

                # Post-parse cleaning for streaming deltas — strip any
                # structural markers that survived the parser's split
                # (e.g. a second `<channel|>` embedded in content when
                # Gemma 4 emits nested channel frames). Must use the
                # DELTA-safe variant that preserves leading/trailing
                # whitespace — otherwise single-space deltas (" the")
                # lose their space, concatenating the entire stream into
                # "Itlooksliketypo" visible garbage (bug 2026-04-21).
                if delta_msg.content:
                    delta_msg.content = strip_marker_tokens_delta(delta_msg.content)
                if delta_msg.reasoning:
                    delta_msg.reasoning = strip_marker_tokens_delta(delta_msg.reasoning)

                # Debug: log parser output when suppress is active
                if suppress_reasoning and (delta_msg.content or delta_msg.reasoning):
                    logger.debug(
                        f"[reasoning-debug] content={repr((delta_msg.content or '')[:30])}, reasoning={repr((delta_msg.reasoning or '')[:30])}, suppress={suppress_reasoning}"
                    )

                # Accumulate for marker detection (before buffering check).
                # Suppressed reasoning stays internal and must not be mirrored
                # into accumulated_content or next-turn visible history.
                if delta_msg.content:
                    accumulated_content += delta_msg.content
                if delta_msg.reasoning:
                    accumulated_reasoning += delta_msg.reasoning

                # Check for tool call markers — separate logic for content vs reasoning:
                # - Content: check accumulated_content (markers can span deltas)
                # - Reasoning: check a trailing window of accumulated_reasoning
                #   (catches markers split across chunk boundaries, but avoids
                #   false positives from earlier reasoning that casually mentions
                #   tool formats like "<function=")
                if tool_call_active and not tool_call_buffering:
                    if delta_msg.content and accumulated_content:
                        tool_call_buffering = _has_tool_marker_or_partial_suffix(
                            accumulated_content
                        )
                    if not tool_call_buffering and delta_msg.reasoning:
                        # Use trailing window (last 30 chars covers longest marker)
                        _reasoning_tail = (
                            accumulated_reasoning[-30:]
                            if len(accumulated_reasoning) > 30
                            else accumulated_reasoning
                        )
                        tool_call_buffering = _has_tool_marker_or_partial_suffix(
                            _reasoning_tail
                        )
                    # GPT-OSS/Harmony native tool format: to=<name> code{...}
                    # Uses regex for specificity (plain "to=" is too broad for markers list)
                    if not tool_call_buffering:
                        _tc_check = (
                            accumulated_content
                            or (
                                accumulated_reasoning[-30:]
                                if accumulated_reasoning
                                else ""
                            )
                            or ""
                        )
                        if re.search(r"\bto=\w[\w.]*\s+code\{", _tc_check):
                            tool_call_buffering = True

                if tool_call_buffering:
                    # Suppress content during tool call buffering, but emit
                    # usage-only chunks so the client TPS counter stays alive.
                    if include_usage:
                        buf_chunk = ChatCompletionChunk(
                            id=response_id,
                            created=_created_ts,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(content=None),
                                    finish_reason=None,
                                )
                            ],
                            usage=get_usage(output),
                            tool_call_generating=True,
                        )
                        if not tool_call_buffering_notified:
                            tool_call_buffering_notified = True
                        yield f"data: {_dump_sse_json(buf_chunk)}\n\n"
                    continue

                # Include usage in every chunk when include_usage is on (for real-time metrics)
                chunk_usage = get_usage(output) if include_usage else None

                # Suppressed reasoning is never redirected into visible content.
                # If a boundary delta carries both reasoning and content, only
                # the content half is user-visible.
                if suppress_reasoning:
                    emit_content = delta_msg.content
                    emit_reasoning = None
                    if not emit_content and not output.finished:
                        heartbeat = ChatCompletionChunk(
                            id=response_id,
                            created=_created_ts,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(content=None),
                                    finish_reason=None,
                                )
                            ],
                            usage=get_usage(output) if include_usage else None,
                        )
                        yield f"data: {_dump_sse_json(heartbeat)}\n\n"
                        continue
                else:
                    emit_reasoning = delta_msg.reasoning
                    emit_content = delta_msg.content

                # Tool-call-only streams often start with harmless whitespace
                # before the native marker (DSV4 commonly emits "\n\n<｜DSML｜…").
                # Do not send that whitespace as visible content until we know
                # the turn is not a tool call; otherwise clients see a blank
                # assistant message before the structured tool_calls delta.
                if (
                    tool_call_active
                    and not content_was_emitted
                    and emit_content
                    and not emit_content.strip()
                ):
                    if include_usage:
                        heartbeat = ChatCompletionChunk(
                            id=response_id,
                            created=_created_ts,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(content=None),
                                    finish_reason=None,
                                )
                            ],
                            usage=get_usage(output),
                        )
                        yield f"data: {_dump_sse_json(heartbeat)}\n\n"
                    continue

                if _suppress_tools and emit_content:
                    emit_content = _suppressed_tool_display_delta(
                        accumulated_content,
                        streamed_content,
                        request,
                    )
                elif emit_content:
                    emit_content = _visual_grounding_display_delta(
                        accumulated_content,
                        streamed_content,
                    )

                # Skip chunks that have nothing to emit after conversion
                if not emit_content and not emit_reasoning and not output.finished:
                    continue

                if emit_reasoning:
                    reasoning_was_streamed = True
                if emit_content:
                    content_was_emitted = True
                    streamed_content += emit_content

                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=_created_ts,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=emit_content,
                                reasoning=emit_reasoning,
                            ),
                            logprobs=chunk_logprobs,
                            finish_reason=output.finish_reason
                            if output.finished
                            else None,
                        )
                    ],
                    usage=chunk_usage,
                )
                yield f"data: {_dump_sse_json(chunk)}\n\n"
            elif not request_parser:
                # Standard path without reasoning parsing (no parser configured)
                content = delta_text

                # Check for tool call markers — start buffering when detected
                if tool_call_active and not tool_call_buffering and content:
                    tool_call_buffering = _has_tool_marker_or_partial_suffix(
                        accumulated_text
                    )

                if tool_call_buffering:
                    # Suppress content but emit usage-only chunks for TPS tracking
                    if include_usage:
                        buf_chunk = ChatCompletionChunk(
                            id=response_id,
                            created=_created_ts,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(content=None),
                                    finish_reason=None,
                                )
                            ],
                            usage=get_usage(output),
                            tool_call_generating=True,
                        )
                        if not tool_call_buffering_notified:
                            tool_call_buffering_notified = True
                        yield f"data: {_dump_sse_json(buf_chunk)}\n\n"
                    continue

                # Same leading-whitespace guard as the reasoning-parser path.
                if (
                    tool_call_active
                    and not content_was_emitted
                    and content
                    and not content.strip()
                ):
                    if include_usage:
                        heartbeat = ChatCompletionChunk(
                            id=response_id,
                            created=_created_ts,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(content=None),
                                    finish_reason=None,
                                )
                            ],
                            usage=get_usage(output),
                        )
                        yield f"data: {_dump_sse_json(heartbeat)}\n\n"
                    continue

                # Add <think> prefix on first content chunk for thinking models
                if is_thinking_model and not think_prefix_sent and content:
                    content = "<think>" + content
                    think_prefix_sent = True

                if _suppress_tools and content:
                    content = _suppressed_tool_display_delta(
                        accumulated_text,
                        streamed_content,
                        request,
                    )
                elif content:
                    content = _visual_grounding_display_delta(
                        accumulated_text,
                        streamed_content,
                    )

                if content:
                    content_was_emitted = True
                    streamed_content += content

                # Include usage in every chunk when include_usage is on (for real-time metrics)
                chunk_usage = get_usage(output) if include_usage else None
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=_created_ts,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=content if content else None
                            ),
                            logprobs=chunk_logprobs,
                            finish_reason=output.finish_reason
                            if output.finished
                            else None,
                        )
                    ],
                    usage=chunk_usage,
                )
                yield f"data: {_dump_sse_json(chunk)}\n\n"
            else:
                # Reasoning parser is active but delta_text is empty/None
                # (progress-only engine chunk). Skip — don't emit empty delta.
                continue

    except PromptTooLongError as e:
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        error_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "prompt_too_long",
            },
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except VLMImagePrefillBudgetError as e:
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        error_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": VLMImagePrefillBudgetError.code,
            },
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        # On any error, abort the request and emit an error SSE event
        # so the client knows what happened instead of a silent EOF.
        logger.error(f"Stream error for {response_id}: {e}", exc_info=True)
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        error_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "error": {
                "message": _generation_error_detail(
                    e, prefix="Stream generation failed"
                ),
                "type": "server_error",
                "code": "internal_error",
            },
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        # M8: Send usage even on error so client gets partial token counts
        if include_usage and (prompt_tokens > 0 or completion_tokens > 0):
            _err_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            if cached_tokens > 0 or cache_detail:
                _err_usage.prompt_tokens_details = PromptTokensDetails(
                    cached_tokens=cached_tokens, cache_detail=cache_detail
                )
            err_usage_chunk = ChatCompletionChunk(
                id=response_id,
                created=_created_ts,
                model=request.model,
                choices=[],
                usage=_err_usage,
            )
            yield f"data: {_dump_sse_json(err_usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ─── Post-stream: tool call extraction ───────────────────────────────
    # If we buffered text because of tool call markers, parse it now
    if tool_call_buffering and accumulated_text and not _suppress_tools:
        # Use content-only text when reasoning parser separated it (avoids losing
        # tool calls that appear inside <think> blocks during regex stripping).
        # If content is empty but reasoning has tool markers, check reasoning too
        # (model may emit tool calls in analysis channel instead of final channel).
        # Fall back to accumulated_text with think-tag stripping when no parser was active.
        if request_parser and accumulated_content.strip():
            parse_text = accumulated_content.strip()
        elif request_parser and accumulated_reasoning.strip():
            # Tool call markers were in reasoning — try parsing reasoning text
            parse_text = accumulated_reasoning.strip()
        else:
            parse_text = _strip_think_for_tool_parse(accumulated_text)
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(
            parse_text or accumulated_text, request
        )
        if tool_calls:
            # Emit any remaining content text before the tool calls,
            # but ONLY the portion that wasn't already streamed.
            # streamed_content tracks what was ACTUALLY yielded to the client
            # (not just accumulated — content in the same delta as a tool marker
            # may be accumulated but never yielded due to buffering).
            unemitted_content = None
            if cleaned_text and cleaned_text.strip():
                already_sent = streamed_content.strip()
                candidate = cleaned_text.strip()
                if not already_sent:
                    # Nothing was streamed yet — emit all cleaned content
                    unemitted_content = candidate
                elif candidate.startswith(already_sent):
                    # Subtract the already-streamed portion
                    remainder = candidate[len(already_sent) :].strip()
                    unemitted_content = remainder if remainder else None
                else:
                    # Content doesn't overlap (rare: reasoning redirect, etc.)
                    # Emit the full candidate only if nothing was sent
                    unemitted_content = candidate if not content_was_emitted else None

            if unemitted_content:
                content_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=_created_ts,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=unemitted_content),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {_dump_sse_json(content_chunk)}\n\n"

            # Emit tool calls in OpenAI-compatible streaming format (#46):
            # Chunk 1: tool_calls data with finish_reason=null
            # Chunk 2: empty delta with finish_reason="tool_calls"
            # This split is required for CLI clients (Claude Code, OpenCode, etc.)
            tc_deltas = [
                {
                    "index": i,
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            # Build tool_calls chunk matching OpenAI's exact format (#46):
            # - role: "assistant" required for SDK message assembly
            # - content omitted (not null) — OpenAI omits it in tool_calls deltas
            tc_data_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": _created_ts,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": tc_deltas,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(tc_data_chunk, ensure_ascii=True)}\n\n"
            # Finish chunk: empty delta with finish_reason="tool_calls"
            tc_finish_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": _created_ts,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            }
            yield f"data: {json.dumps(tc_finish_chunk, ensure_ascii=True)}\n\n"
            # Skip normal end-of-stream handling — we already set finish_reason
            if include_usage:
                _tc_usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                if cached_tokens > 0 or cache_detail:
                    _tc_usage.prompt_tokens_details = PromptTokensDetails(
                        cached_tokens=cached_tokens, cache_detail=cache_detail
                    )
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=_created_ts,
                    model=request.model,
                    choices=[],
                    usage=_tc_usage,
                )
                yield f"data: {_dump_sse_json(usage_chunk)}\n\n"
            tool_calls_emitted = True
            yield "data: [DONE]\n\n"
            return
        else:
            # No tool calls found despite markers — flush only the UN-STREAMED
            # portion. streamed_content tracks what was actually yielded to client;
            # the remainder needs flushing.
            # When reasoning parser is active, use accumulated_content (content-only)
            # instead of accumulated_text (which includes reasoning and would leak it).
            already_sent = streamed_content.strip()
            full = (
                accumulated_content.strip()
                if request_parser
                else accumulated_text.strip()
            )
            if already_sent and full.startswith(already_sent):
                remainder = full[len(already_sent) :].strip()
            else:
                remainder = full if not content_was_emitted else ""
            if remainder:
                remainder = _strip_tool_markup_residue_for_display(remainder)
            if remainder:
                # Use the engine's actual finish_reason (e.g., "length" if max_tokens
                # was hit) instead of hardcoding "stop"
                _flush_reason = (
                    last_output.finish_reason
                    if last_output and last_output.finish_reason
                    else "stop"
                )
                flush_chunk = ChatCompletionChunk(
                    id=response_id,
                    created=_created_ts,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=remainder),
                            finish_reason=_flush_reason,
                        )
                    ],
                )
                yield f"data: {_dump_sse_json(flush_chunk)}\n\n"

    # Fallback: if reasoning parser produced only reasoning with no content,
    # emit the reasoning text as content so clients always get a usable response.
    # This handles models that wrap everything in <think>...</think> without
    # producing content after the closing tag.
    # Skip if reasoning was already streamed as reasoning_content chunks — the
    # client already has the text and echoing it as content creates duplicates.
    if (
        request_parser
        and not content_was_emitted
        and accumulated_reasoning
        and not reasoning_was_streamed
        and not suppress_reasoning
    ):
        fallback_chunk = ChatCompletionChunk(
            id=response_id,
            created=_created_ts,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        content=accumulated_reasoning,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {_dump_sse_json(fallback_chunk)}\n\n"

    # When reasoning is suppressed and the model produced ONLY reasoning (no content),
    # try to extract tool calls from reasoning before showing the diagnostic.
    # MiniMax M2.7 and similar models may emit tool calls inside <think> blocks
    # when reasoning OFF — the tool call parser needs to see the reasoning text.
    if suppress_reasoning and not content_was_emitted and accumulated_reasoning:
        # Last-chance tool call extraction from reasoning text
        _has_tc_markers = any(m in accumulated_reasoning for m in _TOOL_CALL_MARKERS)
        if _has_tc_markers and not _suppress_tools:
            logger.info(f"Request {response_id}: tool call markers found in suppressed reasoning — extracting")
            _tc_text = accumulated_reasoning.strip()
            _tc_cleaned, _tc_calls = _parse_tool_calls_with_parser(_tc_text, request)
            if _tc_calls:
                # Emit tool calls as if they came from content
                _tc_content = (_tc_cleaned or "").strip()
                if _tc_content:
                    tc_content_chunk = ChatCompletionChunk(
                        id=response_id, created=_created_ts, model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=_tc_content),
                            finish_reason=None,
                        )],
                    )
                    yield f"data: {_dump_sse_json(tc_content_chunk)}\n\n"
                tc_chunk = ChatCompletionChunk(
                    id=response_id, created=_created_ts, model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            tool_calls=[{
                                "index": i, "id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            } for i, tc in enumerate(_tc_calls)],
                        ),
                        finish_reason="tool_calls",
                    )],
                )
                tool_calls_emitted = True
                yield f"data: {_dump_sse_json(tc_chunk)}\n\n"
            else:
                logger.info(
                    f"Request {response_id}: tool markers found in suppressed "
                    "reasoning but parsing failed — preserving hidden reasoning"
                )
                diag_chunk = ChatCompletionChunk(
                    id=response_id, created=_created_ts, model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )],
                )
                yield f"data: {_dump_sse_json(diag_chunk)}\n\n"
        else:
            logger.info(
                f"Request {response_id}: model produced only reasoning ({len(accumulated_reasoning)} chars) — suppressed per user setting"
            )
            diag_chunk = ChatCompletionChunk(
                id=response_id, created=_created_ts, model=request.model,
                choices=[ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )],
            )
            yield f"data: {_dump_sse_json(diag_chunk)}\n\n"

    _stream_chat_warnings = None
    if (
        accumulated_reasoning
        and not content_was_emitted
        and not tool_calls_emitted
        and (suppress_reasoning or reasoning_was_streamed)
    ):
        _stream_chat_warnings = _chat_completion_warnings_for_reasoning_only(
            content=None,
            reasoning=accumulated_reasoning,
            tool_calls=None,
        )
    if _stream_chat_warnings:
        warning_chunk = ChatCompletionChunk(
            id=response_id,
            created=_created_ts,
            model=request.model,
            choices=[],
            warnings=_stream_chat_warnings,
        )
        yield f"data: {_dump_sse_json(warning_chunk)}\n\n"

    # Safeguard: if model generated zero tokens (empty stream), finish the stream
    # without injecting diagnostic prose as assistant output. The warning belongs
    # in logs/diagnostics, not in next-turn model history.
    _zero_tokens = (last_output is None) or (
        not content_was_emitted
        and not accumulated_reasoning
        and getattr(last_output, "completion_tokens", 1) == 0
    )
    if _zero_tokens and not content_was_emitted and not accumulated_reasoning:
        logger.warning(f"Request {response_id}: model generated zero tokens")
        empty_chunk = ChatCompletionChunk(
            id=response_id,
            created=_created_ts,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {_dump_sse_json(empty_chunk)}\n\n"

    # Enforce tool_choice="required" in streaming: emit error SSE if no tool calls
    if getattr(request, "tool_choice", None) == "required" and not tool_calls_emitted:
        # tool_calls_emitted is set True only when tool calls were actually parsed and yielded.
        # tool_call_buffering can be True even on false-positive marker detection with no actual calls.
        logger.warning(
            f"Stream {response_id}: tool_choice='required' but no tool calls produced"
        )
        error_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "error": {
                "message": (
                    "tool_choice='required' was set but the model did not produce "
                    "any tool calls. Try rephrasing your prompt or using a model "
                    "with better tool-calling support."
                ),
                "type": "invalid_request_error",
                "code": "tool_calls_required",
            },
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    # H4: Validate response_format (json_schema/json_object) at end of stream.
    # Streaming can't reject mid-stream, so validate the accumulated output and
    # log a warning. For strict mode, emit an error SSE event.
    _rf = getattr(request, "response_format", None)
    if _rf and content_was_emitted and not tool_call_buffering:
        _final_text = (
            accumulated_content.strip() if request_parser else accumulated_text.strip()
        )
        if _final_text:
            _, parsed_json, is_valid, rf_error = parse_json_output(_final_text, _rf)
            if not is_valid:
                _rf_strict = False
                if isinstance(_rf, dict):
                    _rf_strict = _rf.get("json_schema", {}).get("strict", False)
                elif hasattr(_rf, "json_schema") and _rf.json_schema:
                    _rf_strict = getattr(_rf.json_schema, "strict", False)
                if _rf_strict:
                    logger.warning(
                        f"Stream {response_id}: JSON schema validation failed (strict): {rf_error}"
                    )
                    error_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "error": {
                            "message": f"response_format strict mode: {rf_error}",
                            "type": "invalid_request_error",
                            "code": "json_validation_failed",
                        },
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                else:
                    logger.warning(
                        f"Stream {response_id}: JSON validation failed: {rf_error}"
                    )

    # Send final chunk with usage if requested
    if include_usage:
        _usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if cached_tokens > 0 or cache_detail:
            _usage.prompt_tokens_details = PromptTokensDetails(
                cached_tokens=cached_tokens, cache_detail=cache_detail
            )
        usage_chunk = ChatCompletionChunk(
            id=response_id,
            created=_created_ts,
            model=request.model,
            choices=[],  # Empty choices for usage-only chunk
            usage=_usage,
        )
        yield f"data: {_dump_sse_json(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_responses_api(
    engine: BaseEngine,
    messages: list,
    request: ResponsesRequest,
    fastapi_request: Request | None = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream response in OpenAI Responses API SSE format.

    Streams text deltas incrementally for real-time display. If tool call
    markers are detected mid-stream, switches to buffered mode and emits
    structured function_call events at the end.
    """
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    kwargs["request_id"] = response_id
    seq = 0
    created_at = int(time.time())

    # Check if client wants per-chunk usage reporting
    include_usage = request.stream_options and request.stream_options.include_usage

    def _sse(event_type: str, data: dict) -> str:
        nonlocal seq
        data["sequence_number"] = seq
        seq += 1
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"

    # Emit response.created
    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": request.model,
                "output": [],
            },
        },
    )

    # Create text message output item immediately for streaming
    msg_id = f"item_{uuid.uuid4().hex[:12]}"
    yield _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield _sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    _suppress_tools = getattr(request, "tool_choice", None) == "none"
    _request_has_tools = bool(getattr(request, "tools", None))
    _stream_tools_available = bool(kwargs.get("tools")) or _request_has_tools
    tool_call_active = _stream_tools_available and not _suppress_tools
    tool_call_buffering = False

    full_text = ""
    streamed_text = ""  # Text already sent as deltas (before tool call marker)
    accumulated_content = ""  # Content-only text for tool call marker detection
    accumulated_reasoning = ""  # Reasoning text for fallback
    content_was_emitted = False
    reasoning_was_streamed = False  # Whether reasoning was sent to client as deltas
    prompt_tokens = 0
    completion_tokens = 0
    _cached = 0
    _cache_detail: str | None = None
    last_output = (
        None  # Track last engine output for finish_reason (status: incomplete)
    )

    # Resolve effective enable_thinking:
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    # (None = fall through to template/tokenizer auto-detect below — this site
    # drives SSE parser behavior, not engine kwargs, so no Gemma4+tools override.)
    _ct_kwargs = _merge_ct_kwargs(request.chat_template_kwargs)
    _effective_thinking = _resolve_enable_thinking(
        request_value=request.enable_thinking,
        ct_kwargs=_ct_kwargs,
        tools_present=bool(getattr(request, "tools", None)),
        model_key=_model_path or _model_name or request.model,
        engine=engine,
        auto_detect=True,
    )

    # Reasoning parser setup (mirrors stream_chat_completion)
    from .model_config_registry import get_model_config_registry

    _model_config = get_model_config_registry().lookup(
        _model_path or _model_name or request.model
    )
    think_in_template = _model_config.think_in_template

    # Fallback only for unknown configs. Known families may expose thinking
    # tokens in vocab without opening a reasoning rail in the rendered prompt.
    think_in_template = _apply_tokenizer_thinking_vocab_fallback(
        think_in_template,
        reasoning_parser=_reasoning_parser,
        tokenizer=engine.tokenizer,
        model_config=_model_config,
    )

    # S5 seed detection (mirrors Chat Completions path)
    if think_in_template:
        if _template_completes_thinking(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # When user explicitly disables thinking, check if template actually respects it.
    # The final parser seed below uses the engine's post-render prompt contract.
    if _effective_thinking is False and think_in_template:
        if not _template_always_thinks(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    if _reasoning_parser:
        _prompt_enable = True if _effective_thinking is None else bool(_effective_thinking)
        _tools_present_for_prompt = bool(getattr(request, "tools", None) or kwargs.get("tools"))
        if _engine_prompt_starts_in_reasoning(
            engine.tokenizer,
            model_name=_model_name or request.model,
            enable_thinking=_prompt_enable,
            family_name=getattr(_model_config, "family_name", None),
            tools_present=_tools_present_for_prompt,
        ):
            think_in_template = True
        else:
            think_in_template = False

    if _hy3_prompt_starts_in_reasoning(
        model_key=_model_path or _model_name or request.model,
        enable_thinking=_effective_thinking,
        ct_kwargs=kwargs.get("chat_template_kwargs") or _ct_kwargs,
    ):
        think_in_template = True

    if _synthetic_mllm_think_prompt_starts(
        model_key=_model_path or _model_name or request.model,
        enable_thinking=_effective_thinking,
        engine=engine,
    ):
        think_in_template = True

    # The parser seed must match the final prompt the engine renders. Tool
    # requests may intentionally leave reasoning open; no-tool thinking-off
    # requests may close it before decode.
    effective_think_in_template = think_in_template

    # For thinking models without reasoning parser, prepend <think>
    is_thinking_model = effective_think_in_template and not _reasoning_parser
    think_prefix_sent = False

    # Suppress reasoning output when user explicitly disabled thinking
    suppress_reasoning = _effective_thinking is False

    # Create a per-request parser instance to avoid mutable-state conflicts
    # when multiple requests stream concurrently.
    _harmony_prefix_active = "prompt_suffix" in kwargs and kwargs.get(
        "prompt_suffix", ""
    ).startswith("<|start|>assistant<|channel|>analysis")
    request_parser = None
    if _reasoning_parser:
        request_parser = _reasoning_parser.__class__()
        request_parser.reset_state(
            think_in_prompt=effective_think_in_template,
            harmony_active=_harmony_prefix_active,
        )
        logger.debug(
            f"[responses] Reasoning parser: {type(request_parser).__name__} (think_in_template={effective_think_in_template}, suppress={suppress_reasoning}, harmony_active={_harmony_prefix_active})"
        )
    else:
        logger.debug("[responses] No reasoning parser active for this request")

    # Use per-request timeout if provided, otherwise server default
    _stream_timeout = (
        request.timeout if request.timeout is not None else _default_timeout
    )

    if (
        _responses_exact_reply_target(request)
        and _responses_messages_have_tool_result_after_latest_user(messages)
    ):
        try:
            output = await _await_chat_with_disconnect_abort(
                engine,
                messages=messages,
                chat_kwargs=kwargs,
                timeout=_stream_timeout,
                fastapi_request=fastapi_request,
                request_id=response_id,
                endpoint="Responses API streaming exact-reply finalization",
            )
        except PromptTooLongError as e:
            if hasattr(engine, "abort_request"):
                await engine.abort_request(response_id)
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": str(e),
                        "code": "prompt_too_long",
                    },
                },
            )
            return
        except VLMImagePrefillBudgetError as e:
            if hasattr(engine, "abort_request"):
                await engine.abort_request(response_id)
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": str(e),
                        "code": VLMImagePrefillBudgetError.code,
                    },
                },
            )
            return
        except Exception as e:
            logger.error(
                "Responses API streaming exact-reply finalization failed: %s",
                e,
                exc_info=True,
            )
            if hasattr(engine, "abort_request"):
                await engine.abort_request(response_id)
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "message": f"Generation failed: {type(e).__name__}: {e}",
                        "code": "internal_error",
                    },
                },
            )
            return

        display_text = _responses_fast_path_visible_text(output, request)
        if display_text:
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": display_text,
                },
            )
        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": display_text,
            },
        )
        yield _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": display_text,
                    "annotations": [],
                },
            },
        )
        output_item = {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": display_text, "annotations": []}
            ],
        }
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": output_item,
            },
        )
        _resp_finish = getattr(output, "finish_reason", None)
        _resp_status = "incomplete" if _resp_finish == "length" else "completed"
        _resp_extra: dict = {}
        if _resp_status == "incomplete":
            _resp_extra["incomplete_details"] = {"reason": "max_output_tokens"}
        all_output_items = [output_item] if display_text else []
        usage_obj = _get_responses_usage(output).model_dump(exclude_none=True)
        completed_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": _resp_status,
            "model": request.model,
            "output_text": display_text,
            "output": all_output_items,
            **_resp_extra,
            "usage": usage_obj,
        }
        _responses_store_history(
            response_id,
            messages + _responses_output_to_assistant_messages(all_output_items),
            reasoning_only=False,
        )
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": completed_response,
            },
        )
        return

    try:
        async for output in _stream_with_keepalive(
            engine.stream_chat(messages=messages, **kwargs),
            total_timeout=_stream_timeout,
        ):
            # Keep-alive sentinel — emit SSE comment to prevent connection timeout
            if output is None:
                yield ": keep-alive\n\n"
                continue

            if fastapi_request and await fastapi_request.is_disconnected():
                logger.info(f"Client disconnected, aborting request {response_id}")
                if hasattr(engine, "abort_request"):
                    await engine.abort_request(response_id)
                break

            last_output = output
            delta_text = output.new_text

            # Track token counts from output BEFORE any continue statements
            # (must run unconditionally — reasoning parser's continue skips
            # code below, so token tracking must happen first)
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            _cached = getattr(output, "cached_tokens", 0)
            _detail_chunk = getattr(output, "cache_detail", "") or None
            if _detail_chunk is not None:
                _cache_detail = _detail_chunk

            if delta_text:
                full_text += delta_text

                # Use reasoning parser if enabled
                if request_parser:
                    previous_text = (
                        full_text[: -len(delta_text)] if delta_text else full_text
                    )
                    delta_msg = request_parser.extract_reasoning_streaming(
                        previous_text, full_text, delta_text
                    )

                    if delta_msg is None:
                        # Skip this chunk entirely (e.g., <think> token itself)
                        # Must use continue (not pass) to avoid falling through
                        # to token tracking and usage emission below.
                        continue
                    else:
                        # Post-parse cleaning — mirror the chat_completions
                        # streaming path so Responses API clients don't see
                        # residual `<channel|>` / `thought\n` / `<turn|>` etc.
                        # that the parser's split leaves in content. Use the
                        # delta-safe stripper that preserves whitespace —
                        # clean_output_text's trailing `.strip()` eats per-
                        # delta leading spaces and concatenates the stream
                        # into " Itlooksliketypo" garbage.
                        if delta_msg.content:
                            delta_msg.content = strip_marker_tokens_delta(delta_msg.content)
                        if delta_msg.reasoning:
                            delta_msg.reasoning = strip_marker_tokens_delta(delta_msg.reasoning)
                        # Accumulate for marker detection (before buffering check).
                        # Suppressed reasoning stays internal and must not be
                        # mirrored into visible output_text/history.
                        if delta_msg.content:
                            accumulated_content += delta_msg.content
                        if delta_msg.reasoning:
                            accumulated_reasoning += delta_msg.reasoning

                        # Check for tool call markers in content and reasoning
                        if tool_call_active and not tool_call_buffering:
                            if delta_msg.content and accumulated_content:
                                tool_call_buffering = _has_tool_marker_or_partial_suffix(
                                    accumulated_content
                                )
                            if not tool_call_buffering and delta_msg.reasoning:
                                _reasoning_tail = (
                                    accumulated_reasoning[-30:]
                                    if len(accumulated_reasoning) > 30
                                    else accumulated_reasoning
                                )
                                tool_call_buffering = _has_tool_marker_or_partial_suffix(
                                    _reasoning_tail
                                )
                            # GPT-OSS/Harmony native tool format: to=<name> code{...}
                            if not tool_call_buffering:
                                _tc_check = (
                                    accumulated_content
                                    or (
                                        accumulated_reasoning[-30:]
                                        if accumulated_reasoning
                                        else ""
                                    )
                                    or ""
                                )
                                if re.search(r"\bto=\w[\w.]*\s+code\{", _tc_check):
                                    tool_call_buffering = True

                        if tool_call_buffering:
                            # Emit heartbeat during tool call buffering so the client
                            # sees activity (matches ChatCompletion heartbeat behavior).
                            yield _sse(
                                "response.heartbeat",
                                {
                                    "type": "response.heartbeat",
                                    "tool_call_generating": True,
                                },
                            )
                            continue

                        if not tool_call_buffering:
                            # Suppressed reasoning is never redirected into
                            # visible output_text. Boundary deltas can still
                            # carry post-</think> content, which remains
                            # visible.
                            if suppress_reasoning:
                                emit_content = delta_msg.content
                                emit_reasoning = None
                                if not emit_content and not output.finished:
                                    yield _sse(
                                        "response.in_progress",
                                        {"type": "response.in_progress", "output_index": 0},
                                    )
                                    continue
                            else:
                                emit_reasoning = delta_msg.reasoning
                                emit_content = delta_msg.content

                            if _suppress_tools and emit_content:
                                emit_content = _suppressed_tool_display_delta(
                                    accumulated_content,
                                    streamed_text,
                                    request,
                                )
                            elif emit_content:
                                emit_content = _visual_grounding_display_delta(
                                    accumulated_content,
                                    streamed_text,
                                )

                            # Emit reasoning as OpenAI Responses reasoning-summary events.
                            if emit_reasoning:
                                reasoning_was_streamed = True
                                yield _sse(
                                    "response.reasoning_summary_text.delta",
                                    {
                                        "type": "response.reasoning_summary_text.delta",
                                        "item_id": msg_id,
                                        "output_index": 0,
                                        "summary_index": 0,
                                        "delta": emit_reasoning,
                                    },
                                )
                            # Emit content as standard text delta
                            if emit_content:
                                content_was_emitted = True
                                streamed_text += emit_content
                                yield _sse(
                                    "response.output_text.delta",
                                    {
                                        "type": "response.output_text.delta",
                                        "item_id": msg_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "delta": emit_content,
                                    },
                                )
                else:
                    # Standard path without reasoning parsing
                    content = delta_text

                    # Check for tool call markers
                    if tool_call_active and not tool_call_buffering:
                        tool_call_buffering = _has_tool_marker_or_partial_suffix(
                            full_text
                        )

                    if tool_call_buffering:
                        # Emit heartbeat during tool call buffering (standard path)
                        yield _sse(
                            "response.heartbeat",
                            {
                                "type": "response.heartbeat",
                                "tool_call_generating": True,
                            },
                        )
                        continue
                    else:
                        # Add <think> prefix on first chunk for thinking models
                        if is_thinking_model and not think_prefix_sent and content:
                            content = "<think>" + content
                            think_prefix_sent = True

                        if _suppress_tools and content:
                            content = _suppressed_tool_display_delta(
                                full_text,
                                streamed_text,
                                request,
                            )
                        elif content:
                            content = _visual_grounding_display_delta(
                                full_text,
                                streamed_text,
                            )

                        if content:
                            content_was_emitted = True
                            streamed_text += content
                            yield _sse(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": content,
                                },
                            )

            # Emit per-chunk usage when include_usage is enabled (for real-time metrics)
            if include_usage and (prompt_tokens or completion_tokens):
                usage_obj = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                if _cached > 0 or _cache_detail:
                    _ictd = {"cached_tokens": _cached}
                    if _cache_detail:
                        _ictd["cache_detail"] = _cache_detail
                    usage_obj["input_tokens_details"] = _ictd
                yield _sse(
                    "response.usage",
                    {
                        "type": "response.usage",
                        "usage": usage_obj,
                    },
                )
    except PromptTooLongError as e:
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e),
                    "code": "prompt_too_long",
                },
            },
        )
    except VLMImagePrefillBudgetError as e:
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e),
                    "code": VLMImagePrefillBudgetError.code,
                },
            },
        )
    except Exception as e:
        # On any error, abort the request and emit an error SSE event
        # so the client knows what happened instead of a silent EOF.
        logger.error(f"Stream error for {response_id}: {e}", exc_info=True)
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": f"Stream generation failed: {e}",
                    "code": "internal_error",
                },
            },
        )
        # M8: Include partial usage in failed response so client gets token counts
        _err_usage = {}
        if prompt_tokens > 0 or completion_tokens > 0:
            _err_usage = {
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "failed",
                    "error": {"type": "server_error", "message": str(e)},
                    **_err_usage,
                },
            },
        )
        return

    # Emit reasoning summary done event if reasoning was produced (skip when suppressed)
    if accumulated_reasoning and not suppress_reasoning:
        yield _sse(
            "response.reasoning_summary_text.done",
            {
                "type": "response.reasoning_summary_text.done",
                "item_id": msg_id,
                "output_index": 0,
                "summary_index": 0,
                "text": accumulated_reasoning,
            },
        )

    # Build output items list for the completed response
    all_output_items = []
    output_index = 0

    # Parse tool calls from the accumulated text (skip when tool_choice="none").
    # Strip think tags before parsing — reasoning text may precede tool calls
    # (mirrors the Chat Completions streaming path at lines 2468-2475).
    tool_calls = None
    cleaned_text = full_text
    if not _suppress_tools:
        # Use content-only text when reasoning parser separated it (avoids losing
        # tool calls that appear inside <think> blocks during regex stripping).
        # If content is empty but reasoning has tool markers, check reasoning too.
        if request_parser and accumulated_content.strip():
            parse_text = accumulated_content.strip()
        elif request_parser and accumulated_reasoning.strip():
            parse_text = accumulated_reasoning.strip()
        else:
            parse_text = _strip_think_for_tool_parse(full_text)
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(
            parse_text or full_text, request
        )
    else:
        cleaned_text = _clean_suppressed_tool_markup_for_display(
            full_text,
            request,
        )

    if (
        not tool_calls
        and suppress_reasoning
        and accumulated_reasoning
        and not _suppress_tools
        and any(m in accumulated_reasoning for m in _TOOL_CALL_MARKERS)
    ):
        logger.info(
            f"Request {response_id}: tool markers in suppressed reasoning — "
            "extracting before Responses API finalization"
        )
        _tc_cleaned, _tc_calls = _parse_tool_calls_with_parser(
            accumulated_reasoning.strip(), request
        )
        if _tc_calls:
            cleaned_text = (_tc_cleaned or "").strip()
            tool_calls = _tc_calls

    display_text = ""

    if tool_calls:
        # Apply reasoning parser to the cleaned (pre-tool-call) text
        if request_parser and cleaned_text:
            reasoning_text, content_text = request_parser.extract_reasoning(
                cleaned_text
            )
            if content_text:
                cleaned_text = content_text
            elif reasoning_text:
                cleaned_text = reasoning_text

        # Finalize the text message with whatever content was before the tool call
        final_text = (cleaned_text or "").strip()
        final_text = _finalize_visible_text_for_request(final_text, request)
        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "text": final_text,
            },
        )
        yield _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": final_text, "annotations": []},
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_text, "annotations": []}
                    ],
                },
            },
        )
        if final_text:
            all_output_items.append(
                {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_text, "annotations": []}
                    ],
                }
            )
            output_index += 1

        # Emit each tool call as a function_call output item
        for tc in tool_calls:
            func = tc.function if hasattr(tc, "function") else tc
            tc_name = func.name if hasattr(func, "name") else func.get("name", "")
            tc_args = (
                func.arguments
                if hasattr(func, "arguments")
                else func.get("arguments", "")
            )
            call_id = tc.id if hasattr(tc, "id") else f"call_{uuid.uuid4().hex[:8]}"
            fc_id = f"fc_{uuid.uuid4().hex[:12]}"

            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": fc_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": call_id,
                        "name": tc_name,
                        "arguments": "",
                    },
                },
            )
            # Emit arguments incrementally in ~16-char chunks per OpenAI spec
            _ARG_CHUNK = 16
            if tc_args:
                for _ci in range(0, len(tc_args), _ARG_CHUNK):
                    yield _sse(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": fc_id,
                            "output_index": output_index,
                            "delta": tc_args[_ci : _ci + _ARG_CHUNK],
                        },
                    )
            else:
                # Empty arguments — emit one delta to satisfy clients
                yield _sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": fc_id,
                        "output_index": output_index,
                        "delta": "",
                    },
                )
            yield _sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": fc_id,
                    "output_index": output_index,
                    "arguments": tc_args,
                },
            )
            fc_item = {
                "id": fc_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": tc_name,
                "arguments": tc_args,
            }
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": fc_item,
                },
            )
            all_output_items.append(fc_item)
            output_index += 1
    else:
        # No tool calls — use content accumulated during streaming (reasoning already separated)
        display_text = cleaned_text or ""
        if request_parser:
            # Reasoning was already emitted during streaming — use content-only text
            if accumulated_content:
                display_text = accumulated_content
            elif (
                accumulated_reasoning
                and not reasoning_was_streamed
                and not suppress_reasoning
            ):
                # Edge case: reasoning was accumulated but NEVER streamed as
                # deltas (e.g., parser produced full block at end without
                # incremental events). Use as fallback. NEVER falls through
                # here when reasoning_was_streamed=True — that would be the
                # B1 bug: reasoning leaks into output_text →
                # client sees internal thinking as visible answer →
                # next-turn history pollution).
                display_text = accumulated_reasoning
            elif display_text:
                # Fallback: re-parse if streaming didn't accumulate
                reasoning_text, content_text = request_parser.extract_reasoning(
                    display_text
                )
                if content_text:
                    display_text = content_text
                elif reasoning_text and not reasoning_was_streamed:
                    # Only fall back to reasoning if it
                    # wasn't already streamed. Otherwise we double-emit and
                    # pollute next-turn history.
                    display_text = reasoning_text
                else:
                    # Reasoning streamed + no content extractable → empty
                    # final text. Caller (panel) should fall back to
                    # reasoning_content for display, not output_text.
                    display_text = ""

        if not display_text:
            # If reasoning was streamed and we have no
            # content, keep display_text EMPTY rather than falling back to
            # full_text (which contains the reasoning text). Empty
            # output_text is the correct signal — client renders the
            # already-streamed reasoning_content. Falling back to full_text
            # here was the bug that pollutes history.
            if not reasoning_was_streamed:
                display_text = clean_output_text(full_text) if full_text else ""
        if not display_text:
            # If reasoning was suppressed and model produced reasoning, the response
            # is intentionally empty. Report it as a diagnostic warning, not as
            # assistant output that would pollute user-visible text/history.
            if suppress_reasoning and accumulated_reasoning:
                display_text = ""
                logger.info(
                    f"Request {response_id}: model produced only reasoning ({len(accumulated_reasoning)} chars) — suppressed per user setting"
                )
                yield _sse(
                    "response.warning",
                    {
                        "type": "response.warning",
                        "code": "reasoning_only_no_content",
                        "message": (
                            "The model produced only internal reasoning, and "
                            "reasoning was suppressed for this request."
                        ),
                    },
                )
            elif reasoning_was_streamed:
                # Reasoning-only completion is now intentional (B1 fix).
                # Empty display_text means the panel should render the
                # reasoning_content that was already streamed.
                display_text = ""
                logger.info(
                    f"Request {response_id}: reasoning-only completion "
                    f"({len(accumulated_reasoning)} chars reasoning, no content) — "
                    f"empty output_text is correct; client renders reasoning_content"
                )
            else:
                display_text = ""
                logger.warning(
                    f"Request {response_id}: empty response in Responses API; "
                    "leaving output_text empty instead of injecting a diagnostic assistant message"
                )
                yield _sse(
                    "response.warning",
                    {
                        "type": "response.warning",
                        "code": "empty_model_response",
                        "message": (
                            "The model produced no visible response. Check server logs "
                            "for generation details."
                        ),
                    },
                )

        if _suppress_tools and display_text:
            display_text = _clean_suppressed_tool_markup_for_display(
                display_text,
                request,
            )
        if display_text:
            display_text = _finalize_visible_text_for_request(display_text, request)

        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "text": display_text,
            },
        )
        yield _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": display_text,
                    "annotations": [],
                },
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": display_text, "annotations": []}
                    ],
                },
            },
        )
        all_output_items.append(
            {
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": display_text, "annotations": []}
                ],
            }
        )

    # H4: Validate text format (json_schema/json_object) at end of stream.
    _text_fmt = getattr(request, "text", None)
    # Convert Pydantic model to dict for uniform access
    if _text_fmt and hasattr(_text_fmt, "model_dump"):
        _text_fmt = _text_fmt.model_dump(exclude_none=True)
    if (
        _text_fmt
        and isinstance(_text_fmt, dict)
        and _text_fmt.get("type") not in (None, "text")
    ):
        _rf_type = _text_fmt.get("type", "")
        if (
            _rf_type in ("json_schema", "json_object")
            and display_text
        ):
            _, _pj, _valid, _err = parse_json_output(display_text, _text_fmt)
            if not _valid:
                _strict = _text_fmt.get("json_schema", {}).get("strict", False)
                if _strict:
                    logger.warning(
                        f"Stream {response_id}: JSON schema validation failed (strict): {_err}"
                    )
                    yield _sse(
                        "error",
                        {
                            "type": "error",
                            "message": f"response_format strict mode: {_err}",
                            "code": "json_validation_failed",
                        },
                    )
                else:
                    logger.warning(
                        f"Stream {response_id}: JSON validation failed: {_err}"
                    )

    # Enforce tool_choice="required" in Responses API streaming
    _resp_stream_tc = getattr(request, "tool_choice", None)
    if _resp_stream_tc == "required" and not tool_calls:
        logger.warning(
            f"Stream {response_id}: tool_choice='required' but no tool calls produced"
        )
        yield _sse(
            "error",
            {
                "type": "error",
                "message": (
                    "tool_choice='required' was set but the model did not produce "
                    "any tool calls. Try rephrasing your prompt or using a model "
                    "with better tool-calling support."
                ),
                "code": "tool_calls_required",
            },
        )

    # Emit response.completed — use "incomplete" status when max_tokens was hit
    _resp_finish = getattr(last_output, "finish_reason", None) if last_output else None
    _resp_status = "incomplete" if _resp_finish == "length" else "completed"
    _resp_extra: dict = {}
    if _resp_status == "incomplete":
        _resp_extra["incomplete_details"] = {"reason": "max_output_tokens"}
    if accumulated_reasoning and not suppress_reasoning:
        all_output_items.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:12]}",
                "type": "reasoning",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "reasoning", "text": accumulated_reasoning}],
            }
        )

    # Reasoning-only detection mirrors the non-stream path. Empty streaming
    # output_text shells do not count as visible output.
    _stream_reasoning_only = _responses_output_is_reasoning_only(all_output_items)
    _stream_chain_warnings = _chain_warnings_for_previous_response_id(
        getattr(request, "previous_response_id", None)
    )
    _stream_warnings = _merge_responses_warnings(
        _stream_chain_warnings,
        _current_response_warnings_for_reasoning_only(_stream_reasoning_only),
    )

    completed_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": _resp_status,
        "model": request.model,
        "output_text": display_text,
        "output": all_output_items,
        **_resp_extra,
        **({"warnings": _stream_warnings} if _stream_warnings else {}),
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            **(
                {
                    "input_tokens_details": (
                        {"cached_tokens": _cached, "cache_detail": _cache_detail}
                        if _cache_detail
                        else {"cached_tokens": _cached}
                    )
                }
                if _cached > 0 or _cache_detail
                else {}
            ),
        },
    }
    _responses_store_history(
        response_id,
        messages + _responses_output_to_assistant_messages(all_output_items),
        reasoning_only=_stream_reasoning_only,
    )
    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": completed_response,
        },
    )


# =============================================================================
# MCP Initialization
# =============================================================================


def _discover_mcp_config_for_startup() -> str | None:
    """Return an MCP config path worth starting, or None.

    ``load_mcp_config(None)`` already implements the expected search order:
    env var, cwd ``mcp.json``/``mcp.yaml``, then the user config directory.
    Startup should use that path when it declares servers, but it should not
    initialize the MCP manager for an absent or empty config.
    """
    explicit_env_config = False
    try:
        from pathlib import Path

        from vmlx_engine.mcp.config import (
            CONFIG_ENV_VAR,
            _find_config_file,
            load_mcp_config,
        )

        env_config = os.environ.get(CONFIG_ENV_VAR)
        explicit_env_config = bool(env_config)
        if explicit_env_config:
            config_path = Path(env_config).expanduser()
            config = load_mcp_config(config_path)
        else:
            config = load_mcp_config(None)
            config_path = _find_config_file(None)
            if config_path is None:
                return None

        config_path = Path(config_path).expanduser().resolve()
        if not config.servers:
            logger.info("MCP config discovered at %s but contains no servers", config_path)
            return None
        return str(config_path)
    except FileNotFoundError:
        if explicit_env_config:
            raise
        return None
    except Exception as e:
        if explicit_env_config:
            raise
        logger.error("Failed to discover MCP config — continuing without MCP: %s", e)
        return None


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager, _mcp_policy

    try:
        from vmlx_engine.mcp import MCPClientManager, load_mcp_config

        config = load_mcp_config(config_path)
        _mcp_policy = _load_mcp_policy_from_env()
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        logger.info(
            f"MCP initialized with {len(_mcp_manager.get_all_tools(policy=_mcp_policy))} effective tools"
        )

    except ImportError:
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        raise


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="vmlx-engine OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vmlx_engine.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vmlx_engine.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vmlx_engine.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )
    parser.add_argument(
        "--continuous-batching",
        dest="continuous_batching",
        action="store_true",
        default=True,
        help="Enable continuous batching/cache-stack path (default: enabled)",
    )
    parser.add_argument(
        "--no-continuous-batching",
        dest="continuous_batching",
        action="store_false",
        help="Use SimpleEngine and disable cache-stack features that require batching",
    )
    # Smelt options
    parser.add_argument(
        "--smelt",
        action="store_true",
        help="Enable MoE Smelt mode (partial expert loading for JANG MoE models)",
    )
    parser.add_argument(
        "--smelt-experts",
        type=int,
        default=50,
        help="Percentage of experts to load in Smelt mode (default: 50)",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--mcp-enabled-servers",
        type=str,
        default=None,
        help="Comma-separated MCP server allow-list for this model session",
    )
    parser.add_argument(
        "--mcp-disabled-servers",
        type=str,
        default=None,
        help="Comma-separated MCP server deny-list for this model session",
    )
    parser.add_argument(
        "--mcp-enabled-tools",
        type=str,
        default=None,
        help="Comma-separated MCP tool allow-list for this model session",
    )
    parser.add_argument(
        "--mcp-disabled-tools",
        type=str,
        default=None,
        help="Comma-separated MCP tool deny-list for this model session",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_FALLBACK_MAX_OUTPUT_TOKENS,
        help=(
            "Default max tokens for generation when explicitly supplied. "
            "If omitted, request > bundle max_new_tokens > bounded engine fallback applies."
        ),
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Maximum prompt/context tokens accepted before prefill. "
        "If omitted, vMLX uses the automatic memory-safe prompt limit.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=["auto", "none"] + reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Use 'auto' to detect from model name, or 'none' to disable explicitly. "
            f"Options: auto, none, {', '.join(reasoning_choices)}."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/all-MiniLM-L6-v2-4bit)",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default temperature for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top_p for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Default top_k for generation when not specified in request",
    )
    parser.add_argument(
        "--default-min-p",
        type=float,
        default=None,
        help="Default min_p for generation when not specified in request",
    )
    parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help=(
            "Default repetition penalty when not specified in request. "
            "1.0 = no penalty, higher = less repetition. "
            "Without this flag, external API clients (curl, Ollama, OpenAI SDK, "
            "Anthropic SDK) use bundle metadata when present; otherwise no "
            "server-wide repetition penalty is applied."
        ),
    )
    parser.add_argument(
        "--native-mtp-depth",
        type=int,
        default=None,
        help="Depth for native in-model MTP heads on preserved-MTP bundles (1-3).",
    )
    parser.add_argument(
        "--native-mtp-sampling-policy",
        choices=["compatible-only", "deterministic-defaults"],
        default="compatible-only",
        help="Native MTP request policy for startup diagnostics and app defaults.",
    )
    parser.add_argument(
        "--disable-native-mtp",
        action="store_true",
        default=False,
        help="Disable native in-model MTP even when bundle metadata is available.",
    )
    parser.add_argument(
        "--omni-backend",
        choices=["stage1", "stage2"],
        default=None,
        help="Nemotron-Omni multimodal backend. stage1 is correctness-first "
        "PyTorch/MPS; stage2 opts into native MLX RADIO + Parakeet via "
        "VMLINUX_OMNI_BACKEND=stage2.",
    )
    parser.add_argument(
        "--default-enable-thinking",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Server-level default for enable_thinking when not specified in request. "
        "Without this, models with reasoning parsers default to thinking ON. "
        "Use 'false' to disable thinking for external API clients (OpenCode, etc.)",
    )
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable automatic tool choice for supported models",
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        help="Tool call parser to use (auto, hermes, qwen, llama, step3p5, mistral, etc.)",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Custom model name exposed via the API (overrides actual model name in /v1/models)",
    )

    max_tokens_explicit = _argv_has_option(sys.argv[1:], "--max-tokens")
    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _default_temperature, _default_top_p, _default_top_k, _default_min_p, _default_repetition_penalty, _default_enable_thinking
    global _native_mtp_sampling_policy
    global _inference_endpoints, _wake_timeout
    global _smelt_enabled, _smelt_experts

    _api_key = args.api_key or os.environ.get("VLLM_API_KEY")
    _default_timeout = args.timeout
    if getattr(args, "inference_endpoints", None):
        _inference_endpoints = [
            e.strip() for e in args.inference_endpoints.split(",") if e.strip()
        ]
    if getattr(args, "wake_timeout", None) is not None:
        _wake_timeout = args.wake_timeout

    if getattr(args, "smelt", False):
        _smelt_enabled = True
        _smelt_experts = getattr(args, "smelt_experts", 50)

    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p
    if getattr(args, "default_top_k", None) is not None:
        _default_top_k = args.default_top_k
    if getattr(args, "default_min_p", None) is not None:
        _default_min_p = args.default_min_p
    if getattr(args, "default_repetition_penalty", None) is not None:
        _default_repetition_penalty = args.default_repetition_penalty
    if getattr(args, "disable_native_mtp", False):
        os.environ["VMLINUX_NATIVE_MTP"] = "0"
    if getattr(args, "native_mtp_depth", None) is not None:
        os.environ["VMLINUX_NATIVE_MTP_DEPTH"] = str(max(1, min(3, int(args.native_mtp_depth))))
    _native_mtp_sampling_policy = getattr(args, "native_mtp_sampling_policy", "compatible-only")
    os.environ["VMLINUX_NATIVE_MTP_SAMPLING_POLICY"] = _native_mtp_sampling_policy
    if _native_mtp_sampling_policy == "deterministic-defaults":
        if _default_temperature is None:
            _default_temperature = 0.0
        if _default_top_p is None:
            _default_top_p = 1.0
        if _default_top_k is None:
            _default_top_k = 0
        if _default_min_p is None:
            _default_min_p = 0.0
    if getattr(args, "omni_backend", None):
        os.environ["VMLINUX_OMNI_BACKEND"] = args.omni_backend
    if args.default_enable_thinking is not None:
        _default_enable_thinking = args.default_enable_thinking == "true"

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config
    if args.mcp_enabled_servers:
        os.environ["VLLM_MLX_MCP_ENABLED_SERVERS"] = args.mcp_enabled_servers
    if args.mcp_disabled_servers:
        os.environ["VLLM_MLX_MCP_DISABLED_SERVERS"] = args.mcp_disabled_servers
    if args.mcp_enabled_tools:
        os.environ["VLLM_MLX_MCP_ENABLED_TOOLS"] = args.mcp_enabled_tools
    if args.mcp_disabled_tools:
        os.environ["VLLM_MLX_MCP_DISABLED_TOOLS"] = args.mcp_disabled_tools

    # Auto-detect reasoning and tool parsers from model name
    from .model_config_registry import get_model_config_registry

    model_name = args.model or ""
    registry = get_model_config_registry()

    # Initialize reasoning parser (auto-detect or explicit)
    global _reasoning_parser
    from .reasoning import get_parser

    parser_name = args.reasoning_parser
    # Auto-detect from model config if not explicitly set
    if not parser_name:
        detected = registry.get_reasoning_parser(model_name)
        if detected:
            parser_name = detected
            logger.info(
                f"Auto-detected reasoning parser: {parser_name} (from model config)"
            )
    elif parser_name == "none":
        # Explicitly disabled by user
        parser_name = None
        logger.info("Reasoning parser explicitly disabled via --reasoning-parser none")
    elif parser_name == "auto":
        detected = registry.get_reasoning_parser(model_name)
        if detected:
            parser_name = detected
            logger.info(
                f"Auto-detected reasoning parser: {parser_name} (from model name)"
            )
        else:
            logger.info("Reasoning parser 'auto': no parser detected for this model")
            parser_name = None

    if parser_name:
        try:
            parser_cls = get_parser(parser_name)
            _reasoning_parser = parser_cls()
            logger.info(f"Reasoning parser enabled: {parser_name}")
        except KeyError:
            logger.warning(
                f"Reasoning parser '{parser_name}' not found. "
                f"Available: {list_parsers()}"
            )

    # Initialize tool call parser from CLI args
    global _tool_call_parser, _enable_auto_tool_choice
    if args.enable_auto_tool_choice:
        _enable_auto_tool_choice = True
        _tool_call_parser = args.tool_call_parser or "auto"
        if _tool_call_parser == "none":
            # Explicitly disabled by user
            _enable_auto_tool_choice = False
            _tool_call_parser = None
            logger.info("Tool calling explicitly disabled via --tool-call-parser none")
        elif _tool_call_parser == "auto":
            detected_tool = registry.get_tool_parser(model_name)
            if detected_tool:
                _tool_call_parser = detected_tool
                logger.info(
                    f"Auto-detected tool call parser: {_tool_call_parser} (from model name)"
                )
            else:
                logger.info(
                    "Tool call parser 'auto': no specific parser detected, using generic"
                )
                _tool_call_parser = None
        if _tool_call_parser:
            logger.info(f"Tool calling enabled (parser: {_tool_call_parser})")

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=args.continuous_batching,
        max_tokens=args.max_tokens,
        max_tokens_explicit=max_tokens_explicit,
        max_prompt_tokens=args.max_prompt_tokens,
        force_mllm=args.mllm,
        served_model_name=args.served_model_name,
        smelt=getattr(args, "smelt", False),
        smelt_experts=getattr(args, "smelt_experts", 50),
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
