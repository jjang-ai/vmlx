# SPDX-License-Identifier: Apache-2.0
# MLX Studio — eric@mlx.studio — Created by Jinho Jang
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
import asyncio
import base64
import json
import logging
import os
import platform
import re
import secrets
import subprocess
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine
from .reasoning.gptoss_parser import GptOssReasoningParser
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: BaseEngine | None = None
_standby_state: str | None = None  # None = active, 'soft' = caches cleared, 'deep' = model unloaded
_pre_sleep_cache_limit: int | None = None  # Saved cache limit before soft sleep
_cli_args: dict = {}  # Saved CLI args for model reload on wake
_wake_lock: asyncio.Lock | None = None  # Lazy-init mutex for JIT wake (prevents double load_model)
_model_name: str | None = None
_model_path: str | None = None  # Full local path for config.json lookups
_served_model_name: str | None = None  # Custom name for API (--served-model-name)
_default_max_tokens: int = 32768
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_default_enable_thinking: bool | None = None  # Set via --default-enable-thinking
_last_request_time: float = 0.0  # Epoch timestamp of last API request (for idle sleep timer)
_model_load_error: str | None = None  # Surfaced via /health when model fails to load

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_temperature is not None:
        return _default_temperature
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_top_p is not None:
        return _default_top_p
    return _FALLBACK_TOP_P


# Global MCP manager
_mcp_manager = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used
_embedding_lock: asyncio.Lock | None = None  # Lazy-init to avoid binding to wrong event loop

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

# Reasoning parser (for models like Qwen3, DeepSeek-R1)
_reasoning_parser = None  # ReasoningParser instance when enabled

# Cache: does a model's template inject <think> even when enable_thinking=False?
# Some templates (e.g., MiniMax M2.5) unconditionally inject <think> regardless.
_template_always_thinks_cache: dict[str, bool] = {}

# Tool call markers to detect in streaming output for buffering
_TOOL_CALL_MARKERS = [
    "<tool_call>",
    "<|tool_call|>",
    "[TOOL_CALLS]",
    "<function=",
    "<minimax:tool_call>",
    "[Calling tool:",
    "<|recipient|>",
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "<\uff5ctool\u2581calls\u2581begin\uff5c>",  # DeepSeek Unicode variant (U+FF5C, U+2581)
]


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
                test_msgs, enable_thinking=False,
                add_generation_prompt=True, tokenize=False,
            )
        except TypeError:
            # Tokenizer doesn't accept enable_thinking — render without it
            # and check if <think> is always present (it's an always-thinking template)
            rendered = tokenizer.apply_chat_template(
                test_msgs, add_generation_prompt=True, tokenize=False,
            )
        # Check if an UNCLOSED <think> appears after the user message.
        # Templates that output <think></think> (empty, closed) ARE honoring
        # enable_thinking=False — the model won't reason. Only flag templates
        # that inject an open <think> without immediate </think> (those truly
        # ignore the flag and the model will reason regardless).
        after_user = rendered.rsplit("__test__", 1)[-1]
        has_think = "<think>" in after_user
        # Strip <think></think> (empty closed block) — that's a proper "no think" signal
        cleaned = after_user.replace("<think></think>", "")
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
                test_msgs, enable_thinking=True,
                add_generation_prompt=True, tokenize=False,
            )
        except TypeError:
            # Tokenizer doesn't accept enable_thinking — render without it
            rendered = tokenizer.apply_chat_template(
                test_msgs, add_generation_prompt=True, tokenize=False,
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

# reasoning_effort → token budget mapping (mirrors OpenAI o-series behavior)
_EFFORT_THINKING_BUDGET = {"low": 1024, "medium": 8192, "high": 32768}
_EFFORT_MAX_TOKENS = {"low": 4096, "medium": 16384, "high": 32768}


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
                logger.info(f"Applied custom chat template for {_mc.family_name} model (batched)")
        except Exception as e:
            logger.warning(f"Failed to apply custom chat template (batched): {e}")

    # Apply JIT compilation for BatchedEngine (which just started above).
    # SimpleEngine JIT is applied in load_model() where it starts synchronously.
    if _enable_jit and _engine is not None:
        _apply_jit_compilation()

    # Initialize MCP if config provided — failure should not crash server
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        try:
            await init_mcp(mcp_config)
        except Exception as e:
            logger.error(f"Failed to initialize MCP — continuing without tool support: {e}")

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
            _image_gen.unload()
            logger.info("Image engine unloaded")
        except Exception as e:
            logger.warning(f"Image engine shutdown error (continuing): {e}")
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
    global _last_request_time, _standby_state
    path = request.url.path
    # Identify actual inference endpoints (not metadata like /v1/models, /v1/cache, /v1/mcp/tools)
    # This list drives BOTH idle timer reset AND JIT wake
    # Cancel endpoints should NOT trigger JIT wake or reset idle timer —
    # they are lightweight checks, not inference requests.
    is_cancel = "/cancel" in path
    is_inference = not is_cancel and any(path.startswith(p) for p in [
        "/v1/chat/", "/v1/completions", "/v1/images/", "/v1/mcp/execute",
        "/v1/messages", "/v1/responses", "/v1/embeddings",
        "/v1/audio/transcriptions", "/v1/audio/speech", "/v1/rerank"
    ])
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
                logger.info(f"JIT wake: request to {path} while in {_standby_state} sleep")
                try:
                    from starlette.responses import JSONResponse
                    wake_result = await admin_wake()
                    if isinstance(wake_result, dict) and wake_result.get("error"):
                        return JSONResponse(
                            status_code=503,
                            content={"error": {"message": f"Failed to wake model: {wake_result['error']}", "type": "server_error"}}
                        )
                    # For deep sleep, model takes time to reload — wait up to 300s
                    # (large models like 60GB+ JANG MoE can take 30-60s to mmap load)
                    for i in range(600):
                        if _standby_state is None:
                            break
                        if i > 0 and i % 20 == 0:
                            logger.info(f"JIT wake: still loading after {i * 0.5:.0f}s...")
                        await asyncio.sleep(0.5)
                    if _standby_state is not None:
                        return JSONResponse(
                            status_code=503,
                            content={"error": {"message": "Model still loading after 300s wake timeout", "type": "server_error"}}
                        )
                except Exception as e:
                    logger.error(f"JIT wake failed: {e}")
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        status_code=503,
                        content={"error": {"message": f"JIT wake failed: {e}", "type": "server_error"}}
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
_max_context_length: int = 0
_enable_jit: bool = False
_jang_metadata: dict | None = None  # Cached at model load time for /health
_model_type: str = "text"  # "text" or "image" — auto-detected from model directory
_image_quantize: int | None = None  # Image model quantization bits (set by cli.py)


def _apply_jit_compilation():
    """Apply mx.compile to the model forward pass for JIT-optimized inference.

    Wraps the model's __call__ with mx.compile for Metal kernel fusion.
    Falls back gracefully if compilation fails (some models have dynamic
    shapes or operations that mx.compile cannot handle).
    """
    global _engine
    if _engine is None:
        return

    try:
        import mlx.core as mx

        # Get the inner model object — SimpleEngine and BatchedEngine both store it
        model_obj = getattr(_engine, '_model', None)
        if model_obj is None:
            model_obj = getattr(_engine, 'model', None)
        if model_obj is None:
            logger.warning("JIT: Could not find model object on engine — skipping")
            return

        # The actual nn.Module is often nested: engine._model.model
        inner = getattr(model_obj, 'model', model_obj)
        if inner is None or not callable(inner):
            logger.warning("JIT: Model object is not callable — skipping")
            return

        logger.info("JIT: Applying mx.compile to model forward pass...")
        compiled = mx.compile(inner)

        # Replace in-place on the wrapper and verify
        replaced = False
        if hasattr(model_obj, 'model') and model_obj.model is inner:
            model_obj.model = compiled
            replaced = (model_obj.model is compiled)
        elif hasattr(_engine, '_model'):
            _engine._model = compiled
            replaced = (_engine._model is compiled)
        elif hasattr(_engine, 'model'):
            _engine.model = compiled
            replaced = (_engine.model is compiled)

        if replaced:
            logger.info("JIT: mx.compile applied successfully — first inference call will trigger Metal compilation")
        else:
            logger.warning("JIT: mx.compile created but could not verify replacement — model structure may have changed")
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
    # If auto tool choice is not enabled, use the generic parser
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return parse_tool_calls(output_text)

    # Create a fresh parser instance per call for thread-safety.
    # Non-streaming requests can be concurrent — sharing a global instance
    # risks interleaved state even with reset().
    try:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        tokenizer = None
        if _engine is not None and hasattr(_engine, "_tokenizer"):
            tokenizer = _engine._tokenizer
        parser_instance = parser_cls(tokenizer)
    except Exception as e:
        logger.warning(
            f"Failed to initialize tool parser '{_tool_call_parser}': {e}"
        )
        return parse_tool_calls(output_text)

    # Use the configured parser, fall back to generic if it finds nothing
    try:
        # Convert request to dict format for parsers that need schema info (e.g., Step3p5 type coercion)
        parser_request = None
        if request and request.tools:
            parser_request = {"tools": convert_tools_for_template(request.tools)}
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
            return result.content or "", tool_calls
        else:
            # Specific parser found nothing — try generic parser as fallback
            # (handles Nemotron, Llama, raw JSON, etc.)
            return parse_tool_calls(output_text)
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return parse_tool_calls(output_text)


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
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        if parser_cls.supports_native_format():
            return True
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{_tool_call_parser}' not found. "
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
        logger.debug("Model config registry lookup failed, skipping native format detection")

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

        "mlx-community/Llama-3.2-3B-Instruct-4bit"
        → "mlx-community/Llama-3.2-3B-Instruct-4bit" (unchanged)
    """
    if os.path.sep in model_name or model_name.startswith("/"):
        # Local path — extract last two components as "org/model"
        parts = model_name.rstrip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return parts[-1]
    return model_name


def _resolve_model_name() -> str:
    """Return the model name to expose via the API.

    Priority: _served_model_name > _model_name > 'default'
    """
    return _served_model_name or _model_name or "default"


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
    served_model_name: str | None = None,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        force_mllm: Force loading as MLLM even if not auto-detected
    """
    global _engine, _model_name, _model_path, _default_max_tokens, _served_model_name, _model_load_error, _jang_metadata, _cli_args

    # Save CLI args for model reload on wake from deep sleep
    _cli_args = {
        'use_batching': use_batching,
        'scheduler_config': scheduler_config,
        'stream_interval': stream_interval,
        'max_tokens': max_tokens,
        'force_mllm': force_mllm,
        'served_model_name': served_model_name,
    }

    # Stop previous engine before loading new model — frees GPU memory, disk cache threads, etc.
    # Note: load_model() is only called from CLI startup (before uvicorn), never during live serving.
    if _engine is not None:
        try:
            if hasattr(_engine, '_scheduler') and hasattr(_engine._scheduler, 'deep_reset'):
                _engine._scheduler.deep_reset()
        except Exception as e:
            logger.warning(f"Failed to stop previous engine: {e}")
        _engine = None

    _model_load_error = None  # Clear any previous error
    _jang_metadata = None  # Clear any previous JANG metadata

    _default_max_tokens = max_tokens
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
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
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

    # Apply JIT compilation if enabled — only for SimpleEngine (already started above).
    # BatchedEngine starts in lifespan(), so JIT is applied there instead.
    if _enable_jit and _engine is not None and hasattr(_engine, '_loaded') and _engine._loaded:
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

    # Cache JANG metadata at load time (avoids sync file IO in async /health handler)
    try:
        from .utils.jang_loader import is_jang_model, JANG_CONFIG_FILENAMES
        if is_jang_model(model_name):
            from pathlib import Path
            for cfg_name in JANG_CONFIG_FILENAMES:
                cfg_path = Path(model_name) / cfg_name
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
            active_gb = mx.get_active_memory() / (1024 ** 3)
            peak_gb = mx.get_peak_memory() / (1024 ** 3)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            active_gb = mx.metal.get_active_memory() / (1024 ** 3)
            peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)
        if active_gb is not None:
            logger.info(f"Metal GPU memory after load: {active_gb:.2f}GB active, {peak_gb:.2f}GB peak")
    except Exception:
        pass

    # Log system memory after model load
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        logger.info(f"System memory after load: {available_gb:.1f}GB available ({mem.percent}% used)")
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

    logger.info(f"Default max tokens: {_default_max_tokens}")


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    cached = getattr(output, "cached_tokens", 0)
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached) if cached > 0 else None,
    )


def _get_responses_usage(output: GenerationOutput) -> "ResponsesUsage":
    """Extract usage metrics in Responses API format."""
    cached = getattr(output, "cached_tokens", 0)
    return ResponsesUsage(
        input_tokens=output.prompt_tokens,
        output_tokens=output.completion_tokens,
        total_tokens=output.prompt_tokens + output.completion_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached) if cached > 0 else None,
    )


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
            "tools_available": len(_mcp_manager.get_all_tools()),
        }

    engine_stats = _engine.get_stats() if _engine else {}

    # Differentiate status: "healthy" when model is loaded, "no_model" otherwise
    # Image models don't use _engine — they use _image_gen
    if _standby_state:
        status = f"standby_{_standby_state}"  # "standby_soft" or "standby_deep"
    elif _model_type == "image":
        status = "healthy" if (_image_gen is not None and _image_gen.is_loaded) else "no_model"
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

    # Include KV cache quantization status for diagnostics
    kv_quant_info = None
    if _engine and hasattr(_engine, '_engine') and _engine._engine:
        scheduler = getattr(_engine._engine.engine, 'scheduler', None)
        if scheduler:
            kv_bits = getattr(scheduler, '_kv_cache_bits', 0)
            if kv_bits:
                kv_quant_info = {
                    "enabled": True,
                    "bits": kv_bits,
                    "group_size": getattr(scheduler, '_kv_cache_group_size', 64),
                }
            else:
                kv_quant_info = {"enabled": False}

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
        sanitized = re.sub(r'(/[^\s:]+)+', '<path>', _model_load_error)
        result["error"] = sanitized
    if memory_info:
        result["memory"] = memory_info
    if kv_quant_info:
        result["kv_cache_quantization"] = kv_quant_info
    if spec_info:
        result["speculative_decoding"] = spec_info.get("speculative_decoding", spec_info)

    # JANG format: report cached quantization metadata (populated at load time)
    if _jang_metadata:
        result["quantization_format"] = _jang_metadata

    return result


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

@app.post("/admin/soft-sleep")
async def admin_soft_sleep():
    """Enter soft sleep: clear all caches, reduce Metal cache limit. Model stays loaded."""
    global _standby_state, _pre_sleep_cache_limit, _wake_lock
    from starlette.responses import JSONResponse
    if _wake_lock is None:
        _wake_lock = asyncio.Lock()

    async with _wake_lock:
        if _standby_state == 'deep':
            return JSONResponse(status_code=409, content={"error": "Already in deep sleep"})
        if _standby_state == 'soft':
            return {"status": "already_soft"}

        try:
            import mlx.core as mx

            scheduler = _get_scheduler()
            if scheduler is not None:
                if hasattr(scheduler, 'deep_reset'):
                    scheduler.deep_reset()
                elif hasattr(scheduler, '_prefix_cache') and scheduler._prefix_cache:
                    scheduler._prefix_cache.clear()

            _set_cache = getattr(mx, 'set_cache_limit', None) or getattr(mx.metal, 'set_cache_limit', None)
            if _set_cache:
                _pre_sleep_cache_limit = _set_cache(512 * 1024 * 1024)

            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            elif hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()

            _standby_state = 'soft'
            logger.info("Entered soft sleep — caches cleared, model loaded")
            return {"status": "soft_sleep"}

        except Exception as e:
            logger.error(f"Failed to enter soft sleep: {e}")
            return {"error": str(e)}


@app.post("/admin/deep-sleep")
async def admin_deep_sleep():
    """Enter deep sleep: unload model entirely. Process stays alive, port stays allocated."""
    global _engine, _standby_state, _pre_sleep_cache_limit, _wake_lock
    from starlette.responses import JSONResponse
    if _wake_lock is None:
        _wake_lock = asyncio.Lock()

    async with _wake_lock:
        if _standby_state == 'deep':
            return JSONResponse(status_code=409, content={"error": "Already in deep sleep"})

        try:
            import mlx.core as mx
            import gc

            # Save Metal cache limit before deep sleep (if not already saved by soft sleep)
            if _pre_sleep_cache_limit is None:
                _set_cache = getattr(mx, 'set_cache_limit', None) or getattr(mx.metal, 'set_cache_limit', None)
                if _set_cache:
                    _pre_sleep_cache_limit = _set_cache(512 * 1024 * 1024)

            scheduler = _get_scheduler()
            if scheduler is not None:
                if hasattr(scheduler, 'deep_reset'):
                    scheduler.deep_reset()

            if _model_type == "image" and _image_gen is not None:
                _image_gen.unload()
            elif _engine is not None:
                if hasattr(_engine, 'stop'):
                    try:
                        await _engine.stop()
                    except Exception:
                        pass
                _engine = None

            gc.collect()
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            elif hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()

            _standby_state = 'deep'
            logger.info("Entered deep sleep — model unloaded, process alive")
            return {"status": "deep_sleep"}

        except Exception as e:
            logger.error(f"Failed to enter deep sleep: {e}")
            return {"error": str(e)}


@app.post("/admin/wake")
async def admin_wake():
    """Wake from sleep: reload model. Triggered by JIT or manual wake."""
    global _engine, _standby_state, _pre_sleep_cache_limit

    if _standby_state is None:
        return {"status": "already_active"}

    try:
        import mlx.core as mx

        if _standby_state == 'soft':
            # Soft sleep: model is still loaded, just restore cache limit
            _set_cache = getattr(mx, 'set_cache_limit', None) or getattr(mx.metal, 'set_cache_limit', None)
            if _set_cache and _pre_sleep_cache_limit:
                _set_cache(_pre_sleep_cache_limit)
            _standby_state = None
            _pre_sleep_cache_limit = None
            logger.info("Woke from soft sleep — cache limit restored")
            return {"status": "active"}

        elif _standby_state == 'deep':
            # Deep sleep: need to reload model
            # Restore cache limit first
            _set_cache = getattr(mx, 'set_cache_limit', None) or getattr(mx.metal, 'set_cache_limit', None)
            if _set_cache and _pre_sleep_cache_limit:
                _set_cache(_pre_sleep_cache_limit)
            _pre_sleep_cache_limit = None

            if _model_type == "image" and _image_gen is not None:
                # Reload image model — run in thread to avoid blocking event loop
                # (loading Flux models takes 15-60s)
                await asyncio.to_thread(
                    _image_gen.load,
                    _model_name,
                    quantize=getattr(_image_gen, '_quantize', None),
                    model_path=getattr(_image_gen, '_model_path', None),
                    mflux_class=getattr(_image_gen, '_mflux_class', None),
                )
                _standby_state = None
                logger.info(f"Woke from deep sleep — image model {_model_name} reloaded")
                return {"status": "active"}
            elif _model_path or _model_name:
                # Reload text model — run in thread to avoid blocking event loop
                # (loading large models takes 10-60s; _wake_lock prevents concurrent
                # access to the globals that load_model modifies)
                await asyncio.to_thread(
                    load_model,
                    _model_path or _model_name,
                    use_batching=_cli_args.get('use_batching', False),
                    scheduler_config=_cli_args.get('scheduler_config'),
                    stream_interval=_cli_args.get('stream_interval', 1),
                    max_tokens=_cli_args.get('max_tokens', 32768),
                    force_mllm=_cli_args.get('force_mllm', False),
                    served_model_name=_cli_args.get('served_model_name'),
                )
                # SimpleEngine needs async start (load_model defers it inside event loop)
                if _engine and hasattr(_engine, '_needs_async_start') and _engine._needs_async_start:
                    await _engine.start()
                    _engine._needs_async_start = False
                # Re-apply JIT compilation after deep wake (load_model skips it
                # because _engine._loaded is False at check time inside event loop)
                if _enable_jit and _engine is not None:
                    _apply_jit_compilation()
                _standby_state = None
                logger.info(f"Woke from deep sleep — model {_model_name} reloaded")
                return {"status": "active"}
            else:
                return {"error": "No model name saved — cannot reload"}

    except Exception as e:
        logger.error(f"Failed to wake from sleep: {e}")
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
            "num_requests_processed": stats.get("num_requests_processed", 0),
            "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
            "total_completion_tokens": stats.get("total_completion_tokens", 0),
        }
        # KV cache quantization info
        if hasattr(scheduler, "_kv_cache_bits"):
            result["kv_cache_quantization"] = {
                "bits": scheduler._kv_cache_bits,
                "group_size": scheduler._kv_cache_group_size,
            }

    # Disk cache (L2) stats — prompt-level
    if scheduler and getattr(scheduler, "disk_cache", None) is not None:
        result["disk_cache"] = scheduler.disk_cache.stats()

    # Block disk store (L2) stats — paged cache blocks
    paged_mgr = getattr(scheduler, "paged_cache_manager", None) if scheduler else None
    block_store = getattr(paged_mgr, "_disk_store", None) if paged_mgr else None
    if block_store is not None and hasattr(block_store, "get_stats"):
        result["block_disk_cache"] = block_store.get_stats()

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
            entries.append({
                "tokens_count": len(tokens_key),
                "memory_bytes": entry.memory_bytes,
                "memory_mb": round(entry.memory_bytes / _BYTES_PER_MB, 2),
                "cache_type": "memory_aware",
            })
    elif getattr(scheduler, "block_aware_cache", None) is not None:
        cache_type = "paged"
        for block_id, block in scheduler.block_aware_cache.paged_cache.allocated_blocks.items():
            entries.append({
                "block_id": block_id,
                "tokens_count": block.token_count,
                "ref_count": block.ref_count,
                "has_data": block.cache_data is not None,
                "cache_type": "paged",
            })
    elif getattr(scheduler, "prefix_cache", None) is not None:
        cache_type = "legacy"
        for _, tokens_tuple in scheduler.prefix_cache._lru:
            entries.append({
                "tokens_count": len(tokens_tuple),
                "cache_type": "legacy",
            })

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

    def _do_warm():
        """Run prefill warming in a thread to avoid blocking the event loop."""
        warmed = 0
        token_counts = []
        errors = []

        for i, prompt in enumerate(prompts):
            try:
                if hasattr(scheduler, "_actual_tokenizer"):
                    tokens = scheduler._actual_tokenizer.encode(prompt)
                else:
                    tokens = scheduler.tokenizer.encode(prompt)

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
                    errors.append(f"Prompt {i}: cache store rejected (too large or full)")
            except Exception as e:
                errors.append(f"Prompt {i}: {type(e).__name__}: {e}")

        return {"warmed": warmed, "token_counts": token_counts, "errors": errors if errors else None}

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


@app.post(
    "/v1/chat/completions/{request_id}/cancel",
    dependencies=[Depends(verify_api_key)]
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
        raise HTTPException(status_code=404, detail="No model loaded — no active requests to cancel")

    # Abort the request (returns True if found, False if not found/already finished)
    success = await _engine.abort_request(request_id)

    if success:
        logger.info(f"Request {request_id} cancelled via API")
        return {"success": True, "message": f"Request {request_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request_id} not found or already finished"
        )


@app.post(
    "/v1/responses/{response_id}/cancel",
    dependencies=[Depends(verify_api_key)]
)
async def cancel_response(response_id: str):
    """
    Cancel an ongoing Responses API request.

    Args:
        response_id: The response ID (e.g., resp_abc123def456)

    Returns:
        Success message or 404 if request not found
    """
    if _engine is None:
        raise HTTPException(status_code=404, detail="No model loaded — no active requests to cancel")
    success = await _engine.abort_request(response_id)

    if success:
        logger.info(f"Response {response_id} cancelled via API")
        return {"success": True, "message": f"Response {response_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Response {response_id} not found or already finished"
        )


@app.post(
    "/v1/completions/{request_id}/cancel",
    dependencies=[Depends(verify_api_key)]
)
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
        raise HTTPException(status_code=404, detail="No model loaded — no active requests to cancel")
    success = await _engine.abort_request(request_id)

    if success:
        logger.info(f"Request {request_id} cancelled via API")
        return {"success": True, "message": f"Request {request_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Request {request_id} not found or already finished"
        )


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models."""
    models = []
    resolved = _resolve_model_name()
    if resolved and resolved != "default":
        models.append(ModelInfo(id=resolved))
        # If served_model_name differs from actual, also list the actual name
        # so clients using either name can find the model
        if _served_model_name and _model_name and _served_model_name != _model_name:
            models.append(ModelInfo(id=_model_name))
    return ModelsResponse(data=models)


# =============================================================================
# Anthropic Messages API
# =============================================================================


@app.post(
    "/v1/messages",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_anthropic_message(
    fastapi_request: Request,
):
    """
    Anthropic Messages API endpoint.

    Enables Claude Code and other Anthropic SDK clients to use vMLX as a
    local inference backend. Converts Anthropic wire format to internal
    Chat Completions pipeline.
    """
    from .api.anthropic_adapter import (
        AnthropicRequest,
        AnthropicStreamAdapter,
        to_chat_completion,
    )

    try:
        body = await fastapi_request.json()
    except Exception:
        return JSONResponse(status_code=400, content={
            "type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON in request body"}
        })
    try:
        anthropic_req = AnthropicRequest(**body)
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "type": "error", "error": {"type": "invalid_request_error", "message": str(e)}
        })

    # Convert to chat completion request
    chat_req = to_chat_completion(anthropic_req)

    # Resolve model name
    resolved_name = _resolve_model_name()
    chat_req.model = resolved_name

    engine = get_engine()

    # Build generation kwargs from the converted chat request (shared by streaming + non-streaming)
    _msg_kwargs: dict = {
        "temperature": _resolve_temperature(chat_req.temperature),
        "top_p": _resolve_top_p(chat_req.top_p),
        "max_tokens": chat_req.max_tokens or _default_max_tokens,
    }
    if chat_req.top_k is not None:
        _msg_kwargs["top_k"] = chat_req.top_k
    if chat_req.min_p is not None:
        _msg_kwargs["min_p"] = chat_req.min_p
    if chat_req.repetition_penalty is not None:
        _msg_kwargs["repetition_penalty"] = chat_req.repetition_penalty
    if chat_req.stop:
        _msg_kwargs["stop"] = chat_req.stop
    # Forward enable_thinking to engine — without this, the model always thinks
    # internally even when the client sends thinking: {type: "disabled"}
    if chat_req.enable_thinking is not None:
        _msg_kwargs["enable_thinking"] = chat_req.enable_thinking
    elif _default_enable_thinking is not None:
        _msg_kwargs["enable_thinking"] = _default_enable_thinking

    messages_dump = [m.model_dump(exclude_none=True) for m in chat_req.messages]

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
                                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                            tool_calls[tc_idx] = {"id": tc["id"], "function": tc.get("function", {})}
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
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
                "input": input_data,
            })

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
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
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
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens "
            f"in {elapsed:.2f}s"
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
                "mlx-embeddings not installed. "
                "Install with: pip install mlx-embeddings"
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
_reranker_lock: asyncio.Lock | None = None  # Lazy-init to avoid binding to wrong event loop


@app.post(
    "/v1/rerank",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
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
        raise HTTPException(status_code=400, detail="documents list is required and must be non-empty")
    if len(documents) > 1000:
        raise HTTPException(status_code=400, detail=f"Too many documents ({len(documents)}). Maximum is 1000.")

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
                raise HTTPException(status_code=400, detail="model is required for first rerank request")
            from .reranker import Reranker
            if _reranker is not None:
                _reranker.unload()
            _reranker = Reranker(model)

    try:
        results = _reranker.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_n,
            return_documents=return_documents,
        )
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            "model": model or _reranker.model_path,
        },
    }


# =============================================================================
# Image Generation Endpoint (OpenAI-compatible /v1/images/generations)
# =============================================================================

_image_gen = None  # Lazy-loaded ImageGenEngine
_image_gen_lock: asyncio.Lock | None = None  # Lazy-init to avoid binding to wrong event loop


@app.post(
    "/v1/images/generations",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
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
        raise HTTPException(status_code=400, detail="response_format 'url' is not supported. Use 'b64_json'.")

    # Parse size
    try:
        width, height = [int(x) for x in size.split("x")]
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid size format: {size}. Use WxH (e.g., 1024x1024)")

    # Enforce dimension limits (prevent OOM from absurd sizes)
    MAX_DIM = 4096
    if width < 64 or height < 64:
        raise HTTPException(status_code=400, detail=f"Minimum dimension is 64. Got {width}x{height}")
    if width > MAX_DIM or height > MAX_DIM:
        raise HTTPException(status_code=400, detail=f"Maximum dimension is {MAX_DIM}. Got {width}x{height}")

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
                    detail="mflux not installed. Install with: pip install mflux"
                )

        # Resolve aliases before comparison (e.g., "flux-schnell" → "schnell")
        from .image_gen import SUPPORTED_MODELS as _IMG_MODELS
        resolved_model = _IMG_MODELS.get(model.lower(), model) if model else model

        # If pre-loaded (from serve command), skip re-loading unless explicitly different model.
        already_loaded = _image_gen.is_loaded and (
            not model or resolved_model == _image_gen.model_name or
            model == _model_name or model == _served_model_name or
            model == _model_path
        )
        if not already_loaded:
            try:
                if _image_gen.is_loaded:
                    _image_gen.unload()
                _image_gen.load(model, quantize=quantize, model_path=model_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load image model '{model}': {e}")

        # If source image provided (img2img), save to temp file for mflux
        source_image_path = None
        images = []
        try:
            if source_image_b64 and image_strength is not None:
                import tempfile
                from pathlib import Path
                raw_b64 = re.sub(r'^data:image/[^;]+;base64,', '', source_image_b64)
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
                    result = await asyncio.to_thread(
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
                    raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

                images.append({
                    "b64_json": result.b64_json,
                    "revised_prompt": prompt,
                    "seed": result.seed,
                })
        finally:
            # Clean up temp source image
            if source_image_path:
                try:
                    os.unlink(source_image_path)
                except OSError:
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


# =============================================================================
# Image Editing Endpoint
# =============================================================================


@app.post("/v1/images/edits", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
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

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

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
        raise HTTPException(status_code=400, detail=f"Width must be between 64 and 4096, got {width}")
    if height < 64 or height > 4096:
        raise HTTPException(status_code=400, detail=f"Height must be between 64 and 4096, got {height}")

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
        image_b64 = re.sub(r'^data:image/[^;]+;base64,', '', image_b64)
        image_data = base64.b64decode(image_b64)
        try:
            from PIL import Image as _PILImage, ImageOps as _ImageOps
            import io as _io
            src_img = _PILImage.open(_io.BytesIO(image_data))
            src_format = src_img.format
            src_original_mode = src_img.mode

            # For animated images (GIF, APNG, WebP), use only the first frame
            if getattr(src_img, 'is_animated', False):
                src_img.seek(0)
                logger.info(f"Edit: animated {src_format} detected, using first frame only")

            # Apply EXIF orientation (iPhone/DSLR photos may be rotated)
            try:
                src_img = _ImageOps.exif_transpose(src_img)
            except Exception:
                pass  # No EXIF data or unsupported — continue without rotation

            # Convert to 8-bit RGB (handles RGBA, CMYK, P, L, I, I;16, F, LA, PA modes)
            if src_img.mode in ('RGBA', 'LA', 'PA'):
                # Composite alpha onto white background before converting
                background = _PILImage.new('RGB', src_img.size, (255, 255, 255))
                background.paste(src_img, mask=src_img.split()[-1])
                src_img = background
            elif src_img.mode != 'RGB':
                src_img = src_img.convert('RGB')

            src_img.save(str(image_path), format='PNG')
            logger.info(f"Edit: received image {len(image_b64)} b64 chars -> {len(image_data)} bytes "
                        f"(format={src_format}, mode={src_original_mode}, size={src_img.size}), "
                        f"converted to RGB PNG at {image_path}")
        except Exception as conv_err:
            # Provide actionable error instead of silently saving raw bytes
            err_msg = str(conv_err)
            if 'HEIF' in err_msg.upper() or 'HEIC' in err_msg.upper() or 'cannot identify' in err_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format: {conv_err}. "
                           f"HEIC/HEIF images require the pillow-heif package. "
                           f"Please convert to JPEG or PNG before uploading."
                )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process input image: {conv_err}"
            )

        # Decode and save mask if provided — also convert via PIL
        if mask_b64:
            mask_path = Path(tmp_dir) / "mask.png"
            mask_b64_clean = re.sub(r'^data:image/[^;]+;base64,', '', mask_b64)
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
                # Masks should be grayscale (L mode) — convert if needed
                if mask_img.mode != 'L':
                    mask_img = mask_img.convert('L')
                mask_img.save(str(mask_path), format='PNG')
                logger.info(f"Edit: mask converted to grayscale PNG at {mask_path}")
            except Exception as mask_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process mask image: {mask_err}"
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
                        detail="mflux not installed. Install with: pip install mflux"
                    )

            from .image_gen import EDIT_MODELS
            resolved = EDIT_MODELS.get(model.lower(), model)

            # Check if we need to load a different model
            already_loaded = _image_gen.is_loaded and (
                resolved == _image_gen.model_name
            )
            if not already_loaded:
                try:
                    if _image_gen.is_loaded:
                        _image_gen.unload()
                    edit_model_path = body.get("model_path") or _model_path
                    _image_gen.load(model, model_path=edit_model_path, quantize=_image_quantize)
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load edit model '{model}': {e}"
                    )

            # Generate edited images (inside lock)
            # Verify the input image can be opened
            try:
                from PIL import Image as _PILImage
                _src = _PILImage.open(str(image_path))
                logger.info(f"Edit: source image verified: {_src.size} mode={_src.mode} format={_src.format}")
            except Exception as _e:
                logger.error(f"Edit: source image INVALID: {_e}")
                raise HTTPException(status_code=400, detail=f"Invalid source image: {_e}")

            logger.info(f"Edit: model={model} resolved={resolved} prompt={prompt[:50]!r} "
                        f"size={width}x{height} steps={steps} guidance={guidance} "
                        f"strength={strength} seed={seed}")

            for i in range(n):
                if await request.is_disconnected():
                    logger.info("Image edit cancelled: client disconnected")
                    break
                img_seed = (seed + i) if seed is not None else None
                try:
                    result = await asyncio.to_thread(
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
                        status_code=500,
                        detail=f"Image editing failed: {e}"
                    )

                images.append({
                    "b64_json": result.b64_json,
                    "revised_prompt": prompt,
                    "seed": result.seed,
                })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image edit endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image editing failed: {e}"
        )
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


# =============================================================================
# MCP Endpoints
# =============================================================================


@app.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    if _mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in _mcp_manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@app.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    if _mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in _mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    return MCPServersResponse(servers=servers)


@app.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    if _mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    result = await _mcp_manager.execute_tool(
        request.tool_name,
        request.arguments,
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


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
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
    global _stt_engine

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

        # Load engine if needed
        if _stt_engine is None or _stt_engine.model_name != model_name:
            _stt_engine = STTEngine(model_name)
            _stt_engine.load()

        # Save uploaded file temporarily
        ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
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


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def create_speech(request: AudioSpeechRequest):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supported models:
    - kokoro (fast, lightweight)
    - chatterbox (multilingual, expressive)
    - vibevoice (realtime)
    - voxcpm (Chinese/English)
    """
    global _tts_engine

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

        # Load engine if needed
        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(request.input, voice=request.voice, speed=request.speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=request.response_format)

        content_type = (
            "audio/wav" if request.response_format == "wav" else f"audio/{request.response_format}"
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
        raise HTTPException(status_code=500, detail=str(e))


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
    "/v1/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    engine = get_engine()

    # Normalize model name (consistent with chat/responses endpoints)
    resolved_name = _resolve_model_name()
    request.model = resolved_name

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

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
                    "max_tokens": request.max_tokens if request.max_tokens is not None else _default_max_tokens,
                    "temperature": _resolve_temperature(request.temperature),
                    "top_p": _resolve_top_p(request.top_p),
                    "stop": request.stop,
                }
            if request.top_k is not None:
                gen_kwargs["top_k"] = request.top_k
            if request.min_p is not None:
                gen_kwargs["min_p"] = request.min_p
            if request.repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = request.repetition_penalty
            output = await asyncio.wait_for(
                engine.generate(**gen_kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504, detail=f"Request timed out after {timeout:.1f} seconds"
            )
        except Exception as e:
            logger.error(f"Completion failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {type(e).__name__}: {e}",
            )

        choices.append(
            CompletionChoice(
                index=i,
                text=output.text,
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
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
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
    if request.model and request.model != resolved_name and request.model != _model_name:
        logger.info(
            f"Request model '{request.model}' differs from served model "
            f"'{resolved_name}' — using loaded model (single-model server)"
        )
    # Normalize response model field to the resolved name
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

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        json_instruction = build_json_system_prompt(response_format)
        if json_instruction:
            # Inject JSON instruction into messages
            messages = _inject_json_instruction(messages, json_instruction)

    # When thinking is explicitly disabled, strip <think> blocks from prior assistant
    # messages in the conversation history. Without this, the model sees prior thinking
    # in context and mimics the pattern, producing reasoning even when the generation
    # prompt doesn't inject <think>. This is the root cause of "thinking OFF but model
    # still thinks on 2nd message" bugs.
    _ct_kwargs = request.chat_template_kwargs or {}
    _explicit_thinking_off = (
        request.enable_thinking is False
        or (_ct_kwargs.get("enable_thinking") is False)
    )
    if _explicit_thinking_off and messages:
        _think_strip_re = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
        cleaned = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                stripped = _think_strip_re.sub('', content).strip()
                if stripped != content:
                    if not stripped:
                        continue  # Drop assistant messages that were ONLY thinking
                    msg = {**msg, "content": stripped}
            cleaned.append(msg)
        messages = cleaned

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": request.max_tokens if request.max_tokens is not None else _default_max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
    }
    # Forward stop sequences from request to engine
    if request.stop:
        chat_kwargs["stop"] = request.stop
    # Extended sampling params (only pass if explicitly set)
    if request.top_k is not None:
        chat_kwargs["top_k"] = request.top_k
    if request.min_p is not None:
        chat_kwargs["min_p"] = request.min_p
    if request.repetition_penalty is not None:
        chat_kwargs["repetition_penalty"] = request.repetition_penalty

    # Pass enable_thinking to engine
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        chat_kwargs["enable_thinking"] = bool(_ct_kwargs["enable_thinking"])
    elif _default_enable_thinking is not None:
        # Server-level default (--default-enable-thinking flag)
        chat_kwargs["enable_thinking"] = _default_enable_thinking
    else:
        # Auto-detect from model config + tokenizer vocabulary.
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(_model_path or _model_name or request.model)
        _enable = _mc.think_in_template
        if not _enable and _mc.reasoning_parser:
            # Model has a reasoning parser — enable thinking by default.
            # For <think>-based models, this makes the template inject thinking tags.
            # For Harmony models (GPT-OSS), the template ignores it but the server
            # uses it to trigger analysis prefix injection.
            _enable = True
        if not _enable:
            try:
                _tok = engine.tokenizer
                if getattr(_tok, 'has_thinking', False):
                    _enable = True
            except Exception:
                pass
        chat_kwargs["enable_thinking"] = _enable

    # Pass reasoning_effort if provided (for GPT-OSS and models that support thinking levels).
    # Also map to thinking_budget (Qwen3) and max_tokens ceiling when not explicitly set.
    if request.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = request.reasoning_effort
        _effort_lower = request.reasoning_effort.lower()
        _budget = _EFFORT_THINKING_BUDGET.get(_effort_lower)
        if _budget:
            _ct_kwargs.setdefault("thinking_budget", _budget)
        _effort_max = _EFFORT_MAX_TOKENS.get(_effort_lower)
        if _effort_max and request.max_tokens is None:
            chat_kwargs["max_tokens"] = _effort_max

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            chat_kwargs["chat_template_kwargs"] = extra_ct

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
    _suppress_tools = (_tool_choice == "none")

    # Merge MCP tools with user-provided tools
    all_tools = []

    if not _suppress_tools:
        # Add MCP tools if available
        if _mcp_manager is not None:
            mcp_tools = _mcp_manager.get_all_tools_openai()
            all_tools.extend(mcp_tools)
            if mcp_tools:
                logger.debug(f"Added {len(mcp_tools)} MCP tools")

        # Add user-provided tools
        if request.tools:
            # If tool_choice is a specific tool dict, filter to only that tool
            if isinstance(_tool_choice, dict):
                target_name = (_tool_choice.get("function", {}).get("name")
                               or _tool_choice.get("name"))
                if target_name:
                    # request.tools are ToolDefinition Pydantic models (type + function dict)
                    filtered = [t for t in request.tools
                                if t.function.get("name") == target_name]
                    all_tools.extend(filtered if filtered else request.tools)
                else:
                    all_tools.extend(request.tools)
            else:
                all_tools.extend(request.tools)
            logger.debug(f"Added {len(all_tools)} tools (tool_choice={_tool_choice})")

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
            # Reasoning effort: scale total max_tokens to control how much the model
            # can reason. Like OpenAI's reasoning_effort, lower effort = tighter token
            # budget = less reasoning before the model must produce an answer.
            # This is the genuine mechanism — the model allocates its token budget
            # between analysis and final answer within the limit.
            _effort = chat_kwargs.get("reasoning_effort", "").lower()
            _original_max = chat_kwargs.get("max_tokens", 4096)
            if _effort == "low":
                # Cap total output to force concise analysis + answer
                chat_kwargs["max_tokens"] = min(_original_max, 512)
            elif _effort == "high":
                # Allow generous budget for deep analysis
                chat_kwargs["max_tokens"] = max(_original_max, 16384)
            # medium/auto: use request's max_tokens as-is
            _adjusted = chat_kwargs.get("max_tokens", _original_max)
            if _adjusted != _original_max:
                logger.info(
                    f"[chat] reasoning_effort='{_effort}' adjusted max_tokens: "
                    f"{_original_max} -> {_adjusted}"
                )
            logger.info(f"[chat] Injecting Harmony analysis prefix for GPT-OSS model (effort={_effort or 'default'})")
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
                **chat_kwargs
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout if request.timeout is not None else _default_timeout

    try:
        output = await asyncio.wait_for(
            engine.chat(messages=messages, **chat_kwargs), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"Request timed out after {timeout:.1f} seconds"
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {type(e).__name__}: {e}",
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
        reasoning_text, remaining_text = request_parser.extract_reasoning(
            output.text
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
        if not _suppress and request.chat_template_kwargs:
            _suppress = request.chat_template_kwargs.get("enable_thinking") is False
        if _suppress:
            reasoning_text = None

    # Strip any residual think tags before tool call parsing
    _cc_parse_text = re.sub(r'<think>.*?</think>', '', content_for_parsing, flags=re.DOTALL)
    if _cc_parse_text == content_for_parsing and '</think>' in content_for_parsing:
        _, _, _cc_parse_text = content_for_parsing.partition('</think>')
    _cc_parse_text = _cc_parse_text.strip()

    # Parse tool calls from output using configured parser (skip when tool_choice="none")
    cleaned_text, tool_calls = (
        _parse_tool_calls_with_parser(_cc_parse_text, request)
        if not _suppress_tools else (_cc_parse_text or content_for_parsing, None)
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
            elif hasattr(response_format, "json_schema") and response_format.json_schema:
                _rf_strict = getattr(response_format.json_schema, "strict", False)
            if _rf_strict:
                raise HTTPException(
                    status_code=400,
                    detail=f"response_format strict mode: {error}"
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

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=clean_output_text(cleaned_text) if cleaned_text else None,
                    reasoning=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=get_usage(output),
    )


# =============================================================================
# Responses API (OpenAI /v1/responses)
# =============================================================================


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


def _responses_input_to_messages(
    input_data: str | list, instructions: str | None = None,
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
            has_media = any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image", "video_url", "video")
                for p in clean_parts
            )
            if has_media:
                return clean_parts
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
                    msg["tool_calls"] = item.tool_calls if isinstance(item.tool_calls, list) else [item.tool_calls]
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
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": args_parsed,
                    },
                }],
            })
            continue

        # function_call_output → tool message with result
        if item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": item["tool_call_id"],
                    "content": _extract_text_from_content(item.get("content", "")),
                })
            # Preserve assistant messages with tool_calls (Chat Completions format)
            elif role == "assistant" and "tool_calls" in item:
                messages.append({
                    "role": "assistant",
                    "content": item.get("content"),
                    "tool_calls": item["tool_calls"],
                })
            else:
                content = _resolve_content(item.get("content", ""))
                messages.append({"role": role, "content": content})
        # Skip unknown item types (e.g. reasoning, web_search_call, etc.)

    return messages


@app.post(
    "/v1/responses",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
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
    if request.model and request.model != resolved_name and request.model != _model_name:
        logger.info(
            f"Request model '{request.model}' differs from served model "
            f"'{resolved_name}' — using loaded model (single-model server)"
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

    # Convert Responses API input to chat messages
    # For MLLM models, preserve content arrays with image/video parts — MLLM engines
    # extract images from message content internally (same as Chat Completions path)
    messages = _responses_input_to_messages(
        request.input, request.instructions,
        preserve_multimodal=engine.is_mllm,
    )

    # Handle text format (json_object / json_schema) — translate to response_format
    if request.text and isinstance(request.text, dict) and request.text.get("type") != "text":
        # Convert Responses API text format to Chat Completions response_format
        json_instruction = build_json_system_prompt(request.text)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)
    elif hasattr(request.text, "type") and request.text.type != "text":
        text_dict = request.text.model_dump(exclude_none=True) if hasattr(request.text, "model_dump") else {"type": request.text.type}
        json_instruction = build_json_system_prompt(text_dict)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Strip <think> blocks from history when thinking is OFF (same as Chat Completions path)
    _ct_kwargs = request.chat_template_kwargs or {}
    _explicit_thinking_off = (
        request.enable_thinking is False
        or (_ct_kwargs.get("enable_thinking") is False)
    )
    if _explicit_thinking_off and messages:
        _think_strip_re = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
        cleaned = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                stripped = _think_strip_re.sub('', content).strip()
                if stripped != content:
                    if not stripped:
                        continue  # Drop assistant messages that were ONLY thinking
                    msg = {**msg, "content": stripped}
            cleaned.append(msg)
        messages = cleaned

    # Build kwargs
    chat_kwargs = {
        "max_tokens": request.max_output_tokens if request.max_output_tokens is not None else _default_max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
    }
    # Forward stop sequences from request to engine
    if request.stop:
        chat_kwargs["stop"] = request.stop
    # Extended sampling params (only pass if explicitly set)
    if request.top_k is not None:
        chat_kwargs["top_k"] = request.top_k
    if request.min_p is not None:
        chat_kwargs["min_p"] = request.min_p
    if request.repetition_penalty is not None:
        chat_kwargs["repetition_penalty"] = request.repetition_penalty

    # Pass enable_thinking to engine
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        chat_kwargs["enable_thinking"] = bool(_ct_kwargs["enable_thinking"])
    elif _default_enable_thinking is not None:
        # Server-level default (--default-enable-thinking flag)
        chat_kwargs["enable_thinking"] = _default_enable_thinking
    else:
        # Auto-detect from model config + tokenizer vocabulary.
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(_model_path or _model_name or request.model)
        _enable = _mc.think_in_template
        if not _enable and _mc.reasoning_parser:
            # Model has a reasoning parser — enable thinking by default.
            # For <think>-based models, this makes the template inject thinking tags.
            # For Harmony models (GPT-OSS), the template ignores it but the server
            # uses it to trigger analysis prefix injection.
            _enable = True
        if not _enable:
            try:
                _tok = engine.tokenizer
                if getattr(_tok, 'has_thinking', False):
                    _enable = True
            except Exception:
                pass
        chat_kwargs["enable_thinking"] = _enable

    # Pass reasoning_effort if provided (for GPT-OSS and models that support thinking levels).
    # Also map to thinking_budget (Qwen3) and max_output_tokens ceiling when not explicitly set.
    if request.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = request.reasoning_effort
        _effort_lower = request.reasoning_effort.lower()
        _budget = _EFFORT_THINKING_BUDGET.get(_effort_lower)
        if _budget:
            _ct_kwargs.setdefault("thinking_budget", _budget)
        _effort_max = _EFFORT_MAX_TOKENS.get(_effort_lower)
        if _effort_max and request.max_output_tokens is None:
            chat_kwargs["max_tokens"] = _effort_max

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            chat_kwargs["chat_template_kwargs"] = extra_ct

    # Video processing controls (MLLM models)
    if request.video_fps:
        chat_kwargs["video_fps"] = request.video_fps
    if request.video_max_frames:
        chat_kwargs["video_max_frames"] = request.video_max_frames

    # Handle tool_choice: "none" suppresses tools entirely, "auto"/None is default,
    # "required" or specific tool dict are handled post-generation.
    _tool_choice = request.tool_choice
    _suppress_tools = (_tool_choice == "none")

    # Merge MCP tools with user-provided tools
    all_tools = []

    if not _suppress_tools:
        # Add MCP tools if available
        if _mcp_manager is not None:
            mcp_tools = _mcp_manager.get_all_tools_openai()
            all_tools.extend(mcp_tools)
            if mcp_tools:
                logger.debug(f"Added {len(mcp_tools)} MCP tools")

        # Add user-provided tools — convert from Responses API flat format to Chat Completions nested format
        if request.tools:
            # If tool_choice is a specific tool dict, filter to only that tool
            if isinstance(_tool_choice, dict):
                target_name = (_tool_choice.get("function", {}).get("name")
                               or _tool_choice.get("name"))
            else:
                target_name = None

            for tool in request.tools:
                tool_type = tool.get("type", "")
                # Skip built-in Responses API tools (web_search, code_interpreter, file_search, etc.)
                if tool_type != "function" and "function" not in tool and "name" not in tool:
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
                    all_tools.append(ToolDefinition(**flat.to_chat_completions_format()))
            logger.debug(f"Added {len(all_tools)} tools (tool_choice={_tool_choice})")

    # Pass merged tools to engine
    if all_tools:
        chat_kwargs["tools"] = convert_tools_for_template(all_tools)

    # Inject Harmony analysis prefix for GPT-OSS models (same as Chat Completions path)
    if isinstance(_reasoning_parser, GptOssReasoningParser):
        _think_val = chat_kwargs.get("enable_thinking")
        if _think_val is True:
            _analysis_hint = "<|start|>assistant<|channel|>analysis<|message|>"
            chat_kwargs["prompt_suffix"] = _analysis_hint
            chat_kwargs["skip_generation_prompt"] = True
            _effort = chat_kwargs.get("reasoning_effort", "").lower()
            _original_max = chat_kwargs.get("max_tokens", 4096)
            if _effort == "low":
                chat_kwargs["max_tokens"] = min(_original_max, 512)
            elif _effort == "high":
                chat_kwargs["max_tokens"] = max(_original_max, 16384)
            _adjusted = chat_kwargs.get("max_tokens", _original_max)
            if _adjusted != _original_max:
                logger.info(
                    f"[responses] reasoning_effort='{_effort}' adjusted max_tokens: "
                    f"{_original_max} -> {_adjusted}"
                )
            logger.info(f"[responses] Injecting Harmony analysis prefix for GPT-OSS model (effort={_effort or 'default'})")
        # GPT-OSS/GLM Flash: add <|im_end|> as string stop sequence (see Chat Completions path)
        _stop = chat_kwargs.get("stop") or []
        if "<|im_end|>" not in _stop:
            _stop = list(_stop) + ["<|im_end|>"]
            chat_kwargs["stop"] = _stop

    if request.stream:
        return StreamingResponse(
            stream_responses_api(engine, messages, request, fastapi_request, **chat_kwargs),
            media_type="text/event-stream",
        )

    # Non-streaming response
    start_time = time.perf_counter()
    timeout = request.timeout if request.timeout is not None else _default_timeout

    try:
        output = await asyncio.wait_for(
            engine.chat(messages=messages, **chat_kwargs), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"Request timed out after {timeout:.1f} seconds"
        )
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
        reasoning_text, remaining_text = request_parser.extract_reasoning(
            output.text
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
        if not _suppress and request.chat_template_kwargs:
            _suppress = request.chat_template_kwargs.get("enable_thinking") is False
        if _suppress:
            reasoning_text = None

    # Strip any residual think tags before tool call parsing
    parse_text = re.sub(r'<think>.*?</think>', '', content_for_parsing, flags=re.DOTALL)
    if parse_text == content_for_parsing and '</think>' in content_for_parsing:
        _, _, parse_text = content_for_parsing.partition('</think>')
    parse_text = parse_text.strip()

    # Parse tool calls (skip when tool_choice="none")
    cleaned_text, tool_calls = (
        _parse_tool_calls_with_parser(parse_text, request)
        if not _suppress_tools else (parse_text or content_for_parsing, None)
    )

    # Process text format (json_schema with strict) if specified — mirrors Chat Completions behavior
    _text_format = request.text if hasattr(request, "text") else None
    if _text_format and hasattr(_text_format, 'model_dump'):
        _text_format = _text_format.model_dump(exclude_none=True)
    if _text_format and isinstance(_text_format, dict) and _text_format.get("type") not in (None, "text") and not tool_calls:
        # Build a response_format-compatible dict for parse_json_output
        _rf_type = _text_format.get("type", "")
        if _rf_type in ("json_schema", "json_object"):
            response_format = _text_format
            _out_text = cleaned_text or content_for_parsing
            _out_text, parsed_json, is_valid, error = parse_json_output(_out_text, response_format)
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                _rf_strict = _text_format.get("json_schema", {}).get("strict", False)
                if _rf_strict:
                    raise HTTPException(
                        status_code=400,
                        detail=f"response_format strict mode: {error}"
                    )
                logger.warning(f"[responses] JSON validation failed: {error}")

    # Enforce tool_choice="required": model MUST produce at least one tool call
    _resp_tool_choice = getattr(request, 'tool_choice', None)
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
        output_items.append(ResponsesOutputMessage(
            type="reasoning",
            role="assistant",
            content=[ResponsesOutputText(type="reasoning", text=reasoning_text)],
        ))

    # Add main message (use cleaned text if tool calls were extracted, otherwise raw text)
    final_text = clean_output_text(cleaned_text) if cleaned_text else ("" if tool_calls else (output.text or ""))
    if final_text:
        output_items.append(ResponsesOutputMessage(
            role="assistant",
            content=[ResponsesOutputText(text=final_text)],
        ))

    # Add tool calls as separate function_call output items (Responses API format)
    if tool_calls:
        for tc in tool_calls:
            func = tc.function if hasattr(tc, "function") else tc
            # Forward tc.id from parser so stateful tool-use loops can correlate results
            tc_call_id = tc.id if hasattr(tc, "id") and tc.id else None
            fc_kwargs = dict(
                name=func.name if hasattr(func, "name") else func.get("name", ""),
                arguments=func.arguments if hasattr(func, "arguments") else func.get("arguments", ""),
            )
            if tc_call_id:
                fc_kwargs["call_id"] = tc_call_id
            output_items.append(ResponsesFunctionCall(**fc_kwargs))

    return ResponsesObject(
        model=request.model,
        output=output_items,
        usage=_get_responses_usage(output),
        previous_response_id=request.previous_response_id,
    )


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

    for prompt_index, prompt in enumerate(prompts):
        # Per-prompt request_id so abort targets the correct engine request
        prompt_request_id = f"{response_id}-{prompt_index}"
        try:
            gen_kwargs: dict = {
                "prompt": prompt,
                "max_tokens": request.max_tokens if request.max_tokens is not None else _default_max_tokens,
                "temperature": _resolve_temperature(request.temperature),
                "top_p": _resolve_top_p(request.top_p),
                "stop": request.stop,
                "request_id": prompt_request_id,
            }
            if request.top_k is not None:
                gen_kwargs["top_k"] = request.top_k
            if request.min_p is not None:
                gen_kwargs["min_p"] = request.min_p
            if request.repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = request.repetition_penalty
            async for output in engine.stream_generate(**gen_kwargs):
                data = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": prompt_index,
                            "text": output.new_text,
                            "finish_reason": output.finish_reason if output.finished else None,
                        }
                    ],
                }
                if output.finished:
                    data["usage"] = get_usage(output).model_dump(exclude_none=True)
                yield f"data: {json.dumps(data, ensure_ascii=True)}\n\n"
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


async def _stream_with_keepalive(async_gen, interval: float = _SSE_KEEPALIVE_INTERVAL, total_timeout: float | None = None):
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
    kwargs['request_id'] = response_id

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
    _ct_kwargs = request.chat_template_kwargs or {}
    if request.enable_thinking is not None:
        _effective_thinking = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        _effective_thinking = bool(_ct_kwargs["enable_thinking"])
    elif _default_enable_thinking is not None:
        _effective_thinking = _default_enable_thinking
    else:
        _effective_thinking = None  # auto-detect below

    # Check if model's chat template injects <think> in the assistant prefix
    # Use _model_name (actual model path) not request.model (which may be "default")
    from .model_config_registry import get_model_config_registry
    _model_config = get_model_config_registry().lookup(_model_path or _model_name or request.model)
    think_in_template = _model_config.think_in_template

    # Fallback: detect from tokenizer if model config doesn't know this model.
    # If the tokenizer has <think>/<\/think> in its vocabulary, the template
    # almost certainly injects <think> in the generation prompt.
    if not think_in_template and _reasoning_parser:
        try:
            _tok = engine.tokenizer
            if getattr(_tok, 'has_thinking', False):
                think_in_template = True
                logger.info("Detected think_in_template from tokenizer vocabulary")
        except Exception:
            pass

    # S5 seed detection: some templates (e.g., Nemotron CRACK) complete the
    # thinking block in the generation prompt (<think>\nOK.\n</think>\n).
    # The model output is plain text — think_in_template must be False so
    # the parser doesn't misclassify all output as reasoning.
    if think_in_template:
        if _template_completes_thinking(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # When user explicitly disables thinking, check if template actually respects it.
    # Templates like Qwen3 honor enable_thinking=False (no <think> injected).
    # Templates like MiniMax ignore it (always inject <think>).
    # For templates that ignore it, keep think_in_template=True so the parser
    # correctly classifies reasoning, and suppress_reasoning hides it.
    if _effective_thinking is False and think_in_template:
        if not _template_always_thinks(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # Keep think_in_prompt active even when tool results are present.
    # Models like Qwen3/Qwen3.5-VL DO produce <think> blocks on follow-up
    # requests after tool execution. The parser's streaming extraction handles
    # <think>→</think>→content transitions correctly. Disabling think_in_prompt
    # here caused reasoning text to leak as visible content before the <think>
    # token accumulated in the stream.
    #
    # After the _template_always_thinks() check above, think_in_template already
    # correctly reflects reality:
    # - Template respects enable_thinking=False → think_in_template is False
    # - Template always injects <think> (e.g., MiniMax) → think_in_template stays True
    # We MUST NOT override it to False here — the parser needs think_in_prompt=True
    # to correctly classify implicit reasoning from always-thinking templates.
    # suppress_reasoning (below) handles hiding reasoning from the user.
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
    _harmony_prefix_active = "prompt_suffix" in kwargs and kwargs.get("prompt_suffix", "").startswith("<|start|>assistant<|channel|>analysis")
    request_parser = None
    if _reasoning_parser:
        request_parser = _reasoning_parser.__class__()
        request_parser.reset_state(
            think_in_prompt=effective_think_in_template,
            harmony_active=_harmony_prefix_active,
        )
        logger.debug(f"[chat] Reasoning parser: {type(request_parser).__name__} (think_in_template={effective_think_in_template}, suppress={suppress_reasoning}, harmony_active={_harmony_prefix_active})")
    else:
        logger.debug("[chat] No reasoning parser active for this request")

    # Track accumulated text for reasoning parser and tool call detection
    accumulated_text = ""
    accumulated_reasoning = ""  # Track reasoning text for fallback
    accumulated_content = ""  # Track content-only text for tool call marker detection
    streamed_content = ""  # Track content actually yielded to client (for post-stream dedup)
    content_was_emitted = False  # Whether any content chunk was actually sent
    reasoning_was_streamed = False  # Whether any reasoning_content chunk was sent

    # Tool call buffering: when we detect a tool call marker in the stream,
    # we stop emitting content and buffer the rest. At end of stream, we parse
    # the buffer for tool calls and emit them as proper tool_calls chunks.
    tool_call_buffering = False  # Are we currently buffering for tool calls?
    tool_call_buffering_notified = False  # Have we sent the buffering signal?
    tool_calls_emitted = False  # Were actual tool calls parsed and emitted?
    _suppress_tools = (getattr(request, 'tool_choice', None) == "none")
    tool_call_active = (_enable_auto_tool_choice or _tool_call_parser is not None) and not _suppress_tools

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    last_output = None

    try:
        # Stream content (with SSE keep-alive during long prefills)
        async for output in _stream_with_keepalive(engine.stream_chat(messages=messages, **kwargs), total_timeout=_default_timeout):
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

            # Always accumulate full text (needed for both reasoning and tool call parsing)
            accumulated_text += delta_text if delta_text else ""

            # Use reasoning parser if enabled
            if request_parser and delta_text:
                previous_text = accumulated_text[:-len(delta_text)] if delta_text else accumulated_text
                delta_msg = request_parser.extract_reasoning_streaming(
                    previous_text, accumulated_text, delta_text
                )

                if delta_msg is None:
                    # Skip this chunk (e.g., <think> token itself)
                    if suppress_reasoning:
                        logger.debug(f"[reasoning-debug] parser returned None for delta={repr(delta_text[:50])}, suppress={suppress_reasoning}, think_in_prompt={effective_think_in_template}")
                    continue

                # Debug: log parser output when suppress is active
                if suppress_reasoning and (delta_msg.content or delta_msg.reasoning):
                    logger.debug(f"[reasoning-debug] content={repr((delta_msg.content or '')[:30])}, reasoning={repr((delta_msg.reasoning or '')[:30])}, suppress={suppress_reasoning}")

                # Accumulate for marker detection (before buffering check)
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
                        for marker in _TOOL_CALL_MARKERS:
                            if marker in accumulated_content:
                                tool_call_buffering = True
                                break
                    if not tool_call_buffering and delta_msg.reasoning:
                        # Use trailing window (last 30 chars covers longest marker)
                        _reasoning_tail = accumulated_reasoning[-30:] if len(accumulated_reasoning) > 30 else accumulated_reasoning
                        for marker in _TOOL_CALL_MARKERS:
                            if marker in _reasoning_tail:
                                tool_call_buffering = True
                                break
                    # GPT-OSS/Harmony native tool format: to=<name> code{...}
                    # Uses regex for specificity (plain "to=" is too broad for markers list)
                    if not tool_call_buffering:
                        _tc_check = accumulated_content or (accumulated_reasoning[-30:] if accumulated_reasoning else "") or ""
                        if re.search(r'\bto=\w[\w.]*\s+code\{', _tc_check):
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
                        )
                        # Signal the client on first buffering chunk
                        if not tool_call_buffering_notified:
                            tool_call_buffering_notified = True
                            buf_chunk.tool_call_generating = True
                        yield f"data: {_dump_sse_json(buf_chunk)}\n\n"
                    continue

                # Include usage in every chunk when include_usage is on (for real-time metrics)
                chunk_usage = get_usage(output) if include_usage else None

                # When reasoning is suppressed (client requested enable_thinking=False but model forces it),
                # drop reasoning chunks entirely so the user only sees the final answer.
                # The model still thinks internally (template always injects <think>),
                # but the UI won't show any thinking text — just a brief pause then the answer.
                if suppress_reasoning:
                    emit_content = delta_msg.content  # Only emit actual content after </think>
                    emit_reasoning = None
                    # Note: reasoning tool-call markers are already detected at lines above
                    # via delta_msg.reasoning check — no need to add reasoning to accumulated_content
                else:
                    emit_reasoning = delta_msg.reasoning
                    emit_content = delta_msg.content

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
                            finish_reason=output.finish_reason if output.finished else None,
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
                    for marker in _TOOL_CALL_MARKERS:
                        if marker in accumulated_text:
                            tool_call_buffering = True
                            break

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
                        )
                        if not tool_call_buffering_notified:
                            tool_call_buffering_notified = True
                            buf_chunk.tool_call_generating = True
                        yield f"data: {_dump_sse_json(buf_chunk)}\n\n"
                    continue

                # Add <think> prefix on first content chunk for thinking models
                if is_thinking_model and not think_prefix_sent and content:
                    content = "<think>" + content
                    think_prefix_sent = True

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
                            finish_reason=output.finish_reason if output.finished else None,
                        )
                    ],
                    usage=chunk_usage,
                )
                yield f"data: {_dump_sse_json(chunk)}\n\n"
            else:
                # Reasoning parser is active but delta_text is empty/None
                # (progress-only engine chunk). Skip — don't emit empty delta.
                continue

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
                "message": f"Stream generation failed: {e}",
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
            if cached_tokens > 0:
                _err_usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)
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
            parse_text = re.sub(r'<think>.*?</think>', '', accumulated_text, flags=re.DOTALL)
            if parse_text == accumulated_text and '</think>' in parse_text:
                # Implicit mode: strip everything before </think>
                _, _, parse_text = parse_text.partition('</think>')
            parse_text = parse_text.strip()
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(parse_text or accumulated_text, request)
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
                    remainder = candidate[len(already_sent):].strip()
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

            # Emit tool calls as proper structured chunk
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
            tool_chunk = ChatCompletionChunk(
                id=response_id,
                created=_created_ts,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(tool_calls=tc_deltas),
                        finish_reason="tool_calls",
                    )
                ],
            )
            yield f"data: {_dump_sse_json(tool_chunk)}\n\n"
            # Skip normal end-of-stream handling — we already set finish_reason
            if include_usage:
                _tc_usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                if cached_tokens > 0:
                    _tc_usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)
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
            full = accumulated_content.strip() if request_parser else accumulated_text.strip()
            if already_sent and full.startswith(already_sent):
                remainder = full[len(already_sent):].strip()
            else:
                remainder = full if not content_was_emitted else ""
            if remainder:
                # Use the engine's actual finish_reason (e.g., "length" if max_tokens
                # was hit) instead of hardcoding "stop"
                _flush_reason = (last_output.finish_reason if last_output and last_output.finish_reason else "stop")
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
    if request_parser and not content_was_emitted and accumulated_reasoning and not reasoning_was_streamed and not suppress_reasoning:
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
    # emit a diagnostic so the client isn't left with a completely silent empty response.
    if suppress_reasoning and not content_was_emitted and accumulated_reasoning:
        logger.info(f"Request {response_id}: model produced only reasoning ({len(accumulated_reasoning)} chars) — suppressed per user setting")
        diag_chunk = ChatCompletionChunk(
            id=response_id,
            created=_created_ts,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        content="[Model produced only internal reasoning with no visible response. Try enabling thinking to see the reasoning, or rephrase your prompt.]",
                    ),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {_dump_sse_json(diag_chunk)}\n\n"

    # Safeguard: if model generated zero tokens (empty stream), emit a diagnostic
    # chunk so clients always get feedback instead of a silent empty response.
    # Also catches the case where engine yielded an output with 0 completion tokens
    # (e.g., immediate EOS) — last_output is set but nothing was emitted.
    _zero_tokens = (last_output is None) or (
        not content_was_emitted and not accumulated_reasoning
        and getattr(last_output, 'completion_tokens', 1) == 0
    )
    if _zero_tokens and not content_was_emitted and not accumulated_reasoning:
        logger.warning(f"Request {response_id}: model generated zero tokens")
        empty_chunk = ChatCompletionChunk(
            id=response_id,
            created=_created_ts,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        content="[Model produced no response. Check server logs for details.]",
                    ),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {_dump_sse_json(empty_chunk)}\n\n"

    # Enforce tool_choice="required" in streaming: emit error SSE if no tool calls
    if getattr(request, 'tool_choice', None) == "required" and not tool_calls_emitted:
        # tool_calls_emitted is set True only when tool calls were actually parsed and yielded.
        # tool_call_buffering can be True even on false-positive marker detection with no actual calls.
        logger.warning(f"Stream {response_id}: tool_choice='required' but no tool calls produced")
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
    _rf = getattr(request, 'response_format', None)
    if _rf and content_was_emitted and not tool_call_buffering:
        _final_text = accumulated_content.strip() if request_parser else accumulated_text.strip()
        if _final_text:
            _, parsed_json, is_valid, rf_error = parse_json_output(_final_text, _rf)
            if not is_valid:
                _rf_strict = False
                if isinstance(_rf, dict):
                    _rf_strict = _rf.get("json_schema", {}).get("strict", False)
                elif hasattr(_rf, "json_schema") and _rf.json_schema:
                    _rf_strict = getattr(_rf.json_schema, "strict", False)
                if _rf_strict:
                    logger.warning(f"Stream {response_id}: JSON schema validation failed (strict): {rf_error}")
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
                    logger.warning(f"Stream {response_id}: JSON validation failed: {rf_error}")

    # Send final chunk with usage if requested
    if include_usage:
        _usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if cached_tokens > 0:
            _usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)
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
    kwargs['request_id'] = response_id
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
    yield _sse("response.created", {
        "type": "response.created",
        "response": {
            "id": response_id, "object": "response", "created_at": created_at,
            "status": "in_progress", "model": request.model, "output": [],
        },
    })

    # Create text message output item immediately for streaming
    msg_id = f"item_{uuid.uuid4().hex[:12]}"
    yield _sse("response.output_item.added", {
        "type": "response.output_item.added", "output_index": 0,
        "item": {"id": msg_id, "type": "message", "status": "in_progress",
                 "role": "assistant", "content": []},
    })
    yield _sse("response.content_part.added", {
        "type": "response.content_part.added", "item_id": msg_id,
        "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })

    _suppress_tools = (getattr(request, 'tool_choice', None) == "none")
    tool_call_active = (_enable_auto_tool_choice or _tool_call_parser is not None) and not _suppress_tools
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
    last_output = None  # Track last engine output for finish_reason (status: incomplete)

    # Resolve effective enable_thinking:
    # Priority: top-level field > chat_template_kwargs > server default > auto-detect
    _ct_kwargs = request.chat_template_kwargs or {}
    if request.enable_thinking is not None:
        _effective_thinking = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        _effective_thinking = bool(_ct_kwargs["enable_thinking"])
    elif _default_enable_thinking is not None:
        _effective_thinking = _default_enable_thinking
    else:
        _effective_thinking = None  # auto-detect below

    # Reasoning parser setup (mirrors stream_chat_completion)
    from .model_config_registry import get_model_config_registry
    _model_config = get_model_config_registry().lookup(_model_path or _model_name or request.model)
    think_in_template = _model_config.think_in_template

    # Fallback: detect from tokenizer if model config doesn't know this model
    if not think_in_template and _reasoning_parser:
        try:
            _tok = engine.tokenizer
            if getattr(_tok, 'has_thinking', False):
                think_in_template = True
        except Exception:
            pass

    # S5 seed detection (mirrors Chat Completions path)
    if think_in_template:
        if _template_completes_thinking(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # When user explicitly disables thinking, check if template actually respects it
    if _effective_thinking is False and think_in_template:
        if not _template_always_thinks(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # Keep think_in_prompt active even when tool results are present.
    # (Same rationale as Chat Completions path — see stream_chat_completion.)
    # After _template_always_thinks() above, think_in_template already correctly
    # reflects reality — do NOT override to False for always-thinking templates.
    effective_think_in_template = think_in_template

    # For thinking models without reasoning parser, prepend <think>
    is_thinking_model = effective_think_in_template and not _reasoning_parser
    think_prefix_sent = False

    # Suppress reasoning output when user explicitly disabled thinking
    suppress_reasoning = _effective_thinking is False

    # Create a per-request parser instance to avoid mutable-state conflicts
    # when multiple requests stream concurrently.
    _harmony_prefix_active = "prompt_suffix" in kwargs and kwargs.get("prompt_suffix", "").startswith("<|start|>assistant<|channel|>analysis")
    request_parser = None
    if _reasoning_parser:
        request_parser = _reasoning_parser.__class__()
        request_parser.reset_state(
            think_in_prompt=effective_think_in_template,
            harmony_active=_harmony_prefix_active,
        )
        logger.debug(f"[responses] Reasoning parser: {type(request_parser).__name__} (think_in_template={effective_think_in_template}, suppress={suppress_reasoning}, harmony_active={_harmony_prefix_active})")
    else:
        logger.debug("[responses] No reasoning parser active for this request")

    try:
        async for output in _stream_with_keepalive(engine.stream_chat(messages=messages, **kwargs), total_timeout=_default_timeout):
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

            if delta_text:
                full_text += delta_text

                # Use reasoning parser if enabled
                if request_parser:
                    previous_text = full_text[:-len(delta_text)] if delta_text else full_text
                    delta_msg = request_parser.extract_reasoning_streaming(
                        previous_text, full_text, delta_text
                    )

                    if delta_msg is None:
                        # Skip this chunk entirely (e.g., <think> token itself)
                        # Must use continue (not pass) to avoid falling through
                        # to token tracking and usage emission below.
                        continue
                    else:
                        # Accumulate for marker detection (before buffering check)
                        if delta_msg.content:
                            accumulated_content += delta_msg.content
                        if delta_msg.reasoning:
                            accumulated_reasoning += delta_msg.reasoning

                        # Check for tool call markers in content and reasoning
                        if tool_call_active and not tool_call_buffering:
                            if delta_msg.content and accumulated_content:
                                for marker in _TOOL_CALL_MARKERS:
                                    if marker in accumulated_content:
                                        tool_call_buffering = True
                                        break
                            if not tool_call_buffering and delta_msg.reasoning:
                                _reasoning_tail = accumulated_reasoning[-30:] if len(accumulated_reasoning) > 30 else accumulated_reasoning
                                for marker in _TOOL_CALL_MARKERS:
                                    if marker in _reasoning_tail:
                                        tool_call_buffering = True
                                        break
                            # GPT-OSS/Harmony native tool format: to=<name> code{...}
                            if not tool_call_buffering:
                                _tc_check = accumulated_content or (accumulated_reasoning[-30:] if accumulated_reasoning else "") or ""
                                if re.search(r'\bto=\w[\w.]*\s+code\{', _tc_check):
                                    tool_call_buffering = True

                        if tool_call_buffering:
                            # Emit heartbeat during tool call buffering so the client
                            # sees activity (matches ChatCompletion heartbeat behavior).
                            yield _sse("response.heartbeat", {
                                "type": "response.heartbeat",
                            })
                            continue

                        if not tool_call_buffering:
                            # When reasoning is suppressed (client requested enable_thinking=False
                            # but model forces it), drop reasoning entirely so the user only
                            # sees the final answer after the model finishes thinking.
                            if suppress_reasoning:
                                emit_content = delta_msg.content  # Only actual content after </think>
                                emit_reasoning = None
                                # Note: reasoning tool-call markers are already detected above
                                # via delta_msg.reasoning check — no need to add reasoning to accumulated_content
                            else:
                                emit_reasoning = delta_msg.reasoning
                                emit_content = delta_msg.content

                            # Emit reasoning as custom event
                            if emit_reasoning:
                                reasoning_was_streamed = True
                                yield _sse("response.reasoning.delta", {
                                    "type": "response.reasoning.delta",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "delta": emit_reasoning,
                                })
                            # Emit content as standard text delta
                            if emit_content:
                                content_was_emitted = True
                                streamed_text += emit_content
                                yield _sse("response.output_text.delta", {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id,
                                    "output_index": 0, "content_index": 0,
                                    "delta": emit_content,
                                })
                else:
                    # Standard path without reasoning parsing
                    content = delta_text

                    # Check for tool call markers
                    if tool_call_active and not tool_call_buffering:
                        for marker in _TOOL_CALL_MARKERS:
                            if marker in full_text:
                                tool_call_buffering = True
                                break

                    if tool_call_buffering:
                        # Emit heartbeat during tool call buffering (standard path)
                        yield _sse("response.heartbeat", {
                            "type": "response.heartbeat",
                        })
                        continue
                    else:
                        # Add <think> prefix on first chunk for thinking models
                        if is_thinking_model and not think_prefix_sent and content:
                            content = "<think>" + content
                            think_prefix_sent = True

                        if content:
                            content_was_emitted = True
                            streamed_text += content
                            yield _sse("response.output_text.delta", {
                                "type": "response.output_text.delta", "item_id": msg_id,
                                "output_index": 0, "content_index": 0,
                                "delta": content,
                            })

            # Emit per-chunk usage when include_usage is enabled (for real-time metrics)
            if include_usage and (prompt_tokens or completion_tokens):
                usage_obj = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                if _cached > 0:
                    usage_obj["input_tokens_details"] = {"cached_tokens": _cached}
                yield _sse("response.usage", {
                    "type": "response.usage",
                    "usage": usage_obj,
                })
    except Exception as e:
        # On any error, abort the request and emit an error SSE event
        # so the client knows what happened instead of a silent EOF.
        logger.error(f"Stream error for {response_id}: {e}", exc_info=True)
        if hasattr(engine, "abort_request"):
            await engine.abort_request(response_id)
        yield _sse("error", {
            "type": "error",
            "error": {
                "type": "server_error",
                "message": f"Stream generation failed: {e}",
                "code": "internal_error",
            },
        })
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
        yield _sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": response_id, "object": "response",
                "status": "failed",
                "error": {"type": "server_error", "message": str(e)},
                **_err_usage,
            },
        })
        return

    # Emit reasoning done event if reasoning was produced (skip when suppressed)
    if accumulated_reasoning and not suppress_reasoning:
        yield _sse("response.reasoning.done", {
            "type": "response.reasoning.done",
            "item_id": msg_id,
            "output_index": 0,
            "reasoning": accumulated_reasoning,
        })

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
            parse_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)
            if parse_text == full_text and '</think>' in full_text:
                # Implicit mode: strip everything before </think>
                _, _, parse_text = full_text.partition('</think>')
            parse_text = parse_text.strip()
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(parse_text or full_text, request)

    display_text = ""

    if tool_calls:
        # Apply reasoning parser to the cleaned (pre-tool-call) text
        if request_parser and cleaned_text:
            reasoning_text, content_text = request_parser.extract_reasoning(cleaned_text)
            if content_text:
                cleaned_text = content_text
            elif reasoning_text:
                cleaned_text = reasoning_text

        # Finalize the text message with whatever content was before the tool call
        final_text = (cleaned_text or "").strip()
        yield _sse("response.output_text.done", {
            "type": "response.output_text.done", "item_id": msg_id,
            "output_index": output_index, "content_index": 0, "text": final_text,
        })
        yield _sse("response.content_part.done", {
            "type": "response.content_part.done", "item_id": msg_id,
            "output_index": output_index, "content_index": 0,
            "part": {"type": "output_text", "text": final_text, "annotations": []},
        })
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": output_index,
            "item": {"id": msg_id, "type": "message", "status": "completed",
                     "role": "assistant",
                     "content": [{"type": "output_text", "text": final_text, "annotations": []}]},
        })
        if final_text:
            all_output_items.append({
                "id": msg_id, "type": "message", "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text, "annotations": []}],
            })
            output_index += 1

        # Emit each tool call as a function_call output item
        for tc in tool_calls:
            func = tc.function if hasattr(tc, "function") else tc
            tc_name = func.name if hasattr(func, "name") else func.get("name", "")
            tc_args = func.arguments if hasattr(func, "arguments") else func.get("arguments", "")
            call_id = tc.id if hasattr(tc, "id") else f"call_{uuid.uuid4().hex[:8]}"
            fc_id = f"fc_{uuid.uuid4().hex[:12]}"

            yield _sse("response.output_item.added", {
                "type": "response.output_item.added", "output_index": output_index,
                "item": {"id": fc_id, "type": "function_call", "status": "in_progress",
                         "call_id": call_id, "name": tc_name, "arguments": ""},
            })
            # Emit arguments incrementally in ~16-char chunks per OpenAI spec
            _ARG_CHUNK = 16
            if tc_args:
                for _ci in range(0, len(tc_args), _ARG_CHUNK):
                    yield _sse("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta", "item_id": fc_id,
                        "output_index": output_index, "delta": tc_args[_ci:_ci + _ARG_CHUNK],
                    })
            else:
                # Empty arguments — emit one delta to satisfy clients
                yield _sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta", "item_id": fc_id,
                    "output_index": output_index, "delta": "",
                })
            yield _sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done", "item_id": fc_id,
                "output_index": output_index, "arguments": tc_args,
            })
            fc_item = {
                "id": fc_id, "type": "function_call", "status": "completed",
                "call_id": call_id, "name": tc_name, "arguments": tc_args,
            }
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done", "output_index": output_index,
                "item": fc_item,
            })
            all_output_items.append(fc_item)
            output_index += 1
    else:
        # No tool calls — use content accumulated during streaming (reasoning already separated)
        display_text = cleaned_text or ""
        if request_parser:
            # Reasoning was already emitted during streaming — use content-only text
            if accumulated_content:
                display_text = accumulated_content
            elif accumulated_reasoning and not reasoning_was_streamed and not suppress_reasoning:
                # Model only produced reasoning with no content — use as fallback.
                # Skip if reasoning was already streamed as deltas (avoids duplication).
                display_text = accumulated_reasoning
            elif display_text:
                # Fallback: re-parse if streaming didn't accumulate
                reasoning_text, content_text = request_parser.extract_reasoning(display_text)
                if content_text:
                    display_text = content_text
                elif reasoning_text:
                    display_text = reasoning_text

        if not display_text:
            display_text = clean_output_text(full_text) if full_text else ""
        if not display_text:
            # If reasoning was suppressed and model produced reasoning, the response
            # is intentionally empty — don't show error fallback
            if suppress_reasoning and accumulated_reasoning:
                display_text = "[Model produced only internal reasoning with no visible response. Try enabling thinking to see the reasoning, or rephrase your prompt.]"
                logger.info(f"Request {response_id}: model produced only reasoning ({len(accumulated_reasoning)} chars) — suppressed per user setting")
            else:
                display_text = "[Model produced no response. Check server logs for details.]"
                logger.warning(f"Request {response_id}: empty response in Responses API")

        yield _sse("response.output_text.done", {
            "type": "response.output_text.done", "item_id": msg_id,
            "output_index": output_index, "content_index": 0, "text": display_text,
        })
        yield _sse("response.content_part.done", {
            "type": "response.content_part.done", "item_id": msg_id,
            "output_index": output_index, "content_index": 0,
            "part": {"type": "output_text", "text": display_text, "annotations": []},
        })
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": output_index,
            "item": {"id": msg_id, "type": "message", "status": "completed",
                     "role": "assistant",
                     "content": [{"type": "output_text", "text": display_text, "annotations": []}]},
        })
        all_output_items.append({
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": display_text, "annotations": []}],
        })

    # H4: Validate text format (json_schema/json_object) at end of stream.
    _text_fmt = getattr(request, 'text', None)
    # Convert Pydantic model to dict for uniform access
    if _text_fmt and hasattr(_text_fmt, 'model_dump'):
        _text_fmt = _text_fmt.model_dump(exclude_none=True)
    if _text_fmt and isinstance(_text_fmt, dict) and _text_fmt.get("type") not in (None, "text"):
        _rf_type = _text_fmt.get("type", "")
        _FALLBACK_MSG = "[Model produced no response. Check server logs for details.]"
        if _rf_type in ("json_schema", "json_object") and display_text and display_text != _FALLBACK_MSG:
            _, _pj, _valid, _err = parse_json_output(display_text, _text_fmt)
            if not _valid:
                _strict = _text_fmt.get("json_schema", {}).get("strict", False)
                if _strict:
                    logger.warning(f"Stream {response_id}: JSON schema validation failed (strict): {_err}")
                    yield _sse("error", {
                        "type": "error",
                        "message": f"response_format strict mode: {_err}",
                        "code": "json_validation_failed",
                    })
                else:
                    logger.warning(f"Stream {response_id}: JSON validation failed: {_err}")

    # Enforce tool_choice="required" in Responses API streaming
    _resp_stream_tc = getattr(request, 'tool_choice', None)
    if _resp_stream_tc == "required" and not tool_calls:
        logger.warning(f"Stream {response_id}: tool_choice='required' but no tool calls produced")
        yield _sse("error", {
            "type": "error",
            "message": (
                "tool_choice='required' was set but the model did not produce "
                "any tool calls. Try rephrasing your prompt or using a model "
                "with better tool-calling support."
            ),
            "code": "tool_calls_required",
        })

    # Emit response.completed — use "incomplete" status when max_tokens was hit
    _resp_finish = getattr(last_output, 'finish_reason', None) if last_output else None
    _resp_status = "incomplete" if _resp_finish == "length" else "completed"
    _resp_extra: dict = {}
    if _resp_status == "incomplete":
        _resp_extra["incomplete_details"] = {"reason": "max_output_tokens"}
    yield _sse("response.completed", {
        "type": "response.completed",
        "response": {
            "id": response_id, "object": "response", "created_at": created_at,
            "status": _resp_status, "model": request.model,
            "output": all_output_items,
            **_resp_extra,
            "usage": {
                "input_tokens": prompt_tokens, "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                **({"input_tokens_details": {"cached_tokens": _cached}} if _cached > 0 else {}),
            },
        },
    })


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager

    try:
        from vmlx_engine.mcp import MCPClientManager, load_mcp_config

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        logger.info(f"MCP initialized with {len(_mcp_manager.get_all_tools())} tools")

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
        default="0.0.0.0",
        help="Host to bind to",
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
        action="store_true",
        help="Enable continuous batching for multiple concurrent users",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation",
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

    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _default_temperature, _default_top_p, _default_enable_thinking
    _api_key = args.api_key or os.environ.get("VLLM_API_KEY")
    _default_timeout = args.timeout
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p
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
            logger.info(f"Auto-detected reasoning parser: {parser_name} (from model config)")
    elif parser_name == "none":
        # Explicitly disabled by user
        parser_name = None
        logger.info("Reasoning parser explicitly disabled via --reasoning-parser none")
    elif parser_name == "auto":
        detected = registry.get_reasoning_parser(model_name)
        if detected:
            parser_name = detected
            logger.info(f"Auto-detected reasoning parser: {parser_name} (from model name)")
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
                logger.info(f"Auto-detected tool call parser: {_tool_call_parser} (from model name)")
            else:
                logger.info("Tool call parser 'auto': no specific parser detected, using generic")
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
        force_mllm=args.mllm,
        served_model_name=args.served_model_name,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
