# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for vllm-mlx.

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
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

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
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
_engine: BaseEngine | None = None
_model_name: str | None = None
_default_max_tokens: int = 32768
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p

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
]


def _template_always_thinks(tokenizer, model_name: str) -> bool:
    """Check if model's template injects <think> even when enable_thinking=False.

    For templates that ignore the flag, we keep think_in_template=True so the
    streaming parser correctly classifies reasoning (suppress_reasoning hides it).
    Results are cached per model name since the template doesn't change at runtime.
    """
    if model_name in _template_always_thinks_cache:
        return _template_always_thinks_cache[model_name]

    # Qwen models natively inject <think> but they are explicitly patched in batched.py/mllm.py to strip it.
    if "qwen" in model_name.lower() or "exploit" in model_name.lower():
        _template_always_thinks_cache[model_name] = False
        return False

    result = False
    try:
        test_msgs = [{"role": "user", "content": "__test__"}]
        rendered = tokenizer.apply_chat_template(
            test_msgs, enable_thinking=False,
            add_generation_prompt=True, tokenize=False,
        )
        # Check if <think> appears after the user message (in the generation prefix)
        after_user = rendered.rsplit("__test__", 1)[-1]
        result = "<think>" in after_user
        if result:
            logger.info(
                f"Template for {model_name} always injects <think> "
                "(ignores enable_thinking=False)"
            )
    except Exception as e:
        logger.debug(f"_template_always_thinks check failed for {model_name}: {e}")

    _template_always_thinks_cache[model_name] = result
    return result


# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_parser_instance = None  # Instantiated parser


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
            _mc = get_model_config_registry().lookup(_model_name or "")
            if _mc.chat_template_custom and _engine.tokenizer:
                _engine.tokenizer.chat_template = _mc.chat_template_custom
                logger.info(f"Applied custom chat template for {_mc.family_name} model (batched)")
        except Exception as e:
            logger.warning(f"Failed to apply custom chat template (batched): {e}")

    # Initialize MCP if config provided
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(mcp_config)

    yield

    # Shutdown: Close MCP connections and stop engine
    if _mcp_manager is not None:
        await _mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _engine is not None:
        await _engine.stop()
        logger.info("Engine stopped")


app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.2.1",
    lifespan=lifespan,
)

security = HTTPBearer(auto_error=False)


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


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    # Use API key as client ID if available, otherwise use IP
    client_id = request.headers.get(
        "Authorization", request.client.host if request.client else "unknown"
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
    global _tool_parser_instance

    # If auto tool choice is not enabled, use the generic parser
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return parse_tool_calls(output_text)

    # Initialize parser if needed
    if _tool_parser_instance is None:
        try:
            parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
            # Get tokenizer from engine if available
            tokenizer = None
            if _engine is not None and hasattr(_engine, "_tokenizer"):
                tokenizer = _engine._tokenizer
            _tool_parser_instance = parser_cls(tokenizer)
            logger.info(f"Initialized tool call parser: {_tool_call_parser}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize tool parser '{_tool_call_parser}': {e}"
            )
            logger.warning("Falling back to generic parser")
            return parse_tool_calls(output_text)

    # Use the configured parser, fall back to generic if it finds nothing
    try:
        # Reset parser state between requests
        _tool_parser_instance.reset()
        # Convert request to dict format for parsers that need schema info (e.g., Step3p5 type coercion)
        parser_request = None
        if request and request.tools:
            parser_request = {"tools": convert_tools_for_template(request.tools)}
        result = _tool_parser_instance.extract_tool_calls(output_text, request=parser_request)
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

    Returns:
        True if native format should be preserved
    """
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        return parser_cls.supports_native_format()
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
        "/Users/eric/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit"
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


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
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
    global _engine, _model_name, _default_max_tokens, _tool_parser_instance

    _default_max_tokens = max_tokens
    # Normalize model name: extract "org/model" from full local paths
    # e.g. "/Users/eric/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit"
    #    → "mlx-community/Llama-3.2-3B-Instruct-4bit"
    _model_name = _normalize_model_name(model_name)
    # Reset tool parser instance when model is reloaded (tokenizer may change)
    _tool_parser_instance = None

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
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_engine.start())
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

    # Apply chat template override from model config registry (e.g. Harmony for GPT-OSS)
    try:
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(model_name)
        if _mc.chat_template_custom and _engine and _engine.tokenizer:
            _engine.tokenizer.chat_template = _mc.chat_template_custom
            logger.info(f"Applied custom chat template for {_mc.family_name} model")
    except Exception as e:
        logger.warning(f"Failed to apply custom chat template: {e}")

    # Log Metal GPU memory after model load
    try:
        import mlx.core as mx
        if hasattr(mx, "get_active_memory"):
            active_gb = mx.get_active_memory() / (1024 ** 3)
            peak_gb = mx.get_peak_memory() / (1024 ** 3)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            active_gb = mx.metal.get_active_memory() / (1024 ** 3)
            peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)
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
    except (ImportError, Exception):
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
    status = "healthy" if _engine is not None else "no_model"

    # Include Metal GPU memory info when available
    memory_info = None
    try:
        import mlx.core as mx
        if hasattr(mx, "get_active_memory"):
            active = mx.get_active_memory()
            peak = mx.get_peak_memory()
            cache = mx.get_cache_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            active = mx.metal.get_active_memory()
            peak = mx.metal.get_peak_memory()
            cache = mx.metal.get_cache_memory()
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

    result = {
        "status": status,
        "model_loaded": _engine is not None,
        "model_name": _model_name,
        "model_type": "mllm" if (_engine and _engine.is_mllm) else "llm",
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }
    if memory_info:
        result["memory"] = memory_info
    if kv_quant_info:
        result["kv_cache_quantization"] = kv_quant_info

    return result


def _get_scheduler():
    """Get the scheduler from the engine, or None if not available."""
    if _engine is None:
        return None
    # BatchedEngine._engine is AsyncEngineCore, which has .engine (EngineCore)
    # EngineCore has .scheduler
    async_core = getattr(_engine, "_engine", None)
    if async_core is not None:
        engine_core = getattr(async_core, "engine", None)
        if engine_core is not None:
            return getattr(engine_core, "scheduler", None)
    return None


@app.get("/v1/cache/stats")
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

    # Disk cache (L2) stats
    if scheduler and scheduler.disk_cache is not None:
        result["disk_cache"] = scheduler.disk_cache.stats()

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

    if not result:
        return {"error": "No cache stats available"}
    return result


@app.get("/v1/cache/entries")
async def cache_entries():
    """List cached prefix entries with metadata."""
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"error": "Scheduler not available (SimpleEngine mode)"}

    entries = []
    cache_type = "none"

    if scheduler.memory_aware_cache is not None:
        cache_type = "memory_aware"
        from .memory_cache import _BYTES_PER_MB
        for tokens_key, entry in scheduler.memory_aware_cache._entries.items():
            entries.append({
                "tokens_count": len(tokens_key),
                "memory_bytes": entry.memory_bytes,
                "memory_mb": round(entry.memory_bytes / _BYTES_PER_MB, 2),
                "cache_type": "memory_aware",
            })
    elif scheduler.block_aware_cache is not None:
        cache_type = "paged"
        for block_id, block in scheduler.block_aware_cache.paged_cache.allocated_blocks.items():
            entries.append({
                "block_id": block_id,
                "tokens_count": block.token_count,
                "ref_count": block.ref_count,
                "has_data": block.cache_data is not None,
                "cache_type": "paged",
            })
    elif scheduler.prefix_cache is not None:
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


@app.post("/v1/cache/warm")
async def cache_warm(request: dict):
    """
    Warm the prefix cache by pre-computing KV states for given prompts.

    Note: This runs model inference synchronously and will block other
    requests while warming. Use for system prompts before serving traffic.

    Request body: {"prompts": ["system prompt text", ...]}
    """
    scheduler = _get_scheduler()
    if scheduler is None:
        return {"error": "Scheduler not available (SimpleEngine mode)"}

    prompts = request.get("prompts", [])
    if not prompts:
        return {"error": "No prompts provided"}

    warmed = 0
    token_counts = []
    errors = []

    for i, prompt in enumerate(prompts):
        try:
            # Tokenize the prompt
            if hasattr(scheduler, "_actual_tokenizer"):
                tokens = scheduler._actual_tokenizer.encode(prompt)
            else:
                tokens = scheduler.tokenizer.encode(prompt)

            if not tokens:
                errors.append(f"Prompt {i}: empty after tokenization")
                continue

            # Run prefill to get cache state
            cache_tokens = tokens[:-1]  # Store N-1 tokens
            if not cache_tokens:
                errors.append(f"Prompt {i}: too short (1 token)")
                continue

            cache = scheduler._prefill_for_prompt_only_cache(cache_tokens)
            if cache is None:
                errors.append(f"Prompt {i}: prefill failed")
                continue

            # Store in the appropriate cache
            stored = False
            if scheduler.memory_aware_cache is not None:
                stored = scheduler.memory_aware_cache.store(tokens, cache)
            elif scheduler.block_aware_cache is not None:
                extracted = scheduler._extract_cache_states(cache)
                if extracted:
                    result = scheduler.block_aware_cache.store_cache(
                        f"warm-{i}-{uuid.uuid4().hex[:8]}", tokens, extracted
                    )
                    stored = result is not None
            elif scheduler.prefix_cache is not None:
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

    return {
        "warmed": warmed,
        "token_counts": token_counts,
        "errors": errors if errors else None,
    }


@app.delete("/v1/cache")
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
            if scheduler.memory_aware_cache is not None:
                scheduler.memory_aware_cache.clear()
                cleared.append("memory_aware_prefix")
            if scheduler.block_aware_cache is not None:
                scheduler.block_aware_cache.clear()
                cleared.append("paged_prefix")
            if scheduler.prefix_cache is not None:
                scheduler.prefix_cache.clear()
                cleared.append("legacy_prefix")
            if scheduler.disk_cache is not None:
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
    engine = get_engine()

    # Abort the request (returns True if found, False if not found/already finished)
    success = await engine.abort_request(request_id)

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
    engine = get_engine()
    success = await engine.abort_request(response_id)

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
    engine = get_engine()
    success = await engine.abort_request(request_id)

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
    if _model_name:
        models.append(ModelInfo(id=_model_name))
    return ModelsResponse(data=models)


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

        # Lazy-load or swap embedding engine
        load_embedding_model(model_name, lock=False, reuse_existing=True)

        # Normalise input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

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


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            return result.text

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
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
        model_name = model_map.get(model, model)

        # Load engine if needed
        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
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

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    if request.stream:
        return StreamingResponse(
            stream_completion(engine, prompts[0], request),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or _default_timeout
    choices = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    for i, prompt in enumerate(prompts):
        try:
            gen_kwargs = {
                    "prompt": prompt,
                    "max_tokens": request.max_tokens or _default_max_tokens,
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
    engine = get_engine()

    # For MLLM models, keep original messages with embedded images
    # (MLLM.chat() extracts images from message content internally)
    if engine.is_mllm:
        # Convert Pydantic messages to dicts preserving full content
        messages = []
        for msg in request.messages:
            msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
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

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": request.max_tokens or _default_max_tokens,
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
    # Priority: top-level field > chat_template_kwargs > auto-detect from model config
    _ct_kwargs = request.chat_template_kwargs or {}
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        chat_kwargs["enable_thinking"] = bool(_ct_kwargs["enable_thinking"])
    else:
        # Auto-detect from model config + tokenizer fallback
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(_model_name or request.model)
        _enable = _mc.think_in_template
        if not _enable:
            # Models with a reasoning_parser are reasoning models — enable thinking
            # so the appropriate prefix/protocol is activated (e.g., Harmony analysis channel)
            if _mc.reasoning_parser:
                _enable = True
            else:
                try:
                    _tok = engine.tokenizer
                    if getattr(_tok, 'has_thinking', False):
                        _enable = True
                except Exception:
                    pass
        chat_kwargs["enable_thinking"] = _enable

    # Pass reasoning_effort if provided (for GPT-OSS and models that support thinking levels)
    if request.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = request.reasoning_effort

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            chat_kwargs["chat_template_kwargs"] = extra_ct

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Merge MCP tools with user-provided tools
    all_tools = []
    
    # Add MCP tools if available
    if _mcp_manager is not None:
        mcp_tools = _mcp_manager.get_all_tools_openai()
        all_tools.extend(mcp_tools)
        if mcp_tools:
            logger.debug(f"Added {len(mcp_tools)} MCP tools")
    
    # Add user-provided tools
    if request.tools:
        all_tools.extend(request.tools)
        logger.debug(f"Added {len(request.tools)} user tools")

    # Pass merged tools to engine (normalize all to template format)
    if all_tools:
        chat_kwargs["tools"] = convert_tools_for_template(all_tools)

    # Inject Harmony analysis prefix for GPT-OSS models when thinking is enabled.
    # The suffix replaces the template's generation prompt (<|start|>assistant<|message|>)
    # with the analysis channel prefix to guide the model into reasoning mode.
    from .reasoning.gptoss_parser import GptOssReasoningParser
    if isinstance(_reasoning_parser, GptOssReasoningParser):
        _think_val = chat_kwargs.get("enable_thinking")
        if _think_val is True:
            # Build analysis prefix with optional effort hint
            _effort = chat_kwargs.get("reasoning_effort", "").lower()
            if _effort == "low":
                _analysis_hint = "<|start|>assistant<|channel|>analysis (brief)<|message|>"
            elif _effort == "high":
                _analysis_hint = "<|start|>assistant<|channel|>analysis (thorough)<|message|>"
            else:
                _analysis_hint = "<|start|>assistant<|channel|>analysis<|message|>"
            chat_kwargs["prompt_suffix"] = _analysis_hint
            # Skip template's generation prompt — suffix provides the full assistant prefix
            chat_kwargs["skip_generation_prompt"] = True
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
    timeout = request.timeout or _default_timeout

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

    # Strip think tags before tool call parsing — reasoning text may precede
    # tool calls in thinking model output
    _cc_parse_text = re.sub(r'<think>.*?</think>', '', output.text, flags=re.DOTALL)
    if _cc_parse_text == output.text and '</think>' in output.text:
        _, _, _cc_parse_text = output.text.partition('</think>')
    _cc_parse_text = _cc_parse_text.strip()

    # Parse tool calls from output using configured parser
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(_cc_parse_text or output.text, request)

    # Process response_format if specified
    if response_format and not tool_calls:
        cleaned_text, parsed_json, is_valid, error = parse_json_output(
            cleaned_text or output.text, response_format
        )
        if parsed_json is not None:
            # Return JSON as string
            cleaned_text = json.dumps(parsed_json)
        if not is_valid:
            logger.warning(f"JSON validation failed: {error}")

    # Extract reasoning content if parser is enabled
    # Always run reasoning extraction, even when tool calls are present —
    # thinking models emit <think>reasoning</think> before tool call text
    reasoning_text = None
    if _reasoning_parser:
        text_to_parse = cleaned_text or output.text
        reasoning_text, remaining_text = _reasoning_parser.extract_reasoning(
            text_to_parse
        )
        if remaining_text is not None:
            cleaned_text = remaining_text
        # Suppress reasoning when user explicitly disabled thinking
        # Check both top-level and chat_template_kwargs
        _suppress = request.enable_thinking is False
        if not _suppress and request.chat_template_kwargs:
            _suppress = request.chat_template_kwargs.get("enable_thinking") is False
        if _suppress:
            reasoning_text = None

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
            # Check if any part has image/video — if so, preserve the full array
            has_media = any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image", "video_url", "video")
                for p in raw_content
            )
            if has_media:
                return raw_content
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

    for item in input_data:
        if not isinstance(item, dict):
            if hasattr(item, "role"):
                raw = item.content if hasattr(item, "content") else ""
                messages.append({"role": item.role, "content": _resolve_content(raw)})
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
            role = item.get("role", "user")
            content = _resolve_content(item.get("content", ""))
            messages.append({"role": role, "content": content})
        # Standard role-based message (no type field, or type is not a special one)
        elif "role" in item:
            role = item.get("role", "user")
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
        text_dict = request.text.model_dump() if hasattr(request.text, "model_dump") else {"type": request.text.type}
        json_instruction = build_json_system_prompt(text_dict)
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Build kwargs
    chat_kwargs = {
        "max_tokens": request.max_output_tokens or _default_max_tokens,
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
    # Priority: top-level field > chat_template_kwargs > auto-detect from model config
    _ct_kwargs = request.chat_template_kwargs or {}
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        chat_kwargs["enable_thinking"] = bool(_ct_kwargs["enable_thinking"])
    else:
        # Auto-detect from model config + tokenizer fallback
        from .model_config_registry import get_model_config_registry
        _mc = get_model_config_registry().lookup(_model_name or request.model)
        _enable = _mc.think_in_template
        if not _enable:
            if _mc.reasoning_parser:
                _enable = True
            else:
                try:
                    _tok = engine.tokenizer
                    if getattr(_tok, 'has_thinking', False):
                        _enable = True
                except Exception:
                    pass
        chat_kwargs["enable_thinking"] = _enable

    # Pass reasoning_effort if provided (for GPT-OSS and models that support thinking levels)
    if request.reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = request.reasoning_effort

    # Forward extra chat_template_kwargs to engine (exclude enable_thinking, already handled)
    if _ct_kwargs:
        extra_ct = {k: v for k, v in _ct_kwargs.items() if k != "enable_thinking"}
        if extra_ct:
            chat_kwargs["chat_template_kwargs"] = extra_ct

    # Merge MCP tools with user-provided tools
    all_tools = []

    # Add MCP tools if available
    if _mcp_manager is not None:
        mcp_tools = _mcp_manager.get_all_tools_openai()
        all_tools.extend(mcp_tools)
        if mcp_tools:
            logger.debug(f"Added {len(mcp_tools)} MCP tools")

    # Add user-provided tools — convert from Responses API flat format to Chat Completions nested format
    if request.tools:
        from .api.tool_calling import convert_tools_for_template
        for tool in request.tools:
            tool_type = tool.get("type", "")
            # Skip built-in Responses API tools (web_search, code_interpreter, file_search, etc.)
            if tool_type != "function" and "function" not in tool and "name" not in tool:
                continue
            # Chat Completions nested format: {"type": "function", "function": {...}}
            if "function" in tool:
                all_tools.append(ToolDefinition(**tool))
            # Responses API flat format: {"type": "function", "name": "...", "parameters": {...}}
            elif "name" in tool:
                flat = ResponsesToolDefinition(**tool)
                all_tools.append(ToolDefinition(**flat.to_chat_completions_format()))
        logger.debug(f"Added {len(request.tools)} user tools")
    
    # Pass merged tools to engine
    if all_tools:
        from .api.tool_calling import convert_tools_for_template
        chat_kwargs["tools"] = convert_tools_for_template(all_tools)

    # Inject Harmony analysis prefix for GPT-OSS models (same as Chat Completions path)
    from .reasoning.gptoss_parser import GptOssReasoningParser
    if isinstance(_reasoning_parser, GptOssReasoningParser):
        _think_val = chat_kwargs.get("enable_thinking")
        if _think_val is True:
            _effort = chat_kwargs.get("reasoning_effort", "").lower()
            if _effort == "low":
                _analysis_hint = "<|start|>assistant<|channel|>analysis (brief)<|message|>"
            elif _effort == "high":
                _analysis_hint = "<|start|>assistant<|channel|>analysis (thorough)<|message|>"
            else:
                _analysis_hint = "<|start|>assistant<|channel|>analysis<|message|>"
            chat_kwargs["prompt_suffix"] = _analysis_hint
            chat_kwargs["skip_generation_prompt"] = True
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
    timeout = _default_timeout

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

    # Strip think tags before tool call parsing — reasoning text may precede
    # tool calls in thinking model output
    parse_text = re.sub(r'<think>.*?</think>', '', output.text, flags=re.DOTALL)
    if parse_text == output.text and '</think>' in output.text:
        _, _, parse_text = output.text.partition('</think>')
    parse_text = parse_text.strip()

    # Parse tool calls
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(parse_text or output.text, request)

    # Extract reasoning if parser is enabled
    # Always run reasoning extraction, even when tool calls are present —
    # thinking models emit <think>reasoning</think> before tool call text
    reasoning_text = None
    if _reasoning_parser:
        text_to_parse = cleaned_text or output.text
        reasoning_text, remaining_text = _reasoning_parser.extract_reasoning(text_to_parse)
        if remaining_text is not None:
            cleaned_text = remaining_text
        # Suppress reasoning when user explicitly disabled thinking
        # Check both top-level and chat_template_kwargs
        _suppress = request.enable_thinking is False
        if not _suppress and request.chat_template_kwargs:
            _suppress = request.chat_template_kwargs.get("enable_thinking") is False
        if _suppress:
            reasoning_text = None

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
            output_items.append(ResponsesFunctionCall(
                name=func.name if hasattr(func, "name") else func.get("name", ""),
                arguments=func.arguments if hasattr(func, "arguments") else func.get("arguments", ""),
            ))

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


async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    try:
        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=request.max_tokens or _default_max_tokens,
            temperature=_resolve_temperature(request.temperature),
            top_p=_resolve_top_p(request.top_p),
            stop=request.stop,
        ):
            data = {
                "id": response_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
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
            await engine.abort_request(response_id)
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


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    fastapi_request: Request = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion with auto-detection of closed connections."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Pass response_id as request_id to engine for unified tracking
    kwargs['request_id'] = response_id

    # Check if we should include usage in the final chunk
    include_usage = request.stream_options and request.stream_options.include_usage

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {_dump_sse_json(first_chunk)}\n\n"

    # Resolve effective enable_thinking:
    # Priority: top-level field > chat_template_kwargs > auto-detect
    _ct_kwargs = request.chat_template_kwargs or {}
    if request.enable_thinking is not None:
        _effective_thinking = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        _effective_thinking = bool(_ct_kwargs["enable_thinking"])
    else:
        _effective_thinking = None  # auto-detect below

    # Check if model's chat template injects <think> in the assistant prefix
    # Use _model_name (actual model path) not request.model (which may be "default")
    from .model_config_registry import get_model_config_registry
    _model_config = get_model_config_registry().lookup(_model_name or request.model)
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

    # When user explicitly disables thinking, check if template actually respects it.
    # Templates like Qwen3 honor enable_thinking=False (no <think> injected).
    # Templates like MiniMax ignore it (always inject <think>).
    # For templates that ignore it, keep think_in_template=True so the parser
    # correctly classifies reasoning, and suppress_reasoning hides it.
    if _effective_thinking is False and think_in_template:
        if not _template_always_thinks(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # When the last message is a tool result, the model typically answers directly
    # without thinking. Don't assume implicit thinking mode in this case — if the
    # model still outputs </think>, the parser will detect it dynamically (Case 2).
    # This prevents the entire answer from being classified as reasoning.
    has_tool_results = False
    if request.messages:
        msgs = request.messages
        # Check if any message has role='tool' (list of dicts or objects)
        for m in msgs:
            role = m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)
            if role == 'tool':
                has_tool_results = True
                break
    effective_think_in_template = think_in_template and not has_tool_results
    if _effective_thinking is False:
        effective_think_in_template = False

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
    content_was_emitted = False  # Whether any content chunk was sent

    # Tool call buffering: when we detect a tool call marker in the stream,
    # we stop emitting content and buffer the rest. At end of stream, we parse
    # the buffer for tool calls and emit them as proper tool_calls chunks.
    tool_call_buffering = False  # Are we currently buffering for tool calls?
    tool_call_buffering_notified = False  # Have we sent the buffering signal?
    tool_call_active = _enable_auto_tool_choice or _tool_call_parser is not None

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    last_output = None

    def _emit_content_chunk(content_text, output_obj, is_finished=False):
        """Helper to create a content SSE chunk."""
        chunk_usage = get_usage(output_obj) if include_usage else None
        return ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        content=content_text if content_text else None,
                    ),
                    finish_reason=output_obj.finish_reason if is_finished else None,
                )
            ],
            usage=chunk_usage,
        )

    try:
        # Stream content
        async for output in engine.stream_chat(messages=messages, **kwargs):
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
                    continue

                # Track what was emitted for end-of-stream fallback
                if delta_msg.content:
                    content_was_emitted = True
                    accumulated_content += delta_msg.content
                if delta_msg.reasoning:
                    accumulated_reasoning += delta_msg.reasoning

                # Check for tool call markers — separate logic for content vs reasoning:
                # - Content: check accumulated_content (markers can span deltas)
                # - Reasoning: check ONLY current delta (avoids false positives from
                #   reasoning that casually mentions tool formats like "<function=")
                if tool_call_active and not tool_call_buffering:
                    if delta_msg.content and accumulated_content:
                        for marker in _TOOL_CALL_MARKERS:
                            if marker in accumulated_content:
                                tool_call_buffering = True
                                break
                    if not tool_call_buffering and delta_msg.reasoning:
                        for marker in _TOOL_CALL_MARKERS:
                            if marker in delta_msg.reasoning:
                                tool_call_buffering = True
                                break
                    # GPT-OSS/Harmony native tool format: to=<name> code{...}
                    # Uses regex for specificity (plain "to=" is too broad for markers list)
                    if not tool_call_buffering:
                        _tc_check = accumulated_content or delta_msg.reasoning or ""
                        if re.search(r'\bto=\w[\w.]*\s+code\{', _tc_check):
                            tool_call_buffering = True

                if tool_call_buffering:
                    # Suppress content during tool call buffering, but emit
                    # usage-only chunks so the client TPS counter stays alive.
                    if include_usage:
                        buf_chunk = ChatCompletionChunk(
                            id=response_id,
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
                # convert the hidden reasoning stream into standard content stream. This prevents the UI
                # from hanging/buffering since it expects immediate visible content.
                if suppress_reasoning:
                    emit_content = delta_msg.reasoning or delta_msg.content
                    emit_reasoning = None
                    if emit_content:
                        content_was_emitted = True
                        if delta_msg.reasoning:
                            # Transfer accumulated text if we shifted reasoning to content
                            accumulated_content += delta_msg.reasoning
                else:
                    emit_reasoning = delta_msg.reasoning
                    emit_content = delta_msg.content

                # Skip chunks that have nothing to emit after conversion
                if not emit_content and not emit_reasoning and not output.finished:
                    continue

                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=emit_content,
                                reasoning_content=emit_reasoning,
                            ),
                            finish_reason=output.finish_reason if output.finished else None,
                        )
                    ],
                    usage=chunk_usage,
                )
                out_j = _dump_sse_json(chunk)
                import logging
                logging.getLogger(__name__).warning(f"SSE_JSON_DUMP: {out_j}")
                yield f"data: {out_j}\n\n"
            else:
                # Standard path without reasoning parsing
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

                # Include usage in every chunk when include_usage is on (for real-time metrics)
                chunk_usage = get_usage(output) if include_usage else None
                chunk = ChatCompletionChunk(
                    id=response_id,
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
        yield "data: [DONE]\n\n"
        return

    # ─── Post-stream: tool call extraction ───────────────────────────────
    # If we buffered text because of tool call markers, parse it now
    if tool_call_buffering and accumulated_text:
        # Strip think tags before parsing — the accumulated text may contain
        # <think>reasoning</think> before the tool call, or the entire text
        # may be inside implicit think context (think_in_template models).
        parse_text = re.sub(r'<think>.*?</think>', '', accumulated_text, flags=re.DOTALL)
        if parse_text == accumulated_text and '</think>' in parse_text:
            # Implicit mode: strip everything before </think>
            _, _, parse_text = parse_text.partition('</think>')
        parse_text = parse_text.strip()
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(parse_text or accumulated_text, request)
        if tool_calls:
            # Emit any remaining content text before the tool calls
            if cleaned_text and cleaned_text.strip():
                content_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=cleaned_text.strip()),
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
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield f"data: {_dump_sse_json(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        else:
            # No tool calls found despite markers — flush buffer as content
            # Find the unbuffered portion and emit it
            buffered_text = accumulated_text
            if buffered_text:
                flush_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=buffered_text),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {_dump_sse_json(flush_chunk)}\n\n"

    # Fallback: if reasoning parser produced only reasoning with no content,
    # emit the reasoning text as content so clients always get a usable response.
    # This handles models that wrap everything in <think>...</think> without
    # producing content after the closing tag.
    if request_parser and not content_was_emitted and accumulated_reasoning:
        fallback_chunk = ChatCompletionChunk(
            id=response_id,
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

    # Safeguard: if model generated zero tokens (empty stream), emit a diagnostic
    # chunk so clients always get feedback instead of a silent empty response.
    if last_output is None and not content_was_emitted and not accumulated_reasoning:
        logger.warning(f"Request {response_id}: model generated zero tokens")
        empty_chunk = ChatCompletionChunk(
            id=response_id,
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

    # Send final chunk with usage if requested
    if include_usage:
        usage_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[],  # Empty choices for usage-only chunk
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        yield f"data: {_dump_sse_json(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_responses_api(
    engine: BaseEngine,
    messages: list,
    request: ResponsesRequest,
    fastapi_request: Request = None,
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

    tool_call_active = _enable_auto_tool_choice or _tool_call_parser is not None
    tool_call_buffering = False

    full_text = ""
    streamed_text = ""  # Text already sent as deltas (before tool call marker)
    accumulated_content = ""  # Content-only text for tool call marker detection
    accumulated_reasoning = ""  # Reasoning text for fallback
    content_was_emitted = False
    prompt_tokens = 0
    completion_tokens = 0
    _cached = 0

    # Resolve effective enable_thinking:
    # Priority: top-level field > chat_template_kwargs > auto-detect
    _ct_kwargs = request.chat_template_kwargs or {}
    if request.enable_thinking is not None:
        _effective_thinking = request.enable_thinking
    elif "enable_thinking" in _ct_kwargs:
        _effective_thinking = bool(_ct_kwargs["enable_thinking"])
    else:
        _effective_thinking = None  # auto-detect below

    # Reasoning parser setup (mirrors stream_chat_completion)
    from .model_config_registry import get_model_config_registry
    _model_config = get_model_config_registry().lookup(_model_name or request.model)
    think_in_template = _model_config.think_in_template

    # Fallback: detect from tokenizer if model config doesn't know this model
    if not think_in_template and _reasoning_parser:
        try:
            _tok = engine.tokenizer
            if getattr(_tok, 'has_thinking', False):
                think_in_template = True
        except Exception:
            pass

    # When user explicitly disables thinking, check if template actually respects it
    if _effective_thinking is False and think_in_template:
        if not _template_always_thinks(engine.tokenizer, _model_name or request.model):
            think_in_template = False

    # Don't assume implicit thinking when last message is tool result
    has_tool_results = any(
        (m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)) == 'tool'
        for m in messages
    )
    effective_think_in_template = think_in_template and not has_tool_results

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
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if fastapi_request and await fastapi_request.is_disconnected():
                logger.info(f"Client disconnected, aborting request {response_id}")
                if hasattr(engine, "abort_request"):
                    await engine.abort_request(response_id)
                break

            delta_text = output.new_text
            if delta_text:
                full_text += delta_text

                # Use reasoning parser if enabled
                if request_parser:
                    previous_text = full_text[:-len(delta_text)] if delta_text else full_text
                    delta_msg = request_parser.extract_reasoning_streaming(
                        previous_text, full_text, delta_text
                    )

                    if delta_msg is None:
                        # Skip this chunk (e.g., <think> token itself)
                        pass
                    else:
                        # Track what was emitted
                        if delta_msg.content:
                            content_was_emitted = True
                            accumulated_content += delta_msg.content
                        if delta_msg.reasoning:
                            accumulated_reasoning += delta_msg.reasoning

                        # Check for tool call markers in content
                        if tool_call_active and not tool_call_buffering:
                            if delta_msg.content and accumulated_content:
                                for marker in _TOOL_CALL_MARKERS:
                                    if marker in accumulated_content:
                                        tool_call_buffering = True
                                        break
                            if not tool_call_buffering and delta_msg.reasoning:
                                for marker in _TOOL_CALL_MARKERS:
                                    if marker in delta_msg.reasoning:
                                        tool_call_buffering = True
                                        break
                            # GPT-OSS/Harmony native tool format: to=<name> code{...}
                            # Uses regex for specificity (plain "to=" is too broad for markers list)
                            if not tool_call_buffering:
                                _tc_check = accumulated_content or delta_msg.reasoning or ""
                                if re.search(r'\bto=\w[\w.]*\s+code\{', _tc_check):
                                    tool_call_buffering = True

                        if not tool_call_buffering:
                            # Emit reasoning as custom event (unless suppressed)
                            if delta_msg.reasoning and not suppress_reasoning:
                                yield _sse("response.reasoning.delta", {
                                    "type": "response.reasoning.delta",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "delta": delta_msg.reasoning,
                                })
                            # Emit content as standard text delta
                            if delta_msg.content:
                                streamed_text += delta_msg.content
                                yield _sse("response.output_text.delta", {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id,
                                    "output_index": 0, "content_index": 0,
                                    "delta": delta_msg.content,
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

                    if not tool_call_buffering:
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

            # Track token counts from output (updated each chunk)
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            _cached = getattr(output, "cached_tokens", 0)

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
        yield _sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": response_id, "object": "response",
                "status": "failed",
                "error": {"type": "server_error", "message": str(e)},
            },
        })
        return

    # Emit reasoning done event if reasoning was produced
    if accumulated_reasoning:
        yield _sse("response.reasoning.done", {
            "type": "response.reasoning.done",
            "item_id": msg_id,
            "output_index": 0,
            "reasoning": accumulated_reasoning,
        })

    # Build output items list for the completed response
    all_output_items = []
    output_index = 0

    # Parse tool calls from the accumulated text.
    # Strip think tags before parsing — reasoning text may precede tool calls
    # (mirrors the Chat Completions streaming path at lines 2468-2475).
    parse_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)
    if parse_text == full_text and '</think>' in full_text:
        # Implicit mode: strip everything before </think>
        _, _, parse_text = full_text.partition('</think>')
    parse_text = parse_text.strip()
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(parse_text or full_text, request)

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
            yield _sse("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta", "item_id": fc_id,
                "output_index": output_index, "delta": tc_args,
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
            elif accumulated_reasoning:
                # Model only produced reasoning with no content — use as fallback
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

    # Emit response.completed
    yield _sse("response.completed", {
        "type": "response.completed",
        "response": {
            "id": response_id, "object": "response", "created_at": created_at,
            "status": "completed", "model": request.model,
            "output": all_output_items,
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
        from vllm_mlx.mcp import MCPClientManager, load_mcp_config

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
        description="vllm-mlx OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
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

    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _default_temperature, _default_top_p
    _api_key = args.api_key or os.environ.get("VLLM_API_KEY")
    _default_timeout = args.timeout
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p

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
    from .reasoning import get_parser, list_parsers

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
    elif _tool_call_parser == "auto":
        # Fallback: if CLI module already set _tool_call_parser to "auto"
        detected_tool = registry.get_tool_parser(model_name)
        if detected_tool:
            _tool_call_parser = detected_tool
            logger.info(f"Auto-detected tool call parser: {_tool_call_parser} (from model name)")
        else:
            _tool_call_parser = None

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=args.continuous_batching,
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
