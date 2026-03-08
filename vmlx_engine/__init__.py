# SPDX-License-Identifier: Apache-2.0
"""
vmlx-engine: Apple Silicon MLX backend for vLLM

This package provides native Apple Silicon GPU acceleration for vLLM
using Apple's MLX framework, mlx-lm for LLMs, and mlx-vlm for
vision-language models.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Support for LLM and multimodal models
"""

__version__ = "0.2.16"

# All imports are lazy to allow usage on non-Apple Silicon platforms
# (e.g., CI running on Linux) where mlx_lm is not available.


def __getattr__(name):
    """Lazy load all components to avoid mlx_lm import on non-Apple platforms."""
    # Request management
    if name in ("Request", "RequestOutput", "RequestStatus", "SamplingParams"):
        from vmlx_engine import request

        return getattr(request, name)

    # Scheduler
    if name in ("Scheduler", "SchedulerConfig", "SchedulerOutput"):
        from vmlx_engine import scheduler

        return getattr(scheduler, name)

    # Engine
    if name in ("EngineCore", "AsyncEngineCore", "EngineConfig"):
        from vmlx_engine import engine_core

        return getattr(engine_core, name)

    # Prefix cache
    if name in ("PrefixCacheManager", "PrefixCacheStats", "BlockAwarePrefixCache"):
        from vmlx_engine import prefix_cache

        return getattr(prefix_cache, name)

    # Paged cache
    if name in ("PagedCacheManager", "CacheBlock", "BlockTable", "CacheStats"):
        from vmlx_engine import paged_cache

        return getattr(paged_cache, name)

    # MLLM cache (with legacy VLM aliases)
    if name in (
        "MLLMCacheManager",
        "MLLMCacheStats",
        "VLMCacheManager",
        "VLMCacheStats",
    ):
        from vmlx_engine import mllm_cache

        # Map legacy VLM names to MLLM
        mllm_name = name.replace("VLM", "MLLM") if name.startswith("VLM") else name
        return getattr(mllm_cache, mllm_name)

    # Model registry
    if name in ("get_registry", "ModelOwnershipError"):
        from vmlx_engine import model_registry

        return getattr(model_registry, name)

    # Model config registry
    if name in ("get_model_config_registry", "ModelConfigRegistry", "ModelConfig"):
        from vmlx_engine import model_config_registry

        return getattr(model_config_registry, name)

    # vLLM integration components (require torch)
    if name == "MLXPlatform":
        from vmlx_engine.mlx_platform import MLXPlatform

        return MLXPlatform
    if name == "MLXWorker":
        from vmlx_engine.worker import MLXWorker

        return MLXWorker
    if name == "MLXModelRunner":
        from vmlx_engine.model_runner import MLXModelRunner

        return MLXModelRunner
    if name == "MLXAttentionBackend":
        from vmlx_engine.attention import MLXAttentionBackend

        return MLXAttentionBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core (lazy loaded, require torch)
    "MLXPlatform",
    "MLXWorker",
    "MLXModelRunner",
    "MLXAttentionBackend",
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (LLM)
    "PrefixCacheManager",
    "PrefixCacheStats",
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "CacheStats",
    # MLLM cache (images/videos)
    "MLLMCacheManager",
    "MLLMCacheStats",
    # Legacy aliases
    "VLMCacheManager",
    "VLMCacheStats",
    # Version
    "__version__",
]
