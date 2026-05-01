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

__version__ = "1.4.2"

# mlx_lm 0.31.x changed create_attention_mask() to require return_array and
# window_size positional args.  mlx_vlm's qwen3_5/language.py calls
# create_ssm_mask(h, cache) which calls cache.make_mask(N) without those args.
# KVCache.make_mask passes them through to the local create_attention_mask()
# in cache.py which now requires all 4 positional args -> TypeError.
# Patch create_ssm_mask to always pass defaults for the new args.
try:
    import mlx_lm.models.base as _base

    if hasattr(_base, "create_ssm_mask"):
        _orig_ssm_mask = _base.create_ssm_mask

        def _patched_ssm_mask(h, cache=None):
            if cache is not None and hasattr(cache, "make_mask"):
                try:
                    return cache.make_mask(h.shape[1])
                except TypeError:
                    return cache.make_mask(
                        h.shape[1], return_array=False, window_size=None
                    )
            return None

        _base.create_ssm_mask = _patched_ssm_mask
except Exception:
    pass

# Also patch KVCache.make_mask to provide defaults for the new positional args
# in mlx_lm.models.cache.create_attention_mask(N, offset, return_array, window_size).
try:
    import mlx_lm.models.cache as _cache_mod

    if hasattr(_cache_mod, "KVCache") and hasattr(_cache_mod.KVCache, "make_mask"):
        _orig_kv_make_mask = _cache_mod.KVCache.make_mask
        _cache_create_attention_mask = getattr(
            _cache_mod, "create_attention_mask", None
        )

        def _patched_kv_make_mask(self, *args, **kwargs):
            kwargs.setdefault("return_array", False)
            kwargs.setdefault("window_size", None)
            if _cache_create_attention_mask is not None:
                return _cache_create_attention_mask(*args, offset=self.offset, **kwargs)
            return _orig_kv_make_mask(self, *args, **kwargs)

        _cache_mod.KVCache.make_mask = _patched_kv_make_mask
except Exception:
    pass

# mlx_vlm's MessageFormatter does not yet have a gemma4 entry in MODEL_CONFIG
# (mlx-vlm 0.3.x ships gemma3/gemma3n but not gemma4). Without this patch,
# `mlx_vlm.prompt_utils.apply_chat_template()` raises
# `ValueError: Unsupported model: gemma4` for any Gemma 4 VLM (including
# Gemma 4 JANG), and the BatchedEngine silently falls back to the text-only
# tokenizer path — which drops image content parts entirely, so the model
# receives the prompt WITHOUT any image placeholder tokens and answers as
# if no image were provided (see GitHub issue reported 2026-04-09).
#
# Fix: register `gemma4` (and the `gemma4_text` sub-type for safety) under
# `LIST_WITH_IMAGE_TYPE`, which builds the same
# `[{type: 'image'}, {type: 'text', text: ...}]` content list the Gemma 4
# Jinja chat template already knows how to render into `<|image|>` tokens.
try:
    from mlx_vlm.prompt_utils import MODEL_CONFIG as _VLM_MODEL_CONFIG, MessageFormat as _VLMMF

    _VLM_MODEL_CONFIG.setdefault("gemma4", _VLMMF.LIST_WITH_IMAGE_TYPE)
    _VLM_MODEL_CONFIG.setdefault("gemma4_text", _VLMMF.LIST_WITH_IMAGE_TYPE)
except Exception:
    pass

# Kimi K2.6 (`kimi_k25` model_type) reuses the `kimi_vl` module in mlx_vlm
# for vision + multi-modal projector. Install the remap at import time
# (setdefault = idempotent) so apply_chat_template + get_model_and_args
# route kimi_k25 through kimi_vl without needing jang_tools to be
# imported first. See research/KIMI-K2.6-VMLX-INTEGRATION.md §1.
try:
    from mlx_vlm import utils as _mlx_vlm_utils
    _mapping = getattr(_mlx_vlm_utils, "MODEL_REMAPPING", None)
    if _mapping is not None:
        _mapping.setdefault("kimi_k25", "kimi_vl")
except Exception:
    pass

try:
    from mlx_vlm.prompt_utils import MODEL_CONFIG as _KV_MODEL_CONFIG
    if "kimi_k25" not in _KV_MODEL_CONFIG and "kimi_vl" in _KV_MODEL_CONFIG:
        _KV_MODEL_CONFIG["kimi_k25"] = _KV_MODEL_CONFIG["kimi_vl"]
except Exception:
    pass

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
