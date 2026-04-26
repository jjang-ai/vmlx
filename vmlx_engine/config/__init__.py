"""
vMLX Configuration System.

Zero hardcoded values - every parameter is configurable via YAML, env vars, CLI, or runtime API.

Usage:
    from vmlx_engine.config import ConfigManager

    # Load config for a specific model
    config = ConfigManager(model_name="Qwen3.5-35B-A3B-CODEBOOK-TEST")

    # Access values
    codebook_enabled = config.get("codebook.enabled")
    memory_limit = config.get("memory.codebook_cache.memory_limit_mb")

    # Runtime updates
    config.update({"kernel.use_metal": False})

    # Get nested config object
    turboquant_cfg = config.get_section("turboquant")
"""

from .models import (
    Config,
    MemoryConfig,
    CodebookConfig,
    CodebookCacheConfig,
    HybridConfig,
    InferenceConfig,
    KernelConfig,
    DebugConfig,
    TurboquantSettings,
    KVCacheConfig,
    SamplingConfig,
)
from .manager import ConfigManager, get_config, reset_config

__all__ = [
    # Models
    "Config",
    "MemoryConfig",
    "CodebookConfig",
    "CodebookCacheConfig",
    "HybridConfig",
    "InferenceConfig",
    "KernelConfig",
    "DebugConfig",
    "TurboquantSettings",
    "KVCacheConfig",
    "SamplingConfig",
    # Manager
    "ConfigManager",
    "get_config",
    "reset_config",
]
