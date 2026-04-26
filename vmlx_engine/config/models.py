"""
vMLX Configuration Models - Pydantic schemas for all configurable options.

Zero hardcoded values - every parameter is configurable via YAML, env vars, CLI, or runtime API.

Usage:
    from vmlx_engine.config import ConfigManager, Config

    # Load config with auto-detection
    config = ConfigManager(model_name="Qwen3.5-35B-A3B-CODEBOOK-TEST")

    # Access values
    memory_limit = config.get("memory.codebook_cache.memory_limit_mb")
    use_metal = config.get("kernel.use_metal")

    # Runtime update
    config.update({"kernel.use_metal": False})
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
import os


# =============================================================================
# TurboQuant Configuration
# =============================================================================


class TurboquantSettings(BaseModel):
    """TurboQuant KV cache compression settings."""

    default_key_bits: int = 3
    default_value_bits: int = 3
    critical_key_bits: int = 4
    critical_value_bits: int = 4
    critical_layers: List[int] = Field(default_factory=lambda: [0, 1, 2, -3, -2, -1])
    sink_tokens: int = 4
    seed: int = 42

    @field_validator("critical_layers", mode="before")
    @classmethod
    def resolve_negative_indices(cls, v):
        """Allow negative indices - they'll be resolved against n_layers at runtime."""
        return v


class KVCacheConfig(BaseModel):
    """KV cache memory configuration."""

    quantization: str = "turboquant"  # "none", "q4", "q8", "turboquant"
    memory_limit_mb: Optional[int] = None
    turboquant: TurboquantSettings = Field(default_factory=TurboquantSettings)


# =============================================================================
# Codebook Cache Configuration
# =============================================================================


class CodebookCacheConfig(BaseModel):
    """Codebook weight cache (LRU with disk spillover) configuration."""

    enabled: bool = True
    memory_limit_mb: Optional[int] = None  # null = unlimited (LRU to disk)
    disk_cache_dir: Optional[str] = "~/.cache/vmlx-engine/codebook-cache"
    disk_max_gb: int = 1000
    eviction_batch_size: int = 4


class CodebookKernelConfig(BaseModel):
    """Metal kernel configuration for codebook operations."""

    threads_per_threadgroup: int = 256
    max_batch_size: int = 32


class CodebookConfig(BaseModel):
    """Codebook VQ model configuration."""

    enabled: str = "auto"  # "auto", "true", "false"
    loading: str = "lazy"  # "lazy", "eager", "hybrid"
    hybrid_eager_layers: List[int] = Field(default_factory=list)
    per_layer_settings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    kernel: str = "metal"  # "metal", "mlx"
    kernel_config: CodebookKernelConfig = Field(default_factory=CodebookKernelConfig)
    cache: CodebookCacheConfig = Field(default_factory=CodebookCacheConfig)

    @field_validator("enabled")
    @classmethod
    def validate_enabled(cls, v):
        if v not in ("auto", "true", "false"):
            raise ValueError(f"enabled must be 'auto', 'true', or 'false', got {v}")
        return v

    @field_validator("loading")
    @classmethod
    def validate_loading(cls, v):
        if v not in ("lazy", "eager", "hybrid"):
            raise ValueError(f"loading must be 'lazy', 'eager', or 'hybrid', got {v}")
        return v


# =============================================================================
# Memory Configuration
# =============================================================================


class PrefixCacheConfig(BaseModel):
    """Prefix cache configuration."""

    enabled: bool = True
    memory_limit_mb: Optional[int] = None
    max_entries: int = 10000


class PagedCacheConfig(BaseModel):
    """Paged KV cache configuration."""

    enabled: bool = False
    block_size: int = 64
    max_blocks: int = 10000
    memory_limit_mb: Optional[int] = None


class DiskCacheConfig(BaseModel):
    """Disk cache (L2 persistence) configuration."""

    enabled: bool = True
    cache_dir: Optional[str] = None  # null = auto (~/.cache/vmlx-engine/prompt-cache)
    max_gb: float = 100.0
    memory_limit_mb: Optional[int] = None


class MemoryConfig(BaseModel):
    """Memory budget configuration for all cache tiers."""

    total_budget_percent: float = 0.85
    kv_cache: KVCacheConfig = Field(default_factory=KVCacheConfig)
    codebook_cache: CodebookCacheConfig = Field(default_factory=CodebookCacheConfig)
    prefix_cache: PrefixCacheConfig = Field(default_factory=PrefixCacheConfig)
    paged_cache: PagedCacheConfig = Field(default_factory=PagedCacheConfig)
    disk_cache: DiskCacheConfig = Field(default_factory=DiskCacheConfig)


# =============================================================================
# Hybrid Architecture Configuration
# =============================================================================


class HybridConfig(BaseModel):
    """Hybrid SSM/attention architecture configuration."""

    ssm_recompute: str = "full"  # "full", "checkpoint"
    checkpoint_interval_tokens: int = 512
    detection: str = "auto"  # "auto", or path to layer_types.json
    cache_types: Dict[str, str] = Field(
        default_factory=dict
    )  # "layer_idx": "turboquant"|"arrays"|"none"

    @field_validator("ssm_recompute")
    @classmethod
    def validate_ssm_recompute(cls, v):
        if v not in ("full", "checkpoint"):
            raise ValueError(f"ssm_recompute must be 'full' or 'checkpoint', got {v}")
        return v


# =============================================================================
# Inference Configuration
# =============================================================================


class SamplingConfig(BaseModel):
    """Default sampling parameters (can be overridden per-request)."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0


class InferenceConfig(BaseModel):
    """Inference engine configuration."""

    max_concurrent_requests: int = 256
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    max_tokens_per_step: int = 8192
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    max_tokens: int = 4096
    stop_tokens: List[int] = Field(default_factory=list)


# =============================================================================
# Kernel Configuration
# =============================================================================


class KernelConfig(BaseModel):
    """Metal/kernel configuration."""

    use_metal: bool = True
    num_threads: Optional[int] = None  # null = auto-detect
    compute_precision: str = "float16"  # "float16", "bfloat16", "float32"
    kv_compute_precision: str = "float16"

    @field_validator("compute_precision", "kv_compute_precision")
    @classmethod
    def validate_precision(cls, v):
        if v not in ("float16", "bfloat16", "float32"):
            raise ValueError(
                f"precision must be 'float16', 'bfloat16', or 'float32', got {v}"
            )
        return v


# =============================================================================
# Debug/Logging Configuration
# =============================================================================


class DebugConfig(BaseModel):
    """Debugging and logging configuration."""

    log_cache_hits: bool = False
    log_cache_misses: bool = False
    log_memory_usage: bool = False
    profile_kernel_times: bool = False
    stats_interval_seconds: int = 60


# =============================================================================
# Root Configuration
# =============================================================================


class Config(BaseModel):
    """Root configuration - encompasses all vMLX settings."""

    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    codebook: CodebookConfig = Field(default_factory=CodebookConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    turboquant: TurboquantSettings = Field(default_factory=TurboquantSettings)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    kernel: KernelConfig = Field(default_factory=KernelConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    class Config:
        """Pydantic config."""

        extra = "allow"  # Allow extra fields for forward compatibility


# =============================================================================
# Model-Specific Configuration
# =============================================================================


class PerModelConfig(BaseModel):
    """Per-model configuration overrides."""

    quantization: Optional[Dict[str, Any]] = None
    codebook: Optional[Dict[str, Any]] = None
    hybrid: Optional[Dict[str, Any]] = None
    turboquant: Optional[Dict[str, Any]] = None
    inference: Optional[Dict[str, Any]] = None


class ConfigRoot(BaseModel):
    """Top-level config file structure."""

    memory: Optional[MemoryConfig] = None
    codebook: Optional[CodebookConfig] = None
    hybrid: Optional[HybridConfig] = None
    turboquant: Optional[TurboquantSettings] = None
    inference: Optional[InferenceConfig] = None
    kernel: Optional[KernelConfig] = None
    debug: Optional[DebugConfig] = None
    per_model: Dict[str, PerModelConfig] = Field(default_factory=dict)

    class Config:
        extra = "allow"
