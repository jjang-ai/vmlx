"""
Configuration source for environment variables.

Environment variables take precedence over YAML files.
Prefix: VMLX_

Examples:
    VMLX_CODEBOOK_MEMORY_LIMIT_MB=65536
    VMLX_KV_CACHE_QUANTIZATION=turboquant
    VMLX_TURBOQUANT_KEY_BITS=4
    VMLX_HYBRID_SSM_RECOMPUTE=full
    VMLX_USE_METAL=true
    VMLX_INFERENCE_MAX_CONCURRENT_REQUESTS=128
"""

import os
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

ENV_PREFIX = "VMLX_"

# Mapping from env var names to config paths
ENV_TO_PATH = {
    # Memory settings
    "VMLX_MEMORY_TOTAL_BUDGET_PERCENT": "memory.total_budget_percent",
    "VMLX_MEMORY_KV_CACHE_QUANTIZATION": "memory.kv_cache.quantization",
    "VMLX_MEMORY_KV_CACHE_MEMORY_LIMIT_MB": "memory.kv_cache.memory_limit_mb",
    "VMLX_CODEBOOK_MEMORY_LIMIT_MB": "memory.codebook_cache.memory_limit_mb",
    "VMLX_CODEBOOK_DISK_CACHE_DIR": "memory.codebook_cache.disk_cache_dir",
    "VMLX_CODEBOOK_DISK_MAX_GB": "memory.codebook_cache.disk_max_gb",
    "VMLX_PREFIX_CACHE_ENABLED": "memory.prefix_cache.enabled",
    "VMLX_PREFIX_CACHE_MEMORY_LIMIT_MB": "memory.prefix_cache.memory_limit_mb",
    "VMLX_PAGED_CACHE_ENABLED": "memory.paged_cache.enabled",
    "VMLX_PAGED_CACHE_BLOCK_SIZE": "memory.paged_cache.block_size",
    "VMLX_DISK_CACHE_ENABLED": "memory.disk_cache.enabled",
    "VMLX_DISK_CACHE_DIR": "memory.disk_cache.cache_dir",
    "VMLX_DISK_CACHE_MAX_GB": "memory.disk_cache.max_gb",
    # Codebook settings
    "VMLX_CODEBOOK_ENABLED": "codebook.enabled",
    "VMLX_CODEBOOK_LOADING": "codebook.loading",
    "VMLX_CODEBOOK_KERNEL": "codebook.kernel",
    "VMLX_CODEBOOK_METAL_THREADS": "codebook.kernel_config.threads_per_threadgroup",
    "VMLX_CODEBOOK_METAL_BATCH_SIZE": "codebook.kernel_config.max_batch_size",
    # TurboQuant settings
    "VMLX_TURBOQUANT_KEY_BITS": "turboquant.default_key_bits",
    "VMLX_TURBOQUANT_VALUE_BITS": "turboquant.default_value_bits",
    "VMLX_TURBOQUANT_CRITICAL_KEY_BITS": "turboquant.critical_key_bits",
    "VMLX_TURBOQUANT_CRITICAL_VALUE_BITS": "turboquant.critical_value_bits",
    "VMLX_TURBOQUANT_SINK_TOKENS": "turboquant.sink_tokens",
    "VMLX_TURBOQUANT_SEED": "turboquant.seed",
    # Hybrid settings
    "VMLX_HYBRID_SSM_RECOMPUTE": "hybrid.ssm_recompute",
    "VMLX_HYBRID_CHECKPOINT_INTERVAL": "hybrid.checkpoint_interval_tokens",
    "VMLX_HYBRID_DETECTION": "hybrid.detection",
    # Inference settings
    "VMLX_INFERENCE_MAX_CONCURRENT_REQUESTS": "inference.max_concurrent_requests",
    "VMLX_INFERENCE_PREFILL_BATCH_SIZE": "inference.prefill_batch_size",
    "VMLX_INFERENCE_COMPLETION_BATCH_SIZE": "inference.completion_batch_size",
    "VMLX_INFERENCE_MAX_TOKENS": "inference.max_tokens",
    "VMLX_INFERENCE_MAX_TOKENS_PER_STEP": "inference.max_tokens_per_step",
    "VMLX_SAMPLING_TEMPERATURE": "inference.sampling.temperature",
    "VMLX_SAMPLING_TOP_P": "inference.sampling.top_p",
    "VMLX_SAMPLING_TOP_K": "inference.sampling.top_k",
    "VMLX_SAMPLING_MIN_P": "inference.sampling.min_p",
    "VMLX_SAMPLING_REPETITION_PENALTY": "inference.sampling.repetition_penalty",
    # Kernel settings
    "VMLX_USE_METAL": "kernel.use_metal",
    "VMLX_NUM_THREADS": "kernel.num_threads",
    "VMLX_COMPUTE_PRECISION": "kernel.compute_precision",
    "VMLX_KV_COMPUTE_PRECISION": "kernel.kv_compute_precision",
    # Debug settings
    "VMLX_DEBUG_LOG_CACHE_HITS": "debug.log_cache_hits",
    "VMLX_DEBUG_LOG_CACHE_MISSES": "debug.log_cache_misses",
    "VMLX_DEBUG_LOG_MEMORY_USAGE": "debug.log_memory_usage",
    "VMLX_DEBUG_PROFILE_KERNEL_TIMES": "debug.profile_kernel_times",
    "VMLX_DEBUG_STATS_INTERVAL": "debug.stats_interval_seconds",
}


def _parse_env_value(value: str, path: str) -> Any:
    """Parse environment variable value to appropriate type based on path."""
    # Boolean paths
    if any(
        p in path
        for p in [
            "enabled",
            "use_metal",
            "log_cache_hits",
            "log_cache_misses",
            "log_memory_usage",
            "profile_kernel_times",
        ]
    ):
        return value.lower() in ("true", "1", "yes", "on")

    # Integer paths
    if any(
        p in path
        for p in [
            "memory_limit_mb",
            "max_gb",
            "max_entries",
            "block_size",
            "max_blocks",
            "threads",
            "max_concurrent_requests",
            "batch_size",
            "max_tokens",
            "max_tokens_per_step",
            "top_k",
            "sink_tokens",
            "seed",
            "bits",
            "interval_seconds",
            "checkpoint_interval",
        ]
    ):
        try:
            return int(value)
        except ValueError:
            return value

    # Float paths
    if any(
        p in path
        for p in ["percent", "temperature", "top_p", "min_p", "penalty", "budget"]
    ):
        try:
            return float(value)
        except ValueError:
            return value

    # List paths (comma-separated)
    if any(p in path for p in ["critical_layers"]):
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            return value

    # String otherwise
    return value


class EnvConfigSource:
    """Loads configuration from environment variables."""

    def __init__(self):
        self._config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load all VMLX_* environment variables."""
        self._config = {}

        for env_key, config_path in ENV_TO_PATH.items():
            env_value = os.environ.get(env_key)
            if env_value is not None:
                # Parse to appropriate type
                parsed_value = _parse_env_value(env_value, config_path)

                # Navigate to correct nested dict and set value
                keys = config_path.split(".")
                current = self._config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = parsed_value

                logger.debug(f"Env config: {config_path} = {parsed_value}")

        return self._config

    def get_overrides(self) -> Dict[str, Any]:
        """Return flat dict of env var overrides for display."""
        return {
            path: os.environ.get(env_key)
            for env_key, path in ENV_TO_PATH.items()
            if env_key in os.environ
        }
