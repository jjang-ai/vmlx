"""
vMLX Configuration Manager.

Orchestrates configuration from multiple sources with priority:
    CLI args > Environment > Per-model YAML > User YAML > System YAML > Defaults

Usage:
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

import copy
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path

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
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Central configuration manager for vMLX.

    Manages configuration from multiple sources with priority:
        1. CLI arguments (highest priority)
        2. Environment variables
        3. Per-model YAML (~/.config/vmlx/models/{model_name}.yaml)
        4. User YAML (~/.config/vmlx/config.yaml)
        5. System YAML (/etc/vmlx/config.yaml)
        6. Built-in defaults (lowest priority)

    All values are validated via Pydantic models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration manager.

        Args:
            model_name: Model name for loading per-model config overrides.
            cli_args: Optional dict of CLI arguments already parsed.
        """
        self.model_name = model_name
        self._config: Optional[Config] = None
        self._raw_overrides: Dict[str, Any] = {}

        # Load from all sources
        self._load_all_sources(cli_args)

    def _load_all_sources(self, cli_args: Optional[Dict[str, Any]] = None):
        """Load configuration from all sources with priority."""
        from .sources import YAMLConfigSource, EnvConfigSource, CLIConfigSource

        # 1. Built-in defaults (loaded by YAML source)
        yaml_source = YAMLConfigSource(model_name=self.model_name)
        defaults = yaml_source.load()

        # 2. Environment variables override
        env_source = EnvConfigSource()
        env_overrides = env_source.load()

        # 3. CLI args override (highest priority)
        cli_source = CLIConfigSource(args=cli_args)
        cli_overrides = cli_source.load()

        # Merge with priority (later overrides earlier)
        merged = self._deep_merge(defaults, env_overrides, cli_overrides)

        # Store raw overrides for reference
        self._raw_overrides = {
            "yaml": defaults,
            "env": env_overrides,
            "cli": cli_overrides,
        }

        # Validate and create Pydantic model
        try:
            self._config = Config(**merged)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            # Try with defaults for critical fields
            self._config = Config()

    def _deep_merge(self, *dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple dicts, later dicts override earlier."""
        result = {}
        for d in dicts:
            if not d:
                continue
            for key, value in d.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.

        Args:
            key_path: Dot-notation path, e.g., "memory.codebook_cache.memory_limit_mb"
            default: Default value if key not found.

        Returns:
            Configuration value or default.

        Examples:
            config.get("codebook.enabled")  # -> "auto"
            config.get("memory.total_budget_percent")  # -> 0.85
            config.get("turboquant.default_key_bits")  # -> 3
        """
        if self._config is None:
            return default

        keys = key_path.split(".")
        current: Any = self._config

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
            if current is None:
                return default

        return current

    def get_section(self, section: str) -> Any:
        """
        Get an entire configuration section as a Pydantic model.

        Args:
            section: Section name (e.g., "codebook", "turboquant", "memory")

        Returns:
            Pydantic model for the section, or None if not found.
        """
        return self.get(section)

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration at runtime.

        Args:
            updates: Dict of dot-notation keys to new values.

        Examples:
            config.update({
                "kernel.use_metal": False,
                "inference.max_concurrent_requests": 128,
            })
        """
        if self._config is None:
            return

        # Convert to dict and re-validate
        current_dict = self._config.model_dump()

        for key_path, value in updates.items():
            keys = key_path.split(".")
            current = current_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        # Re-validate
        try:
            self._config = Config(**current_dict)
        except Exception as e:
            logger.error(f"Config update validation failed: {e}")

    def update_section(self, section: str, updates: Dict[str, Any]):
        """Update all values in a section."""
        self.update({f"{section}.{k}": v for k, v in updates.items()})

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dict."""
        if self._config is None:
            return {}
        return self._config.model_dump()

    def get_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return dict showing which sources override which values."""
        return {
            "cli": self._raw_overrides.get("cli", {}),
            "env": self._raw_overrides.get("env", {}),
            "model_yaml": self._raw_overrides.get("yaml", {})
            .get("per_model", {})
            .get(self.model_name, {}),
            "user_yaml": {
                k: v
                for k, v in self._raw_overrides.get("yaml", {}).items()
                if k != "per_model"
            },
        }

    def log_config(self):
        """Log current configuration with source attribution."""
        logger.info(f"vMLX Configuration for model: {self.model_name or 'default'}")
        logger.info(
            f"  Memory budget: {self.get('memory.total_budget_percent') * 100:.0f}% of RAM"
        )
        logger.info(
            f"  KV cache quantization: {self.get('memory.kv_cache.quantization')}"
        )
        logger.info(f"  Codebook enabled: {self.get('codebook.enabled')}")
        logger.info(f"  Codebook loading: {self.get('codebook.loading')}")
        logger.info(f"  Codebook kernel: {self.get('codebook.kernel')}")
        logger.info(f"  TurboQuant key bits: {self.get('turboquant.default_key_bits')}")
        logger.info(
            f"  TurboQuant value bits: {self.get('turboquant.default_value_bits')}"
        )
        logger.info(f"  Hybrid SSM recompute: {self.get('hybrid.ssm_recompute')}")
        logger.info(f"  Kernel use_metal: {self.get('kernel.use_metal')}")
        logger.info(
            f"  Inference max_concurrent: {self.get('inference.max_concurrent_requests')}"
        )

    @property
    def config(self) -> Config:
        """Return the full Config object."""
        return self._config


# =============================================================================
# Global config instance (lazy initialization)
# =============================================================================

_global_config: Optional[ConfigManager] = None


def get_config(model_name: Optional[str] = None) -> ConfigManager:
    """
    Get the global ConfigManager instance.

    Creates a new instance if not yet initialized, or if model_name changes.
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(model_name=model_name)
    return _global_config


def reset_config():
    """Reset the global config instance."""
    global _global_config
    _global_config = None
