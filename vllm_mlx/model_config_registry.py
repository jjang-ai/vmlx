# SPDX-License-Identifier: Apache-2.0
"""
Model configuration registry for vllm-mlx.

Centralizes all model-specific configuration: cache types, EOS tokens,
chat templates, tool parsers, and architecture hints. Replaces scattered
if/elif checks throughout the codebase with a single source of truth.

Usage:
    from vllm_mlx.model_config_registry import get_model_config_registry

    registry = get_model_config_registry()
    config = registry.lookup("mlx-community/Qwen3-8B-Instruct-4bit")
    print(config.eos_tokens)  # ["<|im_end|>"]
    print(config.tool_parser)  # "qwen"
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration profile for a model family."""

    # Identity
    family_name: str
    model_types: List[str]  # e.g. ["llama", "qwen2", "mistral"]

    # Cache configuration
    cache_type: str = "kv"  # "kv" | "mamba" | "hybrid" | "rotating_kv"

    # Tokenizer overrides
    eos_tokens: Optional[List[str]] = None
    special_tokens_to_clean: Optional[List[str]] = None
    tokenizer_fallback: bool = False

    # Chat template
    chat_template_custom: Optional[str] = None
    preserve_native_tool_format: bool = False

    # Tool calling
    tool_parser: Optional[str] = None
    supports_native_tools: bool = False

    # Reasoning
    reasoning_parser: Optional[str] = None
    think_in_template: bool = False  # True if chat template injects <think> in assistant prefix

    # Multimodal
    is_mllm: bool = False

    # Architecture-specific hints
    architecture_hints: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: Optional[str] = None
    priority: int = 100  # Lower = higher priority (matched first)


# Default config for unknown models
_DEFAULT_CONFIG = ModelConfig(
    family_name="unknown",
    model_types=[],
    cache_type="kv",
    description="Default configuration for unknown models",
    priority=999,
)


class ModelConfigRegistry:
    """
    Singleton registry mapping model families to configurations.

    Uses regex-based pattern matching with priority ordering.
    Results are cached for performance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._configs: List[ModelConfig] = []
        self._match_cache: Dict[str, ModelConfig] = {}
        self._rlock = threading.RLock()
        self._initialized = True

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        with self._rlock:
            self._configs.append(config)
            # Sort by priority (lower = higher priority)
            self._configs.sort(key=lambda c: c.priority)
            # Invalidate cache
            self._match_cache.clear()

    def lookup(self, model_name: str) -> ModelConfig:
        """
        Look up configuration for a model name by reading its config.json.

        Returns default config if no match found.
        """
        with self._rlock:
            if model_name in self._match_cache:
                return self._match_cache[model_name]

            model_type = None
            try:
                from pathlib import Path
                from mlx_lm.utils import load_config
                model_config = load_config(Path(model_name))
                model_type = model_config.get("model_type", "").lower()
            except Exception as e:
                logger.warning(f"Could not load config.json for {model_name} to check model_type: {e}")

            if model_type:
                for config in self._configs:
                    if model_type in config.model_types:
                        self._match_cache[model_name] = config
                        return config

            self._match_cache[model_name] = _DEFAULT_CONFIG
            return _DEFAULT_CONFIG

    def get_cache_type(self, model_name: str) -> str:
        """Get cache type for a model."""
        return self.lookup(model_name).cache_type

    def get_eos_tokens(self, model_name: str) -> Optional[List[str]]:
        """Get EOS token overrides."""
        return self.lookup(model_name).eos_tokens

    def is_mllm(self, model_name: str) -> bool:
        """Check if model is multimodal."""
        return self.lookup(model_name).is_mllm

    def needs_tokenizer_fallback(self, model_name: str) -> bool:
        """Check if model needs tokenizer fallback."""
        return self.lookup(model_name).tokenizer_fallback

    def get_tool_parser(self, model_name: str) -> Optional[str]:
        """Get recommended tool parser name."""
        return self.lookup(model_name).tool_parser

    def get_reasoning_parser(self, model_name: str) -> Optional[str]:
        """Get recommended reasoning parser name."""
        return self.lookup(model_name).reasoning_parser

    def get_architecture_hints(self, model_name: str) -> Dict[str, Any]:
        """Get architecture-specific hints."""
        return self.lookup(model_name).architecture_hints

    def list_registered(self) -> List[str]:
        """List all registered model family names."""
        with self._rlock:
            return [c.family_name for c in self._configs]

    def clear_cache(self) -> None:
        """Clear pattern matching cache."""
        with self._rlock:
            self._match_cache.clear()

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        with self._rlock:
            self._configs.clear()
            self._match_cache.clear()


_configs_loaded = False


def get_model_config_registry() -> ModelConfigRegistry:
    """Get the global model config registry, auto-loading configs on first access."""
    global _configs_loaded
    registry = ModelConfigRegistry()
    if not _configs_loaded:
        _configs_loaded = True
        try:
            from .model_configs import register_all
            register_all(registry)
        except ImportError:
            pass
    return registry
