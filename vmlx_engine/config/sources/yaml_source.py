"""
Configuration source for YAML files.

Loads configuration from YAML files with support for:
- Default config: vmlx_engine/config/defaults.yaml
- User config: ~/.config/vmlx/config.yaml
- Per-model config: ~/.config/vmlx/models/{model_name}.yaml
- System config: /etc/vmlx/config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Standard config locations
CONFIG_DIR = Path.home() / ".config" / "vmlx"
SYSTEM_CONFIG = Path("/etc/vmlx/config.yaml")
DEFAULT_CONFIG = Path(__file__).parent / "defaults.yaml"


class YAMLConfigSource:
    """Loads configuration from YAML files."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load all YAML configs in priority order."""
        self._config = {}

        # 1. Built-in defaults
        if DEFAULT_CONFIG.exists():
            defaults = self._load_file(DEFAULT_CONFIG)
            if defaults:
                self._config = self._deep_merge(self._config, defaults)

        # 2. System config
        if SYSTEM_CONFIG.exists():
            system = self._load_file(SYSTEM_CONFIG)
            if system:
                self._config = self._deep_merge(self._config, system)

        # 3. User config
        user_config = CONFIG_DIR / "config.yaml"
        if user_config.exists():
            user = self._load_file(user_config)
            if user:
                self._config = self._deep_merge(self._config, user)

        # 4. Per-model config (highest priority for model-specific settings)
        if self.model_name:
            model_config = CONFIG_DIR / "models" / f"{self.model_name}.yaml"
            if model_config.exists():
                model = self._load_file(model_config)
                if model:
                    self._config = self._deep_merge(self._config, model)

        return self._config

    def _load_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load a single YAML file."""
        try:
            with open(path, "r") as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}
                # Strip top-level keys that are metadata, not config
                content.pop("__comment", None)
                return content
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            return {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base, override takes precedence."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


def get_config_dirs() -> Dict[str, Path]:
    """Return standard config directories."""
    return {
        "default": DEFAULT_CONFIG,
        "system": SYSTEM_CONFIG,
        "user": CONFIG_DIR / "config.yaml",
        "per_model": CONFIG_DIR / "models",
    }
