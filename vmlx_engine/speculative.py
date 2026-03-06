# SPDX-License-Identifier: Apache-2.0
"""
Speculative Decoding support for vmlx-engine.

Enables speculative decoding using a smaller draft model to accelerate
token generation by 20-90% with zero quality loss.

The draft model proposes N tokens, then the target model verifies them
in a single forward pass. Accepted tokens skip individual decode steps.

Usage:
    # CLI
    vmlx-engine serve model --speculative-model draft-model --num-draft-tokens 3

    # Python
    from vmlx_engine.speculative import SpeculativeConfig, load_draft_model
    config = SpeculativeConfig(model="draft-model", num_tokens=3)
    draft_model, draft_tokenizer = load_draft_model(config)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Attributes:
        model: Draft model name/path (HuggingFace or local)
        num_tokens: Number of tokens to draft per step (default: 3)
        disable_by_batch_size: Disable spec decoding when batch > N (0 = never disable)
        enabled: Whether speculative decoding is currently enabled
    """

    model: str = ""
    num_tokens: int = 3
    disable_by_batch_size: int = 0
    enabled: bool = False

    def __post_init__(self):
        self.enabled = bool(self.model)
        if self.num_tokens < 1:
            logger.warning(
                f"num_draft_tokens={self.num_tokens} < 1, setting to 1"
            )
            self.num_tokens = 1
        if self.num_tokens > 20:
            logger.warning(
                f"num_draft_tokens={self.num_tokens} > 20 is unusual, "
                "consider a smaller value for better acceptance rates"
            )


# Global speculative decoding state
_spec_config: Optional[SpeculativeConfig] = None
_draft_model: Any = None
_draft_tokenizer: Any = None


def get_spec_config() -> Optional[SpeculativeConfig]:
    """Get the global speculative decoding configuration."""
    return _spec_config


def get_draft_model() -> Optional[Any]:
    """Get the loaded draft model (returns None if not loaded)."""
    return _draft_model


def is_speculative_enabled() -> bool:
    """Check if speculative decoding is enabled and draft model is loaded."""
    return _spec_config is not None and _spec_config.enabled and _draft_model is not None


def load_draft_model(config: SpeculativeConfig) -> tuple[Any, Any]:
    """Load the draft model for speculative decoding.

    Args:
        config: Speculative decoding configuration

    Returns:
        Tuple of (draft_model, draft_tokenizer)

    Raises:
        ImportError: If mlx-lm is not installed
        ValueError: If model cannot be loaded
    """
    global _spec_config, _draft_model, _draft_tokenizer

    _spec_config = config

    if not config.enabled:
        logger.info("Speculative decoding not configured (no --speculative-model)")
        return None, None

    try:
        from mlx_lm import load as mlx_lm_load
    except ImportError:
        raise ImportError(
            "mlx-lm is required for speculative decoding. "
            "Install with: pip install mlx-lm"
        )

    logger.info(f"Loading draft model for speculative decoding: {config.model}")
    start_time = time.time()

    try:
        draft_model, draft_tokenizer = mlx_lm_load(
            config.model,
            tokenizer_config={"trust_remote_code": True},
        )
    except Exception as e:
        logger.error(f"Failed to load draft model '{config.model}': {e}")
        config.enabled = False
        raise ValueError(f"Cannot load draft model: {e}") from e

    load_time = time.time() - start_time
    logger.info(
        f"Draft model loaded in {load_time:.2f}s: {config.model} "
        f"(num_draft_tokens={config.num_tokens})"
    )

    # Log memory after draft model load
    try:
        import mlx.core as mx

        if hasattr(mx, "get_active_memory"):
            active_gb = mx.get_active_memory() / (1024**3)
            logger.info(f"Metal GPU memory after draft model: {active_gb:.2f}GB active")
    except Exception:
        pass

    _draft_model = draft_model
    _draft_tokenizer = draft_tokenizer

    return draft_model, draft_tokenizer


def unload_draft_model() -> None:
    """Unload the draft model and free memory."""
    global _draft_model, _draft_tokenizer, _spec_config

    if _draft_model is not None:
        logger.info("Unloading draft model")
        _draft_model = None
        _draft_tokenizer = None
        if _spec_config:
            _spec_config.enabled = False

        # Free memory
        try:
            import gc
            import mlx.core as mx

            gc.collect()
            if hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass


def should_use_speculative(is_batched: bool = False, is_mllm: bool = False) -> bool:
    """Check if speculative decoding should be used for this request.

    Speculative decoding is only compatible with:
    - SimpleEngine (not batched)
    - LLM models (not MLLM/VLM)
    - Non-Mamba/SSM models

    Args:
        is_batched: Whether using continuous batching (BatchedEngine)
        is_mllm: Whether the model is multimodal

    Returns:
        True if speculative decoding should be active
    """
    if not is_speculative_enabled():
        return False
    if is_batched:
        logger.debug("Speculative decoding disabled: incompatible with continuous batching")
        return False
    if is_mllm:
        logger.debug("Speculative decoding disabled: incompatible with multimodal models")
        return False
    return True


def get_num_draft_tokens() -> int:
    """Get the configured number of draft tokens per step."""
    if _spec_config is None:
        return 0
    return _spec_config.num_tokens


def validate_draft_tokenizer(target_tokenizer: Any) -> bool:
    """Validate that draft and target models use compatible tokenizers.

    Speculative decoding requires both models to use the same vocabulary.
    A mismatch will produce garbage output silently.

    Args:
        target_tokenizer: The target model's tokenizer

    Returns:
        True if compatible or if validation cannot be performed
    """
    if _draft_tokenizer is None or target_tokenizer is None:
        return True

    try:
        # Quick check: compare vocab sizes
        draft_vocab = len(_draft_tokenizer)
        target_vocab = len(target_tokenizer)
        if draft_vocab != target_vocab:
            logger.warning(
                f"Tokenizer vocab size mismatch: draft={draft_vocab}, target={target_vocab}. "
                "Speculative decoding requires matching tokenizers. Output may be incorrect."
            )
            return False

        # Spot check: encode a test string and compare
        test_str = "Hello, world! This is a test."
        draft_ids = _draft_tokenizer.encode(test_str)
        target_ids = target_tokenizer.encode(test_str)
        if draft_ids != target_ids:
            logger.warning(
                "Tokenizer encoding mismatch between draft and target models. "
                "Speculative decoding requires matching tokenizers. Output may be incorrect."
            )
            return False

        logger.info("Draft/target tokenizer compatibility: VERIFIED")
        return True
    except Exception as e:
        logger.warning(f"Could not validate tokenizer compatibility: {e}")
        return True  # Don't block on validation failure


def get_spec_stats() -> dict:
    """Get speculative decoding statistics for the stats endpoint."""
    if _spec_config is None:
        return {"speculative_decoding": "not_configured"}

    return {
        "speculative_decoding": {
            "enabled": is_speculative_enabled(),
            "draft_model": _spec_config.model if _spec_config else None,
            "num_draft_tokens": _spec_config.num_tokens if _spec_config else 0,
            "draft_model_loaded": _draft_model is not None,
            "disable_by_batch_size": _spec_config.disable_by_batch_size if _spec_config else 0,
        }
    }
