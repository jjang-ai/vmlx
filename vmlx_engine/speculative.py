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

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
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
_spec_kind = "standard"


def get_spec_config() -> Optional[SpeculativeConfig]:
    """Get the global speculative decoding configuration."""
    return _spec_config


def get_draft_model() -> Optional[Any]:
    """Get the loaded draft model (returns None if not loaded)."""
    return _draft_model


def is_speculative_enabled() -> bool:
    """Check if speculative decoding is enabled and draft model is loaded."""
    return _spec_config is not None and _spec_config.enabled and _draft_model is not None


def is_dflash2_enabled() -> bool:
    return is_speculative_enabled() and _spec_kind == "dflash2"


def _is_dflash2_model(model: str) -> bool:
    path = Path(model).expanduser()
    if path.is_dir():
        try:
            config = json.loads((path / "config.json").read_text())
            architectures = config.get("architectures")
            return isinstance(architectures, list) and "DFlash2DraftModel" in architectures
        except Exception:
            return False
    return "dflash2" in model.lower()


def find_bundled_dflash2(model_path: str) -> str | None:
    """A DFlash2 drafter shipped inside the model bundle, or None.

    Any immediate subfolder whose config.json declares DFlash2DraftModel
    (convention: ``<bundle>/dflash2``). Bundles without one return None.
    """
    model_path = str(model_path or "").strip()
    if not model_path:
        return None
    root = Path(model_path).expanduser()
    if not root.is_dir():
        return None
    try:
        children = sorted(root.iterdir(), key=lambda p: (p.name != "dflash2", p.name))
    except OSError:
        return None
    for child in children:
        if child.name.startswith(".") or not child.is_dir():
            continue
        if _is_dflash2_model(str(child)) and (child / "config.json").is_file():
            return str(child)
    return None


def resolve_num_draft_tokens(model: str, requested: int) -> int:
    """Return the verifier width the selected draft runtime will actually use."""
    if not _is_dflash2_model(model):
        return requested

    path = Path(model).expanduser()
    if path.is_dir():
        try:
            config = json.loads((path / "config.json").read_text())
            return min(5, int(config["block_size"]))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            pass

    # DFlash2's published verifier contract is a five-token block. The loaded
    # local config is authoritative when available; this fallback lets the CLI
    # report the effective width before a Hub checkpoint has been downloaded.
    return 5


def external_speculative_incompatibility_reason(
    target_model_name: str | None,
) -> str | None:
    """Return why external draft decoding is unsafe for a target model.

    mlx-lm's external speculative path splits prompt caches by
    ``len(model.layers)``. Nanbeige 4.2 deliberately has 22 shared module
    layers but 44 independent loop cache slots, so that split silently drops
    the second loop's cache. Fail closed until the upstream verifier accepts
    an explicit cache-slot count.
    """

    if not target_model_name:
        return None
    try:
        from .model_config_registry import get_model_config_registry

        config = get_model_config_registry().lookup(target_model_name)
        hints = getattr(config, "architecture_hints", None) or {}
        if (
            getattr(config, "family_name", None) == "nanbeige"
            or hints.get("cache_schema") == "looped_kv_v1"
        ):
            return "Nanbeige looped KV cache (44 slots for 22 shared layers)"
    except Exception:
        # A malformed optional JANG stamp must not let a known local Nanbeige
        # bundle fall through to the unsafe external verifier.
        pass

    try:
        from .utils.nanbeige_runtime import is_nanbeige_model_path

        if is_nanbeige_model_path(target_model_name):
            return "Nanbeige looped KV cache (44 slots for 22 shared layers)"
    except Exception:
        pass
    return None


def _bundled_dflash2_target(draft_path) -> tuple[str | None, int]:
    """(target model_type, drafter bits at load) for a drafter shipped inside a bundle (<bundle>/dflash2).

    The bundle's jang_config.json may declare ``drafter.runtime_quantization`` (e.g. "affine 8-bit g64 at load": the
    GLM-5.3-Flash JANGT bundle ships the CC BY-NC-ND drafter unmodified and asks for 8-bit). Default 4-bit: the
    measured Qwen3.8 setting (R2-21 section 4, no acceptance change vs bf16 codebooks).
    """
    import re

    root = Path(draft_path).expanduser().parent
    target_type, bits = None, 4
    try:
        target_type = json.loads((root / "config.json").read_text()).get("model_type")
    except Exception:
        pass
    try:
        rq = (json.loads((root / "jang_config.json").read_text()).get("drafter") or {}).get("runtime_quantization", "")
        m = re.search(r"(\d+)-bit", str(rq))
        if m and int(m.group(1)) in (4, 8):
            bits = int(m.group(1))
    except Exception:
        pass
    return target_type, bits


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
    global _spec_config, _draft_model, _draft_tokenizer, _spec_kind

    _spec_config = config

    if not config.enabled:
        logger.info("Speculative decoding not configured (no --speculative-model)")
        return None, None

    logger.info(f"Loading draft model for speculative decoding: {config.model}")
    start_time = time.time()

    try:
        from .model_bundle_integrity import prepare_model_bundle_for_load

        resolved_draft, _integrity_report = prepare_model_bundle_for_load(
            config.model,
            allow_download=True,
        )
        if _is_dflash2_model(config.model):
            import dflash.model_mlx as dflash_runtime
            import mlx.core as mx
            import mlx.nn as nn
            from .patches.mlx_lm_mtp import set_mtp_active
            from .patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch

            target_type, draft_bits = _bundled_dflash2_target(resolved_draft)
            if target_type != "glm5_next":
                # DFlash2 reuses the Qwen text-RoPE and hybrid rollback hooks from
                # the native-MTP adapter even though its own external draft wins.
                # (glm5_next has its own adapter, vmlx_engine/glm_dflash2.py; activating MTP there would make the
                # target build an MTP head its bundle does not ship.)
                set_mtp_active(True)
                apply_mlx_vlm_mtp_patch()

            original_download = dflash_runtime.snapshot_download
            if Path(resolved_draft).is_dir():
                dflash_runtime.snapshot_download = lambda model_id, **_kwargs: model_id
            try:
                draft_model = dflash_runtime.load_draft(resolved_draft)
            finally:
                dflash_runtime.snapshot_download = original_download
            # All drafter weights 4-bit, selector codebooks included: keeping
            # the two 248,320 x 256 codebooks bf16 (+190 MB) was measured with
            # no acceptance change (R2-21 §4), so it is not done.
            nn.quantize(draft_model, group_size=64, bits=draft_bits)
            mx.eval(draft_model.parameters())
            logger.info("DFlash2 drafter quantized at load: affine %d-bit g64 (target %s)", draft_bits, target_type or "unknown")
            draft_tokenizer = None
            _spec_kind = "dflash2"
            config.num_tokens = resolve_num_draft_tokens(
                config.model,
                config.num_tokens,
            )
        else:
            from mlx_lm import load as mlx_lm_load

            draft_model, draft_tokenizer = mlx_lm_load(
                resolved_draft,
                tokenizer_config={"trust_remote_code": True},
            )
            _spec_kind = "standard"
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
    global _draft_model, _draft_tokenizer, _spec_config, _spec_kind

    if _draft_model is not None:
        logger.info("Unloading draft model")
        _draft_model = None
        _draft_tokenizer = None
        _spec_kind = "standard"
        if _spec_config:
            _spec_config.enabled = False

        # Free memory
        try:
            import gc
            from .mlx_memory import clear_mlx_memory_cache

            gc.collect()
            clear_mlx_memory_cache(log=logger)
        except Exception:
            pass


def should_use_speculative(
    is_batched: bool = False,
    is_mllm: bool = False,
    target_model_name: str | None = None,
) -> bool:
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
    reason = external_speculative_incompatibility_reason(target_model_name)
    if reason:
        logger.warning("Speculative decoding disabled: %s", reason)
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
