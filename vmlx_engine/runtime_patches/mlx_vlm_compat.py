# SPDX-License-Identifier: Apache-2.0
"""Runtime shims for upstream mlx_vlm fixes not yet present in pinned wheels."""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def install() -> None:
    """Install all mlx_vlm compatibility patches once."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_gemma4_video_processor_hf_kwargs()
    _patch_gemma4_shared_kv_layers()
    _APPLIED = True


def _patch_gemma4_shared_kv_layers() -> None:
    """Backport mlx-vlm#1301: Gemma4 shared-KV layers omit unused KV modules."""
    try:
        from mlx_vlm.models import gemma4
        from mlx_vlm.models.gemma4 import language
    except Exception as exc:
        logger.debug("mlx_vlm_compat: Gemma4 shared-KV classes unavailable: %s", exc)
        return

    text_model_cls = getattr(language, "Gemma4TextModel", None)
    original_init = getattr(text_model_cls, "__init__", None)
    if original_init is not None and not getattr(
        original_init, "_vmlx_gemma4_shared_kv_init_patch", False
    ):

        def _vmlx_gemma4_text_init(self, config, kv_shared_only=False):
            original_init(self, config, kv_shared_only=kv_shared_only)
            num_kv_shared = getattr(config, "num_kv_shared_layers", 0) or 0
            first_shared = config.num_hidden_layers - num_kv_shared
            self.first_kv_shared_layer_idx = first_shared
            if num_kv_shared <= 0:
                return
            for index, layer in enumerate(getattr(self, "layers", [])):
                if index < first_shared:
                    continue
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                attn.kv_shared_only = True
                attn.is_kv_shared_layer = True
                for name in ("k_proj", "v_proj", "k_norm", "v_norm"):
                    if hasattr(attn, name):
                        delattr(attn, name)

        _vmlx_gemma4_text_init._vmlx_gemma4_shared_kv_init_patch = True  # type: ignore[attr-defined]
        text_model_cls.__init__ = _vmlx_gemma4_text_init

    def _is_unused_shared_kv_weight(owner, key: str) -> bool:
        prefix = "language_model.model.layers."
        if not key.startswith(prefix):
            return False
        rest = key[len(prefix) :]
        parts = rest.split(".")
        if len(parts) < 4 or parts[1] != "self_attn":
            return False
        try:
            layer_idx = int(parts[0])
        except ValueError:
            return False

        layers = getattr(owner, "layers", None)
        if layers is None:
            language_model = getattr(owner, "language_model", None)
            inner_model = getattr(language_model, "model", None)
            layers = getattr(inner_model, "layers", None)
        if layers is None or layer_idx >= len(layers):
            return False
        attn = getattr(layers[layer_idx], "self_attn", None)
        return bool(
            getattr(attn, "is_kv_shared_layer", False)
            and parts[2] in {"k_proj", "v_proj", "k_norm", "v_norm"}
        )

    model_cls = getattr(gemma4, "Model", None)
    original_sanitize = getattr(model_cls, "sanitize", None)
    if original_sanitize is not None and not getattr(
        original_sanitize, "_vmlx_gemma4_shared_kv_sanitize_patch", False
    ):

        def _vmlx_gemma4_sanitize(self, weights):
            sanitized = original_sanitize(self, weights)
            return {
                key: value
                for key, value in sanitized.items()
                if not _is_unused_shared_kv_weight(self, key)
            }

        _vmlx_gemma4_sanitize._vmlx_gemma4_shared_kv_sanitize_patch = True  # type: ignore[attr-defined]
        model_cls.sanitize = _vmlx_gemma4_sanitize

    language_model_cls = getattr(language, "LanguageModel", None)
    original_language_sanitize = getattr(language_model_cls, "sanitize", None)
    if original_language_sanitize is None or getattr(
        original_language_sanitize, "_vmlx_gemma4_shared_kv_sanitize_patch", False
    ):
        return

    def _vmlx_gemma4_language_sanitize(self, weights):
        sanitized = original_language_sanitize(self, weights)
        return {
            key: value
            for key, value in sanitized.items()
            if not _is_unused_shared_kv_weight(self, key)
        }

    _vmlx_gemma4_language_sanitize._vmlx_gemma4_shared_kv_sanitize_patch = True  # type: ignore[attr-defined]
    language_model_cls.sanitize = _vmlx_gemma4_language_sanitize


def _patch_gemma4_video_processor_hf_kwargs() -> None:
    """Backport mlx-vlm#1321: ignore unused HF video processor config keys."""
    try:
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4VideoProcessor
    except Exception as exc:
        logger.debug("mlx_vlm_compat: Gemma4VideoProcessor unavailable: %s", exc)
        return

    original_init = getattr(Gemma4VideoProcessor, "__init__", None)
    if original_init is None or getattr(
        original_init, "_vmlx_gemma4_video_hf_kwargs_patch", False
    ):
        return

    signature = inspect.signature(original_init)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return
    accepted = set(signature.parameters) - {"self"}

    def _vmlx_gemma4_video_init(self, *args, **kwargs):
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in accepted
        }
        return original_init(self, *args, **filtered_kwargs)

    _vmlx_gemma4_video_init._vmlx_gemma4_video_hf_kwargs_patch = True  # type: ignore[attr-defined]
    Gemma4VideoProcessor.__init__ = _vmlx_gemma4_video_init
