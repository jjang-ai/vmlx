# SPDX-License-Identifier: Apache-2.0
"""Runtime shims for upstream mlx_vlm fixes not yet present in pinned wheels."""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def _is_gemma4_unused_shared_kv_weight(owner, key: str) -> bool:
    """Return true for Gemma4 k/v weights materialized on shared-KV layers."""
    prefixes = (
        "language_model.model.layers.",
        "model.language_model.model.layers.",
    )
    prefix = next((item for item in prefixes if key.startswith(item)), None)
    if prefix is None:
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


def _drop_gemma4_unused_shared_kv_weights(owner, weights):
    if isinstance(weights, dict):
        return {
            key: value
            for key, value in weights.items()
            if not _is_gemma4_unused_shared_kv_weight(owner, key)
        }
    if isinstance(weights, list):
        return [
            (key, value)
            for key, value in weights
            if not _is_gemma4_unused_shared_kv_weight(owner, key)
        ]
    return weights


def install() -> None:
    """Install all mlx_vlm compatibility patches once."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_qwen3_vl_chunked_prefill_deepstack_alignment()
    _patch_lfm25_vl_projector_layernorm_loading()
    _patch_gemma4_video_processor_hf_kwargs()
    _patch_gemma4_shared_kv_layers()
    _APPLIED = True


def _patch_lfm25_vl_projector_layernorm_loading() -> None:
    """Backport mlx-vlm#1328: always materialize LFM2.5-VL projector layernorm."""
    try:
        from mlx_vlm.models.lfm2_vl import lfm2_vl
    except Exception as exc:
        logger.debug("mlx_vlm_compat: LFM2.5-VL projector unavailable: %s", exc)
        return

    projector_cls = getattr(lfm2_vl, "Lfm2VlMultiModalProjector", None)
    original_init = getattr(projector_cls, "__init__", None)
    if original_init is not None and not getattr(
        original_init, "_vmlx_lfm25_vl_layernorm_load_patch", False
    ):

        def _vmlx_lfm25_vl_projector_init(self, config):
            original_init(self, config)
            self.projector_use_layernorm = bool(
                getattr(config, "projector_use_layernorm", False)
            )
            in_channels = getattr(config.vision_config, "hidden_size") * (
                getattr(config, "downsample_factor") ** 2
            )
            if isinstance(getattr(self, "layer_norm", None), lfm2_vl.nn.Identity):
                self.layer_norm = lfm2_vl.nn.LayerNorm(in_channels)

        _vmlx_lfm25_vl_projector_init._vmlx_lfm25_vl_layernorm_load_patch = True  # type: ignore[attr-defined]
        projector_cls.__init__ = _vmlx_lfm25_vl_projector_init

    original_call = getattr(projector_cls, "__call__", None)
    if original_call is None or getattr(
        original_call, "_vmlx_lfm25_vl_layernorm_load_patch", False
    ):
        return

    def _vmlx_lfm25_vl_projector_call(self, x):
        if getattr(self, "projector_use_layernorm", True):
            x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.linear_2(lfm2_vl.nn.gelu(x))
        return x

    _vmlx_lfm25_vl_projector_call._vmlx_lfm25_vl_layernorm_load_patch = True  # type: ignore[attr-defined]
    projector_cls.__call__ = _vmlx_lfm25_vl_projector_call


def _patch_qwen3_vl_chunked_prefill_deepstack_alignment() -> None:
    """Backport mlx-vlm#1325/#1332: align Qwen3-VL visual state per chunk."""

    def _patch_language_model(module_name: str) -> None:
        try:
            module = __import__(module_name, fromlist=["LanguageModel"])
        except Exception as exc:
            logger.debug("mlx_vlm_compat: %s unavailable: %s", module_name, exc)
            return

        language_model_cls = getattr(module, "LanguageModel", None)
        original_call = getattr(language_model_cls, "__call__", None)
        if original_call is None or getattr(
            original_call, "_vmlx_qwen3_vl_deepstack_alignment_patch", False
        ):
            return

        def _vmlx_qwen3_vl_call(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            n_to_process = kwargs.pop("n_to_process", None)
            if (
                n_to_process is not None
                and visual_pos_masks is not None
                and deepstack_visual_embeds is not None
            ):
                try:
                    start = int(n_to_process)
                    window = int(inputs.shape[1])
                    window_masks = visual_pos_masks[:, start : start + window]
                    n_before = int(visual_pos_masks[:, :start].sum().item())
                    n_window = int(window_masks.sum().item())
                    deepstack_visual_embeds = [
                        embeds[n_before : n_before + n_window]
                        for embeds in deepstack_visual_embeds
                    ]
                    visual_pos_masks = window_masks
                except Exception as exc:
                    logger.debug(
                        "mlx_vlm_compat: Qwen3-VL deepstack alignment skipped: %s",
                        exc,
                    )
                    kwargs["n_to_process"] = n_to_process
            elif n_to_process is not None:
                kwargs["n_to_process"] = n_to_process

            return original_call(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        _vmlx_qwen3_vl_call._vmlx_qwen3_vl_deepstack_alignment_patch = True  # type: ignore[attr-defined]
        language_model_cls.__call__ = _vmlx_qwen3_vl_call

    _patch_language_model("mlx_vlm.models.qwen3_vl.language")
    _patch_language_model("mlx_vlm.models.qwen3_vl_moe.language")


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

    model_cls = getattr(gemma4, "Model", None)
    original_sanitize = getattr(model_cls, "sanitize", None)
    if original_sanitize is not None and not getattr(
        original_sanitize, "_vmlx_gemma4_shared_kv_sanitize_patch", False
    ):

        def _vmlx_gemma4_sanitize(self, weights):
            sanitized = original_sanitize(self, weights)
            return _drop_gemma4_unused_shared_kv_weights(self, sanitized)

        _vmlx_gemma4_sanitize._vmlx_gemma4_shared_kv_sanitize_patch = True  # type: ignore[attr-defined]
        model_cls.sanitize = _vmlx_gemma4_sanitize

    original_load_weights = getattr(model_cls, "load_weights", None)
    if original_load_weights is not None and not getattr(
        original_load_weights, "_vmlx_gemma4_shared_kv_load_weights_patch", False
    ):

        def _vmlx_gemma4_load_weights(self, file_or_weights, strict=True):
            weights = file_or_weights
            if isinstance(weights, str):
                try:
                    import mlx.core as mx

                    weights = list(mx.load(weights).items())
                except Exception:
                    return original_load_weights(self, file_or_weights, strict=strict)
            weights = _drop_gemma4_unused_shared_kv_weights(self, weights)
            return original_load_weights(self, weights, strict=strict)

        _vmlx_gemma4_load_weights._vmlx_gemma4_shared_kv_load_weights_patch = True  # type: ignore[attr-defined]
        model_cls.load_weights = _vmlx_gemma4_load_weights

    language_model_cls = getattr(language, "LanguageModel", None)
    original_language_sanitize = getattr(language_model_cls, "sanitize", None)
    if original_language_sanitize is None or getattr(
        original_language_sanitize, "_vmlx_gemma4_shared_kv_sanitize_patch", False
    ):
        return

    def _vmlx_gemma4_language_sanitize(self, weights):
        sanitized = original_language_sanitize(self, weights)
        return _drop_gemma4_unused_shared_kv_weights(self, sanitized)

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
