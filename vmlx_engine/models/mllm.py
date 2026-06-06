# SPDX-License-Identifier: Apache-2.0
"""
MLX Multimodal Language Model (MLLM) wrapper.

This module provides a wrapper around mlx-vlm for multimodal inference,
supporting vision, audio, and video understanding on Apple Silicon.

Features:
- OpenAI-compatible API format for images and video
- Smart video frame extraction with configurable FPS
- Base64 and URL image support
- Streaming generation
- MLLM KV cache for repeated image/video+prompt combinations
"""

import atexit
import base64
import importlib
import importlib.machinery
import json
import logging
import math
import os
import sys
import tempfile
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import numpy as np
import requests

from vmlx_engine.mlx_memory import clear_mlx_memory_cache
from vmlx_engine.mllm_cache import MLLMPrefixCacheManager
from vmlx_engine.errors import UnsupportedMediaModalityError

logger = logging.getLogger(__name__)


_VLM_STREAM = None


def _register_local_mlx_vlm_runtime_if_needed(model_path: str | Path) -> None:
    """Register vMLX-owned mlx-vlm model classes before stock mlx-vlm import.

    Some locally converted bundles use architectures that are not yet shipped
    by upstream mlx-vlm. Stock `mlx_vlm.utils.load()` resolves the model class
    from `config.json["model_type"]` before weights are inspected, so local
    adapters must be registered at the MLLM load boundary. This is not a
    monkey-patch of an instantiated model; it supplies the missing model module
    that mlx-vlm's normal loader contract already expects.
    """
    try:
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.exists():
            return
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type == "zaya1_vl":
        from ..models.zaya1_vl import register_mlx_vlm_zaya1_vl

        register_mlx_vlm_zaya1_vl()
    if model_type == "step3p7":
        _register_step3p7_mlx_vlm_runtime()
    if model_type == "mimo_v2":
        _register_mimo_v2_mlx_vlm_runtime()


def _register_step3p7_mlx_vlm_runtime() -> None:
    module_name = "mlx_vlm.models.step3p7"
    processor_module_name = f"{module_name}.processing_step3"
    if module_name in sys.modules and processor_module_name in sys.modules:
        return

    source_runtime = importlib.import_module("vmlx_engine.models.step3p7_mlx_vlm")
    module = types.ModuleType(module_name)
    for name in (
        "Model",
        "ModelConfig",
        "TextConfig",
        "VisionConfig",
        "ProjectorConfig",
        "LanguageModel",
        "VisionModel",
        "Step3p7Projector",
        "Step3VisionProcessor",
        "Step3VLProcessor",
    ):
        setattr(module, name, getattr(source_runtime, name))
    module.__file__ = getattr(source_runtime, "__file__", module_name)
    module.__package__ = module_name
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name,
        loader=None,
        origin=module.__file__,
        is_package=True,
    )
    sys.modules[module_name] = module

    processor_module = types.ModuleType(processor_module_name)
    processor_module.Step3VisionProcessor = source_runtime.Step3VisionProcessor
    processor_module.Step3VLProcessor = source_runtime.Step3VLProcessor
    processor_module.__file__ = getattr(source_runtime, "__file__", processor_module_name)
    processor_module.__package__ = module_name
    processor_module.__spec__ = importlib.machinery.ModuleSpec(
        processor_module_name,
        loader=None,
        origin=processor_module.__file__,
        is_package=False,
    )
    sys.modules[processor_module_name] = processor_module


def _register_mimo_v2_mlx_vlm_runtime() -> None:
    module_name = "mlx_vlm.models.mimo_v2"
    if module_name in sys.modules:
        return

    text_runtime = importlib.import_module("jang_tools.mimo_v2.mlx_model")

    import mlx.nn as nn

    TextConfig = getattr(text_runtime, "ModelArgs")
    TextModel = getattr(text_runtime, "Model")

    _INPUTS_EMBEDS_UNSET = object()

    class _MiMoV2CausalLMOutput:
        def __init__(self, logits):
            self.logits = logits
            self.cross_attention_states = None
            self.encoder_outputs = None

    class _MiMoV2LanguageModelCompat(nn.Module):
        """Adapt jang_tools' mlx-lm MiMo model to mlx-vlm's language-model API."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @property
        def model(self):
            return self.inner.model

        @property
        def layers(self):
            return self.inner.layers

        def make_cache(self):
            return self.inner.make_cache()

        def sanitize(self, weights):
            return self.inner.sanitize(weights)

        def load_weights(self, weights, strict=True):
            return self.inner.load_weights(weights, strict=strict)

        def __call__(
            self,
            input_ids=None,
            *,
            inputs_embeds=_INPUTS_EMBEDS_UNSET,
            cache=None,
            mask=None,
            **kwargs,
        ):
            mllm_style_call = inputs_embeds is not _INPUTS_EMBEDS_UNSET
            if mllm_style_call:
                try:
                    logits = self.inner(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        cache=cache,
                        mask=mask,
                        **kwargs,
                    )
                except TypeError as exc:
                    if "inputs_embeds" not in str(exc):
                        raise
                    if inputs_embeds is None:
                        logits = self.inner(input_ids, cache=cache, mask=mask, **kwargs)
                    else:
                        logits = self.inner.model(
                            input_ids=None,
                            inputs_embeds=inputs_embeds,
                            cache=cache,
                            mask=mask,
                            **kwargs,
                        )
                        logits = self.inner.lm_head(logits)
                if hasattr(logits, "logits"):
                    return logits
                return _MiMoV2CausalLMOutput(logits)
            return self.inner(input_ids, cache=cache, mask=mask, **kwargs)

    class VisionConfig:
        @classmethod
        def from_dict(cls, params):
            return cls()

    class AudioConfig:
        @classmethod
        def from_dict(cls, params):
            return cls()

    class ModelConfig:
        def __init__(
            self,
            text_config=None,
            vision_config=None,
            audio_config=None,
            model_type: str = "mimo_v2",
            eos_token_id=None,
            **kwargs,
        ):
            self.model_type = model_type
            self.text_config = text_config
            self.vision_config = vision_config or VisionConfig()
            self.audio_config = audio_config or AudioConfig()
            self.eos_token_id = eos_token_id

        @classmethod
        def from_dict(cls, params):
            merged_text = dict(params)
            if isinstance(params.get("text_config"), dict):
                merged_text.update(params["text_config"])
            params["text_config"] = merged_text
            quantization = params.get("quantization")
            if isinstance(quantization, dict):
                for key, value in list(quantization.items()):
                    if (
                        isinstance(value, dict)
                        and key.startswith(("model.", "lm_head"))
                        and not key.startswith("language_model.")
                    ):
                        quantization.setdefault(f"language_model.{key}", value)
            return cls(
                text_config=TextConfig.from_dict(merged_text),
                vision_config=VisionConfig.from_dict(params.get("vision_config") or {}),
                audio_config=AudioConfig.from_dict(params.get("audio_config") or {}),
                model_type=str(params.get("model_type") or "mimo_v2"),
                eos_token_id=params.get("eos_token_id"),
            )

    class VisionModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()

        def sanitize(self, weights):
            return {
                key: value
                for key, value in weights.items()
                if not (
                    key.startswith("visual.")
                    or key.startswith("vision_tower.")
                    or key.startswith("audio_encoder.")
                    or key.startswith("speech_embeddings.")
                )
            }

    class AudioModel(VisionModel):
        pass

    class Model(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.language_model = _MiMoV2LanguageModelCompat(TextModel(config.text_config))
            self._mimo_v2_quantized_for_load = False

        @property
        def layers(self):
            return self.language_model.layers

        def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
            from mlx_vlm.models.base import InputEmbeddingsFeatures

            if pixel_values is not None:
                raise UnsupportedMediaModalityError(
                    "vision",
                    "MiMo-V2.5 JANG_2L vision weights are preserved, but the "
                    "current Python runtime only wires the text decode path.",
                    family="mimo_v2",
                )
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        def __call__(
            self,
            input_ids,
            pixel_values=None,
            inputs_embeds=_INPUTS_EMBEDS_UNSET,
            mask=None,
            cache=None,
            **kwargs,
        ):
            if pixel_values is not None:
                raise UnsupportedMediaModalityError(
                    "vision",
                    "MiMo-V2.5 JANG_2L vision input is not wired in this Python runtime.",
                    family="mimo_v2",
                )
            if inputs_embeds is not _INPUTS_EMBEDS_UNSET:
                return self.language_model(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    cache=cache,
                    mask=mask,
                    **kwargs,
                )
            return self.language_model(input_ids, cache=cache, mask=mask, **kwargs)

        def load_weights(self, weights, strict=True):
            filtered = {}
            for key, value in weights:
                if key.startswith(("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp.")):
                    continue
                filtered[key] = value

            if hasattr(self.language_model, "sanitize"):
                filtered = self.language_model.sanitize(filtered)

            quantization = getattr(self.config.text_config, "quantization", None)
            if isinstance(quantization, dict) and not self._mimo_v2_quantized_for_load:

                def class_predicate(path, module):
                    if not hasattr(module, "to_quantized"):
                        return False
                    override = quantization.get(path)
                    if isinstance(override, dict):
                        return override
                    return f"{path}.scales" in filtered

                nn.quantize(
                    self.language_model.inner,
                    group_size=quantization.get("group_size"),
                    bits=quantization.get("bits"),
                    mode=quantization.get("mode", "affine"),
                    class_predicate=class_predicate,
                )
                self._mimo_v2_quantized_for_load = True

            return self.language_model.load_weights(list(filtered.items()), strict=strict)

        def sanitize(self, weights):
            return self.language_model.sanitize(weights)

    module = types.ModuleType(module_name)
    module.Model = Model
    module.ModelConfig = ModelConfig
    module.TextConfig = TextConfig
    module.VisionConfig = VisionConfig
    module.AudioConfig = AudioConfig
    module.LanguageModel = TextModel
    module.VisionModel = VisionModel
    module.AudioModel = AudioModel
    module.__file__ = getattr(text_runtime, "__file__", module_name)
    sys.modules[module_name] = module


def _vlm_stream():
    """Create one stream for simple mlx-vlm generation paths.

    The batched VLM scheduler already pins MLX work to a dedicated stream.
    SimpleEngine's direct `mlx_vlm.stream_generate` path must do the same or
    JANGTQ/VL kernels can later materialize on a thread with no MLX stream.
    """
    global _VLM_STREAM
    if _VLM_STREAM is None:
        try:
            import mlx.core as mx

            _VLM_STREAM = mx.new_stream(mx.default_device())
        except Exception:
            try:
                import mlx.core as mx

                _VLM_STREAM = mx.default_stream(mx.default_device())
            except Exception:
                _VLM_STREAM = False
    return None if _VLM_STREAM is False else _VLM_STREAM


def reset_vlm_stream() -> None:
    """Clear the direct mlx-vlm stream handle after model teardown.

    ``_VLM_STREAM`` is owned by the worker thread that first runs direct
    ``mlx_vlm.stream_generate``. Deep sleep, wake, model switch, and direct
    SimpleEngine teardown can replace that worker; retaining the old stream
    lets a later request enter ``mx.stream(old_stream)`` from a different
    thread and raises ``RuntimeError: There is no Stream(gpu, N) in current
    thread``.
    """
    global _VLM_STREAM
    _VLM_STREAM = None


def _set_vlm_inference_mode(model: Any) -> None:
    """Put mlx-vlm model copies in eval mode so inference kernels stay enabled."""
    seen: set[int] = set()
    for candidate in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        eval_fn = getattr(candidate, "eval", None)
        if callable(eval_fn):
            eval_fn()


class _MaybeVLMStream:
    __slots__ = ("_cm",)

    def __init__(self):
        self._cm = None
        stream = _vlm_stream()
        if stream is not None:
            try:
                import mlx.core as mx

                self._cm = mx.stream(stream)
            except Exception:
                self._cm = None

    def __enter__(self):
        if self._cm is not None:
            self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        if self._cm is not None:
            return self._cm.__exit__(*exc)
        return False


def _prompt_cache_state_from_entry(cache_entry: Any) -> Any | None:
    """Build mlx-vlm PromptCacheState from a vMLX MLLM prefix entry."""
    if cache_entry is None:
        return None
    kv_cache = getattr(cache_entry, "kv_cache", None)
    token_ids = getattr(cache_entry, "token_ids", None)
    if kv_cache is None or not token_ids:
        return None
    # Legacy MLLMPrefixCacheManager.store_cache() writes dummy all-zero token IDs
    # when the caller only knows a token count. mlx-vlm PromptCacheState uses
    # token_ids as the authoritative trim boundary, so dummy IDs would turn a
    # stale cache into a fake prefix hit.
    if all(token_id == 0 for token_id in token_ids):
        return None
    try:
        from mlx_vlm.generate import PromptCacheState
    except Exception:
        return None

    state = PromptCacheState()
    state.cache = kv_cache
    state.token_ids = list(token_ids)
    return state


def _copy_prompt_cache_to_prompt_boundary(
    prompt_cache: list[Any], prompt_tokens_count: int
) -> list[Any]:
    """Copy an mlx-vlm prompt cache and trim generated-token tails."""
    import copy

    import mlx.core as mx

    cache_to_store = []
    prompt_tokens_count = max(int(prompt_tokens_count or 0), 0)
    for layer_cache in prompt_cache:
        new_cache = copy.copy(layer_cache)
        if hasattr(layer_cache, "state"):
            state = layer_cache.state
            if state is not None and len(state) >= 2 and state[0] is not None:
                keys = mx.array(state[0])
                values = mx.array(state[1])
                cache_len = keys.shape[2] if len(keys.shape) >= 3 else None
                target_len = prompt_tokens_count
                if cache_len is not None:
                    target_len = min(prompt_tokens_count, cache_len)
                    if cache_len > target_len:
                        keys = keys[:, :, :target_len, :]
                        values = values[:, :, :target_len, :]
                new_cache.keys = keys
                new_cache.values = values
                if len(state) >= 3:
                    new_cache.offset = min(state[2], target_len)
                elif hasattr(layer_cache, "offset"):
                    new_cache.offset = min(layer_cache.offset, target_len)
        cache_to_store.append(new_cache)
    return cache_to_store


class TempFileManager:
    """Thread-safe manager for tracking and cleaning up temporary files."""

    def __init__(self):
        self._files: set[str] = set()
        self._lock = threading.Lock()
        atexit.register(self.cleanup_all)

    def register(self, path: str) -> str:
        """Register a temp file for tracking. Returns the path for convenience."""
        with self._lock:
            self._files.add(path)
        return path

    def cleanup(self, path: str) -> bool:
        """Clean up a specific temp file. Returns True if successful."""
        try:
            if os.path.exists(path):
                os.unlink(path)
                with self._lock:
                    self._files.discard(path)
                logger.debug(f"Cleaned up temp file: {path}")
                return True
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")
        return False

    def cleanup_all(self) -> int:
        """Clean up all tracked temp files. Returns count of cleaned files."""
        with self._lock:
            files_to_clean = list(self._files)
            self._files.clear()

        cleaned = 0
        for path in files_to_clean:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    cleaned += 1
            except OSError:
                pass

        # Do not log from atexit cleanup: logging streams may already be closed
        # by pytest or the app shutdown path, which makes logging print noisy
        # "I/O operation on closed file" diagnostics despite successful cleanup.
        return cleaned


# Global temp file manager
_temp_manager = TempFileManager()



# Video processing constants
FRAME_FACTOR = 2  # Frames must be divisible by this
DEFAULT_FPS = 2.0  # Default frames per second for video
MIN_FRAMES = 4
MAX_FRAMES = 128  # Practical limit for most MLLMs
IMAGE_FACTOR = 28  # For smart resize

# Security: File size limits (in bytes)
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB max for images
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB max for videos
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB max for audio
MAX_BASE64_IMAGE_LENGTH = 30 * 1024 * 1024  # 30 MB base64 string (~22 MB decoded)
MAX_BASE64_VIDEO_LENGTH = 700 * 1024 * 1024  # 700 MB base64 string (~500 MB decoded)
MAX_BASE64_AUDIO_LENGTH = 70 * 1024 * 1024  # 70 MB base64 string (~50 MB decoded)


class FileSizeExceededError(Exception):
    """Raised when a downloaded file exceeds the size limit."""

    pass


@dataclass
class MultimodalInput:
    """Input for multimodal generation."""

    prompt: str
    images: list[str] = field(default_factory=list)  # Paths, URLs, or base64
    videos: list[str] = field(default_factory=list)  # Paths
    audio: list[str] = field(default_factory=list)  # Paths


@dataclass
class MLLMOutput:
    """Output from multimodal language model."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def is_base64_image(s: str) -> bool:
    """Check if string is base64-encoded image data."""
    return s.startswith("data:image/") or (
        len(s) > 100 and not s.startswith(("http://", "https://", "/"))
    )


def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://"))


def is_base64_video(s: str) -> bool:
    """Check if string is base64-encoded video data."""
    return s.startswith("data:video/")


def is_base64_audio(s: str) -> bool:
    """Check if string is base64-encoded audio data."""
    return s.startswith("data:audio/")


def decode_base64_image(
    base64_string: str, max_length: int = MAX_BASE64_IMAGE_LENGTH
) -> bytes:
    """
    Decode base64 image to bytes.

    Args:
        base64_string: Base64 encoded image (optionally with data URL prefix)
        max_length: Maximum allowed length of base64 string

    Returns:
        Decoded image bytes

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 image data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
    if base64_string.startswith("data:"):
        # Extract the base64 part after the comma
        _, data = base64_string.split(",", 1)
        return base64.b64decode(data)
    return base64.b64decode(base64_string)


def download_image(url: str, timeout: int = 30, max_size: int = MAX_IMAGE_SIZE) -> str:
    """
    Download image from URL and return local path.

    Args:
        url: Image URL
        timeout: Download timeout in seconds
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded image

    Raises:
        FileSizeExceededError: If image exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    # First, make a HEAD request to check Content-Length
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Image at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "png" in content_type:
        ext = ".png"
    elif "gif" in content_type:
        ext = ".gif"
    elif "webp" in content_type:
        ext = ".webp"
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = Path(path).suffix or ".jpg"

    # Save to temp file with size checking during download
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Image at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    return _temp_manager.register(temp_file.name)


def download_video(url: str, timeout: int = 120, max_size: int = MAX_VIDEO_SIZE) -> str:
    """
    Download video from URL and return local path.

    Args:
        url: Video URL (http/https)
        timeout: Download timeout in seconds (default 120s for larger videos)
        max_size: Maximum allowed file size in bytes

    Returns:
        Local file path to downloaded video

    Raises:
        FileSizeExceededError: If video exceeds max_size
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    logger.info(f"Downloading video from: {url}")

    # First, make a HEAD request to check Content-Length
    try:
        head_response = requests.head(
            url, timeout=timeout, headers=headers, allow_redirects=True, verify=True
        )
        content_length = head_response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise FileSizeExceededError(
                f"Video at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
                f"{max_size / 1024 / 1024:.1f} MB limit"
            )
    except requests.RequestException:
        # HEAD request failed, proceed with GET and check during download
        pass

    response = requests.get(
        url, timeout=timeout, headers=headers, stream=True, verify=True
    )
    response.raise_for_status()

    # Check Content-Length header from GET response
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise FileSizeExceededError(
            f"Video at {url} exceeds maximum size: {int(content_length) / 1024 / 1024:.1f} MB > "
            f"{max_size / 1024 / 1024:.1f} MB limit"
        )

    # Determine extension from content type or URL
    content_type = response.headers.get("content-type", "")
    if "mp4" in content_type:
        ext = ".mp4"
    elif "webm" in content_type:
        ext = ".webm"
    elif "avi" in content_type:
        ext = ".avi"
    elif "mov" in content_type or "quicktime" in content_type:
        ext = ".mov"
    elif "mkv" in content_type:
        ext = ".mkv"
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = Path(path).suffix or ".mp4"

    # Save to temp file (stream for larger files) with size checking
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    downloaded_size = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileSizeExceededError(
                    f"Video at {url} exceeds maximum size during download: "
                    f"{downloaded_size / 1024 / 1024:.1f} MB > {max_size / 1024 / 1024:.1f} MB limit"
                )
            temp_file.write(chunk)
        temp_file.close()
    except FileSizeExceededError:
        raise
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

    file_size = Path(temp_file.name).stat().st_size
    logger.info(
        f"Video downloaded: {temp_file.name} ({file_size / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_video(
    base64_string: str, max_length: int = MAX_BASE64_VIDEO_LENGTH
) -> str:
    """
    Decode base64 video to temp file and return path.

    Supports format: data:video/mp4;base64,AAAA...

    Args:
        base64_string: Base64-encoded video with data URL prefix
        max_length: Maximum allowed length of base64 string

    Returns:
        Local file path to decoded video

    Raises:
        FileSizeExceededError: If base64 string exceeds max_length
    """
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 video data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )

    # Extract format and data
    if base64_string.startswith("data:video/"):
        # Format: data:video/mp4;base64,AAAA...
        header, data = base64_string.split(",", 1)
        # Extract extension from header (e.g., "data:video/mp4;base64" -> "mp4")
        format_part = header.split(";")[0]  # "data:video/mp4"
        ext = "." + format_part.split("/")[-1]  # ".mp4"
    else:
        # Assume mp4 if no header
        data = base64_string
        ext = ".mp4"

    # Decode, save, and register for cleanup
    video_bytes = base64.b64decode(data)
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(video_bytes)
    temp_file.close()

    logger.info(
        f"Base64 video decoded: {temp_file.name} ({len(video_bytes) / 1024 / 1024:.1f} MB)"
    )

    return _temp_manager.register(temp_file.name)


def decode_base64_audio(
    base64_string: str,
    *,
    default_format: str = "wav",
    max_length: int = MAX_BASE64_AUDIO_LENGTH,
) -> str:
    """Decode base64 audio to a temp file and return its path."""
    if len(base64_string) > max_length:
        raise FileSizeExceededError(
            f"Base64 audio data exceeds maximum size: {len(base64_string) / 1024 / 1024:.1f} MB > "
            f"{max_length / 1024 / 1024:.1f} MB limit"
        )
    ext = "." + (default_format or "wav").lower().lstrip(".")
    if base64_string.startswith("data:audio/"):
        header, data = base64_string.split(",", 1)
        format_part = header.split(";", 1)[0]
        fmt = format_part.split("/", 1)[-1]
        if fmt:
            ext = "." + fmt.lower().replace("x-wav", "wav")
    else:
        data = base64_string
    audio_bytes = base64.b64decode(data)
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise FileSizeExceededError(
            f"Audio data exceeds maximum size: {len(audio_bytes) / 1024 / 1024:.1f} MB > "
            f"{MAX_AUDIO_SIZE / 1024 / 1024:.1f} MB limit"
        )
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    logger.info(
        f"Base64 audio decoded: {temp_file.name} ({len(audio_bytes) / 1024 / 1024:.1f} MB)"
    )
    return _temp_manager.register(temp_file.name)


def process_audio_input(audio: str | dict) -> str:
    """Process audio input in OpenAI/data-url/path/URL formats."""
    if isinstance(audio, dict):
        src = audio.get("input_audio") or audio.get("audio") or audio.get("audio_url") or audio
        if isinstance(src, dict):
            if src.get("data"):
                return decode_base64_audio(
                    str(src["data"]),
                    default_format=str(src.get("format") or "wav"),
                )
            audio = src.get("url") or src.get("path") or ""
        else:
            audio = str(src or "")
    audio = str(audio or "")
    if not audio:
        raise ValueError("Empty audio input")
    if is_base64_audio(audio):
        return decode_base64_audio(audio)
    return audio


def process_video_input(video: str | dict) -> str:
    """
    Process video input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string (data:video/mp4;base64,...)
    - OpenAI format dict: {"url": "..."} or {"url": "data:video/...;base64,..."}

    Args:
        video: Video input in any supported format

    Returns:
        Local file path to video
    """
    # Handle dict format (OpenAI style)
    if isinstance(video, dict):
        url = video.get("url", video.get("video_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        video = url

    if not video:
        raise ValueError("Empty video input")

    # Check URL/data-URL formats before filesystem probing. Long data URLs can
    # exceed platform filename limits if passed through Path.exists().
    if is_base64_video(video):
        return decode_base64_video(video)

    if is_url(video):
        return download_video(video)

    # Check if it's a local file
    try:
        if Path(video).exists():
            return video
    except OSError:
        pass

    raise ValueError(f"Cannot process video: {video[:50]}...")


def _normalize_tool_calls_for_template(tool_calls: Any) -> list[dict] | None:
    """Return OpenAI tool calls as plain mappings with dict arguments."""
    if not tool_calls:
        return None
    normalized: list[dict] = []
    for tool_call in tool_calls:
        if hasattr(tool_call, "model_dump"):
            call = tool_call.model_dump(exclude_none=True)
        elif hasattr(tool_call, "dict"):
            call = tool_call.dict()
            call = {k: v for k, v in call.items() if v is not None}
        elif isinstance(tool_call, dict):
            call = dict(tool_call)
        else:
            continue

        fn = call.get("function")
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump(exclude_none=True)
        elif hasattr(fn, "dict"):
            fn = fn.dict()
            fn = {k: v for k, v in fn.items() if v is not None}
        elif isinstance(fn, dict):
            fn = dict(fn)
        else:
            fn = {}

        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                fn["arguments"] = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        elif args is None:
            fn["arguments"] = {}

        call["function"] = fn
        normalized.append(call)

    return normalized or None


# Cache for base64 images to avoid re-saving the same image
# OrderedDict for LRU eviction with bounded size
from collections import OrderedDict

_base64_image_cache: OrderedDict[str, str] = OrderedDict()  # hash -> temp file path
_BASE64_IMAGE_CACHE_MAX_SIZE = 100


def save_base64_image(base64_string: str) -> str:
    """Save base64 image to temp file and return path. Caches identical images."""
    import hashlib

    # Hash the full base64 string to avoid collisions
    image_hash = hashlib.sha256(base64_string.encode()).hexdigest()

    # Return cached path if available and file still exists
    if image_hash in _base64_image_cache:
        cached_path = _base64_image_cache[image_hash]
        if Path(cached_path).exists():
            _base64_image_cache.move_to_end(image_hash)  # LRU: mark as recently used
            return cached_path
        else:
            # File was cleaned up, remove stale entry
            del _base64_image_cache[image_hash]

    image_bytes = decode_base64_image(base64_string)

    # Detect format from magic bytes
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif image_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        ext = ".gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        ext = ".webp"
    else:
        ext = ".jpg"  # Default

    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    path = _temp_manager.register(temp_file.name)
    _base64_image_cache[image_hash] = path

    # Evict oldest entries if cache exceeds size limit
    # M5: Also delete the temp file on eviction (not just the cache entry)
    while len(_base64_image_cache) > _BASE64_IMAGE_CACHE_MAX_SIZE:
        _, evicted_path = _base64_image_cache.popitem(last=False)
        _temp_manager.cleanup(evicted_path)

    return path


def process_image_input(image: str | dict) -> str:
    """
    Process image input in various formats and return local path.

    Supports:
    - Local file path
    - URL (http/https)
    - Base64 encoded string
    - OpenAI format dict: {"url": "..."} or {"url": "data:image/...;base64,..."}
    """
    # Handle dict format (OpenAI style)
    if isinstance(image, dict):
        url = image.get("url", image.get("image_url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        image = url

    if not image:
        raise ValueError("Empty image input")

    # Check if it's base64 FIRST (before Path.exists() which fails on long strings)
    if is_base64_image(image):
        return save_base64_image(image)

    # Check if it's a URL
    if is_url(image):
        return download_image(image)

    # Check if it's a local file (only for short strings that could be paths)
    if len(image) < 4096 and Path(image).exists():
        return image

    raise ValueError(f"Cannot process image: {image[:50]}...")


def round_by_factor(x: int, factor: int) -> int:
    """Round to nearest multiple of factor."""
    return round(x / factor) * factor


def ceil_by_factor(x: float, factor: int) -> int:
    """Ceiling to next multiple of factor."""
    return math.ceil(x / factor) * factor


def floor_by_factor(x: float, factor: int) -> int:
    """Floor to previous multiple of factor."""
    return math.floor(x / factor) * factor


def smart_nframes(
    total_frames: int,
    video_fps: float,
    target_fps: float = DEFAULT_FPS,
    min_frames: int = MIN_FRAMES,
    max_frames: int = MAX_FRAMES,
) -> int:
    """
    Calculate optimal number of frames to extract from video.

    Uses smart sampling based on video length and target FPS.
    """
    # Calculate duration-based frame count
    duration = total_frames / video_fps if video_fps > 0 else 0
    nframes = duration * target_fps

    # Clamp to min/max
    nframes = max(min_frames, min(nframes, max_frames, total_frames))

    # Round to factor
    nframes = max(FRAME_FACTOR, floor_by_factor(nframes, FRAME_FACTOR))

    return int(nframes)


def extract_video_frames_smart(
    video_path: str,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """
    Extract frames from video with smart sampling.

    Args:
        video_path: Path to video file
        fps: Target frames per second (default: 2.0)
        max_frames: Maximum frames to extract
        resize: Optional (width, height) to resize frames

    Returns:
        List of frame arrays (RGB format)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculate number of frames to extract
    nframes = smart_nframes(
        total_frames=total_frames,
        video_fps=video_fps,
        target_fps=fps,
        max_frames=max_frames,
    )

    # Calculate frame indices (evenly spaced)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    logger.info(
        f"Video: {total_frames} total frames @ {video_fps:.1f} fps, "
        f"extracting {nframes} frames"
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if specified
        if resize:
            frame = cv2.resize(frame, resize)

        frames.append(frame)

    cap.release()

    return frames


def save_frames_to_temp(frames: list[np.ndarray]) -> list[str]:
    """Save frame arrays to temporary files and return paths."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for frame processing")

    paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file.name, "JPEG", quality=85)
        paths.append(_temp_manager.register(temp_file.name))

    return paths


def _expand_video_placeholders_to_image_frames(
    chat_messages: list[dict],
    frame_counts: list[int],
) -> list[dict]:
    """Replace each video placeholder with N image placeholders.

    Some VLMs route videos through sampled image frames in the SimpleEngine
    path. Their prompts must contain image placeholders matching those sampled
    frames; otherwise the model sees no attached media or receives a native
    video marker unsupported by its image-only template.
    """
    if not frame_counts:
        return chat_messages

    counts = iter(frame_counts)
    rewritten: list[dict] = []
    changed = False
    for msg in chat_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            rewritten.append(msg)
            continue
        new_content: list[dict] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "video":
                count = max(1, int(next(counts, 1) or 1))
                new_content.extend({"type": "image"} for _ in range(count))
                changed = True
            else:
                new_content.append(part)
        if changed:
            out = dict(msg)
            out["content"] = new_content
            rewritten.append(out)
        else:
            rewritten.append(msg)
    return rewritten if changed else chat_messages


class MLXMultimodalLM:
    """
    Wrapper around mlx-vlm for multimodal inference.

    This class provides a unified interface for multimodal language models
    using Apple's MLX framework. Supports:
    - Image understanding (single and multi-image)
    - Video understanding (smart frame extraction)
    - Audio understanding (for supported models)
    - OpenAI-compatible API format

    Supported models include:
    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    - LLaVA
    - Idefics3
    - PaliGemma
    - And more via mlx-vlm

    Example:
        >>> model = MLXMultimodalLM("mlx-community/Qwen2-VL-2B-Instruct-4bit")
        >>> model.load()
        >>> output = model.generate(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        enable_cache: bool = True,
        cache_size: int = 50,
    ):
        """
        Initialize the MLX multimodal language model.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable KV cache for repeated image/video+prompt (default: True)
            cache_size: Maximum cache entries (default: 50)
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.enable_cache = enable_cache

        self.model = None
        self.processor = None
        self.config = None
        self._loaded = False

        # Initialize MLLM prefix cache manager (with vision embedding caching)
        self._cache_manager: MLLMPrefixCacheManager | None = None
        if enable_cache:
            self._cache_manager = MLLMPrefixCacheManager(max_entries=cache_size)

    def load(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        # Resolve HF repo IDs (e.g. "Org/Model") to local cache paths so that
        # file-based checks (jang_config.json, config.json) find the files.
        from ..api.utils import resolve_to_local_path
        resolved_name = resolve_to_local_path(self.model_name)
        _register_local_mlx_vlm_runtime_if_needed(resolved_name)

        # Install mlx_vlm registry patches (gemma4 + kimi_k25) on THIS thread
        # so any module-level `mx.new_stream(...)` in mlx_vlm.generate ends
        # up bound to the loader-executor worker rather than uvicorn
        # MainThread. Otherwise scheduler.step() crashes mid-prefill with
        # `RuntimeError: There is no Stream(gpu, N) in current thread.`
        # (mlxstudio#119 follow-up, 2026-05-02).
        try:
            from .. import _install_mlx_vlm_registry_patches
            _install_mlx_vlm_registry_patches()
        except Exception:
            pass

        # Apply mlx_vlm runtime compat patches (Qwen3-VL grid_thw coercion, #69).
        try:
            from ..utils.mlx_vlm_compat import apply as _apply_mlx_vlm_compat
            _apply_mlx_vlm_compat()
        except Exception as _e:
            logger.debug(f"mlx_vlm compat patch skipped: {_e}")

        # JANG VL models: use JANG loader (handles mixed-precision + mlx-vlm sanitization)
        from ..utils.jang_loader import is_jang_model
        if is_jang_model(resolved_name):
            logger.info(f"Loading JANG VL model: {self.model_name}")
            from ..utils.jang_loader import load_jang_vlm_model
            from mlx_vlm.utils import load_config
            try:
                from .. import server as _server_module
                _smelt = getattr(_server_module, '_smelt_enabled', False)
                _smelt_pct = getattr(_server_module, '_smelt_experts', 50)
            except Exception:
                _smelt = False
                _smelt_pct = 50
            if _smelt:
                from ..utils.smelt_loader import smelt_load
                self.model, self.processor = smelt_load(resolved_name, expert_percent=_smelt_pct)
            else:
                self.model, self.processor = load_jang_vlm_model(resolved_name)
            self.config = load_config(resolved_name)
            _set_vlm_inference_mode(self.model)
            self._loaded = True
            logger.info(f"JANG VL model loaded: {self.model_name}")
            return

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            logger.info(f"Loading MLLM: {self.model_name}")

            # Bundled Python doesn't include torch/torchvision (~2GB). Preemptively
            # patch video processor loading and suppress warnings so models with
            # video_processor sub-components (Qwen3.5-VL, InternVL, etc.) load cleanly.
            self._patch_video_processor()
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchvision.*", category=UserWarning)
                self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)
            if str((self.config or {}).get("model_type", "")).lower() == "zaya1_vl":
                from ..utils.jang_loader import _build_vlm_processor

                self.processor = _build_vlm_processor(Path(resolved_name))

            # vmlx#80 follow-up (Flor1an-B, 2026-04-15): mlx_vlm.load() is a
            # DIFFERENT code path from utils.tokenizer.load_model_with_fallback,
            # so neither the monkey-patch of mlx_lm.tokenizer_utils.load nor
            # the wrapper-level _inject_chat_template_if_missing ever fire for
            # standard mlx-community MLLM quants like gemma-4-31b-8bit. These
            # models ship the chat template as a sidecar chat_template.jinja
            # that HF AutoTokenizer silently ignores, and the engine crashes
            # the first time apply_chat_template is called inside batched.py.
            # Fix: inject the sidecar template directly into the processor
            # (and its inner tokenizer) right after load, before the engine
            # is marked loaded. Idempotent — bails out if a template is
            # already present.
            try:
                from ..utils.tokenizer import _inject_chat_template_if_missing as _inj
                _injected = _inj(self.processor, self.model_name)
                if _injected:
                    logger.info(
                        f"MLLM chat template injected from {_injected} for {self.model_name}"
                    )
            except Exception as _e:
                logger.debug(f"MLLM chat_template injection skipped: {_e}")

            # TurboQuant: auto-enable for ALL MLX VLM models
            _lang = getattr(self.model, 'language_model', None)
            if _lang is not None and hasattr(_lang, 'layers'):
                from ..utils.tokenizer import _apply_turboquant_to_model
                _apply_turboquant_to_model(_lang, resolved_name)

            _set_vlm_inference_mode(self.model)
            self._loaded = True
            logger.info(f"MLLM loaded successfully: {self.model_name}")

        except ImportError as e:
            # 2026-05-03: surface the ACTUAL missing module instead of
            # always blaming mlx-vlm. Common upstream causes:
            #   - torch / torchvision missing → Qwen3VLVideoProcessor and
            #     similar transformers VLM processors hard-require them.
            #     Earlier vMLX builds stripped torch from bundled-python
            #     (see panel/scripts/bundle-python.sh:128) — that strip was
            #     reversed 2026-05-03 because every VL bundle hits this.
            #   - mlx_vlm itself absent → bare wheel issue.
            #   - One of mlx_vlm's submodules (e.g. video_processor backend)
            #     transitively imports a missing dep.
            # Pre-fix we always re-raised "mlx-vlm is required" which hid
            # the real cause. Pass the original exception through.
            _msg = str(e).lower()
            if ("torch" in _msg and ("vision" in _msg or "audio" in _msg)) or "torchvision" in _msg:
                raise ImportError(
                    f"VLM bundle requires torch+torchvision in vMLX bundled-python. "
                    f"Install: <bundled-python>/bin/pip install torch torchvision. "
                    f"Original error: {e}"
                ) from e
            if "mlx_vlm" in _msg or "mlx-vlm" in _msg:
                raise ImportError(
                    f"mlx-vlm is required for multimodal inference. "
                    f"Install with: pip install mlx-vlm. "
                    f"Original error: {e}"
                ) from e
            # Unknown ImportError: re-raise WITH original cause attached so
            # the traceback shows the real failure, not a generic message.
            raise ImportError(
                f"VLM model load failed with ImportError (cause not auto-detected): {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load MLLM: {e}")
            raise

    _video_processor_patched = False

    @classmethod
    def _patch_video_processor(cls):
        """Patch AutoVideoProcessor to skip torchvision requirement.

        Some VLM models (Qwen3.5-VL, InternVL) include a video_processor
        sub-component that requires torchvision. Since we only do image inference
        and torchvision requires torch (~2GB), we patch the processor to load
        without it. Called preemptively before first model load. Idempotent.
        """
        if cls._video_processor_patched:
            return

        try:
            from transformers.models.auto import video_processing_auto

            _orig_from_pretrained = video_processing_auto.AutoVideoProcessor.from_pretrained

            @classmethod  # type: ignore[misc]
            def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                try:
                    return _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
                except (ValueError, ImportError) as e:
                    if "torchvision" in str(e):
                        logger.info("Video processor skipped (torchvision not available) — image-only mode")
                        return None
                    raise

            video_processing_auto.AutoVideoProcessor.from_pretrained = _patched_from_pretrained
            cls._video_processor_patched = True
            logger.debug("Patched AutoVideoProcessor for torchvision-free loading")
        except Exception:
            logger.warning("Failed to patch AutoVideoProcessor — torchvision-dependent VLMs may fail to load")
            cls._video_processor_patched = True  # Don't retry on every load, log warning once

    def _prepare_images(self, images: list) -> list[str]:
        """Process image inputs and return local file paths."""
        processed = []
        failed_count = 0
        for img in images:
            try:
                path = process_image_input(img)
                processed.append(path)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to process image: {e}")
        if images and not processed:
            raise ValueError(
                f"All {failed_count} image(s) failed to process. "
                f"Check image URLs/paths and try again."
            )
        return processed

    def _prepare_audio(self, audio_inputs: list) -> list[str]:
        """Process audio inputs and return local paths/URLs loadable by mlx-vlm."""
        processed = []
        failed_count = 0
        for audio in audio_inputs:
            try:
                path = process_audio_input(audio)
                if path:
                    processed.append(path)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to process audio: {e}")
        if audio_inputs and not processed:
            raise ValueError(
                f"All {failed_count} audio input(s) failed to process. "
                f"Check audio URLs/paths and try again."
            )
        return processed

    def _prepare_video(
        self,
        video_input: str | dict,
        fps: float = DEFAULT_FPS,
        max_frames: int = MAX_FRAMES,
    ) -> list[str]:
        """
        Process video input and extract frames.

        Supports:
        - Local file paths
        - URLs (http/https) - will be downloaded
        - Base64 encoded videos (data:video/mp4;base64,...)
        - OpenAI format dicts: {"url": "..."} or {"video_url": {"url": "..."}}

        Args:
            video_input: Video in any supported format
            fps: Frames per second to extract
            max_frames: Maximum frames to extract

        Returns:
            List of paths to extracted frame images
        """
        # Process video input (download if URL, decode if base64)
        video_path = process_video_input(video_input)

        # Extract frames
        frames = extract_video_frames_smart(
            video_path,
            fps=fps,
            max_frames=max_frames,
        )
        return save_frames_to_temp(frames)

    def _guard_simple_image_prefill(
        self,
        formatted_prompt: str,
        has_images: bool,
        *,
        images: list | None = None,
        resize_shape: object | None = None,
    ) -> None:
        """Apply the vmlx#156 image-prefill guard on SimpleEngine paths.

        Continuous-batching VLMs run this guard inside
        ``MLLMBatchGenerator._run_vision_encoding_inner``. Direct
        ``MLXMultimodalLM`` generate/chat paths still call ``mlx_vlm``
        one-shot APIs, so they need the same preflight before Metal sees an
        image-expanded prompt. Use processor-expanded ``input_ids`` when
        possible because each VLM family can expand one image/video placeholder
        into a different number of language tokens.
        """
        if not has_images:
            return
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            mx = None  # type: ignore[assignment]

        try:
            seq_len = self._simple_vlm_prefill_seq_len(
                formatted_prompt,
                images=images,
                resize_shape=resize_shape,
            )
        except Exception as exc:
            logger.debug("Simple VLM image-prefill guard skipped: tokenize failed: %s", exc)
            return

        try:
            if mx is not None:
                mx.clear_cache()
        except Exception:
            pass

        from vmlx_engine.mllm_batch_generator import (
            _raise_if_image_prefill_exceeds_budget,
        )

        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            inner = getattr(self.model, "model", None)
            language_model = getattr(inner, "language_model", None)
        _raise_if_image_prefill_exceeds_budget(
            has_images=True,
            seq_len=seq_len,
            language_model=language_model,
        )

    def _simple_vlm_prefill_seq_len(
        self,
        formatted_prompt: str,
        *,
        images: list | None = None,
        resize_shape: object | None = None,
    ) -> int:
        """Return the best available VLM prefill length for direct paths."""

        def _seq_len_from_input_ids(input_ids: object) -> int | None:
            if input_ids is None:
                return None
            shape = getattr(input_ids, "shape", None)
            if shape:
                if len(shape) >= 2:
                    return int(shape[1])
                return int(shape[0])
            if isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    return int(len(input_ids[0]))
                return int(len(input_ids))
            return None

        if images:
            try:
                from mlx_vlm.generate import normalize_resize_shape, prepare_inputs

                model_config = getattr(self.model, "config", None)
                image_token_index = (
                    getattr(model_config, "image_token_index", None)
                    if model_config is not None else None
                )
                if image_token_index is None and model_config is not None:
                    image_token_index = getattr(model_config, "image_token_id", None)
                model_type = str(getattr(model_config, "model_type", "") or "")
                add_special_tokens = (
                    getattr(self.processor, "chat_template", None) is None
                    if model_type in {"gemma3", "gemma3n", "gemma4"}
                    else True
                )
                inputs = prepare_inputs(
                    self.processor,
                    images=images,
                    prompts=formatted_prompt,
                    image_token_index=image_token_index,
                    resize_shape=normalize_resize_shape(resize_shape),
                    add_special_tokens=add_special_tokens,
                )
                seq_len = _seq_len_from_input_ids(inputs.get("input_ids"))
                if seq_len:
                    return seq_len
            except Exception as exc:
                logger.debug(
                    "Simple VLM image-prefill guard using raw tokenizer length; "
                    "processor expansion probe failed: %s",
                    exc,
                )

        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        return len(tokenizer.encode(formatted_prompt))

    @staticmethod
    def _extract_multimodal_messages(
        messages: list[dict],
    ) -> tuple[list[dict], list[str], list, list]:
        """
        Parse OpenAI-format messages into chat_messages, image URLs, videos, and audio.

        Extracts text, images, and videos from multimodal message content,
        building properly structured chat messages for Qwen3-VL-MoE and similar
        models that expect image tokens before text in user messages.

        Args:
            messages: List of chat messages in OpenAI format

        Returns:
            Tuple of (chat_messages, all_image_urls, videos, audio_inputs) where:
            - chat_messages: Properly structured messages for chat template
            - all_image_urls: Raw image URLs/paths to process
            - videos: Raw video inputs to process
            - audio_inputs: Raw audio inputs to process
        """
        all_image_urls: list[str] = []
        videos: list = []
        audio_inputs: list = []
        chat_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_text = ""
            msg_image_count = 0
            msg_video_count = 0
            msg_audio_count = 0

            if isinstance(content, str):
                msg_text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        msg_text += item
                        continue

                    # Convert Pydantic models to dicts.
                    # exclude_none=True prevents Jinja2 false key-existence bugs
                    # (checking 'if image_url' returns True for None values). If a
                    # field should be explicitly None (not absent), it would need
                    # special handling.
                    if hasattr(item, "model_dump"):
                        item = item.model_dump(exclude_none=True)
                    elif hasattr(item, "dict"):
                        item = item.dict()
                        # Remove None values to match exclude_none behavior
                        item = {k: v for k, v in item.items() if v is not None}

                    if isinstance(item, dict):
                        item_type = item.get("type", "")

                        if item_type in ("text", "input_text"):
                            msg_text += item.get("text", "")

                        elif item_type == "image_url":
                            img_url = item.get("image_url", {})
                            if isinstance(img_url, str):
                                all_image_urls.append(img_url)
                            else:
                                all_image_urls.append(img_url.get("url", ""))
                            msg_image_count += 1

                        elif item_type == "input_image":
                            img_url = item.get("image_url", item.get("url", ""))
                            if isinstance(img_url, str):
                                all_image_urls.append(img_url)
                            elif isinstance(img_url, dict):
                                all_image_urls.append(img_url.get("url", ""))
                            msg_image_count += 1

                        elif item_type == "image":
                            all_image_urls.append(
                                item.get("image", item.get("url", ""))
                            )
                            msg_image_count += 1

                        elif item_type == "video":
                            videos.append(item.get("video", item.get("url", "")))
                            msg_video_count += 1

                        elif item_type == "video_url":
                            vid_url = item.get("video_url", {})
                            if isinstance(vid_url, str):
                                videos.append(vid_url)
                            else:
                                videos.append(vid_url.get("url", ""))
                            msg_video_count += 1

                        elif item_type == "input_video":
                            vid_url = item.get("video_url", item.get("url", item.get("file_id", "")))
                            if isinstance(vid_url, str):
                                videos.append(vid_url)
                            else:
                                videos.append(vid_url.get("url", ""))
                            msg_video_count += 1

                        elif item_type in ("input_audio", "audio", "audio_url"):
                            src = item.get("input_audio") or item.get("audio") or item.get("audio_url") or item
                            audio_inputs.append(src)
                            msg_audio_count += 1

            # Build properly structured message for Qwen3-VL-MoE
            # Format: {"role": "...", "content": [{"type": "image"|"video"}, ..., {"type": "text", "text": "..."}]}
            # Preserve tool_calls on assistant messages and tool role fields
            tool_calls = _normalize_tool_calls_for_template(msg.get("tool_calls"))
            tool_call_id = msg.get("tool_call_id")
            msg_name = msg.get("name")

            if msg_text or msg_image_count > 0 or msg_video_count > 0 or msg_audio_count > 0 or tool_calls or role == "tool":
                if (msg_image_count > 0 or msg_video_count > 0 or msg_audio_count > 0) and role in ("user", "assistant"):
                    # Build multimodal content list with image markers for
                    # any role that carries media (user or assistant).
                    # Some UIs (e.g. klite) send images on assistant messages
                    # when attaching an image to the conversation context.
                    content_list: list[dict] = []
                    if msg_audio_count and not msg_image_count and not msg_video_count:
                        content_list.append(
                            {"type": "text", "text": msg_text, "content": msg_text}
                        )
                        for _ in range(msg_audio_count):
                            content_list.append({"type": "audio"})
                    else:
                        for _ in range(msg_image_count):
                            content_list.append({"type": "image"})
                        for _ in range(msg_video_count):
                            content_list.append({"type": "video"})
                        for _ in range(msg_audio_count):
                            content_list.append({"type": "audio"})
                        content_list.append(
                            {"type": "text", "text": msg_text, "content": msg_text}
                        )
                    out_msg: dict = {"role": role, "content": content_list}
                    if role == "assistant" and tool_calls:
                        out_msg["tool_calls"] = tool_calls
                    chat_messages.append(out_msg)
                elif role == "assistant":
                    out_msg = {"role": role, "content": msg_text}
                    if tool_calls:
                        out_msg["tool_calls"] = tool_calls
                    chat_messages.append(out_msg)
                elif role == "tool":
                    out_msg = {"role": role, "content": msg_text}
                    if tool_call_id:
                        out_msg["tool_call_id"] = tool_call_id
                    if msg_name:
                        out_msg["name"] = msg_name
                    chat_messages.append(out_msg)
                else:
                    chat_messages.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "text", "text": msg_text, "content": msg_text}
                            ],
                        }
                    )

        return chat_messages, all_image_urls, videos, audio_inputs

    def _normalize_text_only_messages_for_processor(
        self,
        chat_messages: list[dict],
        *,
        has_media: bool,
    ) -> list[dict]:
        """Use family-native plain text turns when a VLM request has no media.

        Some local processor templates have two distinct modes: rich list
        content is required for real image/video turns, while text-only turns
        render correctly as plain strings. Feeding a no-media request through
        the generic rich-content MLLM shape can make the template path look
        like a multimodal prompt and weakens exact text, multi-turn, and
        reasoning probes. Keep media turns untouched; only collapse text-only
        content lists that contain text fragments and no modality markers.
        """

        if has_media:
            return chat_messages

        config = getattr(self, "config", None)
        if isinstance(config, dict):
            model_type = str(config.get("model_type", "") or "").lower()
        else:
            model_type = str(getattr(config, "model_type", "") or "").lower()
        if model_type not in {"zaya1_vl", "mimo_v2"}:
            return chat_messages

        normalized: list[dict] = []
        for message in chat_messages:
            content = message.get("content")
            if not isinstance(content, list):
                normalized.append(message)
                continue

            text_parts: list[str] = []
            contains_media_marker = False
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if not isinstance(item, dict):
                    text_parts.append(str(item))
                    continue
                item_type = str(item.get("type", "") or "").lower()
                if item_type and item_type != "text":
                    contains_media_marker = True
                    break
                text_parts.append(str(item.get("text") or item.get("content") or ""))

            if contains_media_marker:
                normalized.append(message)
                continue

            collapsed = dict(message)
            collapsed["content"] = "".join(text_parts)
            normalized.append(collapsed)

        return normalized

    def _synthesizes_thinking_prompt_when_enabled(self) -> bool:
        """Return True for MLLM families whose template needs an explicit open rail.

        Some MLLM bundles declare qwen3 reasoning support while their mlx-vlm
        chat template can be a plain ``assistant: `` suffix. In that case an
        explicit per-chat/API ``enable_thinking=true`` may need to open the
        qwen3 rail in the prompt. This is intentionally family-gated; most VLM
        templates either own their think rail or do not support reasoning. The
        current ZAYA1-VL plain-template artifacts are registry-demoted to
        ``supports_thinking=False`` because live proof produced hidden-only
        output under the synthetic rail.
        """

        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            config = {}

        caps = config.get("capabilities")
        caps = caps if isinstance(caps, dict) else {}
        if caps.get("supports_thinking") is False:
            return False

        model_type = config.get("model_type")
        text_config = config.get("text_config")
        if not model_type and isinstance(text_config, dict):
            model_type = text_config.get("model_type")

        family_names = {
            str(v).lower()
            for v in (caps.get("family"), model_type)
            if isinstance(v, str) and v
        }
        reasoning_parser = caps.get("reasoning_parser")
        supports_thinking = caps.get("supports_thinking")
        think_in_template = caps.get("think_in_template")

        try:
            from vmlx_engine.model_config_registry import get_model_config_registry

            registry = get_model_config_registry()
            family_config = registry.lookup(str(getattr(self, "model_name", "") or ""))
            if getattr(family_config, "family_name", None):
                family_names.add(str(family_config.family_name).lower())
            if getattr(family_config, "reasoning_parser", None):
                reasoning_parser = family_config.reasoning_parser
            if getattr(family_config, "supports_thinking", None) is not None:
                supports_thinking = family_config.supports_thinking
            if getattr(family_config, "think_in_template", None) is not None:
                think_in_template = family_config.think_in_template
        except Exception:
            pass

        if supports_thinking is False:
            return False
        if think_in_template is True:
            return False
        if str(reasoning_parser or "").lower() != "qwen3":
            return False
        return bool(family_names & {"zaya", "zaya1_vl", "zaya1-vl"})

    def _prompt_template_supports_thinking(self) -> bool:
        """Return whether the native VLM prompt template owns the think rail.

        Capability stamps are the source of truth here. A model can support a
        qwen-style reasoning parser while its VLM chat template remains plain;
        those families must not receive template-sentinel assumptions when
        ``enable_thinking=false``.
        """

        config = getattr(self, "config", None)
        caps = config.get("capabilities") if isinstance(config, dict) else None
        caps = caps if isinstance(caps, dict) else {}
        if caps.get("supports_thinking") is False:
            return False
        if caps.get("think_in_template") is not None:
            return bool(caps.get("think_in_template"))

        try:
            from vmlx_engine.model_config_registry import get_model_config_registry

            family_config = get_model_config_registry().lookup(
                str(getattr(self, "model_name", "") or "")
            )
            if getattr(family_config, "supports_thinking", None) is False:
                return False
            if getattr(family_config, "think_in_template", None) is not None:
                return bool(family_config.think_in_template)
        except Exception:
            pass

        return False

    def _apply_chat_template(
        self,
        chat_messages: list[dict],
        enable_thinking: bool | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """
        Apply chat template to structured messages with enable_thinking support.

        Handles TypeError fallback for processors that don't support
        enable_thinking and general exception fallback to last user message.
        The wrapper preserves the native template output; family-specific
        no-thinking prompt sentinels must live in explicit renderer contracts.

        Args:
            chat_messages: Structured chat messages from _extract_multimodal_messages()
            enable_thinking: Whether to enable thinking mode (None = don't pass)

        Returns:
            Formatted prompt string
        """
        from mlx_vlm.prompt_utils import get_chat_template

        template_kwargs = {}
        model_type = ""
        try:
            config = getattr(self, "config", None)
            if isinstance(config, dict):
                model_type = str(config.get("model_type", "") or "").lower()
            else:
                model_type = str(getattr(config, "model_type", "") or "").lower()
        except Exception:
            model_type = ""
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if tools:
            template_kwargs["tools"] = tools

        formatted_prompt = None
        try:
            formatted_prompt = get_chat_template(
                self.processor,
                chat_messages,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError:
            # Processor doesn't support enable_thinking kwarg — use processor
            # WITHOUT the kwarg first. Preserve tools unless the processor
            # rejects them too.
            fallback_kwargs = dict(template_kwargs)
            fallback_kwargs.pop("enable_thinking", None)
            try:
                formatted_prompt = get_chat_template(
                    self.processor,
                    chat_messages,
                    add_generation_prompt=True,
                    **fallback_kwargs,
                )
            except TypeError:
                fallback_kwargs.pop("tools", None)
                try:
                    formatted_prompt = get_chat_template(
                        self.processor,
                        chat_messages,
                        add_generation_prompt=True,
                        **fallback_kwargs,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply chat template: {e}, using last user message"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}, using last user message"
                )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, using last user message"
            )

        # Some processor templates can be plain while the model is still
        # qwen3-reasoning capable. Honor explicit thinking-on by opening the
        # rail in the assistant generation prompt only for registry-approved
        # families; Auto/Off remain plain.
        if (
            enable_thinking is True
            and formatted_prompt
            and self._synthesizes_thinking_prompt_when_enabled()
        ):
            last_think = formatted_prompt.rfind("<think>")
            has_open_unclosed = last_think >= 0 and "</think>" not in formatted_prompt[last_think:]
            if not has_open_unclosed:
                stripped = formatted_prompt.rstrip()
                if stripped.endswith("assistant:"):
                    formatted_prompt = stripped + " <think>\n"
                else:
                    formatted_prompt = stripped + "\n<think>\n"

        if enable_thinking is False and formatted_prompt:
            try:
                from ..utils.chat_template_kwargs import ensure_thinking_off_sentinel

                formatted_prompt = ensure_thinking_off_sentinel(
                    formatted_prompt,
                    family_name=str((self.config or {}).get("model_type", ""))
                    if isinstance(self.config, dict)
                    else None,
                    model_name=self.model_name,
                    tools_present=bool(tools),
                )
            except Exception as exc:
                logger.debug(
                    "MLLM thinking-off prompt sentinel skipped for %s: %s",
                    self.model_name,
                    exc,
                )

        if formatted_prompt is None:
            # Fallback to last user message if template fails
            last_user_msg = ""
            for m in reversed(chat_messages):
                if m["role"] == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                last_user_msg = item.get("text", "")
                                break
                    else:
                        last_user_msg = content
                    break
            formatted_prompt = last_user_msg

        return formatted_prompt

    def generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list | None = None,
        audio: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        use_cache: bool = True,
        **kwargs,
    ) -> MLLMOutput:
        """
        Generate text from multimodal input.

        Args:
            prompt: Text prompt/question
            images: List of image paths, URLs, or base64 strings
            videos: List of video inputs (paths, URLs, base64, or OpenAI format dicts)
            audio: List of audio file paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            video_fps: FPS for video frame extraction (default: 2.0)
            video_max_frames: Max frames to extract from video
            use_cache: Whether to use KV cache (default: True)
            **kwargs: Additional generation parameters

        Returns:
            MLLMOutput with generated text

        Example:
            # With local video
            output = model.generate("Describe this video", videos=["video.mp4"])

            # With video URL
            output = model.generate("What happens?", videos=["https://example.com/video.mp4"])

            # With base64 video
            output = model.generate("Describe", videos=["data:video/mp4;base64,AAAA..."])
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate
        from mlx_vlm.models import cache as vlm_cache
        from mlx_vlm.prompt_utils import apply_chat_template

        images = images or []
        videos = videos or []
        audio = audio or []

        # Process all images (including frames from videos) and audio.
        all_images = []
        all_audio = []
        all_sources = []  # Track original sources for cache key

        # Process image inputs
        if images:
            all_images.extend(self._prepare_images(images))
            all_sources.extend(images)
        if audio:
            all_audio.extend(self._prepare_audio(audio))
            all_sources.extend(audio)

        # Extract frames from videos
        for video_path in videos:
            frames = self._prepare_video(
                video_path,
                fps=video_fps,
                max_frames=video_max_frames,
            )
            all_images.extend(frames)
            # Include video params in cache key
            video_str = video_path if isinstance(video_path, str) else str(video_path)
            all_sources.append(
                f"video:{video_str}:fps{video_fps}:max{video_max_frames}"
            )
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template if needed
        if all_images and hasattr(self.processor, "apply_chat_template"):
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        # Check cache for existing KV state
        prompt_cache = None
        prompt_cache_state = None
        cache_hit = False
        _cache_token_ids: list[int] | None = None  # reused by store path below

        if use_cache and self._cache_manager is not None and all_sources:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = tokenizer.encode(formatted_prompt)
                _cache_token_ids = token_ids
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_sources, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    if all_images:
                        prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                        if prompt_cache_state is None:
                            logger.debug(
                                "[PREFIX CACHE] Generate image hit lacked PromptCacheState; "
                                "falling back to full prompt processing"
                            )
                        else:
                            logger.debug(
                                "[PREFIX CACHE] Generate image hit - using mlx-vlm "
                                "PromptCacheState for prefix trim"
                            )
                    else:
                        prompt_cache = cache_entry.kv_cache
                    cache_hit = True
                    logger.info(
                        f"MLLM cache hit for {len(all_sources)} source(s)"
                        + (f", {prefix_match_len} prefix tokens match" if prefix_match_len > 0 else "")
                    )
            except Exception as e:
                logger.debug(f"Generate cache fetch failed: {e}")

        # Create new cache if needed
        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        # Generate with cache
        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
            prompt_cache=prompt_cache,
            **kwargs,
        )

        # Store cache for future reuse (only on miss).
        # Use `is not None` — MLLMPrefixCacheManager.__len__ exists but
        # __bool__ does not, so plain `if self._cache_manager` evaluates
        # False when the cache is empty (via len fallback). That made
        # store skip the very first attempt, so the cache could never
        # populate and no request ever hit. (2026-04-18 Ralph iter 11)
        if use_cache and self._cache_manager is not None and all_sources and not cache_hit:
            if prompt_cache is not None:
                try:
                    num_tokens = getattr(result, "prompt_tokens", 0)
                    cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                        prompt_cache, num_tokens
                    )
                    # Prefer the real token_ids (computed above during fetch)
                    # so partial-hit prefix matching works. store_cache's
                    # num_tokens-only path writes dummy `[0] * N` which makes
                    # downstream get_prefix_match_length always return 0.
                    if _cache_token_ids:
                        self._cache_manager.store(
                            images=all_sources,
                            prompt=formatted_prompt,
                            vision_embeddings=None,
                            kv_cache=cache_to_store,
                            token_ids=_cache_token_ids,
                            num_image_tokens=0,
                            model_name=self.model_name,
                        )
                    else:
                        self._cache_manager.store_cache(
                            all_sources, formatted_prompt, cache_to_store, num_tokens
                        )
                    logger.info(f"MLLM cache stored for {len(all_sources)} source(s)")
                except Exception as e:
                    logger.debug(f"Failed to store MLLM cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        finish_reason = "length" if generation_tokens >= max_tokens else "stop"

        return MLLMOutput(
            text=output_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
        )

    def stream_generate(
        self,
        prompt: str,
        images: list | None = None,
        videos: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream text generation for multimodal input.

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video paths
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            video_fps: FPS for video frame extraction
            video_max_frames: Maximum video frames to extract
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
            from mlx_vlm.models import cache as vlm_cache
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError:
            # Fallback to non-streaming
            output = self.generate(
                prompt=prompt,
                images=images,
                videos=videos,
                max_tokens=max_tokens,
                temperature=temperature,
                video_fps=video_fps,
                **kwargs,
            )
            yield output.text
            return

        images = images or []
        videos = videos or []

        # Process images
        all_images = []
        all_sources = []
        if images:
            all_images.extend(self._prepare_images(images))
            all_sources.extend(images)
        for video_path in videos:
            frames = self._prepare_video(video_path, fps=video_fps, max_frames=video_max_frames)
            all_images.extend(frames)
            video_str = video_path if isinstance(video_path, str) else str(video_path)
            all_sources.append(
                f"video:{video_str}:fps{video_fps}:max{video_max_frames}"
            )

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        if all_images:
            try:
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    prompt,
                    num_images=len(all_images),
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        prompt_cache = None
        prompt_cache_state = None
        cache_hit = False
        token_ids: list[int] = []
        use_cache = kwargs.pop("use_cache", True)

        if use_cache and self._cache_manager is not None and all_sources:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = tokenizer.encode(formatted_prompt)
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_sources, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    if all_images:
                        prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                        if prompt_cache_state is None:
                            logger.debug(
                                "[PREFIX CACHE] Stream generate image hit lacked "
                                "PromptCacheState; falling back to full prompt processing"
                            )
                    else:
                        prompt_cache = cache_entry.kv_cache
                    cache_hit = True
                    if prefix_match_len > 0:
                        logger.debug(
                            f"Stream generate prefix cache hit: {prefix_match_len} tokens, "
                            f"{len(all_sources)} source(s)"
                        )
            except Exception as e:
                logger.debug(f"Stream generate cache fetch failed: {e}")

        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        last_prompt_tokens = 0
        with _MaybeVLMStream():
            for chunk in stream_generate(
                self.model,
                self.processor,
                formatted_prompt,
                all_images if all_images else None,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_cache=prompt_cache,
                **kwargs,
            ):
                chunk_prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                if chunk_prompt_tokens:
                    last_prompt_tokens = chunk_prompt_tokens
                yield chunk

        if (
            use_cache
            and self._cache_manager is not None
            and all_sources
            and not cache_hit
            and prompt_cache
        ):
            try:
                prompt_tokens_count = last_prompt_tokens or len(token_ids)
                cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                    prompt_cache, prompt_tokens_count
                )
                self._cache_manager.store(
                    images=all_sources,
                    prompt=formatted_prompt,
                    vision_embeddings=None,
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=0,
                    model_name=self.model_name,
                )
                logger.info(
                    f"MLLM stream cache stored for {len(all_sources)} source(s)"
                )
            except Exception as e:
                logger.debug(f"Failed to store MLLM stream cache: {e}")

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> MLLMOutput:
        """
        Chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            MLLMOutput with assistant's response
        """
        if not self._loaded:
            self.load()

        from mlx_vlm import generate

        # Extract text, images, videos, and audio from messages
        chat_messages, all_image_urls, videos, audio_inputs = self._extract_multimodal_messages(messages)
        chat_messages = self._normalize_text_only_messages_for_processor(
            chat_messages,
            has_media=bool(all_image_urls or videos or audio_inputs),
        )

        logger.info(f"MLLM.chat() called with {len(messages)} messages")

        # Process images
        all_images = []
        all_audio = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))
        if audio_inputs:
            all_audio.extend(self._prepare_audio(audio_inputs))

        # Process videos
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        video_frame_counts: list[int] = []
        for video_path in videos:
            frames = self._prepare_video(
                video_path, fps=video_fps, max_frames=video_max_frames
            )
            all_images.extend(frames)
            video_frame_counts.append(len(frames))
            logger.info(f"Added {len(frames)} frames from video: {video_path}")

        model_type = (
            str(self.config.get("model_type", "") or "").lower()
            if isinstance(self.config, dict)
            else str(getattr(self.config, "model_type", "") or "").lower()
        )
        if model_type in {"gemma4_unified", "step3p7"} and video_frame_counts:
            chat_messages = _expand_video_placeholders_to_image_frames(
                chat_messages,
                video_frame_counts,
            )

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        template_tools = kwargs.pop("tools", None)
        logger.info(
            f"Applying chat template with {len(chat_messages)} messages, "
            f"{len(all_images)} images, {len(all_audio)} audio"
        )
        for i, cm in enumerate(chat_messages):
            content_preview = str(cm.get("content", ""))[:80]
            logger.info(
                f"  Chat msg {i}: role={cm['role']}, content={content_preview}..."
            )
        enable_thinking = kwargs.pop("enable_thinking", None)
        formatted_prompt = self._apply_chat_template(
            chat_messages,
            enable_thinking,
            tools=template_tools,
        )

        # Post-template image count guard: VLM chat templates may not expand
        # image placeholders for assistant-role messages. Trim all_images to
        # match actual placeholder token count to prevent crashes in generate().
        if all_images and self.config:
            _img_token_id = getattr(self.config, "image_token_index", None)
            if _img_token_id is not None:
                _tok = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                try:
                    _token_ids = _tok.encode(formatted_prompt)
                    _slot_count = _token_ids.count(_img_token_id)
                    if _slot_count != len(all_images):
                        logger.warning(
                            f"Image count mismatch: {len(all_images)} images but "
                            f"{_slot_count} slots in prompt (likely assistant-role "
                            f"images not supported by template). Trimming to {_slot_count}."
                        )
                        all_images = all_images[:_slot_count]
                except Exception as _e:
                    logger.debug(f"Image slot count check failed: {_e}")

        # Prefix caching with vision embedding support
        # Following LMCache approach: cache vision embeddings to skip encoder on hit
        from mlx_vlm.models import cache as vlm_cache
        import time

        use_cache = kwargs.pop("use_cache", True)
        cache_entry = None
        prefix_match_len = 0
        cache_hit = False

        # Tokenize prompt for cache lookup
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        token_ids = tokenizer.encode(formatted_prompt)

        # Check prefix cache
        if use_cache and self._cache_manager is not None and all_images:
            try:
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry:
                    cache_hit = True
                    # NOTE: cache_entry.vision_embeddings is not used — mlx-vlm's
                    # generate() does not accept pre-computed vision embeddings.
                    # The KV cache hit path below handles the actual speedup.
                    if prefix_match_len > 0:
                        logger.debug(
                            f"[PREFIX CACHE] {prefix_match_len} prefix tokens match"
                        )
            except Exception as e:
                logger.warning(f"Cache fetch failed: {e}")

        # Generate - use KV cache if available from previous identical request
        start_time = time.time()

        # Create or reuse prompt cache for prefix caching speedup
        prompt_cache = None
        prompt_cache_state = None
        skip_prompt_processing = False

        if cache_hit and cache_entry and cache_entry.kv_cache:
            if all_images:
                # mlx-vlm trims multimodal prefixes via PromptCacheState. Passing
                # raw prompt_cache with the full prompt would double-process the
                # image/prefix tokens; dropping the hit would full-prefill every
                # follow-up turn.
                prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                if prompt_cache_state is None:
                    logger.debug(
                        "[PREFIX CACHE] Image prefix hit lacked PromptCacheState; "
                        "falling back to full prompt processing"
                    )
                else:
                    logger.debug(
                        "[PREFIX CACHE] Image prefix hit - using mlx-vlm "
                        "PromptCacheState for prefix trim"
                    )
                skip_prompt_processing = False
            else:
                # Text-only: can use skip_prompt_processing for maximum speedup
                logger.debug(
                    "[PREFIX CACHE] Text-only cache hit - using skip_prompt_processing speedup"
                )
                cached_prompt_cache = cache_entry.kv_cache
                try:
                    import copy

                    prompt_cache = []
                    for layer_cache in cached_prompt_cache:
                        new_cache = copy.copy(layer_cache)
                        if hasattr(layer_cache, "state"):
                            state = layer_cache.state
                            if state is not None:
                                import mlx.core as mx

                                if len(state) >= 2 and state[0] is not None:
                                    new_cache.keys = mx.array(state[0])
                                    new_cache.values = mx.array(state[1])
                                    if len(state) >= 3:
                                        new_cache.offset = state[2]
                                    elif hasattr(layer_cache, "offset"):
                                        new_cache.offset = layer_cache.offset
                        prompt_cache.append(new_cache)
                    skip_prompt_processing = True
                    logger.debug(
                        f"[PREFIX CACHE] Skipping {prefix_match_len} token forward pass"
                    )
                except Exception as e:
                    logger.warning(f"[PREFIX CACHE] Failed to copy cache: {e}")
                    prompt_cache = None
                    skip_prompt_processing = False

        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            # Create fresh cache
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Extract sampling params not natively supported by mlx_vlm.generate()
        # (top_k, min_p require a custom sampler; they're silently ignored otherwise)
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            from ..sampling import make_sampler
            top_p = kwargs.pop("top_p", 1.0)
            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        result = generate(
            self.model,
            self.processor,
            formatted_prompt,
            all_images if all_images else None,
            audio=all_audio if all_audio else None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
            prompt_cache=prompt_cache,
            skip_prompt_processing=skip_prompt_processing,
            **kwargs,
        )

        # Store KV cache for future reuse (on cache miss)
        # IMPORTANT: We need to store only the prompt portion, not generated tokens
        if (
            use_cache
            and self._cache_manager is not None
            and all_images
            and not cache_hit
            and prompt_cache
        ):
            try:
                # Get prompt token count (before generation)
                prompt_tokens_count = getattr(result, "prompt_tokens", 0)

                cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                    prompt_cache, prompt_tokens_count
                )

                # Estimate num_image_tokens from the model config or token IDs.
                # Different models use different counts (e.g., Gemma3=256, Qwen2-VL varies).
                num_img_tokens = 0
                if all_images and self.config:
                    # Try to get image_token_index from config and count occurrences in token_ids
                    img_token_id = getattr(self.config, "image_token_index", None)
                    if img_token_id is not None and token_ids:
                        num_img_tokens = token_ids.count(img_token_id)
                    if num_img_tokens == 0:
                        # Fallback: use prompt_tokens_count minus a rough text estimate
                        # or just leave as 0 (cache stats only, not critical for correctness)
                        num_img_tokens = 0

                self._cache_manager.store(
                    images=all_images,
                    prompt=formatted_prompt,
                    vision_embeddings=None,  # mlx-vlm doesn't support pre-computed embeddings
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=num_img_tokens,
                    model_name=self.model_name,
                )
                logger.info(
                    f"[PREFIX CACHE] Stored KV cache for {len(all_images)} image(s) ({prompt_tokens_count} prompt tokens)"
                )
            except Exception as e:
                logger.warning(f"Failed to cache: {e}")

        # Handle GenerationResult object or plain string
        if hasattr(result, "text"):
            output_text = result.text
            prompt_tokens = getattr(result, "prompt_tokens", 0)
            generation_tokens = getattr(result, "generation_tokens", 0)
        else:
            output_text = str(result)
            prompt_tokens = 0
            generation_tokens = 0

        finish_reason = "length" if generation_tokens >= max_tokens else "stop"

        return MLLMOutput(
            text=output_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=generation_tokens,
        )

    def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[MLLMOutput]:
        """
        Stream chat with OpenAI-compatible message format.

        Supports multimodal content in messages:
        - {"type": "text", "text": "..."}
        - {"type": "image_url", "image_url": {"url": "..."}}
        - {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}

        Args:
            messages: List of chat messages (OpenAI format)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            MLLMOutput with incremental text chunks
        """
        if not self._loaded:
            self.load()

        try:
            from mlx_vlm import stream_generate
        except ImportError:
            # Fallback to non-streaming if stream_generate not available
            output = self.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            yield output
            return

        # Extract text, images, videos, and audio from messages
        chat_messages, all_image_urls, videos, audio_inputs = self._extract_multimodal_messages(messages)
        chat_messages = self._normalize_text_only_messages_for_processor(
            chat_messages,
            has_media=bool(all_image_urls or videos or audio_inputs),
        )

        # Process images
        all_images = []
        all_audio = []
        if all_image_urls:
            all_images.extend(self._prepare_images(all_image_urls))
        if audio_inputs:
            all_audio.extend(self._prepare_audio(audio_inputs))

        # Process videos
        video_fps = kwargs.pop("video_fps", DEFAULT_FPS)
        video_max_frames = kwargs.pop("video_max_frames", MAX_FRAMES)
        video_frame_counts: list[int] = []
        for video_path in videos:
            frames = self._prepare_video(
                video_path, fps=video_fps, max_frames=video_max_frames
            )
            all_images.extend(frames)
            video_frame_counts.append(len(frames))

        model_type = (
            str(self.config.get("model_type", "") or "").lower()
            if isinstance(self.config, dict)
            else str(getattr(self.config, "model_type", "") or "").lower()
        )
        if model_type == "gemma4_unified" and video_frame_counts:
            chat_messages = _expand_video_placeholders_to_image_frames(
                chat_messages,
                video_frame_counts,
            )

        # Guard against excessive total images (including video frames)
        _max_images = kwargs.get("max_images_per_request", 20)
        if len(all_images) > _max_images:
            raise ValueError(
                f"Total image count ({len(all_images)}, including video frames) "
                f"exceeds limit of {_max_images}. Reduce images/videos or increase "
                f"max_images_per_request."
            )

        # Apply chat template
        template_tools = kwargs.pop("tools", None)
        enable_thinking = kwargs.pop("enable_thinking", None)
        formatted_prompt = self._apply_chat_template(
            chat_messages,
            enable_thinking,
            tools=template_tools,
        )

        # Post-template image count guard: VLM chat templates may not expand
        # image placeholders for assistant-role messages. Trim all_images to
        # match actual placeholder token count to prevent crashes in generate().
        if all_images and self.config:
            _img_token_id = getattr(self.config, "image_token_index", None)
            if _img_token_id is not None:
                _tok = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                try:
                    _token_ids = _tok.encode(formatted_prompt)
                    _slot_count = _token_ids.count(_img_token_id)
                    if _slot_count != len(all_images):
                        logger.warning(
                            f"Image count mismatch: {len(all_images)} images but "
                            f"{_slot_count} slots in prompt (likely assistant-role "
                            f"images not supported by template). Trimming to {_slot_count}."
                        )
                        all_images = all_images[:_slot_count]
                except Exception as _e:
                    logger.debug(f"Image slot count check failed: {_e}")

        # Check cache for existing KV state (uses images as cache key)
        from mlx_vlm.models import cache as vlm_cache

        prompt_cache = None
        prompt_cache_state = None
        cache_hit = False
        use_cache = kwargs.pop("use_cache", True)
        token_ids: list[int] = []

        if use_cache and self._cache_manager is not None and all_images:
            try:
                tokenizer = (
                    self.processor.tokenizer
                    if hasattr(self.processor, "tokenizer")
                    else self.processor
                )
                token_ids = tokenizer.encode(formatted_prompt)
                cache_entry, prefix_match_len = self._cache_manager.fetch(
                    all_images, formatted_prompt, token_ids
                )
                if cache_entry and cache_entry.kv_cache is not None:
                    prompt_cache_state = _prompt_cache_state_from_entry(cache_entry)
                    if prompt_cache_state is None:
                        logger.debug(
                            "Stream chat image prefix hit lacked PromptCacheState; "
                            "falling back to full prompt processing"
                        )
                    cache_hit = True
                    if prefix_match_len > 0:
                        logger.debug(
                            f"Stream chat prefix cache hit: {prefix_match_len} tokens, "
                            f"{len(all_images)} image(s)"
                        )
                    else:
                        logger.debug(f"Stream chat cache hit for {len(all_images)} image(s)")
            except Exception as e:
                logger.debug(f"Stream chat cache fetch failed: {e}")

        # Create new cache if needed
        if prompt_cache is None and prompt_cache_state is None and self.model is not None:
            try:
                prompt_cache = vlm_cache.make_prompt_cache(self.model.language_model)
            except Exception:
                prompt_cache = None

        # Stream generate tokens with cache
        accumulated_text = ""
        token_count = 0
        last_prompt_tokens = 0

        # Ensure processor uses NaiveStreamingDetokenizer for all MLLM models.
        # mlx-vlm's default streaming detokenizer can buffer infinitely for some
        # architectures. NaiveStreamingDetokenizer yields tokens immediately.
        if not hasattr(self.processor, "_patched_detok"):
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

            # Monkey-patch NaiveStreamingDetokenizer to accept kwargs from mlx_vlm
            if not hasattr(NaiveStreamingDetokenizer, "_vml_patched"):
                original_add = NaiveStreamingDetokenizer.add_token
                def _patched_add(self, token, **kwargs):
                    return original_add(self, token)
                NaiveStreamingDetokenizer.add_token = _patched_add
                def _patched_copy(self):
                    return type(self)(self._tokenizer)
                NaiveStreamingDetokenizer.__copy__ = _patched_copy
                NaiveStreamingDetokenizer._vml_patched = True

            tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
            self.processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
            self.processor._patched_detok = True

        # Extract sampling params not natively supported by mlx_vlm.stream_generate()
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        if (top_k and top_k > 0) or (min_p and min_p > 0.0):
            from ..sampling import make_sampler
            top_p = kwargs.pop("top_p", 1.0)
            kwargs["sampler"] = make_sampler(
                temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k
            )
        if prompt_cache_state is not None:
            kwargs["prompt_cache_state"] = prompt_cache_state

        self._guard_simple_image_prefill(
            formatted_prompt,
            bool(all_images and prompt_cache_state is None),
            images=all_images,
            resize_shape=kwargs.get("resize_shape"),
        )

        try:
            with _MaybeVLMStream():
                for chunk in stream_generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    all_images if all_images else None,
                    audio=all_audio if all_audio else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prompt_cache=prompt_cache,
                    **kwargs,
                ):
                    token_count += 1
                    # chunk is a GenerationResult with .text attribute containing the new token
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    accumulated_text += new_text

                    chunk_prompt_tokens = getattr(chunk, "prompt_tokens", 0)
                    if chunk_prompt_tokens:
                        last_prompt_tokens = chunk_prompt_tokens

                    yield MLLMOutput(
                        text=new_text,  # Just the new token for streaming
                        finish_reason=None,
                        prompt_tokens=last_prompt_tokens,
                        completion_tokens=token_count,
                    )
        except Exception as e:
            clear_mlx_memory_cache(log=logger)
            logger.error(f"VLM stream_generate error: {type(e).__name__}: {e}")
            raise

        if (
            use_cache
            and self._cache_manager is not None
            and all_images
            and not cache_hit
            and prompt_cache
        ):
            try:
                prompt_tokens_count = last_prompt_tokens or len(token_ids)
                cache_to_store = _copy_prompt_cache_to_prompt_boundary(
                    prompt_cache, prompt_tokens_count
                )
                num_img_tokens = 0
                if all_images and self.config:
                    img_token_id = getattr(self.config, "image_token_index", None)
                    if img_token_id is not None and token_ids:
                        num_img_tokens = token_ids.count(img_token_id)
                self._cache_manager.store(
                    images=all_images,
                    prompt=formatted_prompt,
                    vision_embeddings=None,
                    kv_cache=cache_to_store,
                    token_ids=token_ids,
                    num_image_tokens=num_img_tokens,
                    model_name=self.model_name,
                )
                logger.info(
                    f"[PREFIX CACHE] Stored streaming KV cache for {len(all_images)} "
                    f"image(s) ({prompt_tokens_count} prompt tokens)"
                )
            except Exception as e:
                logger.warning(f"Failed to cache streaming MLLM prompt: {e}")

        # Final yield with finish_reason
        finish_reason = "length" if token_count >= max_tokens else "stop"
        yield MLLMOutput(
            text="",
            finish_reason=finish_reason,
            prompt_tokens=last_prompt_tokens,
            completion_tokens=token_count,
        )

    def describe_image(
        self,
        image: str,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Convenience method to describe an image.

        Args:
            image: Image path, URL, or base64 string
            prompt: Description prompt
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Image description text
        """
        output = self.generate(
            prompt=prompt,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def answer_about_image(
        self,
        image: str,
        question: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: Image path, URL, or base64 string
            question: Question about the image
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Answer text
        """
        output = self.generate(
            prompt=question,
            images=[image],
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def describe_video(
        self,
        video: str | dict,
        prompt: str = "Describe what happens in this video.",
        fps: float = 2.0,
        max_frames: int = 32,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Describe a video using frame extraction.

        Args:
            video: Video file path, URL, base64, or OpenAI format dict
            prompt: Description prompt
            fps: Frames per second to extract
            max_frames: Maximum frames to extract
            max_tokens: Maximum tokens to generate

        Returns:
            Video description text

        Example:
            # Local file
            model.describe_video("video.mp4")

            # URL
            model.describe_video("https://example.com/video.mp4")

            # OpenAI format
            model.describe_video({"url": "https://example.com/video.mp4"})
        """
        output = self.generate(
            prompt=prompt,
            videos=[video],
            video_fps=fps,
            video_max_frames=max_frames,
            max_tokens=max_tokens,
            **kwargs,
        )
        return output.text

    def get_cache_stats(self) -> dict:
        """
        Get MLLM cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit_rate, tokens_saved, etc.)
        """
        if self._cache_manager is None:
            return {"enabled": False}

        stats = self._cache_manager.get_stats()
        stats["enabled"] = True
        stats["cache_entries"] = len(self._cache_manager)
        stats["max_entries"] = self._cache_manager.max_size
        return stats

    def clear_cache(self) -> None:
        """Clear the MLLM KV cache."""
        if self._cache_manager is not None:
            self._cache_manager.clear()
            logger.info("MLLM cache cleared")

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "type": "multimodal-language-model",
            "supports_video": True,
            "supports_streaming": True,
            "cache_enabled": self.enable_cache,
        }

        if self.config:
            info["model_type"] = getattr(self.config, "model_type", "unknown")

        if self._cache_manager is not None:
            info["cache_stats"] = self._cache_manager.get_stats()

        return info

    @staticmethod
    def list_supported_model_families() -> dict[str, str]:
        """
        List supported model families and their patterns.

        Any model on HuggingFace containing these patterns in the name
        is likely compatible with mlx-vlm.
        """
        return {
            "Qwen-VL": "Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.)",
            "LLaVA": "LLaVA vision-language models",
            "Idefics": "Idefics vision-language models",
            "PaliGemma": "PaliGemma multimodal models",
            "Pixtral": "Mistral's Pixtral vision models",
            "Molmo": "Allen AI's Molmo models",
            "Phi-3-Vision": "Microsoft's Phi-3 Vision models",
            "CogVLM": "Tsinghua's CogVLM models",
            "InternVL": "InternVL models",
            "MiniCPM-V": "OpenBMB's MiniCPM-V models",
            "Florence": "Microsoft Florence vision models",
            "DeepSeek-VL": "DeepSeek's vision-language models (DeepSeek-VL, DeepSeek-VL2)",
        }

    @staticmethod
    def is_mllm_model(model_name: str) -> bool:
        """Check if a model name indicates an MLLM model.

        Delegates to the canonical implementation in api.utils which checks
        config.json, model registry, and regex patterns.
        """
        from ..api.utils import is_mllm_model as _canonical
        return _canonical(model_name)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXMultimodalLM model={self.model_name} status={status}>"


# Backwards compatibility aliases
MLXVisionLanguageModel = MLXMultimodalLM
VLMOutput = MLLMOutput
is_vlm_model = MLXMultimodalLM.is_mllm_model
