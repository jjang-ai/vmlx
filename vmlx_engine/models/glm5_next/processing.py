from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from mlx_vlm.models.base import install_auto_processor_patch, load_chat_template
from mlx_vlm.models.glm_ocr.processing import GlmOcrProcessor
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_processing_utils import ImageProcessingMixin

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_LOG = logging.getLogger(__name__)


def _aligned_canvas(
    height: int,
    width: int,
    *,
    factor: int,
    temporal_factor: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[int, int]:
    pixels_per_token = temporal_factor * factor * factor
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    frames = max(temporal_factor, round(temporal_factor / temporal_factor) * temporal_factor)
    target_h, target_w = align(height), align(width)
    total = frames * target_h * target_w
    if total < min_pixels:
        scale = math.sqrt(min_pixels / (temporal_factor * height * width))
        target_h = align(max(1, math.ceil(height * scale)))
        target_w = align(max(1, math.ceil(width * scale)))
        total = frames * target_h * target_w
    if total > max_pixels:
        low, high = 1, height
        best_h = best_w = factor
        while low <= high:
            content_h = (low + high) // 2
            content_w = max(1, math.floor(width * content_h / height))
            candidate_h, candidate_w = align(content_h), align(content_w)
            if frames * candidate_h * candidate_w <= max_pixels:
                best_h, best_w = candidate_h, candidate_w
                low = content_h + 1
            else:
                high = content_h - 1
        target_h, target_w = best_h, best_w
    return target_h, target_w


def _as_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        array = image
        if array.ndim != 3:
            raise ValueError(f"GLM image input must be rank 3, got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.dtype != np.uint8:
            scale = 255.0 if float(np.max(array)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array).convert("RGB")
    else:
        pil = Image.open(image).convert("RGB")
    return np.asarray(pil)


class Glm5NextImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.merge_size = int(merge_size)
        self.min_image_tokens = int(min_image_tokens)
        self.max_image_tokens = int(max_image_tokens)
        self.do_rescale = bool(do_rescale)
        self.rescale_factor = float(rescale_factor)
        self.do_normalize = bool(do_normalize)
        self.image_mean = list(image_mean or _CLIP_MEAN)
        self.image_std = list(image_std or _CLIP_STD)
        # Loader compatibility only; not a one-pixel processing ceiling.
        self.size = {"longest_edge": 1}

    @property
    def min_pixels(self) -> int:
        return self.min_image_tokens * (self.patch_size * self.merge_size) ** 2

    @property
    def max_pixels(self) -> int:
        return self.max_image_tokens * (self.patch_size * self.merge_size) ** 2

    def image_control_size(self, height: int, width: int) -> tuple[int, int]:
        """Native padded canvas, shared by preprocessing and control diagnostics."""
        return _aligned_canvas(
            height, width,
            factor=self.patch_size * self.merge_size,
            temporal_factor=self.temporal_patch_size,
            min_tokens=self.min_image_tokens,
            max_tokens=self.max_image_tokens,
        )

    def fetch_images(self, images):
        if not isinstance(images, list):
            images = [images]
        return [Image.fromarray(_as_rgb_array(image)) for image in images]

    def _process_one(self, image) -> tuple[np.ndarray, list[int]]:
        array = _as_rgb_array(image)
        height, width = array.shape[:2]
        factor = self.patch_size * self.merge_size
        target_h, target_w = self.image_control_size(height, width)
        scale = min(target_h / height, target_w / width)
        if self.temporal_patch_size * height * width >= (
            self.temporal_patch_size * factor * factor * self.min_image_tokens
        ):
            scale = min(1.0, scale)
        content_h = max(1, min(target_h, math.floor(height * scale)))
        content_w = max(1, min(target_w, math.floor(width * scale)))
        if (content_h, content_w) != (height, width):
            array = np.asarray(
                Image.fromarray(array).resize(
                    (content_w, content_h), Image.Resampling.BICUBIC
                )
            )
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:content_h, :content_w] = array
        pixels = canvas.astype(np.float32)
        if self.do_rescale:
            pixels *= self.rescale_factor
        pixels = np.moveaxis(pixels, -1, 0)
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
            pixels = (pixels - mean) / std

        channels, resized_h, resized_w = pixels.shape
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        _LOG.info(
            "GLM image processed: input=%dx%d content=%dx%d canvas=%dx%d "
            "grid=1x%dx%d tokens=%d",
            height, width, content_h, content_w, resized_h, resized_w,
            grid_h, grid_w, grid_h * grid_w // self.merge_size**2,
        )
        patches = pixels.reshape(
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None, :, :],
            (*patches.shape[:5], self.temporal_patch_size, *patches.shape[5:]),
        )
        patches = patches.reshape(
            grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return patches, [1, grid_h, grid_w]

    def __call__(self, images, **kwargs):
        del kwargs
        if not isinstance(images, list):
            images = [images]
        processed = [self._process_one(image) for image in images]
        return {
            "pixel_values": np.concatenate([item[0] for item in processed], axis=0),
            "image_grid_thw": np.asarray([item[1] for item in processed], dtype=np.int64),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)


class Glm5NextProcessor(GlmOcrProcessor):
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        kwargs.pop("trust_remote_code", None)
        use_fast = kwargs.pop("use_fast", True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast, **kwargs
        )
        load_chat_template(tokenizer, model_path)
        config_path = Path(model_path) / "processor_config.json"
        processor_config = json.loads(config_path.read_text()) if config_path.exists() else {}
        image_config = dict(processor_config.get("image_processor") or {})
        image_config.pop("image_processor_type", None)
        image_processor = Glm5NextImageProcessor(**image_config)
        return cls(image_processor=image_processor, tokenizer=tokenizer)


install_auto_processor_patch("glm5_next", Glm5NextProcessor)

__all__ = ["Glm5NextImageProcessor", "Glm5NextProcessor"]
