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
from transformers.feature_extraction_utils import BatchFeature

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
    num_frames: int | None = None,
) -> tuple[int, int]:
    pixels_per_token = temporal_factor * factor * factor
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    # Budget the actual padded temporal extent, including an odd final frame.
    frames = temporal_factor if num_frames is None else math.ceil(num_frames / temporal_factor) * temporal_factor
    target_h, target_w = align(height), align(width)
    total = frames * target_h * target_w
    if total < min_pixels:
        scale = math.sqrt(min_pixels / (frames * height * width))
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
    # The bundled JANG loader wraps __call__; keep timestamp support explicit
    # even when inspect.signature sees that wrapper's **kwargs.
    supports_video_timestamps = True

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # OCR initializes a two-component ProcessorMixin. Recent transformers
        # infer that component count from this signature, so attach our native
        # video processor only after the parent's two-argument initialization.
        video_processor = kwargs.pop("video_processor", None)
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, **kwargs)
        self.video_processor = video_processor

    def __call__(self, images=None, text=None, videos=None, *, fps=None,
                 video_timestamps=None, videos_kwargs=None, **kwargs):
        if videos is None:
            return super().__call__(images=images, text=text, videos=None, **kwargs)
        if self.video_processor is None:
            raise ValueError("GLM bundle has no native video processor configuration")

        # Native GLM uses temporal pairs and timestamped IMAGE placeholders
        # inside video segments. The OCR parent and Qwen JANG fallback do not.
        image_inputs = self.image_processor(images=images) if images is not None else {}
        video_inputs = self.video_processor(videos=videos, **(videos_kwargs or {}))
        texts = [""] if text is None else ([text] if isinstance(text, str) else list(text))
        image_index = video_index = 0
        for i, value in enumerate(texts):
            image_parts = value.split(self.image_token)
            expanded = image_parts[0]
            for part in image_parts[1:]:
                grids = image_inputs.get("image_grid_thw", [])
                if image_index >= len(grids):
                    raise ValueError("GLM image placeholder has no matching image")
                count = int(np.prod(grids[image_index])) // self.image_processor.merge_size**2
                expanded += self.image_token * count + part
                image_index += 1
            video_parts = expanded.split(self.video_token)
            expanded = video_parts[0]
            for part in video_parts[1:]:
                grids = video_inputs["video_grid_thw"]
                if video_index >= len(grids):
                    raise ValueError("GLM video placeholder has no matching clip")
                grid_t, grid_h, grid_w = map(int, grids[video_index])
                count = grid_h * grid_w // self.video_processor.merge_size**2
                if video_timestamps is not None:
                    times = list(video_timestamps[video_index])
                else:
                    rate = fps[video_index] if isinstance(fps, (list, tuple)) else fps
                    rate = float(rate if rate is not None else self.video_processor.fps)
                    if not math.isfinite(rate) or rate <= 0:
                        raise ValueError("GLM sampled video fps must be finite and positive")
                    times = [j / rate for j in range(grid_t * self.video_processor.temporal_patch_size)]
                if not times or any(not math.isfinite(float(t)) or float(t) < 0 for t in times):
                    raise ValueError("GLM video timestamps must be finite and nonnegative")
                selected = times[::self.video_processor.temporal_patch_size][:grid_t]
                selected += [selected[-1]] * (grid_t - len(selected))
                for stamp in selected:
                    expanded += f"<|begin_of_image|>{self.image_token * count}<|end_of_image|>{stamp:.1f} seconds"
                expanded += part
                _LOG.info("GLM video placeholders: clip=%d grid=%dx%dx%d tokens=%d timestamps=%s",
                          video_index, grid_t, grid_h, grid_w, grid_t * count, selected)
                video_index += 1
            texts[i] = expanded
        if image_index != len(image_inputs.get("image_grid_thw", [])) or video_index != len(video_inputs["video_grid_thw"]):
            raise ValueError("GLM media inputs and placeholders disagree")

        return_tensors = kwargs.pop("return_tensors", None)
        kwargs.setdefault("padding", False)
        kwargs.setdefault("return_token_type_ids", False)
        text_inputs = self.tokenizer(texts, **kwargs)
        start_id = self.tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|end_of_video|>")
        types = []
        for row in text_inputs["input_ids"]:
            ids = np.asarray(row)
            video_region = np.cumsum(ids == start_id) > np.cumsum(ids == end_id)
            types.append(np.where(ids == self.image_token_id, np.where(video_region, 2, 1), 0).tolist())
        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs, "mm_token_type_ids": types},
                            tensor_type=return_tensors)

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
        video_config = processor_config.get("video_processor")
        video_processor = None
        if isinstance(video_config, dict):
            video_config = dict(video_config)
            video_config.pop("video_processor_type", None)
            video_processor = Glm5NextVideoProcessor(**video_config)
        return cls(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor)


class Glm5NextVideoProcessor(Glm5NextImageProcessor):
    """NumPy/PIL native GLM video packing; frame sampling belongs to the engine.

    Temporal layout and timestamp contract follow Hugging Face Transformers
    glm5_next at 5474a55e920f358d8382f3ecd3377edca979baa1 (Apache-2.0).
    No torchvision dependency or image-per-frame temporal approximation.
    """

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, max_image_tokens=240000, fps=2, patch_expand_factor=1, **kwargs):
        super().__init__(max_image_tokens=max_image_tokens, **kwargs)
        self.fps = float(fps)
        self.patch_expand_factor = int(patch_expand_factor)
        if min(self.patch_size, self.temporal_patch_size, self.merge_size, self.patch_expand_factor) <= 0:
            raise ValueError("GLM video patch geometry must be positive")

    def __call__(self, videos, **kwargs):
        # Decoded frames have already been sampled/resized by the request's
        # validated video controls. Do not sample them a second time.
        del kwargs
        if isinstance(videos, np.ndarray) and videos.ndim == 4:
            videos = [videos]
        if not isinstance(videos, (list, tuple)) or not videos:
            raise ValueError("GLM video input requires a nonempty list of decoded clips")
        outputs, grids = [], []
        for clip in videos:
            if not isinstance(clip, (list, tuple, np.ndarray)) or len(clip) == 0:
                raise ValueError("GLM video clip must contain decoded frames")
            if isinstance(clip, np.ndarray) and clip.ndim != 4:
                raise ValueError(f"GLM video input must be rank 4, got {clip.shape}")
            frames = [_as_rgb_array(frame) for frame in clip]
            height, width = frames[0].shape[:2]
            if any(frame.shape != frames[0].shape for frame in frames):
                raise ValueError("GLM frames in one clip must share their dimensions")
            factor = self.patch_size * self.merge_size * self.patch_expand_factor
            target_h, target_w = _aligned_canvas(
                height, width, factor=factor, temporal_factor=self.temporal_patch_size,
                min_tokens=self.min_image_tokens, max_tokens=self.max_image_tokens,
                num_frames=len(frames),
            )
            scale = min(target_h / height, target_w / width)
            if len(frames) * height * width >= self.temporal_patch_size * factor**2 * self.min_image_tokens:
                scale = min(1.0, scale)
            content_h = max(1, min(target_h, math.floor(height * scale)))
            content_w = max(1, min(target_w, math.floor(width * scale)))
            pixels = np.zeros((len(frames), target_h, target_w, 3), dtype=np.uint8)
            for index, frame in enumerate(frames):
                if (content_h, content_w) != (height, width):
                    frame = np.asarray(Image.fromarray(frame).resize((content_w, content_h), Image.Resampling.BICUBIC))
                pixels[index, :content_h, :content_w] = frame
            pixels = np.moveaxis(pixels.astype(np.float32), -1, 1)
            if self.do_rescale:
                pixels *= self.rescale_factor
            if self.do_normalize:
                pixels = (pixels - np.asarray(self.image_mean, dtype=np.float32)[None, :, None, None]) / np.asarray(self.image_std, dtype=np.float32)[None, :, None, None]
            if pad := -len(pixels) % self.temporal_patch_size:
                pixels = np.concatenate([pixels, np.repeat(pixels[-1:], pad, axis=0)], axis=0)
            grid_t = len(pixels) // self.temporal_patch_size
            grid_h, grid_w = target_h // self.patch_size, target_w // self.patch_size
            patches = pixels.reshape(grid_t, self.temporal_patch_size, 3,
                                     grid_h // self.merge_size, self.merge_size, self.patch_size,
                                     grid_w // self.merge_size, self.merge_size, self.patch_size)
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
                grid_t * grid_h * grid_w, 3 * self.temporal_patch_size * self.patch_size**2)
            outputs.append(patches)
            grids.append([grid_t, grid_h, grid_w])
            _LOG.info("GLM video processed: frames=%d padded=%d input=%dx%d canvas=%dx%d grid=%dx%dx%d tokens=%d",
                      len(frames), len(pixels), height, width, target_h, target_w,
                      grid_t, grid_h, grid_w, grid_t * grid_h * grid_w // self.merge_size**2)
        return {"pixel_values_videos": np.concatenate(outputs, axis=0),
                "video_grid_thw": np.asarray(grids, dtype=np.int64)}

    def preprocess(self, videos, **kwargs):
        return self(videos, **kwargs)


install_auto_processor_patch("glm5_next", Glm5NextProcessor)

__all__ = ["Glm5NextImageProcessor", "Glm5NextVideoProcessor", "Glm5NextProcessor"]
