# SPDX-License-Identifier: Apache-2.0
"""
Multimodal processor for VLM continuous batching.

This module handles preprocessing of multimodal inputs (images, videos)
for use with the continuous batching scheduler. It extracts processed
inputs that can be batched together efficiently.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .models.mllm import (
    process_image_input,
    process_video_input,
    extract_video_frames_smart,
    save_frames_to_temp,
    DEFAULT_FPS,
    MAX_FRAMES,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMultimodalInput:
    """
    Container for processed multimodal inputs ready for batching.

    Attributes:
        input_ids: Tokenized text with image/video tokens (mx.array)
        pixel_values: Processed image tensors (mx.array)
        attention_mask: Attention mask for the input (mx.array)
        image_grid_thw: Grid info for Qwen-VL models (mx.array)
        num_images: Number of images in this input
        num_tokens: Number of tokens in input_ids
        extra_kwargs: Additional model-specific kwargs
    """

    input_ids: mx.array
    pixel_values: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    image_grid_thw: Optional[mx.array] = None
    num_images: int = 0
    num_tokens: int = 0
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class MultimodalProcessor:
    """
    Processor for preparing multimodal inputs for VLM batching.

    This class wraps mlx_vlm's prepare_inputs function and provides
    a clean interface for the scheduler to preprocess requests.

    Example:
        >>> processor = MultimodalProcessor(model, vlm_processor)
        >>> processed = processor.process(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> # processed.input_ids, processed.pixel_values ready for batching
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: Optional[Any] = None,
    ):
        """
        Initialize the multimodal processor.

        Args:
            model: The VLM model (for config access)
            processor: The VLM processor (tokenizer + image processor)
            config: Optional model config
        """
        self.model = model
        self.processor = processor
        self.config = config or getattr(model, "config", None)

        # Get tokenizer from processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

        # Get image token index if available
        self.image_token_index = (
            getattr(self.config, "image_token_index", None) if self.config else None
        )

    def process(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        video_fps: float = DEFAULT_FPS,
        video_max_frames: int = MAX_FRAMES,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> ProcessedMultimodalInput:
        """
        Process multimodal inputs for batching.

        Args:
            prompt: Text prompt (already formatted with chat template)
            images: List of image paths, URLs, or base64 strings
            videos: List of video inputs
            video_fps: FPS for video frame extraction
            video_max_frames: Max frames per video
            add_special_tokens: Whether to add special tokens
            **kwargs: Additional model-specific parameters

        Returns:
            ProcessedMultimodalInput with all processed tensors
        """
        from mlx_vlm.utils import prepare_inputs

        # Process raw images
        all_images = []
        failed_images = 0
        if images:
            for img in images:
                try:
                    path = process_image_input(img)
                    all_images.append(path)
                except Exception as e:
                    failed_images += 1
                    logger.warning(f"Failed to process image: {e}")

        # Extract frames from videos
        failed_videos = 0
        if videos:
            for video in videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path,
                        fps=video_fps,
                        max_frames=video_max_frames,
                    )
                    frame_paths = save_frames_to_temp(frames)
                    all_images.extend(frame_paths)
                    logger.debug(f"Extracted {len(frame_paths)} frames from video")
                except Exception as e:
                    failed_videos += 1
                    logger.warning(f"Failed to process video: {e}")

        # H3: Raise if ALL media items failed (user would get no visual context)
        total_inputs = len(images or []) + len(videos or [])
        if total_inputs > 0 and not all_images:
            raise ValueError(
                f"All media inputs failed to process "
                f"({failed_images} image(s), {failed_videos} video(s) failed). "
                f"Check URLs/paths and try again."
            )

        # Determine add_special_tokens based on model type
        if self.config and self.config.model_type in ["gemma3", "gemma3n", "gemma4"]:
            add_special_tokens = not hasattr(self.processor, "chat_template")

        # Prepare inputs using mlx_vlm
        inputs = prepare_inputs(
            self.processor,
            images=all_images if all_images else None,
            prompts=prompt,
            image_token_index=self.image_token_index,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        # Extract processed tensors
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        # Extract model-specific kwargs
        extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        # Get image_grid_thw for Qwen-VL models
        image_grid_thw = extra_kwargs.pop("image_grid_thw", None)

        return ProcessedMultimodalInput(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            num_images=len(all_images),
            num_tokens=input_ids.size if input_ids is not None else 0,
            extra_kwargs=extra_kwargs,
        )

