# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for MLX from Transformers' Inkling2 implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.processing_utils import ProcessorMixin

from mlx_vlm.models.base import install_auto_processor_patch, load_chat_template


class Inkling2ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        size=None,
        image_mean=None,
        image_std=None,
        rescale_factor=1 / 255,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size or {"height": 40, "width": 40}
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.rescale_factor = rescale_factor

    def preprocess(self, images, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]
        patches, grids = [], []
        ph, pw = self.size["height"], self.size["width"]
        for image in images:
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    image = np.asarray(
                        opened.convert("RGBA" if opened.has_transparency_data else "RGB")
                    )
            if isinstance(image, Image.Image):
                image = np.asarray(
                    image.convert("RGBA" if image.has_transparency_data else "RGB")
                )
            else:
                image = np.asarray(image)
                if (
                    image.ndim == 3
                    and image.shape[-1] not in (1, 3, 4)
                    and image.shape[0] in (1, 3, 4)
                ):
                    image = image.transpose(1, 2, 0)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            if image.ndim != 3:
                raise ValueError("Inkling Mini expects an image with height and width")
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            height, width, channels = image.shape
            if channels == 4:
                yy, xx = np.indices((height, width))
                background = np.where((yy // 8 + xx // 8) % 2 == 0, 204, 153)[..., None]
                rgba = image.astype(np.int32)
                alpha = rgba[..., 3:4]
                image = (
                    (rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255
                ).astype(np.uint8)
            elif channels != 3:
                raise ValueError("Inkling 2 expects grayscale, RGB, or RGBA images")
            rows, cols = math.ceil(height / ph), math.ceil(width / pw)
            padded = np.full((rows * ph, cols * pw, 3), -1, np.float32)
            padded[:height, :width] = image
            padded = (
                padded * self.rescale_factor - np.array(self.image_mean, np.float32)
            ) / np.array(self.image_std, np.float32)
            patches.append(
                padded.reshape(rows, ph, cols, pw, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows * cols, ph, pw, 3)
            )
            grids.append([1, rows, cols])
        return BatchFeature(
            {
                "pixel_values": mx.array(np.concatenate(patches)),
                "image_grid_thw": mx.array(grids),
            }
        )

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)


class Inkling2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Inkling2ImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template", "image_token"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<|unused_200054|>",
    ):
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(self, images=None, text=None, padding_side="left", **kwargs):
        image_inputs = self.image_processor(images) if images is not None else {}
        texts = [text] if isinstance(text, str) else list(text or [])
        counts = [
            int(np.prod(g))
            for g in np.asarray(image_inputs.get("image_grid_thw", [])).tolist()
        ]
        index = 0
        expanded = []
        for item in texts:
            parts = item.split(self.image_token)
            output = parts[0]
            for part in parts[1:]:
                if index >= len(counts):
                    raise ValueError("More image placeholders than images")
                output += self.image_token * counts[index] + part
                index += 1
            expanded.append(output)
        if texts and index != len(counts):
            raise ValueError("More images than image placeholders")
        text_inputs = {}
        if expanded:
            all_ids = [
                self.tokenizer.encode(t, add_special_tokens=False) for t in expanded
            ]
            length = max(map(len, all_ids))
            input_ids, masks = [], []
            for ids in all_ids:
                padding = [self.tokenizer.pad_token_id or 0] * (length - len(ids))
                mask = [1] * len(ids)
                zeros = [0] * len(padding)
                input_ids.append(
                    padding + ids if padding_side == "left" else ids + padding
                )
                masks.append(zeros + mask if padding_side == "left" else mask + zeros)
            text_inputs = {
                "input_ids": mx.array(input_ids),
                "attention_mask": mx.array(masks),
            }
        return BatchFeature({**text_inputs, **image_inputs})

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from huggingface_hub import snapshot_download

        path = Path(pretrained_model_name_or_path)
        if not path.is_dir():
            path = Path(
                snapshot_download(
                    str(pretrained_model_name_or_path),
                    allow_patterns=["*.json", "*.jinja"],
                    revision=kwargs.get("revision"),
                )
            )
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=False, local_files_only=True
        )
        config = json.loads((path / "config.json").read_text())
        eos = config.get("eos_token_id", 200006)
        eos = eos[0] if isinstance(eos, list) else eos
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(eos)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_chat_template(tokenizer, path)
        processor_config = (
            json.loads((path / "processor_config.json").read_text())
            if (path / "processor_config.json").exists()
            else {}
        )
        return cls(
            Inkling2ImageProcessor(**processor_config.get("image_processor", {})),
            tokenizer,
            tokenizer.chat_template,
            processor_config.get("image_token", "<|unused_200054|>"),
        )

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]


install_auto_processor_patch("inkling2_loop", Inkling2Processor)
