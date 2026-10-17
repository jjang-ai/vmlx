from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(Qwen3VLModel):
    """Qwen3.8-Flash-Next image/video wrapper around the qwen4_exp core."""

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        mask = kwargs.get("mask")
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None and pixel_values_videos is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        if getattr(self, "vision_tower", None) is None:
            raise ValueError("this qwen4_exp bundle is text-only (no vision tower): image/video input is not supported")
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        if pixel_values is not None:
            image_features = (
                kwargs.get("cached_image_features")
                if pixel_values_videos is None
                else None
            )
            if image_features is None:
                image_features, _ = self.vision_tower(
                    pixel_values.astype(dtype), image_grid_thw
                )
            inputs_embeds = self._scatter_media_features(
                image_features,
                inputs_embeds,
                input_ids,
                self.config.image_token_index,
                "image",
            )
        if pixel_values_videos is not None:
            video_features, _ = self.vision_tower(
                pixel_values_videos.astype(dtype), video_grid_thw
            )
            inputs_embeds = self._scatter_media_features(
                video_features,
                inputs_embeds,
                input_ids,
                self.config.video_token_index,
                "video",
            )
        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = rope_deltas
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @staticmethod
    def _scatter_media_features(
        media_features,
        inputs_embeds,
        input_ids,
        token_index,
        modality,
    ):
        special_mask = input_ids == token_index
        n_tokens = special_mask.sum()
        expanded = mx.broadcast_to(special_mask[..., None], inputs_embeds.shape)
        if expanded.sum() != media_features.size:
            raise ValueError(
                f"{modality.capitalize()} features and {modality} tokens do not match: "
                f"tokens={n_tokens}, features={media_features.shape[0]}"
            )
        return masked_scatter(inputs_embeds, expanded, media_features)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_mask = (input_ids == image_token_index) | (
            input_ids == video_token_index
        )
        n_tokens = special_mask.sum()
        expanded = mx.broadcast_to(special_mask[..., None], inputs_embeds.shape)
        if expanded.sum() != image_features.size:
            raise ValueError(
                "Image features and image/video tokens do not match: "
                f"tokens={n_tokens}, features={image_features.shape[0]}"
            )
        return masked_scatter(inputs_embeds, expanded, image_features), expanded

    def sanitize(self, weights):
        """Normalize official HF names without discarding trained MTP state."""
        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        shifted_norm_suffixes = (
            ".hc_norm.weight",
            ".norm_key.weight",
            ".norm_query.weight",
            ".norm_conv.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
        )
        out = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif key.startswith("model.visual."):
                key = key.replace("model.visual", "vision_tower", 1)
            elif key.startswith("lm_head."):
                key = key.replace("lm_head", "language_model.lm_head", 1)

            if key.endswith("ple.conv1d.weight") and value.ndim == 3:
                value = value.squeeze(1)
            elif "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if value.ndim == 1 and key.endswith(shifted_norm_suffixes):
                value = value + 1.0
            out[key] = value
        return out

    def mtp_forward(
        self, hidden_states, next_token_ids, mtp_cache, return_hidden=False
    ):
        return self.language_model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache,
            return_hidden=return_hidden,
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
