from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput

from .config import ModelConfig
from .glm5_next import Model as TextModel
from .vision import VisionModel


class LanguageModel(TextModel):
    def __call__(
        self,
        inputs=None,
        inputs_embeds=None,
        cache=None,
        return_hidden: bool = False,
        **kwargs,
    ):
        kwargs.pop("mask", None)
        kwargs.pop("n_to_process", None)
        result = super().__call__(
            inputs,
            inputs_embeds=inputs_embeds,
            cache=cache,
            return_hidden=return_hidden,
            **kwargs,
        )
        if return_hidden:
            logits, hidden = result
            return LanguageModelOutput(logits=logits, hidden_states=[hidden])
        return LanguageModelOutput(logits=result)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        pixel_values_videos: mx.array | None = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("GLM multimodal input requires input_ids")
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None and pixel_values_videos is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        video_region = None
        if pixel_values_videos is not None:
            # Native video frames use image tokens inside begin/end-of-video,
            # not the unexpanded video placeholder. Keep historical images
            # separate when both modalities occur in one connected request.
            starts = mx.cumsum((input_ids == self.config.video_start_token_id).astype(mx.int32), axis=-1)
            ends = mx.cumsum((input_ids == self.config.video_end_token_id).astype(mx.int32), axis=-1)
            video_region = starts > ends

        if pixel_values is not None:
            grid = kwargs.get("image_grid_thw")
            if grid is None:
                raise ValueError("GLM image input requires image_grid_thw")
            features = kwargs.get("cached_image_features")
            if features is None:
                dtype = self.vision_tower.patch_embed.proj.weight.dtype
                features = self.vision_tower(pixel_values.astype(dtype), grid)
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                features,
                self.config.image_token_id,
                "image",
                region=~video_region if video_region is not None else None,
            )

        if pixel_values_videos is not None:
            grid = kwargs.get("video_grid_thw")
            if grid is None:
                raise ValueError("GLM video input requires video_grid_thw")
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            features = self.vision_tower(pixel_values_videos.astype(dtype), grid)
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                features,
                self.config.image_token_id,
                "video",
                region=video_region,
            )

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @staticmethod
    def _scatter_features(inputs_embeds, input_ids, features, token_id, modality, *, region=None):
        mask = input_ids == token_id
        if region is not None:
            mask = mask & region
        token_count = int(mx.sum(mask).item())
        if token_count != int(features.shape[0]):
            raise ValueError(
                f"GLM {modality} features and placeholders disagree: "
                f"tokens={token_count}, features={features.shape[0]}"
            )
        feature_rows = mx.reshape(features, (-1, inputs_embeds.shape[-1]))
        flat_input = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_mask = mask.reshape(-1)
        feature_index = mx.cumsum(flat_mask.astype(mx.int32)) - 1
        gathered = feature_rows[mx.maximum(feature_index, 0)]
        merged = mx.where(flat_mask[:, None], gathered, flat_input)
        return merged.reshape(inputs_embeds.shape)

    def __call__(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_videos=None,
        cache=None,
        **kwargs,
    ):
        embeddings = self.get_input_embeddings(
            input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            **kwargs,
        )
        return self.language_model(
            input_ids,
            inputs_embeds=embeddings.inputs_embeds,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        text_layers = self.config.text_config.num_hidden_layers
        mtp_prefix = f"model.layers.{text_layers}."
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("visual."):
                key = "vision_tower." + key[len("visual.") :]
            elif key.startswith("model.visual."):
                key = "vision_tower." + key[len("model.visual.") :]
            elif key.startswith(mtp_prefix):
                if hasattr(self.language_model, "mtp"):
                    key = "language_model.mtp." + key[len(mtp_prefix) :]
                else:
                    continue
            elif key.startswith("model.") or key.startswith("lm_head."):
                key = "language_model." + key
            sanitized[key] = value
        return self.vision_tower.sanitize(sanitized)

    def prepare_acceleration(self):
        return self.language_model.prepare_acceleration()

    def mtp_forward(self, *args, **kwargs):
        return self.language_model.mtp_forward(*args, **kwargs)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return getattr(self.language_model, "quant_predicate", None)

    @property
    def cast_predicate(self):
        return getattr(self.language_model, "cast_predicate", None)
