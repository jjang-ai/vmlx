from __future__ import annotations

import inspect
from dataclasses import dataclass

from mlx_vlm.models.base import BaseModelConfig

from .glm5_next import ModelArgs as TextConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm5_next_vision"
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    patch_size: int = 14
    image_size: int = 448
    in_channels: int = 3
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    out_hidden_size: int = 4096
    projection_intermediate_size: int = 10240
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    swiglu_limit: float = 10.0


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "glm5_next"
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False
    eos_token_id: list[int] | int | None = None
    # JANGTQ v2 campaign: carried so the model can install TQ experts and alias per-module quant keys.
    # None for every other bundle (mlx_vlm reads quantization from this attribute when present -> same value).
    jangtq: dict | None = None
    quantization: dict | None = None

    @classmethod
    def from_dict(cls, params):
        values = {
            key: value
            for key, value in params.items()
            if key in inspect.signature(cls).parameters
        }
        return cls(**values)
