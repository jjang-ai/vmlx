import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(Qwen3_5Model):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # ignore mtp weights
        weights = {key: value for key, value in weights.items() if "mtp." not in key}

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.config.text_config.num_hidden_layers):
            for prefix in (
                f"model.language_model.layers.{l}.mlp",
                f"language_model.model.layers.{l}.mlp",
            ):
                fused_key = f"{prefix}.experts.gate_up_proj"
                if fused_key in weights:
                    gate_up_weight = weights.pop(fused_key)
                    gate_weight, up_weights = mx.split(gate_up_weight, 2, axis=-2)
                    weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
                    weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weights
                down_key = f"{prefix}.experts.down_proj"
                if down_key in weights:
                    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                        down_key
                    )

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        # ABSOLUTE import: this file is registered under the mlx_vlm package
        # name, so relative imports resolve against mlx_vlm.
        from vmlx_engine.utils.zero_centered_norms import (
            zero_centered_norm_shift_needed,
        )

        apply_norm_shift = zero_centered_norm_shift_needed(
            weights, norm_keys, model_label="qwen3_5_moe"
        )

        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if apply_norm_shift and any(key.endswith(sfx) for sfx in norm_keys):
                if value.ndim == 1:
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
