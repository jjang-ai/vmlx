from dataclasses import dataclass, field

from mlx_vlm.models.base import BaseModelConfig

from .config import TextConfig as TargetTextConfig


@dataclass
class TextConfig(TargetTextConfig):
    mtp_config: dict = field(default_factory=dict)


@dataclass
class DSparkConfig(BaseModelConfig):
    model_type: str = "inkling2_dspark"
    text_config: TextConfig = field(default_factory=TextConfig)
    block_size: int = 0

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        text = self.text_config.to_dict()
        draft = text.get("mtp_config") or {}
        if not draft:
            raise ValueError("Inkling 2 DSpark requires mtp_config")
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "markov_rank",
            "max_position_embeddings",
            "sliding_window_size",
            "rope_theta",
            "rms_norm_eps",
            "target_layer_ids",
            "local_layer_ids",
        ):
            setattr(self, name, draft[name])
        self.vocab_size = text["vocab_size"]
        self.unpadded_vocab_size = text.get("unpadded_vocab_size", self.vocab_size)
        self.final_logit_softcapping = text.get("final_logit_softcapping", 30.0)
        self.block_size = self.block_size or draft["block_size"]
        if (
            draft.get("markov_head_type") != "rnn_hidden"
            or draft.get("q_bias")
            or draft.get("o_bias")
            or self.hidden_size != text["hidden_size"]
            or self.block_size != draft["block_size"]
            or self.block_size < 2
            or self.num_hidden_layers < 1
            or self.markov_rank < 1
            or self.num_attention_heads % self.num_key_value_heads
            or self.head_dim % 2
            or self.sliding_window_size < 1
            or any(i < 0 or i >= self.num_hidden_layers for i in self.local_layer_ids)
        ):
            raise ValueError("Unsupported Inkling 2 mtp_config")
        ids = self.target_layer_ids
        if (
            len(ids) < 2
            or ids != sorted(set(ids))
            or ids[0] < 0
            or ids[-1] != len(self.text_config.layer_indices)
        ):
            raise ValueError("Invalid Inkling 2 DSpark target_layer_ids")
