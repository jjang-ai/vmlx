from dataclasses import dataclass
from typing import Optional

from mlx_vlm.models.base import BaseModelConfig


@dataclass(frozen=True)
class LoopConfig:
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    is_varloop: bool = False


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "inkling2_text"
    hidden_size: int = 4608
    num_hidden_layers: int = 48
    vocab_size: int = 201024
    unpadded_vocab_size: Optional[int] = 200058
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 48
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    local_layer_ids: Optional[list[int]] = None
    layer_types: Optional[list[str]] = None
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: Optional[int] = 128000
    log_scaling_alpha: float = 0.1
    sconv_kernel_size: int = 4
    intermediate_size: int = 2304
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8
    n_shared_experts: int = 0
    route_scale: float = 8.0
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    attn_res_normalize_values: bool = True
    use_value_norm_no_gamma: bool = True
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    loop_config: LoopConfig | None = None

    def __post_init__(self):
        if isinstance(self.loop_config, dict):
            self.loop_config = LoopConfig(**self.loop_config)
        loop = self.loop_config
        if loop is None:
            if self.model_type == "inkling2_loop_text":
                raise ValueError("Inkling 2 Mini requires loop_config")
            return
        if not (
            0 < loop.core_start < loop.core_end <= self.num_hidden_layers
            and self.n_layers_per_attn_res_block > 0
            and loop.core_start % self.n_layers_per_attn_res_block == 0
            and loop.core_end % self.n_layers_per_attn_res_block == 0
            and loop.num_passes >= 1
            and loop.gate_rank >= 1
            and not loop.is_varloop
        ):
            raise ValueError(f"Unsupported Inkling 2 loop_config: {loop}")

    @property
    def layer_indices(self):
        indices = tuple(range(self.num_hidden_layers))
        loop = self.loop_config
        if loop is None:
            return indices
        return (
            indices[: loop.core_start]
            + indices[loop.core_start : loop.core_end] * loop.num_passes
            + indices[loop.core_end :]
        )

    def layer_is_sliding(self, index):
        if self.layer_types is not None:
            return self.layer_types[index] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return index in self.local_layer_ids
        return (index + 1) % 6 != 0


def validate_sharding(config: dict, world_size: int):
    """Require expert ownership; compatible two-rank attention also partitions."""
    text = config.get("text_config") or config
    count = text.get("n_routed_experts", TextConfig.n_routed_experts)
    if not 1 <= world_size <= count:
        raise ValueError("Each distributed rank must own at least one expert")


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "inkling2_vision"
    native_patch_size: int = 20
    adapter_merge_size: int = 2
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    mlp_ratio: float = 4.0
    posemb_grid_size: int = 32
    layer_norm_eps: float = 1e-5
    text_hidden_size: int = 4608
    rope_theta: float = 10000.0
    out_hidden_size: int = 4608


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "inkling2"
    text_config: Optional[TextConfig] = None
    vision_config: Optional[VisionConfig] = None
    mtp_config: Optional[dict] = None
    image_token_id: int = 200054
    eos_token_id: int = 200006
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        if not isinstance(self.text_config, TextConfig):
            self.text_config = TextConfig.from_dict(self.text_config or {})
        if not isinstance(self.vision_config, VisionConfig):
            self.vision_config = VisionConfig.from_dict(self.vision_config or {})
        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.vision_config.out_hidden_size = self.text_config.hidden_size
