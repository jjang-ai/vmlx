# SPDX-License-Identifier: Apache-2.0
# Architecture reference: Baidu vLLM-ERNIE & upstream MIT mlx-vlm ernie5_5_moe (Copyright 2026 Neywa Labs Inc.).
"""Configuration for ERNIE 5.5 Flash (ernie5_5_moe) models."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelConfig:
    model_type: str = "ernie5_5_moe"
    vocab_size: int = 201216
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)
    hybrid_mla_num_attention_heads: int = 64
    hybrid_mla_q_lora_rank: int = 1024
    hybrid_mla_kv_lora_rank: int = 512
    hybrid_mla_qk_nope_head_dim: int = 192
    hybrid_mla_qk_rope_head_dim: int = 64
    hybrid_mla_v_head_dim: int = 256
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    max_position_embeddings: int = 1048576
    rope_theta: float = 8000000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    moe_intermediate_size: int = 2048
    moe_latent_size: int = 2048
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    num_experts_per_tok: int = 10
    first_k_dense_replace: int = 1
    routed_scaling_factor: float = 3.0
    norm_topk_prob: bool = True
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    vha_rank: int = 16
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    mtp_shared_last_layer: bool = True
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[Union[int, List[int]]] = 2
    pad_token_id: Optional[int] = 0

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in params.items() if k in valid}
        return cls(**filtered)

    def __post_init__(self) -> None:
        if not self.compress_ratios:
            self.compress_ratios = [
                -2 if i > 0 and i % 7 == 0 else 128
                for i in range(self.num_hidden_layers)
            ]
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError("compress_ratios must cover every decoder layer")
        if any(r != -2 and (r <= 0 or r == 4) for r in self.compress_ratios):
            raise ValueError("Expected non-overlapping HCA compression or -2 for MLA")
        if self.hidden_act != "situ":
            raise ValueError("ERNIE 5.5 requires the SiTU activation")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("o_groups must divide num_attention_heads")
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("Invalid number of selected experts")
