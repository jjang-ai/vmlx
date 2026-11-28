# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclasses for Hy4-flash (model_type='hy_v4_flash')."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HyV4FlashConfig:
    """Text + inline DFly2 + Engram configuration parsed strictly from config.json."""

    model_type: str = "hy_v4_flash"
    vocab_size: int = 120832
    hidden_size: int = 3072
    num_hidden_layers: int = 42
    intermediate_size: int = 13824
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    enable_lm_head_fp32: bool = True

    # Special tokens
    bos_token_id: int = 120000
    pad_token_id: int = 120002
    eos_token_id: int = 120025
    image_start_token_id: int = 120118
    image_end_token_id: int = 120119
    image_token_id: int = 120120
    image_newline_token_id: int = 120121
    video_start_token_id: int = 120122
    video_end_token_id: int = 120123
    video_token_id: int = 120124

    # Identity Hyper-Connections (iHC)
    enable_ihc: bool = True
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6

    # Hybrid MLA Attention: 12 DSA (IndexPool) + 30 SWA (window 512)
    layer_types: List[str] = field(default_factory=list)
    swa_pattern: List[int] = field(default_factory=list)
    num_attention_heads: int = 32
    swa_num_attention_heads: int = 64
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    sliding_window: int = 512
    gated_mla: bool = True
    learnable_sink: bool = True
    sandwich_norm: bool = True
    rope_theta_dsa: float = 10_000_000.0
    rope_theta_swa: float = 10_000.0
    mrope_section: Tuple[int, int, int] = (12, 10, 10)
    mrope_interleaved: bool = True

    # Compressed Indexer (IndexPool)
    use_compressed_indexer: bool = True
    index_n_heads: int = 16
    index_head_dim: int = 128
    index_topk: int = 2048
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_group_topk: int = 512
    compressed_indexer_candidate_topk: int = 2048

    # LatentMoE (layers 1..41)
    first_k_dense_replace: int = 1
    mlp_layer_types: List[str] = field(default_factory=list)
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    num_experts_per_tok: int = 16
    routed_expert_hidden_size: int = 1536
    moe_intermediate_size: int = 1536
    routed_scaling_factor: float = 3.998
    norm_topk_prob: bool = True
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    swiglu_limit: float = 10.0

    # Disk-backed Engram N-Gram Memory (1-based layer IDs [2, 12, 22, 32] -> 0-based [1, 11, 21, 31])
    engram_layer_ids: List[int] = field(default_factory=list)
    engram_max_ngram_size: int = 3
    engram_n_embed_per_ngram: int = 768
    engram_n_head_per_ngram: int = 8
    engram_pad_id: int = 120002
    engram_seed: int = 0
    engram_fp8_embedding: bool = False
    engram_layer_head_vocab_sizes: Dict[str, List[List[int]]] = field(default_factory=dict)

    # Inline DFly2 Speculative Drafter (root config fields)
    dfly_num_hidden_layers: int = 0
    dfly_block_size: int = 6
    dfly_mask_token_id: int = 120009
    dfly_target_kv_layers: List[int] = field(default_factory=list)
    dfly_target_index_layer: List[int] = field(default_factory=list)
    dfly_target_index_topk: int = 2048
    dfly_mtp_fusion: str = "slot0"
    dfly_enable_hidden_correction: bool = True
    dfly_context_local_window: Optional[int] = None

    # Raw quantization metadata
    quantization_config: Dict[str, Any] = field(default_factory=dict)
    vision_config: Optional[Dict[str, Any]] = None

    @property
    def engram_zero_indexed_layers(self) -> Dict[int, int]:
        """Map 0-based decoder layer index -> 1-based engram layer_id (for seed & prime lookup)."""
        return {int(lid) - 1: int(lid) for lid in self.engram_layer_ids}

    def is_dsa_layer(self, layer_idx: int) -> bool:
        if layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] == "deepseek_sparse_attention"
        if layer_idx < len(self.swa_pattern):
            return int(self.swa_pattern[layer_idx]) == 0
        return layer_idx in (0, 1) or ((layer_idx - 1) % 4 == 0)

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer_idx] == "sparse"
        return layer_idx >= self.first_k_dense_replace

    @classmethod
    def from_dict(cls, root: Dict[str, Any]) -> "HyV4FlashConfig":
        tc = root.get("text_config") if isinstance(root.get("text_config"), dict) else root
        rp = tc.get("rope_parameters") or {}
        dsa_rp = rp.get("deepseek_sparse_attention") or {}
        swa_rp = rp.get("sliding_attention") or {}
        mrope_sec = tuple(int(x) for x in (dsa_rp.get("mrope_section") or swa_rp.get("mrope_section") or (12, 10, 10)))

        num_layers = int(tc.get("num_hidden_layers", 42))
        layer_types = list(tc.get("layer_types") or [])
        swa_pattern = [int(x) for x in (tc.get("swa_pattern") or [])]
        if not layer_types and swa_pattern:
            layer_types = ["sliding_attention" if s else "deepseek_sparse_attention" for s in swa_pattern]
        if not layer_types:
            layer_types = [
                "deepseek_sparse_attention" if (i in (0, 1) or (i - 1) % 4 == 0) else "sliding_attention"
                for i in range(num_layers)
            ]

        mlp_types = list(tc.get("mlp_layer_types") or [])
        first_k = int(tc.get("first_k_dense_replace", 1))
        if not mlp_types:
            mlp_types = ["dense" if i < first_k else "sparse" for i in range(num_layers)]

        spec_cfg = root.get("speculative_config") or {}
        dfly_layers = int(
            root.get("dfly_num_hidden_layers")
            or tc.get("dfly_num_hidden_layers")
            or spec_cfg.get("dfly_num_hidden_layers")
            or 0
        )
        dfly_kv = list(
            root.get("dfly_target_kv_layers")
            or tc.get("dfly_target_kv_layers")
            or spec_cfg.get("dfly_target_kv_layers")
            or []
        )
        dfly_idx = list(
            root.get("dfly_target_index_layer")
            or tc.get("dfly_target_index_layer")
            or spec_cfg.get("dfly_target_index_layer")
            or dfly_kv
        )
        dfly_bs = int(
            root.get("dfly_block_size")
            or tc.get("dfly_block_size")
            or spec_cfg.get("num_speculative_tokens", 5) + 1
            if "num_speculative_tokens" in spec_cfg
            else (root.get("dfly_block_size") or tc.get("dfly_block_size") or 6)
        )

        qcfg = root.get("quantization_config") or tc.get("quantization_config") or {}
        eos_raw = tc.get("eos_token_id", 120025)
        eos_id = int(eos_raw[0]) if isinstance(eos_raw, list) else int(eos_raw)

        return cls(
            model_type=str(root.get("model_type") or tc.get("model_type") or "hy_v4_flash"),
            vocab_size=int(tc.get("vocab_size", 120832)),
            hidden_size=int(tc.get("hidden_size", 3072)),
            num_hidden_layers=num_layers,
            intermediate_size=int(tc.get("intermediate_size", 13824)),
            max_position_embeddings=int(tc.get("max_position_embeddings", 1048576)),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
            enable_lm_head_fp32=bool(tc.get("enable_lm_head_fp32", True)),
            bos_token_id=int(tc.get("bos_token_id", 120000)),
            pad_token_id=int(tc.get("pad_token_id", 120002)),
            eos_token_id=eos_id,
            image_start_token_id=int(root.get("image_start_token_id", 120118)),
            image_end_token_id=int(root.get("image_end_token_id", 120119)),
            image_token_id=int(root.get("image_token_id", 120120)),
            image_newline_token_id=int(root.get("image_newline_token_id", 120121)),
            video_start_token_id=int(root.get("video_start_token_id", 120122)),
            video_end_token_id=int(root.get("video_end_token_id", 120123)),
            video_token_id=int(root.get("video_token_id", 120124)),
            enable_ihc=bool(tc.get("enable_ihc", True)),
            hc_mult=int(tc.get("hc_mult", 4)),
            hc_magnitude=float(tc.get("hc_magnitude", 2.0)),
            hc_eps=float(tc.get("hc_eps", 1e-6)),
            layer_types=layer_types,
            swa_pattern=swa_pattern,
            num_attention_heads=int(tc.get("num_attention_heads", 32)),
            swa_num_attention_heads=int(tc.get("swa_num_attention_heads", 64)),
            q_lora_rank=int(tc.get("q_lora_rank", 1024)),
            kv_lora_rank=int(tc.get("kv_lora_rank", 512)),
            qk_nope_head_dim=int(tc.get("qk_nope_head_dim", 192)),
            qk_rope_head_dim=int(tc.get("qk_rope_head_dim", 64)),
            qk_head_dim=int(tc.get("qk_head_dim", 256)),
            v_head_dim=int(tc.get("v_head_dim", 256)),
            sliding_window=int(tc.get("sliding_window", 512)),
            gated_mla=bool(tc.get("gated_mla", True)),
            learnable_sink=bool(tc.get("learnable_sink", True)),
            sandwich_norm=bool(tc.get("sandwich_norm", True)),
            rope_theta_dsa=float(dsa_rp.get("rope_theta", 10_000_000.0)),
            rope_theta_swa=float(swa_rp.get("rope_theta", 10_000.0)),
            mrope_section=(mrope_sec[0], mrope_sec[1], mrope_sec[2]),
            mrope_interleaved=bool(dsa_rp.get("mrope_interleaved", True)),
            use_compressed_indexer=bool(tc.get("use_compressed_indexer", True)),
            index_n_heads=int(tc.get("compressed_indexer_num_attention_heads", tc.get("index_n_heads", 16))),
            index_head_dim=int(tc.get("compressed_indexer_head_dim", tc.get("index_head_dim", 128))),
            index_topk=int(tc.get("index_topk", 2048)),
            compressed_indexer_compress_ratio=int(tc.get("compressed_indexer_compress_ratio", 4)),
            compressed_indexer_group_topk=int(tc.get("compressed_indexer_group_topk", 512)),
            compressed_indexer_candidate_topk=int(tc.get("compressed_indexer_candidate_topk", 2048)),
            first_k_dense_replace=first_k,
            mlp_layer_types=mlp_types,
            n_routed_experts=int(tc.get("n_routed_experts", 512)),
            n_shared_experts=int(tc.get("n_shared_experts", 1)),
            num_experts_per_tok=int(tc.get("num_experts_per_tok", 16)),
            routed_expert_hidden_size=int(tc.get("routed_expert_hidden_size", 1536)),
            moe_intermediate_size=int(tc.get("moe_intermediate_size", 1536)),
            routed_scaling_factor=float(tc.get("routed_scaling_factor", 3.998)),
            norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
            moe_router_use_sigmoid=bool(tc.get("moe_router_use_sigmoid", True)),
            moe_router_enable_expert_bias=bool(tc.get("moe_router_enable_expert_bias", True)),
            swiglu_limit=float(tc.get("swiglu_limit", 10.0)),
            engram_layer_ids=[int(x) for x in (tc.get("engram_layer_ids") or [])],
            engram_max_ngram_size=int(tc.get("engram_max_ngram_size", 3)),
            engram_n_embed_per_ngram=int(tc.get("engram_n_embed_per_ngram", 768)),
            engram_n_head_per_ngram=int(tc.get("engram_n_head_per_ngram", 8)),
            engram_pad_id=int(tc.get("engram_pad_id", 120002)),
            engram_seed=int(tc.get("engram_seed", 0)),
            engram_fp8_embedding=bool(tc.get("engram_fp8_embedding", False)),
            engram_layer_head_vocab_sizes=dict(tc.get("engram_layer_head_vocab_sizes") or {}),
            dfly_num_hidden_layers=dfly_layers,
            dfly_block_size=dfly_bs,
            dfly_mask_token_id=int(root.get("dfly_mask_token_id", tc.get("dfly_mask_token_id", 120009))),
            dfly_target_kv_layers=[int(x) for x in dfly_kv],
            dfly_target_index_layer=[int(x) for x in dfly_idx],
            dfly_target_index_topk=int(root.get("dfly_target_index_topk", tc.get("dfly_target_index_topk", 2048))),
            dfly_mtp_fusion=str(root.get("dfly_mtp_fusion", tc.get("dfly_mtp_fusion", "slot0"))),
            dfly_enable_hidden_correction=bool(root.get("dfly_enable_hidden_correction", True)),
            dfly_context_local_window=root.get("dfly_context_local_window"),
            quantization_config=qcfg,
            vision_config=root.get("vision_config") if isinstance(root.get("vision_config"), dict) else None,
        )
