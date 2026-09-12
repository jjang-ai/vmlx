"""Qwen4-Exp (Qwen3.8-Flash-Next) — MLX runtime.

Mirrors transformers-main modular_qwen4_exp.py. Reuses mlx_lm's battle-tested
GDN kernel, SwitchGLU experts and caches; adds:
  - mHC hyper-connections (GatedResidual, 4x2560 residual stream)
  - PLE n-gram embedding injection (layer id 2, exact int64 hashing)
  - QSA (DSA-style block-sparse indexer on full-attention layers)
  - sigmoid output gate on the GDN gated norm

Norm conventions (VERIFIED against HF source, per component):
  - Qwen4ExpTextRMSNorm family (hc_norm, ple norms, q/k norms, indexer
    layernorms): checkpoint stores weight-1 → sanitize adds +1, module is
    plain RMSNorm.
  - GDN gated norm (Qwen3NextRMSNormGated lineage): plain weight, NO shift.

The language core is shared by the text, image, and video lanes.
"""

from dataclasses import dataclass, field, replace
import logging
import os
import time
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache
from vmlx_engine.metal.qwen4_verify_sdpa import qwen4_verify_sdpa
from vmlx_engine.metal.qwen4_hc_combine import (
    exact_hc_combine,
    exact_hc_combine_requested,
)
from vmlx_engine.metal.qwen4_gdn_blocked_prefill import (
    qwen4_blocked_gated_delta_update as gated_delta_update,
)
from mlx_lm.models.qwen3_5 import GatedDeltaNet as _Qwen35GatedDeltaNet
from mlx_lm.models.qwen3_5 import TextModelArgs as _Qwen35TextArgs
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.qwen3_5.language import (
    LanguageModel as _Qwen35VlmLanguageModel,
)
from mlx_vlm.models.qwen3_5.language import (
    Qwen3_5RotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)

from vmlx_engine.models.minimax_m3.cache import (
    MiniMaxM3SparseCache as _SparseIndexerKVCache,
)
from vmlx_engine.native_mtp_prompt_priming import (
    capture_prefill,
    capture_requested,
)
from vmlx_engine.metal.qwen4_affine_moe_decode import qwen4_affine_switchglu
from vmlx_engine.metal.gated_rmsnorm_decode import (
    fused_gated_rmsnorm_requested,
    sigmoid_gated_rmsnorm_small_rows,
)
from vmlx_engine.metal.ple_conv_decode import (
    fused_ple_conv_requested,
    qwen4_ple_conv_decode,
)
from vmlx_engine.metal.gdn_conv_decode import (
    fused_gdn_conv_requested,
    qwen4_gdn_conv_decode,
)
from vmlx_engine.metal.quantized_projection_group import (
    QuantizedProjectionGroup,
    cached_quantized_projection_group,
    quantized_projection_group_reason,
)
from vmlx_engine.metal.sparse_index_score_decode import (
    fused_sparse_index_score_requested,
    sparse_index_scores_decode,
)

from .ngram import NGramHasher
from .host_profile import profile_decode_forward
from .projection_cache import validated_projection_group


logger = logging.getLogger(__name__)

_HYPER_SPLIT_INDICES = {}
_FAST_PROJECTION_CACHE = os.environ.get("VMLX_QWEN4_FAST_PROJECTION_CACHE") == "1"
_EAGER_DISPATCH_MAX_ROWS = 64


def _load_calibrated_proposal_sidecar(
    bundle_path,
    source_head,
    proposal_bits: int,
) -> Optional[Dict[str, Any]]:
    """Load the stamp-declared calibrated q4 proposal head, or None.

    Flash-Next contract: the bundle stamp's ``draft_artifact`` names a
    sha256-pinned sidecar (``mtp_draft/vmlx_mtp_proposal_head.safetensors``)
    holding ``lm_head.{weight,scales,biases}`` — the imatrix-weighted q4
    refit of the bundle's own calibrated lm_head. Validate everything
    (declared bits/group/mode, path containment, sha256, tensor keys,
    geometry against the actually-loaded head) and return the tensors;
    ANY problem returns None so the caller keeps the RTN rebuild — a bad
    sidecar can never block a load or install a mis-shaped head.
    """

    if not bundle_path:
        return None
    try:
        import hashlib
        from pathlib import Path as _Path

        # Absolute import only: this module also executes under the mlx_vlm
        # package namespace, where a relative import aborts server startup.
        from vmlx_engine.native_mtp_proposal_stamp import read_proposal_stamp

        stamp = read_proposal_stamp(bundle_path)
        if not stamp:
            return None
        artifact = stamp.get("draft_artifact")
        if not isinstance(artifact, dict):
            return None
        declared_sha = artifact.get("sha256")
        rel_file = str(artifact.get("file") or "")
        if (
            not isinstance(declared_sha, str)
            or len(declared_sha) != 64
            or not rel_file
            or int(artifact.get("bits") or 0) != proposal_bits
            or int(artifact.get("group_size") or 0) != int(source_head.group_size)
            or str(artifact.get("mode") or "affine") != "affine"
        ):
            return None

        bundle_dir = _Path(bundle_path).resolve()
        sidecar = (bundle_dir / rel_file).resolve()
        if not sidecar.is_file() or bundle_dir not in sidecar.parents:
            return None
        if hashlib.sha256(sidecar.read_bytes()).hexdigest() != declared_sha:
            logger.warning(
                "qwen4_exp proposal sidecar sha256 mismatch (%s); using RTN "
                "rebuild",
                rel_file,
            )
            return None

        tensors = mx.load(str(sidecar))
        try:
            weight = tensors["lm_head.weight"]
            scales = tensors["lm_head.scales"]
            biases = tensors["lm_head.biases"]
        except KeyError:
            return None

        group_size = int(source_head.group_size)
        vocab = int(source_head.scales.shape[0])
        hidden = int(source_head.scales.shape[1]) * group_size
        if (
            tuple(int(d) for d in weight.shape)
            != (vocab, hidden * proposal_bits // 32)
            or tuple(int(d) for d in scales.shape)
            != (vocab, hidden // group_size)
            or tuple(int(d) for d in biases.shape)
            != (vocab, hidden // group_size)
        ):
            return None
        if scales.dtype != source_head.scales.dtype:
            # Family compute-dtype contract: stray metadata dtype here would
            # re-open the fp32-promotion trap.
            scales = scales.astype(source_head.scales.dtype)
            biases = biases.astype(source_head.biases.dtype)
        return {"weight": weight, "scales": scales, "biases": biases}
    except Exception as exc:  # noqa: BLE001 - sidecar is optional, fail open
        logger.warning(
            "qwen4_exp proposal sidecar unusable (%s); using RTN rebuild", exc
        )
        return None


def _mtp_draft_head_request() -> tuple[Optional[int], str]:
    """Return the requested proposal-head width and its status.

    Default ON at 4 bits since the settled 2026-09-03 A/B: three
    back-to-back same-thermal pairs on Qwen3.8-Flash-Next-JANG_4S at fixed
    D3 all measured the q4 proposal head faster on BOTH probe lanes
    (count +2.3-3.0%, code +1.8-2.8%) — cheaper drafting outweighs its
    ~7.5pt acceptance cost, and correctness is untouched because target
    verification always uses the checkpoint-owned full head. Set
    VMLX_QWEN4_MTP_DRAFT_HEAD_BITS=0 to disable for A/B.
    """

    raw = os.environ.get(
        "VMLINUX_QWEN4_MTP_DRAFT_HEAD_BITS",
        os.environ.get("VMLX_QWEN4_MTP_DRAFT_HEAD_BITS", "4"),
    ).strip().lower()
    if raw in {"", "0", "false", "no", "off", "none"}:
        return None, "disabled"
    try:
        bits = int(raw)
    except (TypeError, ValueError):
        return None, f"invalid_requested_bits:{raw}"
    if bits != 4:
        return None, f"unsupported_requested_bits:{bits}"
    return bits, "not_built"


class _MTPDraftHeadState:
    """Non-module holder so a proposal copy never enters checkpoint traversal."""

    def __init__(self):
        self.requested_bits, self.reason = _mtp_draft_head_request()
        self.attempted = False
        self.head = None
        self.source_bits = None
        self.group_size = None
        self.mode = None
        self.build_ms = 0.0
        self.calls = 0

    def status(self) -> Dict[str, Any]:
        return {
            "configured": self.requested_bits is not None,
            "requested_bits": self.requested_bits,
            "source_bits": self.source_bits,
            "draft_bits": self.requested_bits if self.head is not None else None,
            "group_size": self.group_size,
            "mode": self.mode,
            "available": self.head is not None,
            "active_observed": self.calls > 0,
            "calls": int(self.calls),
            "build_ms": float(self.build_ms),
            "reason": self.reason,
        }


def _layer_profile_enabled(input_ids: Optional[mx.array]) -> bool:
    """Enable an explicit one-token fence after each Qwen4 phase.

    This is deliberately opt-in because the fences destroy normal graph fusion.
    It separates SSD PLE lookup, GDN/QSA, routed MoE, and residual mixing wall
    time instead of attributing the whole lazy graph to a later ``mx.eval``.
    """
    mode = os.environ.get("VMLINUX_QWEN4_PROFILE_LAYERS", "").lower()
    if mode not in {"1", "true", "yes", "on", "prefill", "all"}:
        return False
    if input_ids is None or input_ids.ndim != 2:
        return False
    seq_len = input_ids.shape[-1]
    if mode == "prefill":
        return seq_len > 1
    if mode == "all":
        return True
    return seq_len == 1


def _profile_eval(*values: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(*values)
    return (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Qwen4ExpTextArgs:
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4
    # GDN
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # mHC
    hc_count: int = 4
    hc_lowrank: int = 320
    # PLE / n-gram
    ple_layer_ids: List[int] = field(default_factory=lambda: [2])
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # rope
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    max_position_embeddings: int = 262144
    eos_token_id: int = 248044
    tie_word_embeddings: bool = False
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Qwen4ExpTextArgs":
        text = cfg.get("text_config", cfg)
        rp = text.get("rope_parameters") or {}
        qsa = text.get("sparse_attention_config") or text.get("qsa_config") or {}
        ple = text.get("ple_config") or text.get("ngram_config") or {}
        eos = text.get("eos_token_id", 248044)
        if isinstance(eos, list):
            eos = eos[0]
        kwargs = {}
        for name in cls.__dataclass_fields__:
            if name in text:
                kwargs[name] = text[name]
        kwargs["rope_theta"] = rp.get(
            "rope_theta", text.get("rope_theta", 10_000_000.0)
        )
        kwargs["partial_rotary_factor"] = rp.get(
            "partial_rotary_factor", text.get("partial_rotary_factor", 0.25)
        )
        kwargs["mrope_section"] = rp.get(
            "mrope_section", text.get("mrope_section", [11, 11, 10])
        )
        aliases = {
            "indexer_n_heads": ("indexer_n_heads", "num_attention_heads"),
            "indexer_kv_heads": ("indexer_kv_heads", "num_key_value_heads"),
            "indexer_head_dim": ("indexer_head_dim", "head_dim"),
            "indexer_budget": ("indexer_budget", "budget"),
            "indexer_compress_ratio": (
                "indexer_compress_ratio",
                "block_size",
                "compress_ratio",
            ),
        }
        for target, names in aliases.items():
            if target in text:
                continue
            for name in names:
                if name in qsa:
                    kwargs[target] = qsa[name]
                    break
        for name in (
            "ple_layer_ids",
            "ple_embed_dim",
            "ple_conv_kernel_size",
            "ngram_size",
            "heads_per_ngram",
            "ngram_vocab_size_base",
            "make_ngram_vocab_size_divisible_by",
            "seed",
            "split_ngram_parts",
        ):
            if name not in text and name in ple:
                kwargs[name] = ple[name]
        if "mtp_num_hidden_layers" not in text:
            kwargs["mtp_num_hidden_layers"] = text.get(
                "num_nextn_predict_layers",
                cfg.get("num_nextn_predict_layers", 0),
            )
        kwargs["eos_token_id"] = eos
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "Qwen4ExpTextArgs":
        return cls.from_config(cfg)

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "full_attention" if t == "qwen_sparse_attention" else t
            for t in self.layer_types
        ]
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if sum(self.mrope_section) * 2 != self.rotary_dim:
            raise ValueError(
                "qwen4_exp mrope_section must cover rotary_dim/2 frequencies: "
                f"section={self.mrope_section}, rotary_dim={self.rotary_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("qwen4_exp Q heads must divide evenly over KV heads")
        if self.indexer_budget <= 0 or self.indexer_compress_ratio <= 0:
            raise ValueError("qwen4_exp QSA budget and block size must be positive")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("qwen4_exp QSA token budget must contain whole blocks")
        if not (0 < self.num_experts_per_tok <= self.num_experts):
            raise ValueError("qwen4_exp routed expert count is invalid")
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads <= 0 or self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "qwen4_exp PLE embedding width must divide across n-gram heads"
            )
        for layer_id in self.ple_layer_ids:
            if layer_id <= 0 or layer_id > self.num_hidden_layers:
                raise ValueError(f"qwen4_exp PLE layer id is invalid: {layer_id}")
            if self.layer_types[layer_id - 1] != "linear_attention":
                raise ValueError(
                    "qwen4_exp PLE must share a GDN ArraysCache slot; "
                    f"layer {layer_id} is {self.layer_types[layer_id - 1]}"
                )


# --------------------------------------------------------------------------- #
# Rope helper (manual cos/sin so arbitrary positions work — indexer needs
# block-start positions). Non-traditional rotate-half, matches HF.
# --------------------------------------------------------------------------- #
class RopeTable:
    def __init__(self, dims: int, base: float):
        self.dims = dims
        self.inv_freq = mx.power(base, -mx.arange(0, dims, 2, dtype=mx.float32) / dims)

    def cos_sin(self, positions: mx.array):
        """positions: [S] int → cos,sin [S, dims/2] fp32"""
        freqs = positions.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        return mx.cos(freqs), mx.sin(freqs)

    def apply(self, x: mx.array, cos: mx.array, sin: mx.array, seq_axis: int = 2):
        """x: [..., S, ..., D] with rotary on first self.dims of D.
        cos/sin: [S, dims/2]; seq_axis tells where S lives (default [B,H,S,D])."""
        half = self.dims // 2
        x_rope, x_pass = x[..., : self.dims], x[..., self.dims :]
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        shape = [1] * x.ndim
        shape[seq_axis] = cos.shape[0]
        shape[-1] = half
        c = cos.reshape(shape)
        s = sin.reshape(shape)
        xt = x1.astype(mx.float32)
        yt = x2.astype(mx.float32)
        out1 = xt * c - yt * s
        out2 = yt * c + xt * s
        return mx.concatenate(
            [out1.astype(x.dtype), out2.astype(x.dtype), x_pass], axis=-1
        )


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #
class GroupedRMSNorm(nn.Module):
    """RMSNorm over groups of `group_size` along the last axis, full-width weight.
    Checkpoint weight is stored -1; sanitize shifts +1."""

    def __init__(self, dims: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps)
        return x.reshape(shape) * self.weight


class ZeroCenteredRMSNorm(nn.Module):
    """Gemma-style RMSNorm for weights stored as a delta from one.

    Qwen4-Exp uses this convention only for the MTP input-fusion pre-norms.
    Unlike the backbone's shifted norms, these checkpoint tensors must remain
    zero-centered and are applied as ``1 + weight`` at runtime.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        weight = (
            self.weight
            if getattr(self, "_offset_folded", False)
            else 1.0 + self.weight
        )
        return mx.fast.rms_norm(x, weight, self.eps)


class RMSNormGatedSigmoid(nn.Module):
    """GDN output norm: per-head RMSNorm followed by sigmoid(gate) product.
    Plain weight convention (no +1 shift)."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        self._fused_decode = fused_gated_rmsnorm_requested()

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        fused = sigmoid_gated_rmsnorm_small_rows(
            x,
            gate,
            self.weight,
            self.eps,
            output_dtype=x.dtype,
            enabled=self._fused_decode,
        )
        if fused is not None:
            return fused
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        return (normed.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(
            x.dtype
        )


# --------------------------------------------------------------------------- #
# mHC GatedResidual
# --------------------------------------------------------------------------- #
class GatedResidual(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs, use_combine: bool = True):
        super().__init__()
        self._exact_combine = exact_hc_combine_requested()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.hc_lowrank = args.hc_lowrank
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = GroupedRMSNorm(
            hc_hidden, self.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False)
        self.use_combine = use_combine

    def __call__(self, hyper_input: mx.array):
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} hyper-connection features, "
                f"got {hyper_input.shape[-1]}"
            )
        compiled_forward = getattr(self, "_compiled_forward", None)
        if (
            compiled_forward is not None
            and 1 <= hyper_input.shape[-2] <= _hc_compile_max_rows()
        ):
            return compiled_forward(hyper_input)
        return self._forward(hyper_input)

    def _forward(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        input_inject_weight = getattr(self, "input_inject_weight", None)
        if input_inject_weight is None:
            mix = self.input_mix_weight_down(normed)
            block_injection = (
                self.block_inject_weight(normed)
                if self.use_combine
                else None
            )
        else:
            combined = input_inject_weight(normed)
            mix_indices, injection_indices = _hyper_split_indices(
                self.hc_lowrank, self.hc_count
            )
            mix = mx.take(combined, mix_indices, axis=-1)
            block_injection = mx.take(combined, injection_indices, axis=-1)
        mix = nn.silu(mix / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.astype(normed.dtype)
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        mixed = (
            mix * normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        ).mean(-2).astype(hyper_input.dtype)
        if block_injection is None:
            return mixed
        inject_w = (
            2.0 * mx.sigmoid(block_injection / self.hc_count)
        ).astype(hyper_input.dtype)
        return mixed, hyper_input, inject_w

    def combine(
        self, hyper_input: mx.array, block_out: mx.array, inject_w: mx.array
    ) -> mx.array:
        if self._exact_combine:
            candidate = exact_hc_combine(
                hyper_input, block_out, inject_w, enabled=True
            )
            if candidate is not None:
                return candidate
        inj = block_out[..., None, :] * inject_w[..., :, None]
        return (hyper_input + inj.reshape(
            *inj.shape[:-2], self.hc_count * self.hidden_size
        )).astype(hyper_input.dtype)


def _hyper_split_indices(lowrank: int, hc_count: int) -> tuple[mx.array, mx.array]:
    key = (lowrank, hc_count)
    indices = _HYPER_SPLIT_INDICES.get(key)
    if indices is None:
        indices = (
            mx.arange(lowrank, dtype=mx.int32),
            mx.arange(lowrank, lowrank + hc_count, dtype=mx.int32),
        )
        _HYPER_SPLIT_INDICES[key] = indices
    return indices


def _can_fuse_hyper_connection(module: GatedResidual) -> bool:
    if hasattr(module, "input_inject_weight"):
        return False
    injection = getattr(module, "block_inject_weight", None)
    down = getattr(module, "input_mix_weight_down", None)
    if injection is None or down is None or type(down) is not type(injection):
        return False
    if down.weight.shape[1:] != injection.weight.shape[1:]:
        return False
    if down.weight.dtype != injection.weight.dtype:
        return False
    for attribute in ("group_size", "bits", "mode"):
        if getattr(down, attribute, None) != getattr(injection, attribute, None):
            return False
    for tensor_name in ("scales", "biases", "bias"):
        if hasattr(down, tensor_name) != hasattr(injection, tensor_name):
            return False
    return True


def fuse_hyper_connection_projections(model: nn.Module) -> int:
    """Fuse each mix-down/injection pair into one row-wise projection."""
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    targets = []
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, GatedResidual) and _can_fuse_hyper_connection(module):
            targets.append(module)

    for module in targets:
        down = module.input_mix_weight_down
        injection = module.block_inject_weight
        fused = {"weight": mx.concatenate([down.weight, injection.weight], axis=0)}
        for tensor_name in ("scales", "biases", "bias"):
            if hasattr(down, tensor_name):
                fused[tensor_name] = mx.concatenate(
                    [getattr(down, tensor_name), getattr(injection, tensor_name)],
                    axis=0,
                )
        mx.eval(*fused.values())
        for tensor_name, value in fused.items():
            setattr(down, tensor_name, value)
        module.input_inject_weight = down
        del module.input_mix_weight_down
        del module.block_inject_weight
    return len(targets)


def compile_hyper_connections(model: nn.Module) -> int:
    """Compile the fixed-shape single-token hyper-connection decode path."""
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    compiled = 0
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if not isinstance(module, GatedResidual) or hasattr(
            module, "_compiled_forward"
        ):
            continue
        module._compiled_forward = mx.compile(module._forward)
        compiled += 1
    return compiled


def fold_zero_centered_norm_offsets(model: nn.Module) -> int:
    """Fold trained ``1 + delta`` MTP norm weights once at load."""

    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    targets = []
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, ZeroCenteredRMSNorm) and not getattr(
            module, "_offset_folded", False
        ):
            module.weight = 1.0 + module.weight
            module._offset_folded = True
            targets.append(module.weight)
    if targets:
        mx.eval(*targets)
    return len(targets)


# --------------------------------------------------------------------------- #
# PLE (n-gram) layer
# --------------------------------------------------------------------------- #
class ShardedNGramEmbedding(nn.Module):
    """The 51B table kept as its checkpoint row-shards so a lookup only pages
    in the gathered rows (mmap-friendly); concatenating would materialize
    ~95 GiB on first use. Shard sizes follow the HF ceil-split rule."""

    def __init__(self, padded_vocab_size: int, head_dim: int, n_shards: int):
        super().__init__()
        per = -(-padded_vocab_size // n_shards)  # ceil
        self.per = per
        self.head_dim = head_dim
        self.output_dtype = None
        # Full Qwen3.8 has ~320M rows. Constructing placeholder Embeddings for
        # those rows allocates the very table that this module is designed to
        # leave on SSD. Tiny parity models retain ordinary embeddings.
        if padded_vocab_size <= 2_000_000:
            self.shards = [
                nn.Embedding(min(per, padded_vocab_size - i * per), head_dim)
                for i in range(n_shards)
                if min(per, padded_vocab_size - i * per) > 0
            ]
        else:
            self.shards = []

    def set_file_backed(self, table, output_dtype=None) -> None:
        """Install a FileBackedNGramTable so lookups read only touched pages
        (np.memmap) instead of materializing 800 MB shard tensors."""
        self._file_backed = table
        self.output_dtype = table.output_dtype if output_dtype is None else output_dtype

    def __call__(
        self,
        rows_np: np.ndarray,
        profile: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        """rows_np: int64 [B, S, H] row ids into the concatenated table."""
        fb = getattr(self, "_file_backed", None)
        if fb is not None:
            vals = (
                fb.gather_mlx(rows_np.reshape(-1))
                if profile is None
                else fb.gather_mlx(rows_np.reshape(-1), profile=profile)
            )
            if self.output_dtype is not None and vals.dtype != self.output_dtype:
                vals = vals.astype(self.output_dtype)
            return vals.reshape(*rows_np.shape, self.head_dim)
        if not self.shards:
            raise RuntimeError(
                "qwen4_exp full PLE table requires a file-backed SSD row reader"
            )
        flat = rows_np.reshape(-1)
        shard_idx = flat // self.per
        local = flat % self.per
        head_dim = self.head_dim
        out = mx.zeros((flat.shape[0], head_dim), dtype=self.shards[0].weight.dtype)
        for s in np.unique(shard_idx):
            sel = np.nonzero(shard_idx == s)[0]
            gathered = self.shards[int(s)].weight[
                mx.array(local[sel].astype(np.uint32))
            ]
            out[mx.array(sel.astype(np.uint32))] = gathered
        return out.reshape(*rows_np.shape, head_dim)


class PLELayer(nn.Module):
    """Cache slots (shared ArraysCache with the host GDN layer):
    [2] = previous `context_len` token ids (int32 [B, C])
    [3] = dilated conv state ([B, (K-1)*dilation, hc_hidden])
    """

    def __init__(self, args: Qwen4ExpTextArgs, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = args.hidden_size * args.hc_count
        self.hasher = NGramHasher(
            vocab_size=args.vocab_size,
            eos_token_id=args.eos_token_id,
            ngram_size=args.ngram_size,
            heads_per_ngram=args.heads_per_ngram,
            ngram_vocab_size_base=args.ngram_vocab_size_base,
            make_divisible_by=args.make_ngram_vocab_size_divisible_by,
            seed=args.seed,
            ple_layer_index=ple_layer_index,
        )
        head_dim = args.ple_embed_dim // self.hasher.ngram_heads
        self.ngram_embedding = ShardedNGramEmbedding(
            self.hasher.padded_vocab_size, head_dim, args.split_ngram_parts
        )
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_query = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_conv = GroupedRMSNorm(
            hc_hidden, args.hidden_size, eps=args.rms_norm_eps
        )
        self.conv_kernel_size = args.ple_conv_kernel_size
        self.conv_dilation = args.ngram_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        # depthwise dilated conv taps: checkpoint [C,1,K] → sanitized to [C,K]
        self.conv1d_weight = mx.zeros((hc_hidden, self.conv_kernel_size))
        self._fused_conv_decode = fused_ple_conv_requested()

    def _embed(
        self,
        input_ids: mx.array,
        cache,
        profile: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        started = time.perf_counter() if profile is not None else None
        ids_np = np.asarray(input_ids, dtype=np.int64)
        if cache is not None and cache[2] is not None:
            prev = np.asarray(cache[2], dtype=np.int64)
        else:
            prev = None
        rows = self.hasher.hash_tokens(ids_np, prev)  # [B, S, heads]
        if profile is not None:
            profile["hash_cpu_ms"] = (time.perf_counter() - started) * 1000.0
        emb = self.ngram_embedding(
            rows,
            profile=profile,
        )  # [B, S, heads, head_dim]
        if cache is not None:
            ctx = np.concatenate(
                [
                    prev
                    if prev is not None
                    else np.full(
                        (ids_np.shape[0], self.hasher.context_len),
                        self.hasher.eos_token_id,
                        dtype=np.int64,
                    ),
                    ids_np,
                ],
                axis=1,
            )[:, -self.hasher.context_len :]
            cache[2] = mx.array(ctx.astype(np.int32))
        return emb.reshape(*emb.shape[:-2], -1)

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        """x: [B, S, C]; state carries the previous (K-1)*dilation positions."""
        B, S, C = x.shape
        if cache is not None and cache[3] is not None:
            state = cache[3]
        else:
            state = mx.zeros((B, self.short_conv_state_len, C), dtype=x.dtype)
        fused = qwen4_ple_conv_decode(
            x,
            state,
            self.conv1d_weight,
            dilation=self.conv_dilation,
            enabled=self._fused_conv_decode,
        )
        if fused is not None:
            output, next_state = fused
            if cache is not None:
                cache[3] = next_state
            return output
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(full[:, -self.short_conv_state_len :, :])
        # taps: y[t] = sum_j w[:, j] * full[t + j*dilation], j = 0..K-1 (t in padded coords)
        taps = []
        conv_taps = getattr(self, "_conv_taps", None)
        if conv_taps is None:
            conv_taps = tuple(
                self.conv1d_weight[:, j] for j in range(self.conv_kernel_size)
            )
        for j, weight in enumerate(conv_taps):
            start = j * self.conv_dilation
            taps.append(full[:, start : start + S, :] * weight)
        return nn.silu(sum(taps))

    def prepare_runtime(self) -> None:
        """Materialize contiguous depthwise-convolution taps once at load."""

        self._conv_taps = tuple(
            mx.contiguous(self.conv1d_weight[:, j])
            for j in range(self.conv_kernel_size)
        )
        mx.eval(*self._conv_taps)

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache,
        profile: bool = False,
    ) -> mx.array:
        phases: Optional[Dict[str, float]] = {} if profile else None
        total_started = time.perf_counter() if profile else None
        emb = self._embed(input_ids, cache, profile=phases)
        projection_started = time.perf_counter() if profile else None
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc_count, self.hidden_size)
        value = self.value_proj(emb)
        query = self.norm_query(hidden_states)
        query = query.reshape(*query.shape[:-1], self.hc_count, self.hidden_size)
        if phases is not None:
            mx.eval(key, value, query)
            phases["projections_gpu_ms"] = (
                time.perf_counter() - projection_started
            ) * 1000.0
        finalize_started = time.perf_counter() if profile else None
        gate = (key * query).sum(-1, keepdims=True) / (self.hidden_size**0.5)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated_value = mx.sigmoid(gate) * value[..., None, :]
        gated_value = gated_value.reshape(*gated_value.shape[:-2], -1)
        gated_value_normed = self.norm_conv(gated_value)
        output = gated_value + self._short_conv(gated_value_normed, cache)
        if phases is not None:
            mx.eval(output)
            phases["gate_conv_gpu_ms"] = (
                time.perf_counter() - finalize_started
            ) * 1000.0
            logger.info(
                "QWEN4_PLE_PROFILE seq_len=%d total_ms=%.3f phases_ms=%s",
                input_ids.shape[-1],
                (time.perf_counter() - total_started) * 1000.0,
                ",".join(
                    f"{name}:{value:.3f}" for name, value in phases.items()
                ),
            )
        return output


# --------------------------------------------------------------------------- #
# GDN — qwen3_5 GatedDeltaNet with sigmoid output gate
# --------------------------------------------------------------------------- #
def _env_rows(name: str, default: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


_VERIFY_MAX_ROWS = 4  # depth 3 + 1 bonus row
# Receipts 2026-09-05 (chain, 4S, fixed D3, governor off, two rounds):
# rows<=4 vs rows==1 = +0.6% / +0.7% / +0.7% at 1.7k / 6.6k / 26k with
# byte-identical outputs at every context; source-matched Electron run with
# rows=4 (three connected turns to a 20k-token prompt, coherent) and the raw
# API tool probe on the same app engine (3/3 calls with arguments).
_DEFAULT_GROUP_ROWS = _VERIFY_MAX_ROWS


def _gdn_group_max_rows() -> int:
    """Widest chunk (rows per sequence) the grouped GDN projection accepts.

    Decode is one row; MTP verification runs depth+1 rows (2..4). The
    grouped QMM is bitwise the separate projections for any row count
    (affine rows are packed independently; pinned at S=1..4 for q2/q4 g64,
    f16/bf16 on synthetic geometry). Measured 2026-09-05 on Flash-Next 4S,
    governor off, interleaved x2: rows<=4 vs rows==1 at fixed D3 = +0.9% /
    +0.3% / +0.7% (1.7k / 6.6k / 26k), never negative, byte-identical
    outputs. Default 4 (verify width); VMLX_QWEN4_GDN_GROUP_MAX_ROWS=1
    restores decode-only.
    """
    return _env_rows("VMLX_QWEN4_GDN_GROUP_MAX_ROWS", _DEFAULT_GROUP_ROWS)


def _hc_compile_max_rows() -> int:
    """Widest chunk the compiled hyper-connection path accepts (default 4).
    mx.compile specializes per input shape, so admitting verify widths only
    adds one trace per width (VMLX_QWEN4_HC_COMPILE_MAX_ROWS)."""
    return _env_rows("VMLX_QWEN4_HC_COMPILE_MAX_ROWS", _DEFAULT_GROUP_ROWS)


def _decode_quantized_linears_fused(
    linears: tuple[nn.Module, ...], x: mx.array
) -> tuple[mx.array, ...] | None:
    """Run same-input affine projections as one bit-identical decode QMM.

    Affine quantization packs every output row independently. Concatenating
    compatible projections along that row axis therefore preserves every
    output bit while removing one Metal launch per additional projection.
    Qwen3.8 GDN has four such projections in each of its 36 linear-attention
    layers, so the unfused path paid 108 unnecessary launches per token.
    """
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[1] > _gdn_group_max_rows():
        return None
    if _FAST_PROJECTION_CACHE:
        group = validated_projection_group(linears, x.dtype)
        if group is not None:
            return group(x)
    if quantized_projection_group_reason(
        linears, activation_dtype=x.dtype
    ) is not None:
        return None
    group = cached_quantized_projection_group(
        linears,
        owner=linears[0],
        cache_attr="_qwen4_fused_decode_linears",
    )
    return group(x)


class GatedDeltaNet(_Qwen35GatedDeltaNet):
    def __init__(self, args: Qwen4ExpTextArgs):
        shim = _Qwen35TextArgs(
            model_type="qwen4_exp_text",
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            rms_norm_eps=args.rms_norm_eps,
            vocab_size=args.vocab_size,
            num_key_value_heads=args.num_key_value_heads,
            linear_num_value_heads=args.linear_num_value_heads,
            linear_num_key_heads=args.linear_num_key_heads,
            linear_key_head_dim=args.linear_key_head_dim,
            linear_value_head_dim=args.linear_value_head_dim,
            linear_conv_kernel_dim=args.linear_conv_kernel_dim,
            head_dim=args.head_dim,
        )
        super().__init__(shim)
        if args.output_gate_type == "sigmoid":
            self.norm = RMSNormGatedSigmoid(
                self.head_v_dim, eps=self.layer_norm_epsilon
            )
        # else: keep the inherited silu-gated norm
        self._fused_conv_decode = fused_gdn_conv_requested()

    def _process_chunk(
        self,
        qkv,
        a,
        b,
        conv_state,
        ssm_state,
        mask=None,
        lengths=None,
    ):
        batch_size, seq_len = qkv.shape[:2]
        keep = self.conv_kernel_size - 1
        fused_conv = (
            qwen4_gdn_conv_decode(
                qkv,
                conv_state,
                self.conv1d.weight,
                enabled=self._fused_conv_decode,
            )
            if lengths is None
            else None
        )
        if fused_conv is not None:
            conv_out, new_conv_state = fused_conv
        else:
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if lengths is not None:
                ends = mx.clip(lengths, 0, seq_len)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                new_conv_state = mx.take_along_axis(
                    conv_input, positions, axis=1
                )
            else:
                new_conv_state = mx.contiguous(conv_input[:, -keep:, :])
            conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            tensor.reshape(batch_size, seq_len, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, new_ssm_state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_state,
            mask,
            use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
        prefill_checkpoint_steps: tuple[int, ...] = (),
    ) -> mx.array:
        batch_size, seq_len, _ = inputs.shape
        if self.sharding_group is not None:
            if n_confirmed or prefill_checkpoint_steps:
                raise NotImplementedError(
                    "qwen4_exp MTP rollback is not implemented for sharded GDN"
                )
            return super().__call__(inputs, mask=mask, cache=cache)

        projections = (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
        )
        fused = _decode_quantized_linears_fused(projections, inputs)
        qkv, z, b, a = fused or tuple(
            projection(inputs) for projection in projections
        )
        z = z.reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        conv_state = cache[0] if cache is not None else None
        if conv_state is None or conv_state.shape[0] != batch_size:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        ssm_state = cache[1] if cache is not None else None
        if ssm_state is not None and ssm_state.shape[0] != batch_size:
            ssm_state = None
        if mask is not None:
            if mask.shape[0] != batch_size:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)

        if prefill_checkpoint_steps:
            if cache is None or n_confirmed or not (
                tuple(sorted(set(prefill_checkpoint_steps))) == prefill_checkpoint_steps
                and 0 < prefill_checkpoint_steps[0]
                and prefill_checkpoint_steps[-1] < seq_len
            ):
                raise ValueError("invalid Qwen4 prefill checkpoint positions")
            pieces = []
            checkpoints = {}
            start = 0
            conv_f, ssm_f = conv_state, ssm_state
            for end in (*prefill_checkpoint_steps, seq_len):
                piece, conv_f, ssm_f = self._process_chunk(
                    qkv[:, start:end], a[:, start:end], b[:, start:end],
                    conv_f, ssm_f,
                    mask[:, start:end] if mask is not None else None,
                )
                pieces.append(piece)
                if end in prefill_checkpoint_steps:
                    checkpoints[end] = (conv_f, ssm_f)
                start = end
            cache.prefill_checkpoint_states = checkpoints
            out = mx.concatenate(pieces, axis=1)
        elif 0 < n_confirmed < seq_len:
            confirmed_mask = mask[:, :n_confirmed] if mask is not None else None
            draft_mask = mask[:, n_confirmed:] if mask is not None else None
            out_c, conv_c, ssm_c = self._process_chunk(
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                confirmed_mask,
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)
                draft_qkv = qkv[:, n_confirmed:]
                draft_a = a[:, n_confirmed:]
                draft_b = b[:, n_confirmed:]

                def rollback_to(
                    count,
                    _q=draft_qkv,
                    _a=draft_a,
                    _b=draft_b,
                    _c=conv_c,
                    _s=ssm_c,
                    _m=draft_mask,
                    _self=self,
                ):
                    _, conv_k, ssm_k = _self._process_chunk(
                        _q[:, :count],
                        _a[:, :count],
                        _b[:, :count],
                        _c,
                        _s,
                        _m[:, :count] if _m is not None else None,
                    )
                    return conv_k, ssm_k

                cache.rollback_to = rollback_to
            out_d, conv_f, ssm_f = self._process_chunk(
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_c,
                ssm_c,
                draft_mask,
            )
            out = mx.concatenate([out_c, out_d], axis=1)
        else:
            lengths = getattr(cache, "lengths", None) if cache is not None else None
            out, conv_f, ssm_f = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            advance = getattr(cache, "advance", None)
            if callable(advance):
                advance(seq_len)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch_size, seq_len, -1))


# --------------------------------------------------------------------------- #
# QSA attention
# --------------------------------------------------------------------------- #
# QSA and MiniMax-M3's MSA have the same persistence shape: full K/V plus an
# append-only indexer payload lane. QSA's payload stores each raw index key and
# its three M-RoPE coordinates. Upstream pools raw keys first, normalizes the
# pooled block, then rotates it at the block's first token; rotating token keys
# before pooling is not equivalent. Keeping the coordinates beside the raw key
# makes that exact order restart-safe for image/video prefixes.
# Reuse the engine's already hardened
# three-lane sparse cache transport so RAM cloning, partial-block SSD storage,
# restart restore, logical-offset trimming, and validation preserve all three
# arrays together. The selector algorithm remains QSA-specific; only its typed
# storage protocol is shared.
QSACache = _SparseIndexerKVCache


_QSA_NEG_INF = mx.array(-float("inf"), dtype=mx.float32)
_QSA_ZERO = mx.array(0.0, dtype=mx.float32)


class _QSAPooledFrontier:
    """Retained pooled/normed/roped keys of the COMPLETED 4-token blocks.

    Exact by construction: a completed block's pooled key depends only on its
    own four raw index keys and the block's first-token position, neither of
    which changes once the block is complete. Lives in ``cache.derived`` so a
    trim truncates it (``truncate_to_tokens``) and any restore/replacement
    clears it (the cache's ``state`` setter). Kill switch:
    ``VMLX_QWEN4_QSA_POOL_RETAIN=0`` recomputes every call (the old path).
    """

    __slots__ = (
        "pooled", "blocks", "batch", "ratio", "reused", "recomputed", "evicted"
    )

    def __init__(self, ratio: int, batch: int) -> None:
        self.pooled: Optional[mx.array] = None  # [B, NB, D] float32
        self.blocks = 0
        self.batch = batch
        self.ratio = ratio
        self.reused = 0
        self.recomputed = 0
        self.evicted = 0

    @property
    def nbytes(self) -> int:
        return 0 if self.pooled is None else int(self.pooled.nbytes)

    def truncate_to_tokens(self, tokens: int) -> None:
        keep = max(0, int(tokens)) // self.ratio
        if keep < self.blocks:
            self.blocks = keep
            self.pooled = None if keep == 0 else self.pooled[:, :keep, :]


def _exact_mrope_cos_sin(
    rotary: "Qwen3_5RotaryEmbedding", position_ids: mx.array, dtype
) -> tuple[mx.array, mx.array]:
    """Shape-independent M-RoPE cos/sin for ``position_ids`` ``[3, B, S]``.

    The stock embedding forms ``inv_freq @ positions`` with a K=1 matmul; the
    GEMM kernel MLX picks for multi-row chunks rounds that product differently
    from the single-row GEMV (up to ~3e-4 rad at position 12), so the same
    absolute position gets different angles at prefill and at decode. The
    QSA indexer needs one angle per position regardless of chunk shape so a
    retained pooled block key is bitwise the value a full recompute produces.
    An elementwise product is the exact fp32 product for every shape.
    """
    if position_ids.ndim == 2:
        position_ids = mx.broadcast_to(
            position_ids[None, ...], (3,) + tuple(position_ids.shape)
        )
    pos = position_ids.astype(mx.float32)[..., None]  # [3, B, S, 1]
    freqs = pos * rotary.inv_freq.astype(mx.float32)  # [3, B, S, F]
    freqs = rotary.apply_interleaved_mrope(freqs, rotary.mrope_section)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _qsa_exact_rope_enabled() -> bool:
    """Exact elementwise M-RoPE angles in the QSA indexer (default on).
    ``VMLX_QWEN4_QSA_EXACT_ROPE=0`` restores the stock K=1-matmul rotary for
    old-vs-new qualification runs; retention is then disabled too because its
    parity proof depends on shape-independent angles."""
    return os.environ.get("VMLX_QWEN4_QSA_EXACT_ROPE", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _qsa_exact_rope_attn_enabled() -> bool:
    """Exact angles for the QSA ATTENTION rotary (default off: model-wide
    numerics change awaiting its own long-context quality A/B;
    ``VMLX_QWEN4_EXACT_ROPE_ATTN=1``)."""
    return os.environ.get("VMLX_QWEN4_EXACT_ROPE_ATTN", "0").strip().lower() not in {
        "", "0", "false", "no", "off"
    }


def _qsa_pool_retention_enabled() -> bool:
    if not _qsa_exact_rope_enabled():
        return False
    return os.environ.get("VMLX_QWEN4_QSA_POOL_RETAIN", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _qsa_pool_retention_max_bytes() -> int:
    """Per-cache-object cap on retained pooled bytes (default 256 MiB, which
    is ~2M tokens of one QSA layer at head_dim 128). Above it the frontier is
    evicted and that layer falls back to the exact full recompute; this only
    ever trades speed, never refuses work."""
    raw = os.environ.get("VMLX_QWEN4_QSA_POOL_RETAIN_MAX_MB", "").strip()
    try:
        mb = float(raw) if raw else 256.0
    except ValueError:
        mb = 256.0
    return int(max(0.0, mb) * 1024 * 1024)


class QSAIndexer(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + 1) * self.head_dim, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            args.rotary_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            mrope_section=args.mrope_section,
        )
        self._fused_score_decode = fused_sparse_index_score_requested(
            "qwen4_exp"
        )

    @staticmethod
    def _position_payload(position_ids: mx.array, batch: int, length: int) -> mx.array:
        """Return exact per-token M-RoPE coordinates as ``[B, S, 3]``."""
        if position_ids.ndim == 2:
            if position_ids.shape == (batch, length):
                position_ids = mx.broadcast_to(
                    position_ids[None, ...], (3, batch, length)
                )
            elif batch == 1 and position_ids.shape == (3, length):
                position_ids = position_ids[:, None, :]
            else:
                raise ValueError(
                    "qwen4_exp QSA position_ids must be [B,S], [3,S] for B=1, "
                    "or [3,B,S]"
                )
        if position_ids.ndim != 3 or position_ids.shape != (3, batch, length):
            raise ValueError(
                "qwen4_exp QSA position_ids must resolve to [3,B,S], got "
                f"{position_ids.shape}"
            )
        return position_ids.transpose(1, 2, 0).astype(mx.float32)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[QSACache],
        *,
        offset: Optional[int] = None,
        position_ids: Optional[mx.array] = None,
        return_blocks: bool = False,
    ) -> Optional[mx.array | tuple[mx.array, mx.array]]:
        """Return selected blocks, an additive [B, 1, S, T] mask, or None.

        At or below ``block_topk`` complete micro-blocks, every block is in the
        native QSA budget. Persist the raw index lane, then bypass pooling,
        scoring, and selection exactly as the upstream Qwen4 implementation
        does. Doing an ``argpartition`` whose result cannot hide anything is
        pure decode overhead.
        """
        B, S, _ = hidden_states.shape
        offset = (
            (cache.offset if cache is not None else 0) if offset is None else offset
        )

        qk = self.index_qk_proj(hidden_states)
        q, raw_k = mx.split(qk, [self.n_heads * self.head_dim], axis=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        q = self.q_layernorm(q).transpose(0, 2, 1, 3)

        if position_ids is None:
            position_ids = mx.arange(offset, offset + S)[None, :]
        if _qsa_exact_rope_enabled():
            qcos, qsin = _exact_mrope_cos_sin(self.rotary_emb, position_ids, q.dtype)
        else:
            qcos, qsin = self.rotary_emb(q, position_ids)
        q, _ = apply_multimodal_rotary_pos_emb(q, q, qcos, qsin)
        q = q.transpose(0, 2, 1, 3)

        # A single float32 payload preserves the raw projection values and
        # integer coordinates exactly through the existing three-lane cache
        # transport. Native positions are <=1M, well inside exact float32 int
        # representation. The extra three scalars are split before scoring.
        current_positions = self._position_payload(position_ids, B, S)
        payload = mx.concatenate(
            [raw_k.astype(mx.float32), current_positions], axis=-1
        )

        if cache is not None:
            all_payload = cache.update_index(payload[:, None, :, :])[:, 0, :, :]
        else:
            all_payload = payload
        all_keys = all_payload[..., : self.head_dim]
        all_positions = all_payload[..., self.head_dim :]
        T = all_payload.shape[1]

        # pooled block keys (block b = tokens [4b, 4b+4)). QSA selects 512
        # complete blocks. At or below that count every complete block plus
        # the current incomplete tail is visible, so building scores and an
        # argpartition is both redundant and contrary to the upstream Qwen4
        # fast path.
        num_blocks = T // self.compress_ratio
        if num_blocks <= self.block_topk:
            return None
        pooled = self._pooled_block_keys(cache, all_keys, all_positions, num_blocks, B)

        # scores: relu(q·k) summed over heads / sqrt(D) → [B, S, NB]
        scores = sparse_index_scores_decode(
            q.astype(mx.float32),
            pooled.astype(mx.float32),
            family="qwen4_exp",
            scale=self.head_dim**-0.5,
            enabled=self._fused_score_decode,
        )
        if scores is None:
            scores = mx.einsum(
                "bshd,bnd->bshn",
                q.astype(mx.float32),
                pooled.astype(mx.float32),
            )
            scores = mx.maximum(scores, 0.0).sum(axis=2) / (
                self.head_dim**0.5
            )

        # per query i (absolute pos p = offset+i): visible tokens 0..p
        #   complete blocks for that query: ncb(p) = (p+1)//ratio
        #   keep: topk(block_topk) among blocks [0, ncb) + tail tokens [ncb*ratio, p]
        # Built on device: no per-call NumPy upload for the query positions.
        ncb_mx = (mx.arange(offset + 1, offset + S + 1) // self.compress_ratio)  # [S]
        block_ids = mx.arange(num_blocks)
        complete = block_ids[None, :] < ncb_mx[:, None]  # [S, NB]
        masked_scores = mx.where(complete[None], scores, _QSA_NEG_INF)

        k_sel = min(self.block_topk, num_blocks)
        top_idx = mx.argpartition(-masked_scores, kth=k_sel - 1, axis=-1)[
            ..., :k_sel
        ]  # [B,S,k]
        if return_blocks:
            # The direct native consumer reads a valid chronological prefix.
            # Sorting preserves selection and puts incomplete blocks last.
            selected = mx.sort(top_idx[0], axis=-1).astype(mx.int32)
            return selected, selected < ncb_mx[:, None]
        keep_blocks = mx.zeros((B, S, num_blocks), dtype=mx.bool_)
        keep_blocks = mx.put_along_axis(keep_blocks, top_idx, mx.array(True), axis=-1)
        # queries with fewer complete blocks than k_sel picked -inf entries; drop those
        keep_blocks = keep_blocks & complete[None]

        # expand to tokens
        keep_tokens = mx.repeat(keep_blocks[..., None], self.compress_ratio, axis=-1)
        keep_tokens = keep_tokens.reshape(B, S, num_blocks * self.compress_ratio)
        if T > num_blocks * self.compress_ratio:
            pad = mx.ones((B, S, T - num_blocks * self.compress_ratio), dtype=mx.bool_)
            keep_tokens = mx.concatenate([keep_tokens, pad], axis=-1)
        # tail tokens (incomplete block for THIS query) always visible:
        token_ids = mx.arange(T)
        tail = token_ids[None, :] >= (ncb_mx * self.compress_ratio)[:, None]  # [S, T]
        keep_tokens = keep_tokens | tail[None]

        return mx.where(keep_tokens[:, None], _QSA_ZERO, _QSA_NEG_INF)


    def _pool_blocks(
        self, keys: mx.array, positions: mx.array, first_block: int, last_block: int, B: int
    ) -> mx.array:
        """Pool, normalize and rotate blocks [first_block, last_block) → [B, n, D]."""
        r = self.compress_ratio
        n = last_block - first_block
        pooled = (
            keys[:, first_block * r : last_block * r, :]
            .reshape(B, n, r, self.head_dim)
            .astype(mx.float32)
            .mean(axis=2)
        )
        pooled = self.k_layernorm(pooled[:, :, None, :]).transpose(0, 2, 1, 3)
        block_positions = positions[
            :, first_block * r : last_block * r : r, :
        ].transpose(2, 0, 1)
        if _qsa_exact_rope_enabled():
            block_cos, block_sin = _exact_mrope_cos_sin(
                self.rotary_emb, block_positions, pooled.dtype
            )
        else:
            block_cos, block_sin = self.rotary_emb(pooled, block_positions)
        pooled, _ = apply_multimodal_rotary_pos_emb(
            pooled, pooled, block_cos, block_sin
        )
        return pooled[:, 0, :, :]

    def _pooled_block_keys(
        self,
        cache: Optional[QSACache],
        all_keys: mx.array,
        all_positions: mx.array,
        num_blocks: int,
        B: int,
    ) -> mx.array:
        """Completed-block pooled keys, reusing the retained frontier when the
        cache is the single-sequence QSA cache (batch caches carry per-row
        offsets; they take the full recompute)."""
        frontier = None
        if (
            cache is not None
            and type(cache) is _SparseIndexerKVCache
            and _qsa_pool_retention_enabled()
        ):
            frontier = cache.derived.get("qsa_pooled")
            if frontier is None or frontier.batch != B or frontier.ratio != self.compress_ratio:
                frontier = _QSAPooledFrontier(self.compress_ratio, B)
                cache.derived["qsa_pooled"] = frontier
        if frontier is None:
            return self._pool_blocks(all_keys, all_positions, 0, num_blocks, B)
        have = frontier.blocks if frontier.pooled is not None else 0
        if have > num_blocks:
            # The raw lane shrank without a trim hook firing: never trust the
            # frontier past the raw history.
            frontier.truncate_to_tokens(num_blocks * self.compress_ratio)
            have = frontier.blocks
        if have == num_blocks:
            frontier.reused += num_blocks
            return frontier.pooled
        new = self._pool_blocks(all_keys, all_positions, have, num_blocks, B)
        frontier.recomputed += num_blocks - have
        frontier.reused += have
        pooled = new if have == 0 else mx.concatenate([frontier.pooled, new], axis=1)
        if num_blocks * B * self.head_dim * 4 > _qsa_pool_retention_max_bytes():
            # Size eviction: return the exact answer, drop the retained state.
            frontier.evicted += 1
            frontier.pooled = None
            frontier.blocks = 0
            return pooled
        frontier.pooled = pooled
        frontier.blocks = num_blocks
        return frontier.pooled


class QSAAttention(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            args.rotary_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            mrope_section=args.mrope_section,
        )
        self.qkv_group = None

    def prepare_runtime(self) -> bool:
        """Replace compatible packed q/k/v rows with one exact projection."""

        linears = (self.q_proj, self.k_proj, self.v_proj)
        if quantized_projection_group_reason(linears) is not None:
            return False
        group = QuantizedProjectionGroup(linears)
        mx.eval(group.weight, group.scales, group.biases)
        self.qkv_group = group
        self.q_proj = self.k_proj = self.v_proj = None
        return True

    def _project_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if self.qkv_group is not None:
            return self.qkv_group(x)
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[QSACache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        offset = cache.offset if cache is not None else 0

        qg, key_states, value_states = self._project_qkv(x)
        qg = qg.reshape(B, S, self.num_heads, 2 * self.head_dim)
        queries, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        keys = key_states.reshape(B, S, self.num_kv_heads, self.head_dim)
        values = value_states.reshape(B, S, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if position_ids is None:
            position_ids = mx.arange(offset, offset + S)[None, :]
        if _qsa_exact_rope_attn_enabled():
            # Opt-in numerical correction for the attention rotary (see
            # _exact_mrope_cos_sin): the stock K=1 matmul rounds the angle
            # with ~1e-3 RELATIVE error for any chunk of >= 2 rows, so prefill
            # keys and decode queries disagree by radians on the highest
            # frequencies at long positions. Separate A/B campaign.
            cos, sin = _exact_mrope_cos_sin(self.rotary_emb, position_ids, values.dtype)
        else:
            cos, sin = self.rotary_emb(values, position_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        T = keys.shape[2]

        # Append K/V before the raw indexer lane. The shared sparse transport
        # validates and slices both at the post-append logical offset; pass the
        # saved pre-append offset for this query chunk's absolute positions.
        direct_prefill = False
        if (
            os.environ.get("VMLX_QWEN4_PREFILL_DIRECT", "0") == "1"
            and S >= 256
            and B == 1
            and not self.training
            and self.indexer.compress_ratio == 4
            and self.indexer.block_topk == 512
            # Materialized sparse QK wins from 8K on the qualified host.
            # Shorter contexts retain stock attention to avoid a regression.
            and T >= 8192
        ):
            from vmlx_engine.metal.qwen4_prefill_direct import (
                qsa_prefill_direct,
                qsa_prefill_direct_supported,
                qsa_prefill_direct_ready,
            )
            # Shape-only placeholders: no selector or cache mutation until
            # every static consumer guard and the native pipeline is ready.
            block_shape = mx.zeros((S, 512), dtype=mx.int32)
            direct_prefill = qsa_prefill_direct_supported(
                queries, keys, values, block_shape, block_shape == 0,
                pos_start=offset, total_tokens=T, scale=self.scale,
            ) and qsa_prefill_direct_ready()
        index_mask = self.indexer(
            x,
            cache,
            offset=offset,
            position_ids=position_ids,
            return_blocks=direct_prefill,
        )
        if direct_prefill:
            selected, valid = index_mask
            out = qsa_prefill_direct(
                queries, keys, values, selected, valid,
                pos_start=offset, total_tokens=T, scale=self.scale,
            )
            if not getattr(self, "_direct_prefill_logged", False):
                logger.info(
                    "Powered by MTPLX/oMLX: direct sparse QSA prefill enabled "
                    "https://github.com/youssofal/mtplx"
                )
                self._direct_prefill_logged = True
            out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
            return self.o_proj(out * mx.sigmoid(gate))
        if index_mask is not None and index_mask.dtype != queries.dtype:
            # The indexer scores/selects in F32 for numerical stability, but
            # MLX requires an additive SDPA mask that promotes to the Q/K/V
            # output dtype. Keep the persisted sparse state in F32 and cast
            # only the completed mask at the attention boundary.
            index_mask = index_mask.astype(queries.dtype)

        # At single-token decode the cache contains no future keys, so the
        # causal mask is identically zero. Avoid allocating that F32 tensor.
        if S == 1:
            full_mask = index_mask
        else:
            q_pos = mx.arange(offset, offset + S)[:, None]
            k_pos = mx.arange(T)[None, :]
            causal = mx.where(k_pos <= q_pos, 0.0, -np.inf).astype(queries.dtype)
            causal = causal[None, None]
            full_mask = causal if index_mask is None else causal + index_mask

        out = qwen4_verify_sdpa(
            queries, keys, values, full_mask, scale=self.scale,
            selected_token_bound=(
                self.indexer.block_topk * self.indexer.compress_ratio
                + self.indexer.compress_ratio - 1
                if type(self.indexer) is QSAIndexer
                and T // self.indexer.compress_ratio > self.indexer.block_topk
                else None
            ),
            selected_four_token_block_bound=(
                self.indexer.block_topk + 1
                if type(self.indexer) is QSAIndexer
                and self.indexer.compress_ratio == 4
                and T // self.indexer.compress_ratio > self.indexer.block_topk
                else None
            ),
        )
        if out is None:
            out = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=full_mask,
            )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #
class SharedExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.gate_up_group = None

    def prepare_runtime(self) -> bool:
        """Replace compatible packed gate/up rows with one exact projection."""

        linears = (self.gate_proj, self.up_proj)
        if quantized_projection_group_reason(linears) is not None:
            return False
        group = QuantizedProjectionGroup(linears)
        mx.eval(group.weight, group.scales, group.biases)
        self.gate_up_group = group
        self.gate_proj = self.up_proj = None
        return True

    def __call__(self, x):
        if self.gate_up_group is not None:
            gate, up = self.gate_up_group(x)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(nn.silu(gate) * up)


def prepare_quantized_projection_groups(model: nn.Module) -> Dict[str, int]:
    """Prepare exact QSA and shared-expert groups across backbone and MTP."""

    prepared = {"qsa_qkv": 0, "shared_gate_up": 0}
    modules = [model]
    modules.extend(module for _, module in model.named_modules() if module is not model)
    seen: set[int] = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, QSAAttention) and module.prepare_runtime():
            prepared["qsa_qkv"] += 1
        elif isinstance(module, SharedExpertMLP) and module.prepare_runtime():
            prepared["shared_gate_up"] += 1
    return prepared


class _RouteOverlapProbe:
    """Diagnostic (VMLX_QWEN4_MOE_ROUTE_OVERLAP_LOG=1): for multi-row calls
    (MTP verification, S = depth + 1) count distinct routed experts against
    rows x top_k on a bounded sample of calls, and log a running summary.
    A small-row expert kernel can only reuse packed weights across rows to
    the extent routes overlap; this measures that before any kernel work.
    Costs one host readback per sampled call, so it is never on by default."""

    def __init__(self) -> None:
        self.calls = 0
        self.sampled = 0
        self.rows = 0
        self.slots = 0
        self.distinct = 0
        self.by_rows: Dict[int, List[int]] = {}

    def observe(self, inds: mx.array) -> None:
        self.calls += 1
        if self.calls % 16 != 1:  # sample 1 in 16 multi-row calls
            return
        arr = np.asarray(inds).reshape(-1, inds.shape[-1])
        rows, top_k = arr.shape
        distinct = int(np.unique(arr).size)
        self.sampled += 1
        self.rows += rows
        self.slots += rows * top_k
        self.distinct += distinct
        acc = self.by_rows.setdefault(rows, [0, 0])
        acc[0] += distinct
        acc[1] += rows * top_k
        if self.sampled % 32 == 0:
            per_rows = ", ".join(
                f"S={r}: {d}/{sl} ({100.0 * d / max(1, sl):.0f}%)"
                for r, (d, sl) in sorted(self.by_rows.items())
            )
            logger.info(
                "qwen4_exp MoE route overlap: %d sampled multi-row calls, "
                "distinct experts / (rows x top_k) = %d/%d (%.0f%%); %s",
                self.sampled,
                self.distinct,
                self.slots,
                100.0 * self.distinct / max(1, self.slots),
                per_rows,
            )


_ROUTE_OVERLAP_PROBE: Optional[_RouteOverlapProbe] = (
    _RouteOverlapProbe()
    if os.environ.get("VMLX_QWEN4_MOE_ROUTE_OVERLAP_LOG", "0").strip().lower()
    not in {"0", "false", "no", "off"}
    else None
)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = SharedExpertMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        profile_phases: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if profile_phases is not None:
            profile_phases["moe_router"] = _profile_eval(inds, scores)
        if _ROUTE_OVERLAP_PROBE is not None and x.ndim == 3 and x.shape[1] > 1:
            _ROUTE_OVERLAP_PROBE.observe(inds)

        routed, fused_reduction = qwen4_affine_switchglu(
            self.switch_mlp, x, inds, scores
        )
        if profile_phases is not None:
            profile_phases["moe_routed"] = _profile_eval(routed)
            profile_phases["moe_reduce"] = 0.0 if fused_reduction else 0.0

        shared_gate = mx.sigmoid(self.shared_expert_gate(x))
        if profile_phases is not None:
            profile_phases["moe_shared_gate"] = _profile_eval(shared_gate)

        shared = self.shared_expert(x)
        if profile_phases is not None:
            profile_phases["moe_shared"] = _profile_eval(shared)

        output = routed + shared_gate * shared
        if profile_phases is not None:
            profile_phases["moe_finalize"] = _profile_eval(output)
        return output


# --------------------------------------------------------------------------- #
# Decoder layer / model
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = QSAAttention(args)
        self.mlp = SparseMoeBlock(args)
        if (layer_idx + 1) in args.ple_layer_ids:
            self.ple = PLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
        else:
            self.ple = None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(
        self,
        h,
        mask=None,
        cache=None,
        input_ids=None,
        position_ids=None,
        profile_layer: Optional[int] = None,
        n_confirmed: int = 0,
        prefill_checkpoint_steps: tuple[int, ...] = (),
        last_token_only: bool = False,
    ):
        phase_ms: Dict[str, float] = {}
        if self.ple is not None:
            if cache is not None and prefill_checkpoint_steps:
                pieces = []
                checkpoints = {}
                start = 0
                for end in (*prefill_checkpoint_steps, h.shape[1]):
                    piece = h[:, start:end]
                    pieces.append(piece + self.ple(
                        piece, input_ids[:, start:end], cache,
                        profile=profile_layer is not None,
                    ))
                    if end in prefill_checkpoint_steps:
                        checkpoints[end] = (cache[2], cache[3])
                    start = end
                cache.prefill_checkpoint_aux_states = checkpoints
                h = mx.concatenate(pieces, axis=1)
            elif cache is not None and 0 < n_confirmed < h.shape[1]:
                confirmed_h = h[:, :n_confirmed]
                draft_h = h[:, n_confirmed:]
                confirmed_ids = input_ids[:, :n_confirmed]
                draft_ids = input_ids[:, n_confirmed:]
                confirmed_h = confirmed_h + self.ple(
                    confirmed_h,
                    confirmed_ids,
                    cache,
                    profile=profile_layer is not None,
                )
                ple_context = cache[2]
                ple_conv = cache[3]
                cache.rollback_aux_state = (ple_context, ple_conv)

                def rollback_aux_to(
                    count,
                    _h=draft_h,
                    _ids=draft_ids,
                    _context=ple_context,
                    _conv=ple_conv,
                    _cache=cache,
                    _ple=self.ple,
                ):
                    _cache[2] = _context
                    _cache[3] = _conv
                    if count:
                        _ple(_h[:, :count], _ids[:, :count], _cache)
                    return _cache[2], _cache[3]

                cache.rollback_aux_to = rollback_aux_to
                draft_h = draft_h + self.ple(
                    draft_h,
                    draft_ids,
                    cache,
                    profile=profile_layer is not None,
                )
                h = mx.concatenate([confirmed_h, draft_h], axis=1)
            else:
                h = h + self.ple(
                    h,
                    input_ids,
                    cache,
                    profile=profile_layer is not None,
                )
            if profile_layer is not None:
                phase_ms["ple"] = _profile_eval(h)

        x, hyper, inject = self.attn_hyper_connection(h)
        if profile_layer is not None:
            phase_ms["attn_hc"] = _profile_eval(x)
        if self.is_linear:
            r = self.linear_attn(
                x, mask=mask, cache=cache, n_confirmed=n_confirmed,
                prefill_checkpoint_steps=prefill_checkpoint_steps,
            )
        else:
            r = self.self_attn(
                x,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
            )
        if profile_layer is not None:
            phase_ms["gdn" if self.is_linear else "qsa"] = _profile_eval(r)
        h = self.attn_hyper_connection.combine(hyper, r, inject)
        if profile_layer is not None:
            phase_ms["attn_combine"] = _profile_eval(h)

        # Attention/recurrent cache updates are complete for every token.
        # At the final layer only the last output is needed for next-token
        # logits; all remaining operations are independent per token.
        if last_token_only:
            h = h[:, -1:, :]
        x, hyper, inject = self.mlp_hyper_connection(h)
        if profile_layer is not None:
            phase_ms["mlp_hc"] = _profile_eval(x)
        r = self.mlp(x, phase_ms if profile_layer is not None else None)
        h = self.mlp_hyper_connection.combine(hyper, r, inject)
        if profile_layer is not None:
            phase_ms["mlp_combine"] = _profile_eval(h)
            logger.info(
                "QWEN4_LAYER_PROFILE layer=%d type=%s total_ms=%.3f phases_ms=%s",
                profile_layer,
                "gdn" if self.is_linear else "qsa",
                sum(phase_ms.values()),
                ",".join(f"{name}:{value:.3f}" for name, value in phase_ms.items()),
            )
        return h


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.args = args
        # Qualification only: submit completed small layer graphs while the
        # caller builds the next layer (including host-only PLE SSD reads).
        # Never create a worker/stream here or change logical cache update order.
        self._eager_dispatch = os.environ.get("VMLX_QWEN4_EAGER_DISPATCH") == "1"
        self._eager_dispatch_logged = False
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.fa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear),
            0,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        position_ids=None,
        return_expanded: bool = False,
        n_confirmed: int = 0,
        prefill_checkpoint_steps: tuple[int, ...] = (),
        last_token_only: bool = False,
        **_kwargs,
    ):
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(inputs)
        h = mx.tile(h, (1, 1, self.args.hc_count))
        if prefill_checkpoint_steps:
            if (
                n_confirmed
                or cache is None
                or tuple(sorted(set(prefill_checkpoint_steps))) != prefill_checkpoint_steps
                or not 0 < prefill_checkpoint_steps[0]
                or prefill_checkpoint_steps[-1] >= inputs.shape[1]
            ):
                raise ValueError("Invalid Qwen4 prefill checkpoint request")
            h = self._checkpointed_layers(
                h, inputs, cache, position_ids, prefill_checkpoint_steps,
                last_token_only,
            )
            mixed = self.hyper_connection_mixer(h)
            return (mixed, h) if return_expanded else mixed
        profile = _layer_profile_enabled(inputs)
        if profile:
            logger.info(
                "QWEN4_LAYER_PROFILE begin seq_len=%d",
                inputs.shape[-1],
            )
        if cache is None:
            cache = [None] * len(self.layers)
        _layer_fp = _layer_fingerprint_enabled(inputs)
        eager_dispatch = (
            self._eager_dispatch
            and not profile
            and not _layer_fp
            and 0 < inputs.shape[0] * inputs.shape[1] <= _EAGER_DISPATCH_MAX_ROWS
        )
        if eager_dispatch and not self._eager_dispatch_logged:
            logger.info(
                "Qwen early layer submission active: rows=%d max_rows=%d "
                "layers=%d stream=caller checkpointed=false",
                inputs.shape[0] * inputs.shape[1],
                _EAGER_DISPATCH_MAX_ROWS,
                len(self.layers),
            )
            self._eager_dispatch_logged = True
        if _layer_fp:
            _log_layer_fingerprint(-1, h, cache[0] if cache else None)  # input to layer 0
            if _contiguous_state_experiment_enabled():
                _materialize_recurrent_state(cache)
                logger.info("QWEN4_LAYER_FP contiguous-state experiment applied before step %d", _LAYER_FP_STEPS["n"])
        for layer_index, (layer, c) in enumerate(zip(self.layers, cache)):
            if _layer_fp and layer_index < 2:
                _log_layer_fingerprint(-10 - layer_index, h, c)  # PRE: state before this layer runs
            h = layer(
                h,
                mask=None,
                cache=c,
                input_ids=inputs,
                position_ids=position_ids,
                profile_layer=layer_index if profile else None,
                n_confirmed=n_confirmed,
                prefill_checkpoint_steps=prefill_checkpoint_steps,
                last_token_only=last_token_only and layer_index == len(self.layers) - 1,
            )
            if eager_dispatch:
                # No wait, queue, retained activation cache, or arithmetic
                # substitution. Dependencies remain on the caller's MLX stream;
                # cache consumers and terminal durability fences stay unchanged.
                mx.async_eval(h)
            if _layer_fp:
                _log_layer_fingerprint(layer_index, h, c)
                if layer_index < 2:
                    _log_module_state(layer_index, layer)
        mixed = self.hyper_connection_mixer(h)
        if profile:
            mixer_ms = _profile_eval(mixed)
            logger.info("QWEN4_LAYER_PROFILE final_mixer_ms=%.3f", mixer_ms)
        return (mixed, h) if return_expanded else mixed

    def _checkpointed_layers(
        self, hidden, inputs, cache, position_ids, boundaries, last_token_only
    ):
        """Keep stock matrix shapes while scheduling all segments per layer.

        Changing projection or attention row counts changes low-precision
        reduction order. Retaining the original segments keeps recurrent and
        PLE checkpoints exact while sharing one lazy model graph.
        """
        total = inputs.shape[1]
        for index, (layer, current) in enumerate(zip(self.layers, cache)):
            pieces, states, aux_states = [], {}, {}
            start = 0
            final_layer = index == len(self.layers) - 1
            for end in (*boundaries, total):
                piece = layer(
                    hidden[:, start:end],
                    cache=current,
                    input_ids=inputs[:, start:end],
                    position_ids=(
                        position_ids[..., start:end] if position_ids is not None else None
                    ),
                    # Earlier final-layer pointwise outputs do not affect
                    # caches or the requested final segment's logits.
                    last_token_only=bool(final_layer and last_token_only and end != total),
                )
                pieces.append(piece)
                if end in boundaries and type(current).__name__ == "ArraysCache":
                    states[end] = tuple(current.cache[:2])
                    if len(current.cache) == 4:
                        aux_states[end] = tuple(current.cache[2:4])
                start = end
            if states:
                current.prefill_checkpoint_states = states
                if aux_states:
                    current.prefill_checkpoint_aux_states = aux_states
            hidden = (
                pieces[-1] if final_layer and last_token_only
                else mx.concatenate(pieces, axis=1)
            )
        return hidden


_LAYER_FP_STEPS = {"n": 0}


def _layer_fingerprint_enabled(inputs) -> bool:
    """Opt-in (VMLX_DIAG_RESTORE_FINGERPRINT=1): per-layer hidden-state
    fingerprints for the first few single-token decode steps of the process,
    so the first layer whose output differs between two runs can be read off
    the log."""
    import os

    if os.environ.get("VMLX_DIAG_RESTORE_FINGERPRINT") not in ("1", "true", "True", "yes", "on"):
        return False
    try:
        if int(inputs.shape[-1]) != 1:
            _LAYER_FP_STEPS["n"] = 0  # a prefill starts a new request's trace
            return False
    except Exception:
        return False
    if _LAYER_FP_STEPS["n"] >= 3:
        return False
    _LAYER_FP_STEPS["n"] += 1
    return True


def _contiguous_state_experiment_enabled() -> bool:
    import os

    return os.environ.get("VMLX_DIAG_CONTIGUOUS_STATE") in ("1", "true", "True", "yes", "on")


def _materialize_recurrent_state(cache) -> None:
    """Experiment: rewrite every recurrent state array as a contiguous,
    evaluated array before the decode step, on whichever path runs."""
    for c in cache or []:
        state = getattr(c, "cache", None)
        if isinstance(state, list):
            new = []
            for a in state:
                if a is None:
                    new.append(None)
                else:
                    b = mx.contiguous(a)
                    mx.eval(b)
                    new.append(b)
            c.cache = new


def _log_module_state(layer_index: int, layer) -> None:
    """Array-valued attributes of the layer module (and its direct children)
    that are not registered parameters: carries, scratch buffers, caches."""
    try:
        import hashlib
        import numpy as np

        found = []
        def scan(mod, prefix, depth):
            if depth > 2:
                return
            for k, v in vars(mod).items():
                if k.startswith("__"):
                    continue
                if hasattr(v, "shape") and not k in ("weight", "bias", "scales", "biases"):
                    found.append(f"{prefix}{k}:{list(v.shape)}:{hashlib.sha256(np.asarray(v).tobytes()).hexdigest()[:8]}")
                elif isinstance(v, (int, float, bool, str)) and k.startswith("_"):
                    found.append(f"{prefix}{k}={v}")
                elif hasattr(v, "__dict__") and not hasattr(v, "shape") and depth < 2 and not isinstance(v, (list, dict)):
                    scan(v, prefix + k + ".", depth + 1)
        scan(layer, "", 0)
        logger.info("QWEN4_LAYER_MODSTATE layer=%d %s", layer_index, " ".join(found)[:1500])
    except Exception as exc:  # noqa: BLE001
        logger.info("QWEN4_LAYER_MODSTATE failed at layer %d: %s", layer_index, exc)


def _log_layer_fingerprint(layer_index: int, h, c) -> None:
    try:
        import hashlib
        import numpy as np

        arr = np.asarray(h.astype(mx.float32))
        sha = hashlib.sha256(np.asarray(h).tobytes()).hexdigest()[:12]
        phys = []
        for name in ("keys", "values", "idx_keys"):
            v = getattr(c, name, None)
            if v is not None and hasattr(v, "shape"):
                phys.append(f"{name}{list(v.shape)}")
        state = getattr(c, "cache", None)
        if isinstance(state, list):
            phys.append("state[" + ",".join(str(list(a.shape)) for a in state if a is not None) + "]")
            for j, a in enumerate(state):
                if a is None:
                    phys.append(f"s{j}=None")
                    continue
                aa = np.asarray(a.astype(mx.float32)) if a.dtype != mx.float32 else np.asarray(a)
                phys.append(f"s{j}:{a.dtype} sum={float(aa.astype(np.float64).sum()):.6e} sha={hashlib.sha256(np.asarray(a).tobytes()).hexdigest()[:10]}")
        attrs = {k: v for k, v in vars(c).items() if not hasattr(v, "shape") and not isinstance(v, (list, dict))}
        phys.append("attrs=" + repr(attrs)[:160])
        for k, v in vars(c).items():
            if k == "cache":
                continue
            if hasattr(v, "shape"):
                phys.append(f"attr.{k}:{v.dtype}{list(v.shape)} sha={hashlib.sha256(np.asarray(v).tobytes()).hexdigest()[:10]}")
            elif isinstance(v, (list, tuple)) and v and all(hasattr(x, "shape") for x in v if x is not None):
                phys.append(f"attr.{k}=[" + ",".join(f"{list(x.shape)}:{hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:8]}" for x in v if x is not None) + "]")
        phys.append("keys=" + ",".join(sorted(vars(c).keys())))
        logger.info(
            "QWEN4_LAYER_FP step=%d layer=%d cache=%s idx=%s offset=%s phys=%s h: sum=%.6e max=%.6e sha=%s",
            _LAYER_FP_STEPS["n"], layer_index, type(c).__name__, getattr(c, "_idx", None), getattr(c, "offset", None),
            " ".join(phys), float(arr.astype(np.float64).sum()), float(np.abs(arr).max()), sha,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("QWEN4_LAYER_FP failed at layer %d: %s", layer_index, exc)


class MTPModule(nn.Module):
    """Trained qwen4_exp next-token head with its own full QSA/MoE layer."""

    def __init__(self, args: Qwen4ExpTextArgs):
        super().__init__()
        self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(
            args.hc_count * args.hidden_size, eps=args.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        mtp_args = replace(
            args,
            num_hidden_layers=args.mtp_num_hidden_layers,
            layer_types=["full_attention"] * args.mtp_num_hidden_layers,
            ple_layer_ids=[],
        )
        self.layers = [
            DecoderLayer(mtp_args, i) for i in range(args.mtp_num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)

    def fuse_inputs(self, hidden_states, token_embeddings):
        """Preserve all four trained mHC branches through MTP input fusion."""
        embedded = self.fc_embedding(self.pre_fc_norm_embedding(token_embeddings))
        hidden = self.pre_fc_norm_hidden(hidden_states)
        original_shape = hidden.shape
        hidden = hidden.reshape(
            *hidden.shape[:-1],
            self.hyper_connection_mixer.hc_count,
            self.hyper_connection_mixer.hidden_size,
        )
        hidden = self.fc_hidden(hidden)
        return (embedded[..., None, :] + hidden).reshape(original_shape)

    def __call__(
        self,
        hidden_states,
        next_token_ids,
        embed_tokens,
        cache=None,
        return_expanded: bool = False,
    ):
        fused = self.fuse_inputs(hidden_states, embed_tokens(next_token_ids))
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            fused = layer(
                fused,
                mask=None,
                cache=layer_cache,
                input_ids=next_token_ids,
            )
        mixed = self.hyper_connection_mixer(fused)
        return (mixed, fused) if return_expanded else mixed


class LanguageModel(nn.Module):
    """mlx-vlm language-model interface for the qwen4_exp text core."""

    def __init__(self, args: Qwen4ExpTextArgs, config=None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        self._position_ids = None
        self._rope_deltas = None
        if args.mtp_num_hidden_layers > 0:
            self.mtp = MTPModule(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._mtp_draft_head_state = _MTPDraftHeadState()

    def prepare_mtp_draft_head(self) -> Dict[str, Any]:
        """Build an opt-in lower-bit head used only for MTP proposals.

        The target/full-model path always retains ``self.lm_head``. Exact
        speculative correction therefore continues to verify against the
        checkpoint-owned head; this copy can only change proposal cost.
        """

        state = self._mtp_draft_head_state
        if state.attempted or state.requested_bits is None:
            return state.status()
        state.attempted = True
        source = getattr(self, "lm_head", None)
        state.source_bits = getattr(source, "bits", None)
        state.group_size = getattr(source, "group_size", None)
        state.mode = getattr(source, "mode", None)
        if not isinstance(source, nn.QuantizedLinear):
            state.reason = "unsupported_source_type"
            return state.status()

        # One-time per-bundle eligibility check, persisted as a stamp so
        # every later launch (and other agents reading the bundle) honors
        # the same verdict without re-deriving. An existing stamp whose
        # recorded source layout still matches is authoritative.
        # Absolute imports only: this module is also executed under the
        # mlx_vlm package namespace at load, where a relative ...native_mtp
        # resolves to mlx_vlm.native_mtp and aborts server startup
        # (ModuleNotFoundError, found live on Flash-Next 4S 2026-09-03).
        from vmlx_engine.native_mtp import native_mtp_active_model_path
        from vmlx_engine.native_mtp_proposal_stamp import (
            resolve_proposal_head_plan,
        )

        plan = resolve_proposal_head_plan(
            native_mtp_active_model_path(),
            {
                "bits": state.source_bits,
                "group_size": state.group_size,
                "mode": state.mode,
                "tied": bool(getattr(self.args, "tie_word_embeddings", False)),
            },
            family="qwen4_exp",
        )
        if not plan.get("eligible"):
            state.reason = str(plan.get("reason") or "stamped_ineligible")
            return state.status()

        # Calibrated sidecar first (Flash-Next contract): when the stamp's
        # draft_artifact points at the imatrix-refit q4 head (under the
        # bundle's mtp_draft/ subfolder — the root holds no tensor files
        # except model shards), sha256-verify and use it. ANY failure —
        # missing file, sha mismatch, bad keys, wrong geometry — falls back
        # silently to the RTN rebuild. Verification always uses the
        # checkpoint head and the depth policy is unchanged, so the artifact
        # can only shape proposals, never outputs.
        stamped_tensors = _load_calibrated_proposal_sidecar(
            native_mtp_active_model_path(),
            source,
            int(state.requested_bits or 4),
        )

        started = time.perf_counter()
        try:
            if stamped_tensors is not None:
                weight = stamped_tensors["weight"]
                scales = stamped_tensors["scales"]
                biases = stamped_tensors["biases"]
            else:
                dense = mx.dequantize(
                    source.weight,
                    source.scales,
                    source.biases,
                    group_size=source.group_size,
                    bits=source.bits,
                    mode=source.mode,
                )
                weight, scales, biases = mx.quantize(
                    dense,
                    group_size=source.group_size,
                    bits=state.requested_bits,
                    mode=source.mode,
                )
            # Construct only a tiny shell, then replace all generated arrays.
            # The real output/input geometry lives in the quantized tensors.
            proposal = nn.QuantizedLinear(
                64,
                64,
                bias=False,
                group_size=source.group_size,
                bits=state.requested_bits,
                mode=source.mode,
            )
            proposal.weight = weight
            proposal.scales = scales
            proposal.biases = biases
            mx.eval(proposal.weight, proposal.scales, proposal.biases)
            state.head = proposal
            state.reason = (
                "ready_sidecar" if stamped_tensors is not None else "ready"
            )
            state.build_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "qwen4_exp MTP proposal head ready (%s): q%d/g%d -> q%d/g%d "
                "build_ms=%.2f",
                "sidecar, sha-verified"
                if stamped_tensors is not None
                else "RTN rebuild",
                source.bits,
                source.group_size,
                state.requested_bits,
                source.group_size,
                state.build_ms,
            )
        except Exception as exc:  # noqa: BLE001 - optional path must fail closed
            state.head = None
            state.reason = f"build_failed:{type(exc).__name__}"
            state.build_ms = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "qwen4_exp MTP proposal head disabled after build failure: %s",
                exc,
            )
        return state.status()

    def mtp_draft_head_status(self) -> Dict[str, Any]:
        return self._mtp_draft_head_state.status()

    @profile_decode_forward
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids")
        cache_offset = 0
        if cache:
            anchor = cache[self.model.fa_idx]
            cache_offset = int(getattr(anchor, "offset", 0) or 0)

        if position_ids is None:
            # The vision wrapper computes exact 3-axis M-RoPE positions before
            # chunked prefill. Slice that retained plan at the logical cache
            # offset; never derive media history from a scalar token count.
            if self._position_ids is not None:
                stop = cache_offset + inputs.shape[1]
                if stop <= self._position_ids.shape[-1]:
                    position_ids = self._position_ids[:, :, cache_offset:stop]

            if position_ids is None:
                batch_size, seq_length = inputs.shape
                if self._rope_deltas is not None and cache_offset > 0:
                    delta = mx.array(cache_offset + self._rope_deltas)
                    if delta.ndim == 0:
                        delta = delta[None]
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]
                    text_pos = mx.arange(seq_length)[None, :]
                    text_pos = mx.broadcast_to(text_pos, (batch_size, seq_length))
                    text_pos = text_pos + delta.reshape(batch_size, 1)
                    position_ids = mx.broadcast_to(
                        text_pos[None, ...],
                        (3, batch_size, seq_length),
                    )
                else:
                    position_ids = mx.arange(
                        cache_offset,
                        cache_offset + seq_length,
                    )[None, :]
                    position_ids = mx.broadcast_to(
                        position_ids, (batch_size, seq_length)
                    )

        hidden, expanded_hidden = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            return_expanded=True,
            n_confirmed=int(kwargs.get("n_confirmed", 0) or 0),
            prefill_checkpoint_steps=tuple(kwargs.get("prefill_checkpoint_steps", ())),
            last_token_only=(
                bool(kwargs.get("prefill_last_logits_only", False))
                and not return_hidden and not capture_requested(self)
            ),
        )
        # Prompt-history priming is armed by the scheduler only for an active
        # native-MTP request.  Capture normal prompt forwards, never the
        # return_hidden seed/verify forwards that advance speculative state.
        if (
            not return_hidden
            and inputs_embeds is None
            and capture_requested(self)
        ):
            capture_prefill(self, inputs, expanded_hidden, cache)
        if not return_logits:
            return (hidden, expanded_hidden) if return_hidden else hidden
        logit_hidden = (
            hidden[:, -1:, :]
            if kwargs.get("prefill_last_logits_only", False) and not return_hidden
            else hidden
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logit_hidden)
        else:
            logits = self.lm_head(logit_hidden)
        if return_hidden:
            return logits, expanded_hidden
        return LanguageModelOutput(logits=logits)

    get_rope_index = _Qwen35VlmLanguageModel.get_rope_index

    def make_cache(self):
        caches = []
        for i, t in enumerate(self.args.layer_types):
            if t == "linear_attention":
                size = 4 if (i + 1) in self.args.ple_layer_ids else 2
                caches.append(ArraysCache(size=size))
            else:
                caches.append(QSACache())
        return caches

    def mtp_forward(
        self, hidden_states, next_token_ids, mtp_cache, return_hidden=False
    ):
        mtp_hidden, expanded_hidden = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
            return_expanded=True,
        )
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(mtp_hidden)
        else:
            state = self._mtp_draft_head_state
            if state.requested_bits is not None and not state.attempted:
                self.prepare_mtp_draft_head()
            head = state.head if state.head is not None else self.lm_head
            logits = head(mtp_hidden)
            if state.head is not None:
                state.calls += 1
        return (logits, expanded_hidden) if return_hidden else logits

    def make_mtp_cache(self):
        if not hasattr(self, "mtp"):
            return []
        return [QSACache() for _layer in self.mtp.layers]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            # Router scores and recurrent-state coefficients stay in their
            # checkpoint dtype. Quantizing either changes expert selection or
            # destabilizes the GDN state update.
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            return not (path.endswith("A_log") or path.endswith("dt_bias"))

        return predicate


# Text-only callers in mlx-lm style still import ``Model``.
Model = LanguageModel
