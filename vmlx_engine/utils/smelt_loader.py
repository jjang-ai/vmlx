# SPDX-License-Identifier: Apache-2.0
"""smelt_loader.py — Smelt mode components for partial expert loading.

Contains two main components:

1. ExpertIndex — safetensors expert location scanner
   Scans safetensors file headers to map MoE expert weight locations on disk
   (file path, byte offset, shape, dtype) WITHOUT loading any weight data.

   Supports all expert key naming conventions across 6 model families:
     - backbone.layers.N.mixer.switch_mlp.{up_proj,down_proj}.*
         Nemotron (2-proj SwitchMLP, mixer parent)
     - model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*
         Qwen 3.5, Mistral 4 (mlp parent)
     - model.layers.N.block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}.*
         MiniMax M2.5 (block_sparse_moe parent)
     - model.language_model.layers.N.switch_mlp.{gate_proj,up_proj,down_proj}.*
         Gemma 4 (language_model prefix, no mlp parent)
     - model.language_model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*
         Mistral VLM variant (language_model + mlp parent)

   Each projection (gate_proj, up_proj, down_proj / up_proj, down_proj for
   Nemotron) is tracked for all three tensor suffixes: .weight, .scales, .biases.

2. TurboRouteWrapper — wraps an existing MoE block and injects cache-bias
   routing so that the model prefers loaded experts. Expert compute is
   delegated to the NATIVE compiled SwitchGLU/SwitchMLP kernel — no custom
   matmul, baseline speed.

   Three routing styles:
     softmax    — Qwen 3.5, Mistral 4, Gemma 4 (default)
     sigmoid    — MiniMax M2.5, GLM-5 (e_score_correction_bias + normalize)
     pre_routed — Nemotron Cascade/Super (gate returns (indices, scores)
                  directly; cache_bias NOT injected here — Nemotron's compiled
                  gate owns it)
"""

from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# ExpertIndex — Part 1: Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TensorInfo:
    """Location and metadata for a single tensor in a safetensors file."""

    file_path: Path
    abs_offset: int  # 8 (header-size field) + header_size + data_offsets[0]
    num_bytes: int  # data_offsets[1] - data_offsets[0]
    shape: List[int]
    dtype: str  # safetensors dtype string, e.g. "U32", "F16", "BF16"

    @property
    def total_bytes(self) -> int:
        return self.num_bytes


@dataclass
class ProjectionTensors:
    """All tensors (.weight, .scales, .biases) for one projection."""

    weight: Optional[TensorInfo] = None
    scales: Optional[TensorInfo] = None
    biases: Optional[TensorInfo] = None

    @property
    def total_bytes(self) -> int:
        total = 0
        for t in (self.weight, self.scales, self.biases):
            if t is not None:
                total += t.total_bytes
        return total

    def all_tensors(self) -> List[TensorInfo]:
        """Return all non-None TensorInfo objects."""
        return [t for t in (self.weight, self.scales, self.biases) if t is not None]


@dataclass
class LayerExpertInfo:
    """Expert weight locations for a single transformer layer.

    Nemotron uses 2-projection SwitchMLP (up_proj + down_proj only, no gate).
    All other supported families use 3-projection (gate_proj + up_proj + down_proj).
    """

    layer_idx: int
    gate_proj: Optional[ProjectionTensors] = None
    up_proj: Optional[ProjectionTensors] = None
    down_proj: Optional[ProjectionTensors] = None

    @property
    def total_bytes(self) -> int:
        total = 0
        for proj in (self.gate_proj, self.up_proj, self.down_proj):
            if proj is not None:
                total += proj.total_bytes
        return total

    @property
    def num_experts(self) -> Optional[int]:
        """Infer expert count from the first available weight tensor shape[0]."""
        for proj in (self.gate_proj, self.up_proj, self.down_proj):
            if proj is not None and proj.weight is not None:
                return proj.weight.shape[0]
        return None


@dataclass
class ExpertIndex:
    """Complete expert weight map for a model.

    Attributes:
        layers: Per-layer expert weight locations, keyed by layer index.
        model_path: Directory the model was loaded from.
        num_experts: Expert count inferred from first expert tensor shape[0].
        num_moe_layers: Count of layers that have expert weights.
        expert_size_bytes: Total bytes occupied by expert weights.
        backbone_bytes: Total bytes for non-expert (backbone) weights.
    """

    layers: Dict[int, LayerExpertInfo] = field(default_factory=dict)
    model_path: Optional[Path] = None
    num_experts: int = 0
    num_moe_layers: int = 0
    expert_size_bytes: int = 0
    backbone_bytes: int = 0

    @classmethod
    def build(cls, path: "str | Path") -> "ExpertIndex":
        """Scan a model directory and build an ExpertIndex.

        Reads only safetensors file headers (first 8 + header_size bytes) —
        no weight data is loaded into memory.

        Args:
            path: Path to the model directory.

        Returns:
            ExpertIndex populated with all expert tensor locations.

        Raises:
            FileNotFoundError: If no safetensors files are found.
        """
        return _build_expert_index(Path(path))


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertIndex — Part 2: Key pattern matching
# ═══════════════════════════════════════════════════════════════════════════════

# Regex captures: (layer_idx, proj_name, tensor_suffix)
# Handles all 5 path structures described in the module docstring.
#
# Named groups:
#   layer_nem  — Nemotron: backbone.layers.N.mixer.switch_mlp
#   layer_mm   — MiniMax: *.layers.N.block_sparse_moe.switch_mlp
#   layer_g4   — Gemma 4: *.language_model.layers.N.switch_mlp (no mlp)
#   layer_qw   — Qwen/Mistral: *.layers.N.mlp.switch_mlp
#   proj       — gate_proj | up_proj | down_proj
#   suffix     — weight | scales | biases
#
# Order matters: Gemma 4 pattern (no mlp parent) MUST come before the
# generic mlp pattern to avoid false matches against mlp.switch_mlp paths.

_EXPERT_KEY_RE = re.compile(
    r"(?:"
    # 1. Nemotron: backbone.layers.N.mixer.switch_mlp
    r"backbone\.layers\.(?P<layer_nem>\d+)\.mixer\.switch_mlp"
    r"|"
    # 2. MiniMax: *.layers.N.block_sparse_moe.switch_mlp
    r"(?:[^.]+\.)*layers\.(?P<layer_mm>\d+)\.block_sparse_moe\.switch_mlp"
    r"|"
    # 3. Gemma 4: *.language_model.layers.N.switch_mlp (no mlp parent)
    #    Must precede the mlp pattern to avoid partial matches.
    r"(?:[^.]+\.)*language_model\.layers\.(?P<layer_g4>\d+)\.switch_mlp"
    r"|"
    # 4. Qwen/Mistral text + Mistral VLM: *.layers.N.mlp.switch_mlp
    r"(?:[^.]+\.)*layers\.(?P<layer_qw>\d+)\.mlp\.switch_mlp"
    r")"
    r"\."
    r"(?P<proj>gate_proj|up_proj|down_proj|fc1|fc2)"
    r"\."
    r"(?P<suffix>weight|scales|biases)"
    r"$",
)


_FC_TO_PROJ = {"fc1": "up_proj", "fc2": "down_proj"}


def _match_expert_key(key: str) -> Optional[Tuple[int, str, str]]:
    """Return (layer_idx, proj_name, suffix) if *key* matches an expert pattern.

    Returns None if the key does not match any known expert weight pattern.
    Nemotron uses fc1/fc2 for SwitchMLP — these are normalized to up_proj/down_proj.
    """
    m = _EXPERT_KEY_RE.match(key)
    if not m:
        return None
    # Pick whichever layer group matched (exactly one will be non-None)
    layer_str = (
        m.group("layer_nem")
        or m.group("layer_mm")
        or m.group("layer_g4")
        or m.group("layer_qw")
    )
    proj = m.group("proj")
    # Normalize Nemotron fc1/fc2 → up_proj/down_proj
    proj = _FC_TO_PROJ.get(proj, proj)
    return int(layer_str), proj, m.group("suffix")


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertIndex — Part 3: Safetensors header reader (no weight loading)
# ═══════════════════════════════════════════════════════════════════════════════


def _read_safetensors_header(file_path: Path) -> Tuple[int, dict]:
    """Read safetensors header metadata without loading weight data.

    Safetensors format:
      - Bytes 0..7  : little-endian uint64 = header_size (number of JSON bytes)
      - Bytes 8..8+header_size-1 : UTF-8 JSON header
      - Bytes 8+header_size.. : raw tensor data

    Each tensor entry in the header: {"dtype": str, "shape": [...], "data_offsets": [start, end]}
    Absolute byte offset for a tensor = 8 + header_size + data_offsets[0].

    Returns:
        (header_size, header_dict)
    """
    with open(file_path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"File too small to be safetensors: {file_path}")
        header_size = struct.unpack("<Q", raw)[0]
        header_bytes = f.read(header_size)

    header = json.loads(header_bytes)
    return header_size, header


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertIndex — Part 4: Index builder
# ═══════════════════════════════════════════════════════════════════════════════


def _get_safetensors_files(model_path: Path) -> List[Path]:
    """Return all *.safetensors files in *model_path*, sorted by name."""
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    return files


def _build_expert_index(model_path: Path) -> ExpertIndex:
    """Core implementation for ExpertIndex.build()."""
    st_files = _get_safetensors_files(model_path)
    logger.debug(
        "ExpertIndex: scanning %d safetensors files in %s",
        len(st_files),
        model_path,
    )

    layers: Dict[int, LayerExpertInfo] = {}
    total_expert_bytes = 0
    total_backbone_bytes = 0

    # Audit-2026-04-07 risk §6.6: defensive overlap tracking. We collect every
    # mtp.* key we exclude AND every key the expert regex matches, and at the end
    # assert no key ended up in both sets. The garbled-output bug we already fixed
    # was caused by mtp.layers.0.mlp.switch_mlp.gate_proj.weight masquerading as a
    # real expert; this assertion fails loud if a future model reintroduces that
    # class of regression.
    _excluded_mtp_keys: set[str] = set()
    _matched_expert_keys: set[str] = set()

    for st_file in st_files:
        try:
            header_size, header = _read_safetensors_header(st_file)
        except Exception as e:
            logger.warning("ExpertIndex: skipping %s — %s", st_file.name, e)
            continue

        # Absolute data region start = 8 (the header-size uint64) + header_size
        data_region_start = 8 + header_size

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(meta, dict):
                continue
            # Skip MTP (multi-token prediction) keys — they have the same
            # layers.N.mlp.switch_mlp pattern but are NOT the model's MoE
            # experts. Without this filter, mtp.layers.0.* OVERWRITES the
            # real model.language_model.layers.0.* in the ExpertIndex.
            if key.startswith("mtp."):
                total_backbone_bytes += (
                    meta.get("data_offsets", [0, 0])[1]
                    - meta.get("data_offsets", [0, 0])[0]
                )
                _excluded_mtp_keys.add(key)
                continue

            data_offsets = meta.get("data_offsets")
            if data_offsets is None or len(data_offsets) != 2:
                continue

            shape = meta.get("shape", [])
            dtype = meta.get("dtype", "")
            byte_start, byte_end = data_offsets
            num_bytes = byte_end - byte_start
            abs_offset = data_region_start + byte_start

            match = _match_expert_key(key)
            if match is None:
                # Non-expert weight — count toward backbone bytes
                total_backbone_bytes += num_bytes
                continue

            layer_idx, proj_name, suffix = match
            _matched_expert_keys.add(key)
            total_expert_bytes += num_bytes

            # Ensure layer entry exists
            if layer_idx not in layers:
                layers[layer_idx] = LayerExpertInfo(layer_idx=layer_idx)

            layer_info = layers[layer_idx]

            # Get or create ProjectionTensors for this projection
            proj_obj: Optional[ProjectionTensors] = getattr(layer_info, proj_name, None)
            if proj_obj is None:
                proj_obj = ProjectionTensors()
                setattr(layer_info, proj_name, proj_obj)

            tensor_info = TensorInfo(
                file_path=st_file,
                abs_offset=abs_offset,
                num_bytes=num_bytes,
                shape=list(shape),
                dtype=dtype,
            )

            setattr(proj_obj, suffix, tensor_info)

    # Infer num_experts from first available MoE layer
    num_experts = 0
    for layer_info in sorted(layers.values(), key=lambda li: li.layer_idx):
        n = layer_info.num_experts
        if n is not None:
            num_experts = n
            break

    num_moe_layers = len(layers)
    logger.info(
        "ExpertIndex built: %d MoE layers, %d experts/layer, "
        "expert=%.2fGB backbone=%.2fGB",
        num_moe_layers,
        num_experts,
        total_expert_bytes / 1e9,
        total_backbone_bytes / 1e9,
    )

    # Audit-2026-04-07 risk §6.6: assert mtp.* and expert keys are disjoint.
    # Loud failure here means the smelt MTP regression bug has resurfaced.
    _overlap = _excluded_mtp_keys & _matched_expert_keys
    if _overlap:
        raise RuntimeError(
            f"smelt: ExpertIndex overlap between mtp.* keys and expert keys "
            f"({len(_overlap)} keys, e.g. {sorted(_overlap)[:3]}). This means "
            f"the regex match is treating an MTP weight as a real expert and "
            f"will overwrite the real model.language_model.layers.*. The garbled "
            f"output bug from session 2026-04 has reproduced. Refusing to load."
        )
    if _excluded_mtp_keys:
        logger.debug(
            "ExpertIndex: excluded %d mtp.* keys (no overlap)", len(_excluded_mtp_keys)
        )

    return ExpertIndex(
        layers=layers,
        model_path=model_path,
        num_experts=num_experts,
        num_moe_layers=num_moe_layers,
        expert_size_bytes=total_expert_bytes,
        backbone_bytes=total_backbone_bytes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TurboRouteWrapper (Task 3)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Supported MoE class names (used by _detect_routing_style / _find_moe_block)
# ---------------------------------------------------------------------------
_MOE_CLASSES = {
    # Qwen family
    "Qwen3NextSparseMoeBlock",  # Qwen 3.5 (qwen3_next.py)
    "Qwen3MoeSparseMoeBlock",  # Qwen 3 (qwen3_moe.py)
    "Qwen2MoeSparseMoeBlock",  # Qwen 2 (qwen2_moe.py)
    # MiniMax
    "MiniMaxSparseMoeBlock",  # MiniMax M2.5 (minimax.py)
    # Nemotron
    "NemotronHMoE",  # Nemotron Cascade/Super (nemotron_h.py)
    # Mistral
    "Mistral4MoE",  # Mistral Small 4 (mistral4.py)
    # DeepSeek / GLM-5 / GLM-5.1 (glm_moe_dsa inherits from deepseek_v32)
    "DeepseekV2MoE",  # DeepSeek V2 (deepseek_v2.py)
    "DeepseekV32MoE",  # DeepSeek V3 / GLM-5.1 (deepseek_v32.py, glm_moe_dsa.py)
    "MoE",  # GLM-4 MoE (glm4_moe.py)
    "Glm4MoeLiteMoE",  # GLM-4 MoE Lite (glm4_moe_lite.py)
    # Generic
    "SparseMoeBlock",  # dbrx, bailing_moe_linear, etc.
}


# ---------------------------------------------------------------------------
# Helper: detect routing style from class name
# ---------------------------------------------------------------------------


def _detect_routing_style(moe_block: nn.Module) -> str:
    """Return the routing style for *moe_block* based on its class name.

    Returns one of "softmax", "sigmoid", or "pre_routed".
    """
    class_name = type(moe_block).__name__
    if "MiniMax" in class_name:
        return "sigmoid"
    # DeepSeek V2/V3, GLM-5/5.1 (glm_moe_dsa), GLM-4 MoE, Nemotron —
    # gate returns (indices, scores) directly via compiled routing
    if (
        "NemotronH" in class_name
        or "Deepseek" in class_name
        or class_name in ("MoE", "Glm4MoeLiteMoE")
    ):
        return "pre_routed"
    return "softmax"


# ---------------------------------------------------------------------------
# Helper: find the MoE sub-block inside a transformer layer
# ---------------------------------------------------------------------------


def _find_moe_block(layer: nn.Module):
    """Return ``(block, attr_name)`` for the MoE sub-block inside *layer*.

    Tries the following attribute names in order:
      block_sparse_moe, mlp, mixer

    Also accepts a layer that *is itself* a SwitchMLP/SwitchGLU block
    (direct switch_mlp attribute).

    Returns ``(None, None)`` if no MoE block is found.
    """
    for attr in ("block_sparse_moe", "mlp", "mixer"):
        candidate = getattr(layer, attr, None)
        if candidate is None:
            continue
        class_name = type(candidate).__name__
        if class_name in _MOE_CLASSES:
            return candidate, attr
        # Fallback: anything that owns a switch_mlp is treated as MoE
        if hasattr(candidate, "switch_mlp"):
            return candidate, attr

    # Layer itself might be a direct MoE block (rare, but be defensive)
    if type(layer).__name__ in _MOE_CLASSES or hasattr(layer, "switch_mlp"):
        return layer, None

    return None, None


# ---------------------------------------------------------------------------
# TurboRouteWrapper
# ---------------------------------------------------------------------------


class TurboRouteWrapper(nn.Module):
    """Wraps an existing MoE block, injects cache_bias routing, remaps indices.

    Delegates expert compute to the NATIVE compiled SwitchGLU/SwitchMLP path
    so there is no speed penalty compared to a fully-loaded model.

    Args:
        original:       The original MoE block (e.g. Qwen3NextSparseMoeBlock).
        remap:          Integer array of shape ``(num_loaded_experts,)`` mapping
                        *global* expert index → *local slot* index.  Pass
                        ``None`` when all experts are loaded (identity mapping).
        cache_bias:     Float array of shape ``(num_experts,)`` where unloaded
                        experts receive ``-1000`` and loaded experts receive
                        ``0``.  Routing argpartition will strongly prefer loaded
                        experts.
        routing_style:  One of "softmax", "sigmoid", or "pre_routed".
    """

    def __init__(
        self,
        original: nn.Module,
        remap,
        cache_bias,
        routing_style: str = "softmax",
    ):
        super().__init__()
        self.original = original
        self.remap = remap
        self.cache_bias = cache_bias
        self.routing_style = routing_style

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        orig = self.original

        # Resolve top-k (different models use different attribute names)
        ne: int = getattr(
            orig,
            "num_experts_per_tok",
            getattr(orig, "top_k", getattr(orig, "num_activated_experts", 8)),
        )

        # ------------------------------------------------------------------
        # 1. Routing
        # ------------------------------------------------------------------
        if self.routing_style == "sigmoid":
            # MiniMax M2.5 / GLM-5 style
            gates = orig.gate(x.astype(mx.float32))
            ss = mx.sigmoid(gates)
            sel = ss
            ecb = getattr(orig, "e_score_correction_bias", None)
            if ecb is not None:
                sel = sel + ecb
            sel = sel + self.cache_bias
            inds = mx.argpartition(-sel, kth=ne - 1, axis=-1)[..., :ne]
            # Unbiased scores for weighting, then normalize
            scores = mx.take_along_axis(ss, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = scores.astype(x.dtype)

        elif self.routing_style == "pre_routed":
            if self.remap is None:
                # Full coverage — use native compiled gate (no overhead)
                inds, scores = orig.gate(x)
            else:
                # Partial coverage — re-implement routing with cache_bias.
                # DeepSeek V3 / GLM-5.1: sigmoid + e_score_correction_bias +
                # group selection + cache_bias. Nemotron: similar but simpler.
                gate = orig.gate
                raw_logits = x @ gate.weight.T
                orig_scores = scores = mx.sigmoid(raw_logits.astype(mx.float32))
                ecb = getattr(gate, "e_score_correction_bias", None)
                if ecb is not None:
                    scores = scores + ecb
                # Group selection (DeepSeek/GLM n_group > 1)
                n_group = getattr(gate, "n_group", 1)
                topk_group = getattr(gate, "topk_group", 1)
                if n_group > 1:
                    scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
                    group_scores = mx.topk(scores, 2, axis=-1).sum(
                        axis=-1, keepdims=True
                    )
                    gk = n_group - topk_group
                    group_idx = mx.argpartition(group_scores, kth=gk - 1, axis=-2)[
                        ..., :gk, :
                    ]
                    scores = mx.put_along_axis(
                        scores,
                        mx.stop_gradient(group_idx),
                        mx.array(0.0),
                        axis=-2,
                    )
                    scores = mx.flatten(scores, -2, -1)
                # Inject cache_bias AFTER group selection, BEFORE expert selection
                scores = scores + self.cache_bias
                inds = mx.argpartition(-scores, kth=ne - 1, axis=-1)[..., :ne]
                # Unbiased scores for weighting
                scores = mx.take_along_axis(orig_scores, inds, axis=-1)
                rsf = getattr(gate, "routed_scaling_factor", 1.0)
                ntp = getattr(gate, "norm_topk_prob", True)
                if ne > 1 and ntp:
                    scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
                scores = scores * rsf

        else:
            # Softmax — Qwen 3.5, Mistral 4, Gemma 4 (default)
            # Matches turbosmelt v12 exactly: softmax first, bias on probs
            gates = orig.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            orig_gates = gates
            gates = gates + self.cache_bias
            inds = mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne]
            scores = mx.take_along_axis(orig_gates, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            rsf = getattr(orig, "routed_scaling_factor", 1.0)
            scores = scores * rsf
            # Gemma 4: per_expert_scale applied after normalization
            pes = getattr(orig, "per_expert_scale", None)
            if pes is not None:
                scores = scores * pes[inds]

        # ------------------------------------------------------------------
        # 2. Remap global → local slot indices
        # ------------------------------------------------------------------
        local_inds = self.remap[inds] if self.remap is not None else inds

        # ------------------------------------------------------------------
        # 3. Optional latent projections (Nemotron latent MoE)
        # ------------------------------------------------------------------
        x_expert = x
        fc1 = getattr(orig, "fc1_latent_proj", None)
        fc2 = getattr(orig, "fc2_latent_proj", None)
        if fc1 is not None:
            x_expert = fc1(x)

        # ------------------------------------------------------------------
        # 4. NATIVE SwitchGLU / SwitchMLP forward — compiled, full speed
        # ------------------------------------------------------------------
        y = orig.switch_mlp(x_expert, local_inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if fc2 is not None:
            y = fc2(y)

        # ------------------------------------------------------------------
        # 5. Shared expert contribution
        # ------------------------------------------------------------------
        shared = getattr(orig, "shared_expert", getattr(orig, "shared_experts", None))
        if shared is not None:
            seg = getattr(orig, "shared_expert_gate", None)
            if seg is not None:
                y = y + mx.sigmoid(seg(x)) * shared(x)
            else:
                y = y + shared(x)

        return y


# ═══════════════════════════════════════════════════════════════════════════════
# smelt_load — main Smelt loading function
# ═══════════════════════════════════════════════════════════════════════════════

import gc
import os
import time

import numpy as np

# Safetensors dtype → numpy dtype mapping for expert pread
_NP_DTYPE_MAP = {
    "U32": np.uint32,
    "F16": np.float16,
    "BF16": np.float16,
    "F32": np.float32,
    "I32": np.int32,
    "U8": np.uint8,
}


def _load_expert_subset(ti: TensorInfo, indices: list) -> mx.array:
    """Load a subset of experts from a safetensors file via numpy pread.

    Reads the full [num_experts, ...] tensor from disk using numpy fromfile
    with offset+count, then slices to the selected expert indices.
    For full coverage (all experts selected), returns the entire tensor.
    """
    np_dtype = _NP_DTYPE_MAP.get(ti.dtype, np.uint8)
    # Audit-2026-04-07 risk §6.10 hardening: fail loud with a clear message
    # when the safetensors file is truncated or the tensor lives in a missing
    # shard. Without this, np.fromfile silently returns a partial/empty
    # buffer and the downstream reshape raises a cryptic "cannot reshape
    # array of size N into shape (...)" error that obscures the real cause
    # (incomplete model download — partial JANG shard). See P0.1 retest
    # finding on Nemotron-Cascade-2-30B-A3B-JANG_2L 2026-04-08.
    try:
        file_size = os.path.getsize(ti.file_path)
    except OSError:
        file_size = 0
    expected_end = int(ti.abs_offset) + int(ti.num_bytes)
    if file_size == 0 or expected_end > file_size:
        raise RuntimeError(
            f"smelt: tensor data missing or truncated. "
            f"file='{getattr(ti.file_path, 'name', ti.file_path)}' "
            f"file_size={file_size} bytes, but tensor requires bytes "
            f"{ti.abs_offset}..{expected_end} (shape={ti.shape}, "
            f"dtype={ti.dtype}). This usually means the model shards are "
            f"incomplete — verify all .safetensors files in the model "
            f"directory are fully downloaded."
        )
    data = np.fromfile(
        ti.file_path, dtype=np.uint8, count=ti.num_bytes, offset=ti.abs_offset
    )
    if data.size != ti.num_bytes:
        raise RuntimeError(
            f"smelt: short read on tensor data. expected {ti.num_bytes} "
            f"bytes at offset {ti.abs_offset} in "
            f"'{getattr(ti.file_path, 'name', ti.file_path)}', got "
            f"{data.size} bytes (shape={ti.shape}, dtype={ti.dtype})."
        )
    full_np = data.view(np_dtype).reshape(ti.shape)
    if len(indices) >= ti.shape[0]:
        return mx.array(full_np)
    subset = full_np[np.array(indices)]
    return mx.array(subset)


def _get_layers_list(model):
    """Find the transformer layers list from a model object."""
    for accessor in [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.backbone.layers,
    ]:
        try:
            return accessor(model)
        except AttributeError:
            continue
    raise ValueError("Could not find model layers")


def _is_gemma4_layer(layer) -> bool:
    """Check if a layer uses Gemma 4's separate router + switch_mlp pattern.

    Gemma 4 has layer.router + layer.switch_mlp as siblings (not nested
    inside an MoE block). The router is a separate Router class that computes
    top-k indices/weights, then passes them to switch_mlp.
    """
    return (
        hasattr(layer, "router")
        and hasattr(layer, "experts")
        and not hasattr(layer, "block_sparse_moe")
        and not hasattr(layer, "mixer")
    )


def smelt_load(
    model_path: str,
    expert_percent: int = 50,
) -> tuple:
    """Load a JANG MoE model in Smelt mode (partial expert loading).

    Uses jang_tools.loader.load_jang_model (the proven turbosmelt v12 approach)
    with two monkey-patches:
    1. mx.load: filters expert weights (switch_mlp/switch_glu) from safetensors
    2. mx.eval: deferred until after expert subset fill (prevents 120GB peak RAM)

    jang_tools handles ALL model quirks correctly: VLM key remapping,
    gate dequant, PLE, bfloat16, TurboQuant, hybrid SSM, sanitize, etc.

    Args:
        model_path: Path to a JANG v2 model directory.
        expert_percent: Percentage of experts to load per layer (0-100).

    Returns:
        (model, tokenizer_or_processor) — standard mlx_lm/mlx_vlm interface.
    """
    import struct as _struct
    from jang_tools.loader import load_jang_model as _jt_load
    from jang_tools.loader import _fix_quantized_bits

    # Gemma 4 (model_type=gemma4, text_config.model_type=gemma4_text):
    # jang_tools text-only loader does NOT sanitize Gemma 4's SwitchGLU +
    # Router + Experts + per_expert_scale weights correctly — produces
    # garbage output. Use our internal JANG loader instead, which handles
    # Gemma 4 via the mlx_lm model skeleton with proper weight mapping.
    import json as _json
    _is_gemma4_model = False
    try:
        _cfg = _json.loads(open(f"{model_path}/config.json").read())
        _mt = _cfg.get("model_type", "")
        _tmt = _cfg.get("text_config", {}).get("model_type", "")
        _is_gemma4_model = _mt == "gemma4" or _tmt == "gemma4_text"
    except Exception:
        pass
    if _is_gemma4_model:
        # Use our internal JANG text loader which calls mlx_lm's
        # load_model skeleton with proper Gemma 4 weight sanitization.
        from ..utils.jang_loader import load_jang_model as _internal_load
        _jt_load = lambda p: _internal_load(p)
        logger.info("  Gemma 4 detected: using internal JANG loader (proper weight sanitization)")

    # vmlx#81: detect JANGTQ (weight_format=mxtq) up front. Smelt partial-
    # expert loading is tied to the standard JANG affine-quant runtime —
    # JANGTQ uses custom TurboQuantLinear modules with codebook weights
    # (tq_packed + tq_norms + codebook + signs) that the smelt patches do
    # not know how to subset. Previously this errored deep in the JANG
    # loader with "missing 'format' field" because JANGTQ uses
    # `weight_format: mxtq` instead of `format: jang`, and the user had
    # no idea what was wrong. Emit a clear, actionable error now.
    try:
        _raw_jc_path = Path(model_path) / "jang_config.json"
        if _raw_jc_path.exists():
            _raw_jc = _json.loads(_raw_jc_path.read_text())
            if _raw_jc.get("weight_format") == "mxtq" or _raw_jc.get("format") == "mxtq":
                raise ValueError(
                    "vmlx#81: --smelt is not supported on JANGTQ "
                    "(weight_format=mxtq) models. JANGTQ uses custom "
                    "TurboQuantLinear modules with codebook weights that "
                    "smelt partial-expert loading cannot subset.\n"
                    "\n"
                    "Workarounds:\n"
                    "  (1) Use the standard JANG variant of this model "
                    "(e.g. MiniMax-M2.7-JANG_2L instead of "
                    "MiniMax-M2.7-JANGTQ). Smelt works there.\n"
                    "  (2) Drop --smelt. JANGTQ already quantizes below "
                    "2 bits effective via P3/P15/P17/P18 Metal kernels — "
                    "smelt's memory savings are largely subsumed.\n"
                )
    except FileNotFoundError:
        pass  # not a JANG model at all — let downstream handle

    path = Path(model_path)
    t0 = time.perf_counter()
    logger.info("Smelt loading %s (expert_percent=%d)", path.name, expert_percent)

    # ── Step 0: Apply LatentMoE patch if needed ──────────────────────
    # Must run BEFORE jang_tools creates the model skeleton. Patches
    # nemotron_h.NemotronHBlock to create LatentNemotronHMoE (with
    # latent-sized SwitchMLP) instead of regular NemotronHMoE (hidden-sized).
    # Without this, Nemotron Super's SwitchMLP expects 4096 input but
    # safetensors experts have 1024 (latent) → gather_qmm shape mismatch.
    from ..utils.nemotron_latent_moe import ensure_latent_moe_support

    ensure_latent_moe_support(str(path))

    # ── Step 1: Load backbone via jang_tools with monkey-patched mx.load ──
    # Exact turbosmelt v12 approach (proven coherent at 20.4 tok/s).
    # Patches mx.load to skip expert-only safetensors shards entirely and
    # filter switch_mlp/switch_glu keys from mixed shards. jang_tools handles
    # everything else: VLM key remapping, gate dequant, sanitize, bfloat16, etc.
    #
    # jang_tools calls mx.eval(model.parameters()) which materializes skeleton
    # expert weights as random data (~110GB peak). On machines with enough RAM
    # or swap this works. Expert fill then replaces skeleton with real data.
    _original_mx_load = mx.load
    _original_mx_eval = mx.eval

    def _backbone_only_mx_load(filepath, **kwargs):
        fp = str(filepath)
        if fp.endswith(".safetensors"):
            with open(fp, "rb") as f:
                hdr_size = _struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_size))
            has_backbone = any(
                k != "__metadata__" and "switch_mlp" not in k and "switch_glu" not in k
                for k in hdr
            )
            if not has_backbone:
                return {}  # skip expert-only shards (never mmap'd)
            result = _original_mx_load(fp, **kwargs)
            return {
                k: v
                for k, v in result.items()
                if "switch_mlp" not in k and "switch_glu" not in k
            }
        return _original_mx_load(fp, **kwargs)

    def _deferred_mx_eval(*args, **kwargs):
        # Defer mx.eval(model.parameters()) — the big eval that materializes
        # ALL params including skeleton expert weights (~110GB peak).
        # Individual tensor evals (gate dequant mx.eval(dq)) pass through.
        if len(args) == 1 and isinstance(args[0], dict):
            return  # skip — we'll eval after expert fill
        return _original_mx_eval(*args, **kwargs)

    mx.load = _backbone_only_mx_load
    mx.eval = _deferred_mx_eval
    try:
        model, tokenizer = _jt_load(str(model_path))
    finally:
        mx.load = _original_mx_load
        mx.eval = _original_mx_eval

    logger.info("  Backbone loaded (%.1fs)", time.perf_counter() - t0)

    # ── Step 2: Build ExpertIndex ─────────────────────────────────────
    ei = ExpertIndex.build(str(path))
    num_experts = ei.num_experts

    # ── Step 3: Calculate n_load ──────────────────────────────────────
    n_load = max(1, int(num_experts * expert_percent / 100))
    n_load = min(n_load, num_experts)
    selected = list(range(n_load))
    logger.info(
        "  Smelt: loading %d/%d experts (%d%%)",
        n_load,
        num_experts,
        expert_percent,
    )

    # ── Step 4: Fill experts + wrap MoE blocks ────────────────────────
    layers_list = _get_layers_list(model)
    patched = 0

    for li, li_info in ei.layers.items():
        layer = layers_list[li]

        # ── Gemma 4 special handling ──────────────────────────────────
        if _is_gemma4_layer(layer):
            switch = layer.experts.switch_glu
            for pa, pi in [
                ("gate_proj", li_info.gate_proj),
                ("up_proj", li_info.up_proj),
                ("down_proj", li_info.down_proj),
            ]:
                if pi is None:
                    continue
                proj = getattr(switch, pa, None)
                if proj is None:
                    continue
                proj.weight = _load_expert_subset(pi.weight, selected)
                if pi.scales is not None:
                    proj.scales = _load_expert_subset(pi.scales, selected)
                if pi.biases is not None and hasattr(proj, "biases"):
                    proj.biases = _load_expert_subset(pi.biases, selected)

            if n_load >= num_experts:
                remap = None
            else:
                remap_np = np.zeros(num_experts, dtype=np.uint32)
                for j, g in enumerate(selected):
                    remap_np[g] = j
                remap = mx.array(remap_np)

            bias_np = np.full(num_experts, -1000.0, dtype=np.float32)
            for e in selected:
                bias_np[e] = 0.0
            cache_bias = mx.array(bias_np)

            ne_per_tok = getattr(
                layer.router,
                "num_experts_per_tok",
                getattr(
                    layer.router, "top_k", getattr(layer.config, "top_k_experts", 8)
                ),
            )

            class _Gemma4SmeltRouter(nn.Module):
                def __init__(self, orig_router, cache_bias_arr):
                    super().__init__()
                    self.cache_bias = cache_bias_arr
                    self.config = orig_router.config
                    self.proj = orig_router.proj
                    self.scale = orig_router.scale
                    self.per_expert_scale = orig_router.per_expert_scale
                    self._root_size = orig_router._root_size
                    # mlx_vlm Router uses self.norm (nn.RMSNorm);
                    # mlx_lm gemma4_text Router uses inline rms_norm
                    # with self.eps. Handle both.
                    self._has_norm = hasattr(orig_router, "norm")
                    if self._has_norm:
                        self.norm = orig_router.norm
                    self.eps = getattr(orig_router, "eps", 1e-6)

                def __call__(self, x):
                    if self._has_norm:
                        x = self.norm(x)
                        x = x * self._root_size
                        x = x * self.scale
                    else:
                        # mlx_lm text path: inline rms_norm
                        x = mx.fast.rms_norm(
                            x, self.scale * self._root_size, self.eps
                        )
                    expert_scores = self.proj(x)
                    # Use biased scores for top-k SELECTION (prefer loaded experts)
                    biased_scores = expert_scores + self.cache_bias
                    top_k = self.config.top_k_experts
                    top_k_indices = mx.argpartition(
                        biased_scores, kth=-top_k, axis=-1
                    )[..., -top_k:]
                    # Softmax on extracted top-k weights only (matches upstream Router)
                    top_k_weights = mx.take_along_axis(
                        expert_scores, top_k_indices, axis=-1
                    )
                    top_k_weights = mx.softmax(top_k_weights, axis=-1)
                    top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
                    return top_k_indices, top_k_weights

            class _Gemma4SmeltExperts(nn.Module):
                def __init__(self, orig_experts, remap_table):
                    super().__init__()
                    self.orig_experts = orig_experts
                    self.remap = remap_table

                def __call__(self, x, top_k_indices, top_k_weights):
                    local_inds = (
                        self.remap[top_k_indices]
                        if self.remap is not None
                        else top_k_indices
                    )
                    return self.orig_experts(x, local_inds, top_k_weights)

            if remap is not None:
                layer.router = _Gemma4SmeltRouter(layer.router, cache_bias)
                layer.experts = _Gemma4SmeltExperts(layer.experts, remap)
            patched += 1
            continue

        # ── Standard MoE block path ───────────────────────────────────
        moe, moe_attr = _find_moe_block(layer)
        if moe is None:
            continue
        switch = getattr(moe, "switch_mlp", getattr(moe, "switch_glu", None))
        if switch is None:
            continue

        for pa, pi in [
            ("gate_proj", li_info.gate_proj),
            ("up_proj", li_info.up_proj),
            ("down_proj", li_info.down_proj),
            ("fc1", li_info.up_proj),
            ("fc2", li_info.down_proj),
        ]:
            if pi is None:
                continue
            proj = getattr(switch, pa, None)
            if proj is None:
                continue
            proj.weight = _load_expert_subset(pi.weight, selected)
            if pi.scales is not None:
                proj.scales = _load_expert_subset(pi.scales, selected)
            if pi.biases is not None and hasattr(proj, "biases"):
                proj.biases = _load_expert_subset(pi.biases, selected)

        if n_load >= num_experts:
            remap = None
        else:
            remap_np = np.zeros(num_experts, dtype=np.uint32)
            for j, g in enumerate(selected):
                remap_np[g] = j
            remap = mx.array(remap_np)

        bias_np = np.full(num_experts, -1000.0, dtype=np.float32)
        for e in selected:
            bias_np[e] = 0.0
        cache_bias = mx.array(bias_np)

        routing_style = _detect_routing_style(moe)
        wrapper = TurboRouteWrapper(moe, remap, cache_bias, routing_style)
        setattr(layers_list[li], moe_attr, wrapper)
        patched += 1

    # After filling experts, fix quantization bits again (expert modules
    # may have different bits than the backbone defaults)
    _fix_quantized_bits(model, {})

    logger.info("  Wrapped %d MoE layers", patched)

    # Evaluate replaced expert weights (backbone was already evaluated by jang_tools).
    # Expert weights from _load_expert_subset are concrete numpy data converted to
    # mx.array — they need one eval pass to land in GPU memory.
    logger.info("  Evaluating expert weights (%d%% loaded)...", expert_percent)
    mx.eval(model.parameters())

    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    logger.info(
        "Smelt load complete: %d/%d experts, %d layers, %.1fs",
        n_load,
        num_experts,
        patched,
        elapsed,
    )

    if hasattr(model, "language_model") and hasattr(model.language_model, "make_cache"):
        if not hasattr(model, "make_cache"):
            _lm = model.language_model

            def _vlm_make_cache(self=None):
                return _lm.make_cache()

            model.make_cache = _vlm_make_cache

    if hasattr(model, "language_model"):
        _cls = type(model)
        _orig_call = getattr(_cls, "_smelt_saved_call", None) or _cls.__call__

        def _smelt_vlm_call(
            self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs
        ):
            # Only pass pixel_values/mask if non-None — smelt forces
            # text-only mode so the underlying model may be loaded via
            # mlx_lm (text path) whose __call__ doesn't accept pixel_values.
            fwd_kwargs = {}
            if pixel_values is not None:
                fwd_kwargs["pixel_values"] = pixel_values
            if mask is not None:
                fwd_kwargs["mask"] = mask
            if cache is not None:
                fwd_kwargs["cache"] = cache
            fwd_kwargs.update(kwargs)
            result = _orig_call(self, input_ids, **fwd_kwargs)
            if hasattr(result, "logits"):
                return result.logits
            return result

        if not hasattr(_cls, "_smelt_saved_call"):
            _cls._smelt_saved_call = _cls.__call__
        _cls.__call__ = _smelt_vlm_call

    return model, tokenizer
