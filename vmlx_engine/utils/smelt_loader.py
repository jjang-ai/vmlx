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
    abs_offset: int    # 8 (header-size field) + header_size + data_offsets[0]
    num_bytes: int     # data_offsets[1] - data_offsets[0]
    shape: List[int]
    dtype: str         # safetensors dtype string, e.g. "U32", "F16", "BF16"

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
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path}"
        )
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

    for st_file in st_files:
        try:
            header_size, header = _read_safetensors_header(st_file)
        except Exception as e:
            logger.warning(
                "ExpertIndex: skipping %s — %s", st_file.name, e
            )
            continue

        # Absolute data region start = 8 (the header-size uint64) + header_size
        data_region_start = 8 + header_size

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(meta, dict):
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

    return ExpertIndex(
        layers=layers,
        model_path=model_path,
        num_experts=num_experts,
        num_moe_layers=num_moe_layers,
        expert_size_bytes=total_expert_bytes,
        backbone_bytes=total_backbone_bytes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertIndex — Part 5: smelt_estimate helpers
# ═══════════════════════════════════════════════════════════════════════════════

# JANG config filenames in priority order
_JANG_CONFIG_NAMES = (
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
)


def smelt_estimate(model_path: "str | Path") -> dict:
    """Fast pre-scan (JSON only, no model loading) for Smelt mode compatibility.

    Reads jang_config.json and config.json, then calls ExpertIndex.build() to
    scan safetensors headers.  No weight data is loaded at any point.

    Args:
        model_path: Path to a JANG quantised model directory.

    Returns:
        On success::

            {
                "compatible": True,
                "num_experts": 256,
                "num_moe_layers": 44,
                "backbone_gb": 4.2,
                "expert_size_mb": 0.86,   # per-layer average total expert bytes
                "total_expert_gb": 58.8,
                "baseline_gb": 63.0,
            }

        On failure::

            {
                "compatible": False,
                "reason": "<human-readable explanation>",
            }
    """
    model_path = Path(model_path)

    # ------------------------------------------------------------------
    # 1. Must have a jang_config.json (or equivalent)
    # ------------------------------------------------------------------
    jang_cfg_path: Optional[Path] = None
    for name in _JANG_CONFIG_NAMES:
        candidate = model_path / name
        if candidate.exists():
            jang_cfg_path = candidate
            break

    if jang_cfg_path is None:
        return {
            "compatible": False,
            "reason": (
                "No JANG config found "
                f"({', '.join(_JANG_CONFIG_NAMES)})"
            ),
        }

    try:
        jang_cfg = json.loads(jang_cfg_path.read_text())
    except Exception as exc:
        return {"compatible": False, "reason": f"Cannot read {jang_cfg_path.name}: {exc}"}

    # ------------------------------------------------------------------
    # 2. Must NOT use codebook VQ
    # ------------------------------------------------------------------
    # codebook_vq may live at top level or nested under "quantization"
    quant_block = jang_cfg if isinstance(jang_cfg, dict) else {}
    if isinstance(quant_block.get("quantization"), dict):
        quant_block = quant_block["quantization"]

    if quant_block.get("codebook_vq", False):
        return {
            "compatible": False,
            "reason": "Codebook VQ quantisation is not supported by Smelt mode",
        }

    # ------------------------------------------------------------------
    # 3. Must have MoE experts (num_experts > 1 in config.json)
    # ------------------------------------------------------------------
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {"compatible": False, "reason": "config.json not found"}

    try:
        model_cfg = json.loads(config_path.read_text())
    except Exception as exc:
        return {"compatible": False, "reason": f"Cannot read config.json: {exc}"}

    # Check both top-level and text_config (VLM wrapper)
    text_cfg = model_cfg.get("text_config", model_cfg)
    num_experts_cfg = (
        text_cfg.get("num_experts")
        or text_cfg.get("num_local_experts")
        or text_cfg.get("n_routed_experts")
        or model_cfg.get("num_experts")
        or model_cfg.get("num_local_experts")
        or model_cfg.get("n_routed_experts")
        or 0
    )
    if not isinstance(num_experts_cfg, int) or num_experts_cfg <= 1:
        return {
            "compatible": False,
            "reason": (
                f"Model does not appear to be MoE "
                f"(num_experts={num_experts_cfg!r})"
            ),
        }

    # ------------------------------------------------------------------
    # 4. Must have at least one *.safetensors file
    # ------------------------------------------------------------------
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        return {
            "compatible": False,
            "reason": "No .safetensors files found — cannot build expert index",
        }

    # ------------------------------------------------------------------
    # 5. ExpertIndex.build() must find MoE layers
    # ------------------------------------------------------------------
    try:
        index = ExpertIndex.build(model_path)
    except Exception as exc:
        return {"compatible": False, "reason": f"ExpertIndex.build() failed: {exc}"}

    if index.num_moe_layers == 0:
        return {
            "compatible": False,
            "reason": (
                "ExpertIndex found no MoE layers — "
                "expert key patterns may not match this architecture"
            ),
        }

    # ------------------------------------------------------------------
    # 6. Build result
    # ------------------------------------------------------------------
    backbone_gb = index.backbone_bytes / 1e9
    total_expert_gb = index.expert_size_bytes / 1e9
    baseline_gb = backbone_gb + total_expert_gb

    # per-layer average total expert bytes → MB
    expert_size_mb = (
        (index.expert_size_bytes / index.num_moe_layers) / 1e6
        if index.num_moe_layers > 0
        else 0.0
    )

    return {
        "compatible": True,
        "num_experts": index.num_experts,
        "num_moe_layers": index.num_moe_layers,
        "backbone_gb": round(backbone_gb, 2),
        "expert_size_mb": round(expert_size_mb, 2),
        "total_expert_gb": round(total_expert_gb, 2),
        "baseline_gb": round(baseline_gb, 2),
    }


def smelt_ram_estimate(index: ExpertIndex, expert_fraction: float) -> dict:
    """Estimate RAM usage for Smelt mode given an expert fraction.

    Args:
        index: ExpertIndex from ExpertIndex.build().
        expert_fraction: Fraction of expert weight bytes to keep in RAM.
            0.0 = backbone only, 1.0 = full model loaded.

    Returns:
        Dict with keys:
            total_model_bytes  — full model size in bytes
            backbone_bytes     — non-expert weight bytes
            expert_bytes       — expert weight bytes (full set)
            smelt_ram_bytes    — estimated RAM with smelt_fraction
            expert_fraction    — the input fraction (echoed back)
    """
    expert_bytes = index.expert_size_bytes
    backbone_bytes = index.backbone_bytes
    total = expert_bytes + backbone_bytes
    smelt_ram = backbone_bytes + int(expert_bytes * expert_fraction)

    return {
        "total_model_bytes": total,
        "backbone_bytes": backbone_bytes,
        "expert_bytes": expert_bytes,
        "smelt_ram_bytes": smelt_ram,
        "expert_fraction": expert_fraction,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TurboRouteWrapper (Task 3)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Supported MoE class names (used by _detect_routing_style / _find_moe_block)
# ---------------------------------------------------------------------------
_MOE_CLASSES = {
    # Qwen family
    "Qwen3NextSparseMoeBlock",  # Qwen 3.5 (qwen3_next.py)
    "Qwen3MoeSparseMoeBlock",   # Qwen 3 (qwen3_moe.py)
    "Qwen2MoeSparseMoeBlock",   # Qwen 2 (qwen2_moe.py)
    # MiniMax
    "MiniMaxSparseMoeBlock",    # MiniMax M2.5 (minimax.py)
    # Nemotron
    "NemotronHMoE",             # Nemotron Cascade/Super (nemotron_h.py)
    # Mistral
    "Mistral4MoE",              # Mistral Small 4 (mistral4.py)
    # DeepSeek / GLM-5 (glm_moe_dsa inherits from deepseek_v32)
    "DeepseekV2MoE",            # DeepSeek V2 (deepseek_v2.py)
    "DeepseekV32MoE",           # DeepSeek V3 / GLM-5 (deepseek_v32.py)
    # Generic
    "SparseMoeBlock",           # dbrx, bailing_moe_linear, etc.
}


# ---------------------------------------------------------------------------
# Helper: detect routing style from class name
# ---------------------------------------------------------------------------

def _detect_routing_style(moe_block: nn.Module) -> str:
    """Return the routing style for *moe_block* based on its class name.

    Returns one of "softmax", "sigmoid", or "pre_routed".
    """
    class_name = type(moe_block).__name__
    if "MiniMax" in class_name or "Glm" in class_name:
        return "sigmoid"
    if "NemotronH" in class_name or "Deepseek" in class_name:
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
            # Nemotron Cascade / Super: gate returns (indices, scores) directly.
            # cache_bias is NOT injected — Nemotron's compiled gate owns routing.
            inds, scores = orig.gate(x)

        else:
            # Softmax — Qwen 3.5, Mistral 4, Gemma 4 (default)
            raw_logits = orig.gate(x)
            # Add cache_bias to RAW logits (pre-softmax) so the bias magnitude
            # is compatible with the logit scale. Adding -1000 to post-softmax
            # probabilities (0-1 range) completely overwhelms the scores.
            biased_logits = raw_logits + self.cache_bias
            inds = mx.argpartition(-biased_logits, kth=ne - 1, axis=-1)[..., :ne]
            # Use UNBIASED softmax probabilities for expert weighting
            probs = mx.softmax(raw_logits, axis=-1, precise=True)
            scores = mx.take_along_axis(probs, inds, axis=-1)
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
        shared = getattr(
            orig, "shared_expert", getattr(orig, "shared_experts", None)
        )
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
    data = np.fromfile(
        ti.file_path, dtype=np.uint8, count=ti.num_bytes, offset=ti.abs_offset
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


class _Gemma4Gate(nn.Module):
    """Extracts raw logits from Gemma 4's Router (norm → scale → proj).

    TurboRouteWrapper expects gate(x) to return raw logits before softmax.
    Gemma 4's Router does norm+scale+proj+softmax+argpartition all in one.
    This wrapper reproduces just the norm+scale+proj portion.
    """

    def __init__(self, router):
        super().__init__()
        self.norm = router.norm
        self.proj = router.proj
        self.scale = router.scale
        self._root_size = router._root_size

    def __call__(self, x):
        x = self.norm(x)
        x = x * self._root_size
        x = x * self.scale
        return self.proj(x)


class _Gemma4MoeShim(nn.Module):
    """Thin shim that wraps Gemma 4's separate router + experts as a single MoE block.

    Gemma 4 has ``layer.router`` (computes top-k) and ``layer.experts``
    (wraps SwitchGLU) as siblings.  TurboRouteWrapper expects a single
    MoE block with ``.gate`` (returns logits) and ``.switch_mlp`` (SwitchGLU).
    """

    def __init__(self, router, switch_glu, num_experts_per_tok: int = 8):
        super().__init__()
        self.gate = _Gemma4Gate(router)
        self.switch_mlp = switch_glu
        self.num_experts_per_tok = num_experts_per_tok
        # Expose per_expert_scale for TurboRouteWrapper to apply after score normalization
        self.per_expert_scale = getattr(router, "per_expert_scale", None)


def smelt_load(
    model_path: str,
    expert_percent: int = 50,
) -> tuple:
    """Load a JANG MoE model in Smelt mode (partial expert loading).

    Smelt loads only the model backbone + a fraction of expert weights,
    dramatically reducing RAM usage while maintaining baseline inference speed
    via native SwitchGLU kernels and cache-biased routing.

    Args:
        model_path: Path to a JANG v2 model directory.
        expert_percent: Percentage of experts to load per layer (0-100).

    Returns:
        (model, tokenizer) — standard mlx_lm interface.
    """
    from mlx_lm.utils import load_config, load_model as _load_model_skeleton, load_tokenizer

    from vmlx_engine.utils.jang_loader import (
        JANG_CONFIG_FILENAMES,
        _chunked_eval_params,
        _fix_quantized_bits,
        _get_v2_weight_files,
        _patch_turboquant_make_cache,
        _upgrade_switch_to_quantized,
    )

    path = Path(model_path)
    t0 = time.perf_counter()
    logger.info("Smelt loading %s (expert_percent=%d)", path.name, expert_percent)

    # ── Read configs ──────────────────────────────────────────────────
    model_cfg = json.loads((path / "config.json").read_text())
    jang_cfg = None
    for name in JANG_CONFIG_FILENAMES:
        p = path / name
        if p.exists():
            jang_cfg = json.loads(p.read_text())
            break
    if jang_cfg is None:
        raise FileNotFoundError(f"No JANG config found in {path}")

    config = load_config(path)
    if "quantization" not in config:
        bs = jang_cfg.get("quantization", {}).get("block_size", 64)
        bw = jang_cfg.get("quantization", {}).get("bit_widths_used", [4])
        config["quantization"] = {"group_size": bs, "bits": min(bw)}

    # Detect model features
    top_type = model_cfg.get("model_type", "")
    text_type = model_cfg.get("text_config", {}).get("model_type", "")
    text_cfg = model_cfg.get("text_config", model_cfg)
    is_nemotron = top_type == "nemotron_h"
    is_nemotron_fc_rename = is_nemotron
    # Gate dequant needed for ALL MoE models — JANG packs gate weights as uint32
    # which must be dequantized before routing (sigmoid/softmax need float input).
    # Same logic as jang_loader: _needs_gate_dequant = _needs_fc_rename or _n_experts > 0
    _n_experts = (
        text_cfg.get("num_experts")
        or text_cfg.get("num_local_experts")
        or text_cfg.get("n_routed_experts")
        or 0
    )
    needs_gate_dequant = is_nemotron or text_type == "mistral4" or _n_experts > 0
    nemotron_renames = {
        "switch_mlp.up_proj": "switch_mlp.fc1",
        "switch_mlp.down_proj": "switch_mlp.fc2",
    }

    # ── Step 1: Load backbone only ────────────────────────────────────
    _is_vlm = False
    try:
        model, config = _load_model_skeleton(
            path, lazy=True, strict=False, model_config=config
        )
    except (ValueError, ImportError, ModuleNotFoundError):
        # VLM model (e.g. Gemma 4) — build skeleton via mlx_vlm's model class system
        from mlx_vlm.utils import (
            get_model_and_args,
            load_config as vlm_load_config,
            update_module_configs,
        )
        vlm_cfg = vlm_load_config(path)
        model_class, _ = get_model_and_args(config=vlm_cfg)
        vlm_cfg.setdefault("text_config", {})
        vlm_cfg.setdefault("vision_config", {})
        if vlm_cfg.get("audio_config") is None:
            vlm_cfg.pop("audio_config", None)
        else:
            vlm_cfg.setdefault("audio_config", {})
        model_config_obj = model_class.ModelConfig.from_dict(vlm_cfg)
        modules = [
            m for m in ["text", "vision", "perceiver", "projector", "audio"]
            if vlm_cfg.get(f"{m}_config") is not None
        ]
        model_config_obj = update_module_configs(model_config_obj, model_class, vlm_cfg, modules)
        model = model_class.Model(model_config_obj)
        _is_vlm = True

        # VLM models need nn.quantize() to convert Linear → QuantizedLinear
        # before loading uint32-packed weights. Build quantized_suffixes from
        # weight keys that have .scales (same approach as _load_jang_v2_vlm).
        import mlx.nn as _nn
        from mlx_vlm.utils import skip_multimodal_module
        _weight_files = _get_v2_weight_files(path)
        _all_keys = set()
        for _sf in _weight_files:
            _d = mx.load(str(_sf))
            _all_keys.update(_d.keys())
            del _d
        _qsuffixes = set()
        for _k in _all_keys:
            if _k.endswith(".scales"):
                _qpath = _k[:-len(".scales")]
                _qsuffixes.add(_qpath)
                if "language_model.model." in _qpath:
                    _qsuffixes.add(_qpath.replace("language_model.model.", "model.language_model.", 1))
                if "model.language_model." in _qpath:
                    _qsuffixes.add(_qpath.replace("model.language_model.", "language_model.model.", 1))

        def _vlm_class_pred(p, m):
            if skip_multimodal_module(p):
                return False
            if not hasattr(m, "to_quantized"):
                return False
            if p in _qsuffixes or f"model.{p}" in _qsuffixes:
                return True
            if "language_model.model." in p:
                if p.replace("language_model.model.", "model.language_model.", 1) in _qsuffixes:
                    return True
            if p.endswith("lm_head") or "language_model.lm_head" in p:
                if "lm_head" in _qsuffixes:
                    return True
            return False

        _nn.quantize(
            model,
            group_size=config["quantization"]["group_size"],
            bits=config["quantization"]["bits"],
            class_predicate=_vlm_class_pred,
        )

    _upgrade_switch_to_quantized(
        model, config["quantization"]["bits"], config["quantization"]["group_size"]
    )

    # Load weights, skipping expert (switch_mlp/switch_glu) keys
    weight_files = _get_v2_weight_files(path)
    n_loaded = n_skipped = 0

    for sf in weight_files:
        weights = mx.load(str(sf))
        filtered = {}
        gate_parts = {}

        for k, v in weights.items():
            if k.endswith(".importance") or k.startswith("mtp."):
                continue
            if "switch_mlp" in k or "switch_glu" in k:
                n_skipped += 1
                continue

            # MoE gate dequant: collect gate scales/biases for dequantization.
            # Gate keys: .mlp.gate., .mixer.gate., .block_sparse_moe.gate.
            # but NOT gate_proj (which is a SwitchGLU expert projection).
            if needs_gate_dequant:
                is_gate_meta = (
                    ".gate." in k
                    and "gate_proj" not in k
                    and (k.endswith(".scales") or k.endswith(".biases"))
                )
                if is_gate_meta:
                    prefix = k.rsplit(".", 1)[0]
                    gate_parts.setdefault(prefix, {})[k.rsplit(".", 1)[1]] = v
                    continue
                # Nemotron fc1/fc2 rename
                if is_nemotron_fc_rename:
                    for old, new in nemotron_renames.items():
                        if old in k:
                            k = k.replace(old, new)
                            break

            filtered[k] = v
            n_loaded += 1

        # Dequantize gate weights (routing needs full precision)
        if needs_gate_dequant:
            for prefix, parts in gate_parts.items():
                wkey = f"{prefix}.weight"
                if wkey in filtered and "scales" in parts:
                    qw = filtered[wkey]
                    scales = parts["scales"]
                    biases = parts.get("biases", mx.zeros_like(scales))
                    for bits in [8, 6, 4, 3, 2]:
                        epu = 32 // bits
                        real = qw.shape[-1] * epu
                        n_s = scales.shape[-1]
                        gs = real // n_s if n_s > 0 else 0
                        if gs > 0 and gs * n_s == real:
                            try:
                                dq = mx.dequantize(qw, scales, biases, gs, bits)
                                mx.eval(dq)
                                filtered[wkey] = dq.astype(mx.float32)
                                filtered.pop(f"{prefix}.scales", None)
                                filtered.pop(f"{prefix}.biases", None)
                                break
                            except Exception:
                                continue

        # Gemma 4 PLE: dequantize ScaledLinear + Embedding weights (#52)
        if text_type == "gemma4_text":
            for _ple_name in ("per_layer_model_projection", "embed_tokens_per_layer"):
                for _pfx in (f"language_model.model.{_ple_name}", f"model.language_model.{_ple_name}"):
                    _w_key = f"{_pfx}.weight"
                    if _w_key not in filtered:
                        continue
                    _w = filtered[_w_key]
                    if _w.dtype != mx.uint32:
                        continue
                    _s_key = f"{_pfx}.scales"
                    if _s_key not in filtered:
                        continue
                    _s = filtered[_s_key]
                    _b = filtered.get(f"{_pfx}.biases", mx.zeros_like(_s))
                    for _bits in (8, 6, 4, 3, 2):
                        _real = _w.shape[-1] * (32 // _bits)
                        _gs = _real // _s.shape[-1] if _s.shape[-1] > 0 else 0
                        if _gs >= 2 and _gs * _s.shape[-1] == _real:
                            try:
                                _dq = mx.dequantize(_w, _s, _b, group_size=_gs, bits=_bits)
                                mx.eval(_dq)
                                filtered[_w_key] = _dq.astype(mx.float16)
                                filtered.pop(_s_key, None)
                                filtered.pop(f"{_pfx}.biases", None)
                                logger.info(f"  Dequantized Gemma4 PLE: {_w_key}")
                                break
                            except Exception:
                                continue

        if filtered:
            if hasattr(model, "sanitize"):
                try:
                    filtered = model.sanitize(filtered)
                except (KeyError, ValueError):
                    pass
            # VLM models may need vision/language sanitizers
            if _is_vlm:
                try:
                    from mlx_vlm.utils import sanitize_weights as _vlm_sanitize
                    _vlm_model_class = type(model)
                    if hasattr(_vlm_model_class, "VisionModel"):
                        filtered = _vlm_sanitize(
                            _vlm_model_class.VisionModel, filtered,
                            model_cfg.get("vision_config", {})
                        )
                    if hasattr(_vlm_model_class, "LanguageModel"):
                        filtered = _vlm_sanitize(
                            _vlm_model_class.LanguageModel, filtered,
                            model_cfg.get("text_config", model_cfg)
                        )
                except (KeyError, ValueError, AttributeError):
                    pass
            model.load_weights(list(filtered.items()), strict=False)
        del weights, filtered

    _fix_quantized_bits(model, {})

    # Fix gate weights that got dequantized to float16 but are inside a
    # QuantizedLinear (e.g. CRACK-variant JANG models). The QuantizedLinear
    # would try to dequantize again, producing garbage routing.
    for _, module in model.named_modules():
        class_name = type(module).__name__
        if class_name not in _MOE_CLASSES and not hasattr(module, "switch_mlp"):
            continue
        gate = getattr(module, "gate", None)
        if gate is None:
            continue
        w = gate.get("weight") if hasattr(gate, "get") else getattr(gate, "weight", None)
        if w is not None and w.dtype != mx.uint32 and hasattr(gate, "bits"):
            new_gate = nn.Linear(w.shape[1], w.shape[0], bias=False)
            new_gate.weight = w
            module.gate = new_gate

    logger.info("  Backbone: %d loaded, %d expert skipped", n_loaded, n_skipped)

    tokenizer = load_tokenizer(path)

    # ── Step 2: Build ExpertIndex ─────────────────────────────────────
    ei = ExpertIndex.build(str(path))
    num_experts = ei.num_experts

    # ── Step 3: Calculate n_load ──────────────────────────────────────
    n_load = max(1, int(num_experts * expert_percent / 100))
    n_load = min(n_load, num_experts)
    selected = list(range(n_load))
    logger.info(
        "  Smelt: loading %d/%d experts (%d%%)",
        n_load, num_experts, expert_percent,
    )

    # ── Step 4-6: Fill experts + wrap ─────────────────────────────────
    layers_list = _get_layers_list(model)
    patched = 0

    for li, li_info in ei.layers.items():
        layer = layers_list[li]

        # ── Step 7: Gemma 4 special handling ──────────────────────────
        if _is_gemma4_layer(layer):
            # Gemma 4: layer.experts.switch_glu is the SwitchGLU module
            switch = layer.experts.switch_glu
            # Fill expert weights
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

            # Build remap + cache_bias
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

            # Create shim: _Gemma4Gate extracts raw logits, switch_glu is the expert module
            ne_per_tok = getattr(layer.router, "num_experts_per_tok",
                                 getattr(layer.router, "top_k",
                                         getattr(layer.config, "top_k_experts", 8)))
            shim = _Gemma4MoeShim(layer.router, switch, ne_per_tok)
            wrapper = TurboRouteWrapper(shim, remap, cache_bias, "softmax")

            # Gemma 4's layer.__call__ does:
            #   indices, weights = self.router(h)
            #   h2 = self.experts(h2, indices, weights)
            # We need experts(h2, indices, weights) to use our wrapper(h2) instead.
            # Wrap the wrapper to accept and ignore Gemma 4's pre-computed routing args.
            # Gemma 4 architecture: Router and Experts receive DIFFERENT inputs.
            #   Router gets: h (post-attention, pre-feedforward)
            #   Experts get: h2 = pre_feedforward_layernorm_2(h)
            # So we can't re-route from inside Experts (wrong input for gate).
            #
            # Solution: patch the ROUTER to inject cache_bias into its logits,
            # and patch EXPERTS to remap global indices to local slots.

            class _Gemma4SmeltRouter(nn.Module):
                """Router with cache_bias injected into expert selection."""
                def __init__(self, orig_router, cache_bias_arr):
                    super().__init__()
                    self.orig = orig_router
                    self.cache_bias = cache_bias_arr
                    # Forward config and attributes
                    self.config = orig_router.config
                    self.norm = orig_router.norm
                    self.proj = orig_router.proj
                    self.scale = orig_router.scale
                    self.per_expert_scale = orig_router.per_expert_scale
                    self._root_size = orig_router._root_size

                def __call__(self, x):
                    x = self.norm(x)
                    x = x * self._root_size
                    x = x * self.scale
                    expert_scores = self.proj(x)

                    # Inject cache_bias into raw logits for selection
                    biased_scores = expert_scores + self.cache_bias
                    top_k = self.config.top_k_experts
                    top_k_indices = mx.argpartition(
                        -biased_scores, kth=top_k - 1, axis=-1
                    )[..., :top_k]

                    # Use UNBIASED probs for weighting (same as original)
                    router_probs = mx.softmax(expert_scores, axis=-1)
                    top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
                    top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
                    top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
                    return top_k_indices, top_k_weights

            class _Gemma4SmeltExperts(nn.Module):
                """Experts wrapper that remaps global indices to local slots."""
                def __init__(self, orig_experts, remap_table):
                    super().__init__()
                    self.orig_experts = orig_experts
                    self.remap = remap_table

                def __call__(self, x, top_k_indices, top_k_weights):
                    if self.remap is not None:
                        local_inds = self.remap[top_k_indices]
                    else:
                        local_inds = top_k_indices
                    return self.orig_experts(x, local_inds, top_k_weights)

            if remap is not None:
                layer.router = _Gemma4SmeltRouter(layer.router, cache_bias)
                layer.experts = _Gemma4SmeltExperts(layer.experts, remap)
            else:
                # Full expert coverage — no changes needed
                pass
            patched += 1
            continue

        # ── Standard MoE block path ───────────────────────────────────
        moe, moe_attr = _find_moe_block(layer)
        if moe is None:
            continue
        switch = getattr(moe, "switch_mlp", getattr(moe, "switch_glu", None))
        if switch is None:
            continue

        # Fill expert weights from SSD
        for pa, pi in [
            ("gate_proj", li_info.gate_proj),
            ("up_proj", li_info.up_proj),
            ("down_proj", li_info.down_proj),
            ("fc1", li_info.up_proj),   # Nemotron 2-proj pattern
            ("fc2", li_info.down_proj),  # Nemotron 2-proj pattern
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

        # Build remap + cache_bias
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

        # Wrap with TurboRouteWrapper
        routing_style = _detect_routing_style(moe)
        wrapper = TurboRouteWrapper(moe, remap, cache_bias, routing_style)
        setattr(layers_list[li], moe_attr, wrapper)
        patched += 1

    logger.info("  Wrapped %d MoE layers", patched)

    # ── Step 8: bfloat16 if needed ────────────────────────────────────
    needs_bf16 = (
        is_nemotron
        or text_type == "mistral4"
        or num_experts >= 512
        # MLA detection
        or model_cfg.get("text_config", model_cfg).get("kv_lora_rank", 0) > 0
    )
    lm_obj = model.language_model if hasattr(model, "language_model") else model

    if needs_bf16:
        logger.info("  Converting to bfloat16")
        lm_obj.set_dtype(mx.bfloat16)

    # ── Step 9: Materialize parameters ────────────────────────────────
    _chunked_eval_params(model)

    # ── Step 10: TurboQuant patching ──────────────────────────────────
    text_cfg = model_cfg.get("text_config", model_cfg)
    _patch_turboquant_make_cache(lm_obj, jang_cfg, text_cfg)

    # Ensure make_cache exists
    if not hasattr(lm_obj, "make_cache"):
        from mlx_lm.models.cache import KVCache
        n_layers = len(layers_list)
        lm_obj.make_cache = lambda: [KVCache() for _ in range(n_layers)]

    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    logger.info(
        "Smelt load complete: %d/%d experts, %d layers, %.1fs",
        n_load, num_experts, patched, elapsed,
    )

    if _is_vlm:
        # VLM models need processor (not just tokenizer) for image handling
        try:
            from mlx_vlm.utils import load_processor, load_image_processor
            from .jang_loader import _build_vlm_processor
            image_processor = load_image_processor(path)
            eos_token_id = getattr(model.config, "eos_token_id", None) if hasattr(model, "config") else None
            try:
                processor = load_processor(path, True, eos_token_ids=eos_token_id)
            except (ImportError, ValueError):
                processor = _build_vlm_processor(path, eos_token_id)
            return model, processor
        except Exception as e:
            logger.warning(f"Failed to load VLM processor, falling back to tokenizer: {e}")

    return model, tokenizer
