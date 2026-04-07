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
    r"(?P<proj>gate_proj|up_proj|down_proj)"
    r"\."
    r"(?P<suffix>weight|scales|biases)"
    r"$",
)


def _match_expert_key(key: str) -> Optional[Tuple[int, str, str]]:
    """Return (layer_idx, proj_name, suffix) if *key* matches an expert pattern.

    Returns None if the key does not match any known expert weight pattern.
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
    return int(layer_str), m.group("proj"), m.group("suffix")


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
    "Qwen3NextSparseMoeBlock",
    "MiniMaxMoE",
    "NemotronHMoEBlock",
    "LatentNemotronHMoE",
    "Mistral4MoE",
    "GlmMoeDsaMoE",
    "Gemma4SparseMoeBlock",
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
    if "NemotronH" in class_name or "DeepSeek" in class_name:
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
            gates = orig.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            orig_gates = gates                              # unbiased
            gates = gates + self.cache_bias                 # biased for selection
            inds = mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne]
            # Use UNBIASED scores for weighting
            scores = mx.take_along_axis(orig_gates, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            rsf = getattr(orig, "routed_scaling_factor", 1.0)
            scores = scores * rsf

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
