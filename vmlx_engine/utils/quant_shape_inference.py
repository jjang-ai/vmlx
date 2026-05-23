# SPDX-License-Identifier: Apache-2.0
"""
Runtime quantization-shape inference + config repair for JANG/JANGTQ bundles.

Some JANG/JANGTQ converter revisions wrote the per-module quantization
metadata (`bits`, `group_size`) wrong in `config.json["quantization"]`
while the actual safetensors weights were stored with different (correct)
quantization parameters. Loading those bundles with the config's claimed
bits/gsz produces degenerate outputs because `mx.dequantize` unpacks the
weight bytes with the wrong stride.

This module scans the bundle's safetensors at load time, computes the
ACTUAL (bits, gsz) for every quantized Linear from shape ratios alone,
and patches the in-memory config when it disagrees. The repair is:

  - Idempotent: bundles with correct configs become a no-op.
  - Conservative: when shape allows multiple (bits, gsz) candidates AND
    the config's claim is one of them, the config wins. The patcher only
    overrides when the config is provably wrong.
  - Tiebreaker-aware: when the config is silent or wrong AND the shape
    is ambiguous, module-name patterns choose the most likely candidate.

THE MATH
--------

For an mx-quantized Linear:
    weight (packed uint32):  shape[-1] = in_features × bits / 32
    scales (one per group):  shape[-1] = in_features / group_size

Therefore:
    bits × group_size = 32 × (weight.shape[-1] / scales.shape[-1])

Several (bits, gsz) pairs can give the same product:
    product 8  → (8,32) (4,64) (2,128)
    product 16 → (8,64) (4,128)
    product 32 → (8,128)
    product 4  → (4,32) (2,64)
    product 2  → (2,32)
    product 3  → (3,32)
    product 6  → (6,32)

In practice JANG/JANGTQ converters use ONE group_size per bundle (uniform
gsz) — when that's the case the bits are uniquely determined by the
product, no guessing needed. The ambiguous-product fallback only fires
when a bundle mixes group sizes (rare).

GUARDS APPLIED
--------------

  1. Pre-flight uniform-gsz detection: if every scales tensor in the
     bundle has the same in_features/scales.shape[-1] across all
     quantized modules, gsz is the unique inferred value and bits =
     32 × ratio / gsz is unambiguous.

  2. Ambiguous-product tiebreaker: when two (bits, gsz) candidates fit
     the same shape ratio, prefer based on module-name pattern:
       - Attention projections (q/k/v/o_proj, q_a/q_b/kv_a/kv_b_proj):
         prefer the higher-bit candidate (typically 8-bit attention).
       - Routed MoE experts (switch_mlp, experts, mlp.experts.X):
         prefer the lower-bit candidate (2/3/4-bit experts are standard).
       - Embed / lm_head / shared_expert: prefer 8-bit.
       - Anything else: prefer the higher-bit candidate.

  3. Config-trust: if the config's per-module override is one of the
     valid (bits, gsz) candidates for that module's shape ratio, trust
     the config. Only override when the config's claim is impossible
     for the observed shape.

OUT-OF-SCOPE
------------

  - MXTQ / TurboQuant codebook tensors (`.tq_packed`, `.tq_norms`,
    `.tq_signs`, `.tq_codebook`) — these don't have scales/biases triples;
    the runtime path handles them via TurboQuantSwitchLinear regardless
    of config.
  - Non-quantized layers (norms, biases) — only quantized linears need
    the (bits, gsz) signal.

USAGE
-----

    from vmlx_engine.utils.quant_shape_inference import (
        infer_quant_overrides_for_bundle,
    )

    config = json.loads((bundle_path / "config.json").read_text())
    config = infer_quant_overrides_for_bundle(bundle_path, config)
    # config now has corrected per-module overrides; safe to construct model
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Candidate (bits, gsz) tables ordered by descending bits within each
# product class. The runtime checks each candidate against shape +
# tiebreaker; the FIRST viable one is selected.
#
# Products that map to multiple candidates (the ambiguous cases):
_RATIO_CANDIDATES: Dict[int, List[Tuple[int, int]]] = {
    1: [(2, 16)],                                 # rare, 2-bit gsz=16
    2: [(2, 32)],                                 # 2-bit gsz=32
    3: [(3, 32)],                                 # 3-bit gsz=32
    4: [(4, 32), (2, 64)],
    6: [(6, 32), (3, 64), (2, 96)],
    8: [(8, 32), (4, 64), (2, 128)],
    12: [(6, 64), (4, 96), (3, 128)],
    16: [(8, 64), (4, 128)],
    24: [(6, 128), (3, 256)],
    32: [(8, 128)],
    48: [(6, 256)],
    64: [(8, 256)],
}

# Bits values that any major JANG/JANGTQ converter actually emits.
# Anything else means we mis-inferred and should fall back.
_VALID_BITS = {2, 3, 4, 5, 6, 8}

# Group-size values the converters use.
_VALID_GSZ = {16, 32, 64, 96, 128, 256}

# Group sizes MLX model-weight quantization can construct at runtime.
# KV-cache quantization has separate head-dim constraints; this table is only
# for model weights passed to nn.quantize / mx.quantize.
_MLX_SUPPORTED_WEIGHT_GSZ = {32, 64, 128}
_MLX_SUPPORTED_WEIGHT_BITS = {2, 3, 4, 5, 6, 8}


# Module-name pattern hints used as the ambiguous-product tiebreaker.
_ATTENTION_PATTERNS = [
    r"\.q_proj$", r"\.k_proj$", r"\.v_proj$", r"\.o_proj$",
    r"\.q_a_proj$", r"\.q_b_proj$",
    r"\.kv_a_proj_with_mqa$", r"\.kv_b_proj$",
    r"\.wq$", r"\.wk$", r"\.wv$", r"\.wo$",
    r"\.attention\.\w+$",
]
_ROUTED_EXPERT_PATTERNS = [
    r"\.experts\.\d+\.",
    r"\.switch_mlp\.",
    r"\.mlp\.experts\.\d+\.",
    r"\.routed_experts?\.",
    r"\.expert_\d+\.",
]
_HIGH_BIT_PATTERNS = [
    r"^embed_tokens$",
    r"\.embed_tokens$",
    r"^lm_head$",
    r"\.lm_head$",
    r"\.shared_experts?\.",
    r"\.shared_expert\.",
]


def _classify(weight_name: str) -> str:
    """Return one of {attention, routed, high_bit, generic} from the weight name."""
    for p in _HIGH_BIT_PATTERNS:
        if re.search(p, weight_name):
            return "high_bit"
    for p in _ROUTED_EXPERT_PATTERNS:
        if re.search(p, weight_name):
            return "routed"
    for p in _ATTENTION_PATTERNS:
        if re.search(p, weight_name):
            return "attention"
    return "generic"


def _resolve_ambiguous_candidates(
    candidates: List[Tuple[int, int]], weight_name: str
) -> Tuple[int, int]:
    """Pick the most likely (bits, gsz) for an ambiguous shape ratio.

    Tiebreaker rules: attention/embed/lm_head prefer the highest-bit
    candidate; routed experts prefer the lowest-bit candidate; everything
    else picks the highest-bit candidate (matches typical converter
    output where the unknown class is more likely a non-MoE projection).
    """
    if not candidates:
        raise ValueError("no candidates")
    if len(candidates) == 1:
        return candidates[0]
    cls = _classify(weight_name)
    if cls == "routed":
        # lowest bits wins
        return min(candidates, key=lambda c: c[0])
    # everything else: highest bits wins (matches converter convention)
    return max(candidates, key=lambda c: c[0])


def _safetensors_index_paths(bundle_path: Path) -> List[Path]:
    """Return the list of .safetensors files in a bundle, honoring an index."""
    idx = bundle_path / "model.safetensors.index.json"
    if idx.is_file():
        try:
            data = json.loads(idx.read_text())
            wmap = data.get("weight_map", {}) or {}
            files = sorted({bundle_path / fn for fn in wmap.values()})
            return [f for f in files if f.is_file()]
        except (OSError, ValueError):
            pass
    # Single-file fallback
    single = bundle_path / "model.safetensors"
    if single.is_file():
        return [single]
    # Multi-shard glob fallback
    shards = sorted(bundle_path.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    return []


def _read_safetensors_metadata(path: Path) -> Dict[str, Any]:
    """Read just the JSON header of a safetensors file (no tensor data)."""
    import struct
    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
        return json.loads(header_bytes.decode("utf-8"))
    except (OSError, ValueError, struct.error) as e:
        logger.debug(f"safetensors header read failed for {path.name}: {e}")
        return {}


def _scan_quantized_modules(bundle_path: Path) -> Dict[str, Dict[str, Any]]:
    """Scan all safetensors and return { module_name: {bits_packed, scales_n, weight_dtype} }.

    A module is considered "quantized" iff it has ALL of:
        {module}.weight (uint32)
        {module}.scales
    Optional: {module}.biases (affine)
    """
    out: Dict[str, Dict[str, Any]] = {}
    for f in _safetensors_index_paths(bundle_path):
        meta = _read_safetensors_metadata(f)
        # meta is { tensor_name: {"dtype":..., "shape":[...], "data_offsets":[s,e]} }
        # Plus an optional "__metadata__" key.
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict):
                continue
            shape = info.get("shape") or []
            dtype = info.get("dtype") or ""
            if not shape or not dtype:
                continue
            if name.endswith(".weight") and dtype in ("U32", "uint32"):
                mod = name[:-len(".weight")]
                out.setdefault(mod, {})["weight_packed_cols"] = int(shape[-1])
                out[mod]["weight_dtype"] = dtype
            elif name.endswith(".scales"):
                mod = name[:-len(".scales")]
                out.setdefault(mod, {})["scales_n"] = int(shape[-1])
            elif name.endswith(".biases"):
                mod = name[:-len(".biases")]
                out.setdefault(mod, {})["has_biases"] = True
    # Drop modules that don't have BOTH a packed weight AND scales — those
    # aren't quantized linears (or use a different scheme like MXTQ).
    return {
        m: d for m, d in out.items()
        if "weight_packed_cols" in d and "scales_n" in d
    }


def _infer_uniform_gsz(modules: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """Return the bundle's uniform gsz if it can be UNAMBIGUOUSLY pinned,
    else None.

    Approach:
        1. Collect each module's set of viable gsz values (those for which
           bits = 32 × packed / (scales_n × gsz) is a valid bit-width).
        2. Intersect across all modules.
        3. The result is unambiguous only when the intersection is a
           single value AND the bundle has enough modules with DIFFERENT
           shape ratios that pinning gsz isn't a coincidence.

    Single-module bundles (or bundles where every module shares the same
    ratio) cannot be disambiguated this way — the function returns None
    and the caller falls back to module-name tiebreakers.
    """
    if not modules:
        return None
    # Collect viable gsz set per module
    per_module_viable: List[set] = []
    distinct_ratios: set = set()
    for m, d in modules.items():
        packed = d["weight_packed_cols"]
        scales_n = d["scales_n"]
        if scales_n == 0 or (32 * packed) % scales_n != 0:
            continue
        ratio_x32 = (32 * packed) // scales_n
        distinct_ratios.add(ratio_x32)
        viable = set()
        for gsz in _VALID_GSZ:
            in_features = scales_n * gsz
            num = 32 * packed
            if num % in_features != 0:
                continue
            bits = num // in_features
            if bits in _VALID_BITS:
                viable.add(gsz)
        per_module_viable.append(viable)
    if not per_module_viable:
        return None
    # Need at least 2 distinct ratios across the bundle for gsz to be
    # genuinely pinned by intersection. Single-ratio bundles (every
    # module same shape) can't disambiguate gsz from intersection alone.
    if len(distinct_ratios) < 2:
        return None
    intersection = per_module_viable[0]
    for s in per_module_viable[1:]:
        intersection = intersection & s
        if not intersection:
            return None
    if len(intersection) == 1:
        return next(iter(intersection))
    return None


def _candidates_for_ratio(ratio_x32: int) -> List[Tuple[int, int]]:
    """Return all (bits, gsz) pairs whose product equals ratio_x32 and
    are within the valid bits + gsz sets.
    """
    out: List[Tuple[int, int]] = []
    for bits in _VALID_BITS:
        if ratio_x32 % bits == 0:
            gsz = ratio_x32 // bits
            if gsz in _VALID_GSZ:
                out.append((bits, gsz))
    # Also seed from the explicit table (covers gsz=16 / 96 corners)
    explicit = _RATIO_CANDIDATES.get(ratio_x32, [])
    for c in explicit:
        if c not in out:
            out.append(c)
    # Stable sort: highest bits first
    out.sort(key=lambda c: -c[0])
    return out


def _config_claim_for_module(
    config_quant: Dict[str, Any], module_name: str
) -> Optional[Tuple[int, int]]:
    """Lookup what the config claims for a given module.

    config_quant shape (after legacy normalization):
        { "bits": int, "group_size": int, "<module>": {"bits":..., "group_size":...}, ... }
    """
    if not isinstance(config_quant, dict):
        return None
    # Per-module override (most specific)
    override = config_quant.get(module_name)
    if isinstance(override, dict):
        b = override.get("bits")
        g = override.get("group_size")
        if isinstance(b, int) and isinstance(g, int):
            return (b, g)
    # Top-level fallback
    b = config_quant.get("bits")
    g = config_quant.get("group_size")
    if isinstance(b, int) and isinstance(g, int):
        return (b, g)
    return None


def _per_module_claim_for_module(
    config_quant: Dict[str, Any], module_name: str
) -> Optional[Tuple[int, int]]:
    """Lookup only an explicit per-module quantization claim."""
    if not isinstance(config_quant, dict):
        return None
    override = config_quant.get(module_name)
    if isinstance(override, dict):
        b = override.get("bits")
        g = override.get("group_size")
        if isinstance(b, int) and isinstance(g, int):
            return (b, g)
    return None


def _top_level_claim(config_quant: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if not isinstance(config_quant, dict):
        return None
    b = config_quant.get("bits")
    g = config_quant.get("group_size")
    if isinstance(b, int) and isinstance(g, int):
        return (b, g)
    return None


def _has_per_module_quant_overrides(config_quant: Dict[str, Any]) -> bool:
    if not isinstance(config_quant, dict):
        return False
    return any(isinstance(value, dict) for value in config_quant.values())


def _sidecar_reports_mixed_affine_bits(bundle_path: Path) -> bool:
    """Return True when jang_config says this is a mixed-bit affine bundle.

    Some Qwen3.6 JANG-MTP rebundles preserved correct tensor bytes but lost the
    large per-module override map from config.json, leaving only a stale global
    ``bits=4/group_size=64`` claim. That claim is shape-compatible with 8b/g32
    tensors, so ordinary config-trust logic cannot detect the mismatch.
    """
    try:
        data = json.loads((bundle_path / "jang_config.json").read_text())
    except Exception:
        return False
    quant = data.get("quantization")
    if not isinstance(quant, dict):
        return False
    values = quant.get("bit_widths_used")
    if not isinstance(values, list):
        return False
    bits: set[int] = set()
    for value in values:
        try:
            bits.add(int(value))
        except Exception:
            continue
    return len(bits) > 1 and 8 in bits


def _qwen_hybrid_without_module_overrides(config: Dict[str, Any]) -> bool:
    model_type = str(config.get("model_type") or "").lower()
    text_type = str((config.get("text_config") or {}).get("model_type") or "").lower()
    if model_type not in {"qwen3_5", "qwen3_5_vl"} and text_type not in {
        "qwen3_5",
        "qwen3_5_text",
    }:
        return False
    return bool(config.get("vision_config"))


def _int_config_value(config: Dict[str, Any], key: str) -> Optional[int]:
    value = config.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _qwen_hybrid_expected_input_dim(
    config: Dict[str, Any],
    module_name: str,
) -> Optional[int]:
    """Return the architecture input dim for Qwen3.5/3.6 hybrid modules.

    Qwen affine-JANG bundles can stamp stale per-module quantization metadata
    that is still shape-compatible. In those ambiguous cases, the only safe
    tiebreaker is the module's real architecture input dimension.
    """
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config
    hidden = _int_config_value(text_config, "hidden_size")
    intermediate = _int_config_value(text_config, "intermediate_size")
    num_heads = _int_config_value(text_config, "num_attention_heads")
    head_dim = _int_config_value(text_config, "head_dim")
    linear_v_heads = _int_config_value(text_config, "linear_num_value_heads")
    linear_v_dim = _int_config_value(text_config, "linear_value_head_dim")

    name = module_name
    if name.startswith("model."):
        name = name[len("model.") :]
    if name.endswith("embed_tokens") or name.endswith("lm_head"):
        return hidden

    if ".linear_attn." in name:
        if name.endswith((".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a")):
            return hidden
        if name.endswith(".out_proj") and linear_v_heads and linear_v_dim:
            return linear_v_heads * linear_v_dim

    if ".self_attn." in name:
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return hidden
        if name.endswith(".o_proj") and num_heads and head_dim:
            return num_heads * head_dim

    if ".mlp." in name:
        if name.endswith((".gate_proj", ".up_proj")):
            return hidden
        if name.endswith(".down_proj"):
            return intermediate

    return None


def _deepseek_v4_sanitized_aliases(module_name: str) -> List[str]:
    """Return mlx-lm module paths produced by jang_tools DSV4 sanitize().

    DeepSeek-V4 JANG metadata is keyed with source names such as ``embed`` and
    ``layers.N.attn.wkv``. The runtime model receives weights after
    ``Model.sanitize()``, where those paths become ``model.embed`` and
    ``model.layers.N.self_attn.wkv``. mlx-lm quantizes modules after sanitize,
    so per-module overrides must exist under the sanitized names too.
    """
    if module_name == "embed":
        return ["model.embed"]
    if module_name == "head":
        return ["lm_head"]

    match = re.match(r"^layers\.(\d+)\.(.+)$", module_name)
    if not match:
        return []

    layer_idx, rest = match.group(1), match.group(2)
    layer_prefix = f"model.layers.{layer_idx}"

    if rest.startswith("attn."):
        return [f"{layer_prefix}.self_attn.{rest[len('attn.') :]}"]

    shared = re.match(r"^ffn\.shared_experts\.(w[123])$", rest)
    if shared:
        proj = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }[shared.group(1)]
        return [f"{layer_prefix}.mlp.shared_experts.{proj}"]

    routed = re.match(r"^ffn\.experts\.\d+\.(w[123])$", rest)
    if routed:
        proj = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }[routed.group(1)]
        return [f"{layer_prefix}.mlp.switch_mlp.{proj}"]

    return []


def _sanitized_aliases_for_config(config: Dict[str, Any], module_name: str) -> List[str]:
    model_type = str(config.get("model_type") or "")
    if model_type == "deepseek_v4":
        return _deepseek_v4_sanitized_aliases(module_name)
    return []


def infer_quant_overrides_for_bundle(
    bundle_path: Path | str,
    config: Dict[str, Any],
    *,
    runtime_supported_only: bool = False,
    error_on_unsupported: bool = False,
) -> Dict[str, Any]:
    """Scan the bundle and return a patched config with corrected per-module
    quantization overrides.

    Idempotent: when the existing config matches what the shapes imply
    for every module, returns the input config unchanged. When shapes
    disagree with the config in a provable way, returns a deep copy with
    `quantization.<module>` entries patched in.

    By default the function never raises — on read failure or unexpected
    layout it logs a debug message and returns the input config unchanged.
    When ``runtime_supported_only=True`` it filters inferred candidates to
    MLX's model-weight runtime support ({32,64,128} group sizes). In that
    mode, ``error_on_unsupported=True`` raises a clear bundle-level error
    when tensor shapes prove the weights require an unsupported group size.
    """
    bp = Path(bundle_path)
    try:
        modules = _scan_quantized_modules(bp)
    except Exception as e:
        logger.debug(f"quant_shape_inference: scan failed for {bp}: {e}")
        return config
    if not modules:
        return config  # Non-quantized bundle, or shape scan came back empty

    # Pre-flight: detect uniform group_size across the entire bundle.
    uniform_gsz = _infer_uniform_gsz(modules)
    if runtime_supported_only and uniform_gsz not in _MLX_SUPPORTED_WEIGHT_GSZ:
        uniform_gsz = None

    # Pull out config.quantization (the per-module override block).
    cfg = dict(config)
    qcfg_raw = cfg.get("quantization")
    qcfg = dict(qcfg_raw) if isinstance(qcfg_raw, dict) else {}
    qwen_hybrid_mixed_affine = (
        _qwen_hybrid_without_module_overrides(cfg)
        and _sidecar_reports_mixed_affine_bits(bp)
    )
    distrust_ambiguous_global_claim = (
        qwen_hybrid_mixed_affine and not _has_per_module_quant_overrides(qcfg)
    )

    patched: Dict[str, Tuple[int, int]] = {}
    effective_by_module: Dict[str, Tuple[int, int]] = {}
    unsupported_modules: List[Tuple[str, List[Tuple[int, int]]]] = []
    skipped_count = 0

    for mod, d in modules.items():
        packed = d["weight_packed_cols"]
        scales_n = d["scales_n"]
        if scales_n == 0:
            skipped_count += 1
            continue
        # ratio = bits × gsz / 32; expressed as ratio_x32 = bits × gsz
        ratio_x32 = (32 * packed) // scales_n if (32 * packed) % scales_n == 0 else 0
        if ratio_x32 == 0:
            skipped_count += 1
            continue

        # ALWAYS compute the full candidate set first. Single-module
        # bundles with ambiguous ratios shouldn't get artificially
        # constrained by uniform_gsz inference (which can over-pick when
        # there isn't enough cross-module variance to fix gsz).
        all_candidates = _candidates_for_ratio(ratio_x32)
        candidates = all_candidates
        if runtime_supported_only:
            candidates = [
                (b, g)
                for b, g in all_candidates
                if b in _MLX_SUPPORTED_WEIGHT_BITS
                and g in _MLX_SUPPORTED_WEIGHT_GSZ
            ]
        if not candidates:
            if runtime_supported_only and all_candidates:
                unsupported_modules.append((mod, all_candidates))
            skipped_count += 1
            continue

        # Config-trust: if the config's claim is one of the valid candidates
        # for this shape ratio, trust the config. This is the correct
        # answer for any bundle whose config matches its weights, and
        # also covers the "ambiguous shape ratio + correct config" case.
        per_module_claim = _per_module_claim_for_module(qcfg, mod)
        top_claim = _top_level_claim(qcfg)
        claim = per_module_claim if per_module_claim is not None else top_claim
        claim_is_only_global = per_module_claim is None and top_claim is not None
        effective: Optional[Tuple[int, int]] = None
        dim_inferred: Optional[Tuple[int, int]] = None
        if qwen_hybrid_mixed_affine and len(candidates) > 1:
            expected_dim = _qwen_hybrid_expected_input_dim(cfg, mod)
            if expected_dim:
                matching = [
                    (bits, gsz)
                    for bits, gsz in candidates
                    if scales_n * gsz == expected_dim
                    and (packed * 32) // bits == expected_dim
                ]
                if len(matching) == 1:
                    dim_inferred = matching[0]

        if dim_inferred is not None:
            effective = dim_inferred
            if claim != dim_inferred:
                patched[mod] = dim_inferred
        if effective is not None:
            pass
        elif claim is not None and claim in candidates:
            if (
                distrust_ambiguous_global_claim
                and claim_is_only_global
                and len(candidates) > 1
            ):
                effective = None
            else:
                effective = claim
        else:
            effective = None

        if effective is None:
            # Config is provably wrong (claim not viable for this shape) OR
            # config has no claim. Pick from candidates using the bundle's
            # detected uniform gsz first (when available + viable for this
            # shape), then the module-name tiebreaker.
            inferred: Optional[Tuple[int, int]] = None
            if uniform_gsz is not None:
                inferred = next(
                    ((b, g) for b, g in candidates if g == uniform_gsz), None
                )
            if inferred is None:
                inferred = _resolve_ambiguous_candidates(candidates, mod)
            effective = inferred

            # No-op if our inference happens to match a (silent) claim.
            if claim != inferred:
                patched[mod] = inferred
        effective_by_module[mod] = effective

        for alias in _sanitized_aliases_for_config(config, mod):
            if alias == mod:
                continue
            alias_claim = _config_claim_for_module(qcfg, alias)
            if alias_claim != effective:
                patched[alias] = effective

    if unsupported_modules and error_on_unsupported:
        first_mod, first_candidates = unsupported_modules[0]
        group_sizes = sorted({g for _, g in first_candidates})
        declared = _top_level_claim(qcfg)
        declared_text = (
            f"declared bits={declared[0]} group_size={declared[1]}; "
            if declared is not None
            else ""
        )
        raise ValueError(
            f"{bp}: quantized tensor shapes for {first_mod} require unsupported "
            f"model weight group_size {group_sizes[-1]}; {declared_text}"
            "MLX supports only group sizes 32, 64, and 128 for model weights. "
            "Re-quantize the bundle with a supported group_size."
        )

    top_claim = _top_level_claim(qcfg)
    top_level_runtime_override: Optional[Tuple[int, int]] = None
    if runtime_supported_only:
        top_supported = (
            top_claim is not None
            and top_claim[0] in _MLX_SUPPORTED_WEIGHT_BITS
            and top_claim[1] in _MLX_SUPPORTED_WEIGHT_GSZ
        )
        if not top_supported:
            if effective_by_module:
                counts = Counter(effective_by_module.values())
                top_level_runtime_override = max(
                    counts,
                    key=lambda pair: (counts[pair], pair[0], -pair[1]),
                )
            elif error_on_unsupported:
                declared = (
                    f"bits={top_claim[0]} group_size={top_claim[1]}"
                    if top_claim is not None
                    else "missing quantization bits/group_size"
                )
                raise ValueError(
                    f"{bp}: {declared} is not supported for MLX model-weight "
                    "quantization, and no safetensors affine shapes were "
                    "available to infer a supported value. MLX supports only "
                    "bits 2/3/4/5/6/8 and group sizes 32/64/128."
                )

    if not patched and top_level_runtime_override is None:
        return config  # No overrides needed — config was good for every module

    # Build the patched config (deep copy so we don't mutate the caller's dict)
    new_cfg = json.loads(json.dumps(config))  # safe deep copy via JSON roundtrip
    new_qcfg = new_cfg.setdefault("quantization", {})
    if top_level_runtime_override is not None:
        new_qcfg["bits"] = top_level_runtime_override[0]
        new_qcfg["group_size"] = top_level_runtime_override[1]
    for mod, (bits, gsz) in patched.items():
        new_qcfg[mod] = {"bits": bits, "group_size": gsz}
        # CRACK-bundle key-prefix normalization (Gemma-4-31B-JANG_4M-CRACK
        # discussion #25, 2026-05-04): some VLM CRACK bundles wrote
        # per-module overrides under HF naming `model.language_model.X`,
        # but mlx_lm's class_predicate matches MLX module paths after
        # `sanitize()` which strips `model.` (`language_model.model.X`
        # path on the live module tree). When the override is keyed
        # under HF naming, mlx_lm's predicate looks up the MLX path,
        # finds nothing, and falls back to the global default — which
        # was usually 8-bit even when the layer is 4-bit. The result
        # is shape mismatch on load (8192,672) vs (8192,1344). Patch
        # by writing the override under BOTH HF and MLX paths so
        # whichever the predicate looks up wins.
        #
        # MLX path conventions for VLM wrappers (gemma3n/gemma4/qwen3.5_vl/
        # nemotron_omni): top-level wrapper places the language model at
        # `language_model` (not `model.language_model`). Inside the
        # language_model, the inner stack is at `.model.layers.X` (NOT
        # `.layers.X`). So the rewrite is:
        #   model.language_model.X → language_model.model.X (when
        #     X != model.layers... — only the leading `model.` is
        #     stripped; the inner `language_model.model` path stays).
        # Conservative: only write the alternate key if the rewrite
        # produces a NEW name (i.e. the source had `model.` prefix).
        if mod.startswith("model."):
            alt = mod[len("model."):]
            if alt and alt not in new_qcfg:
                new_qcfg[alt] = {"bits": bits, "group_size": gsz}
        elif mod.startswith("language_model.") or mod.startswith("vision_tower."):
            # The reverse rewrite: bundle had MLX naming, also write HF
            # naming so older mlx_lm pre-sanitize lookups still find it.
            alt = "model." + mod
            if alt not in new_qcfg:
                new_qcfg[alt] = {"bits": bits, "group_size": gsz}

    logger.warning(
        "quant_shape_inference: patched %d module(s) in %s — config "
        "disagreed with safetensors shapes (top-level config claimed "
        "bits=%s group_size=%s; per-module overrides corrected from shape). "
        "Most common cause: an older JANG/JANGTQ converter wrote uniform "
        "bits=%s into config.json while actual weights were stored with "
        "mixed precision per layer.%s",
        len(patched),
        bp.name,
        qcfg.get("bits"),
        qcfg.get("group_size"),
        qcfg.get("bits"),
        (
            " Runtime default was also rewritten to MLX-supported "
            f"bits={top_level_runtime_override[0]} "
            f"group_size={top_level_runtime_override[1]}."
            if top_level_runtime_override is not None
            else ""
        ),
    )
    if skipped_count:
        logger.debug(
            "quant_shape_inference: skipped %d module(s) (couldn't infer "
            "from shape — likely non-standard quant scheme like MXTQ)",
            skipped_count,
        )

    return new_cfg
