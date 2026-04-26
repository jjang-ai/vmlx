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
_VALID_BITS = {2, 3, 4, 6, 8}

# Group-size values the converters use.
_VALID_GSZ = {16, 32, 64, 96, 128, 256}


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


def infer_quant_overrides_for_bundle(
    bundle_path: Path | str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Scan the bundle and return a patched config with corrected per-module
    quantization overrides.

    Idempotent: when the existing config matches what the shapes imply
    for every module, returns the input config unchanged. When shapes
    disagree with the config in a provable way, returns a deep copy with
    `quantization.<module>` entries patched in.

    The function never raises — on read failure or unexpected layout it
    logs a debug message and returns the input config unchanged. Worst
    case is the model loads with the original (potentially bad) config,
    same as the pre-patcher behavior.
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

    # Pull out config.quantization (the per-module override block).
    cfg = dict(config)
    qcfg_raw = cfg.get("quantization")
    qcfg = dict(qcfg_raw) if isinstance(qcfg_raw, dict) else {}

    patched: Dict[str, Tuple[int, int]] = {}
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
        candidates = _candidates_for_ratio(ratio_x32)
        if not candidates:
            skipped_count += 1
            continue

        # Config-trust: if the config's claim is one of the valid candidates
        # for this shape ratio, trust the config. This is the correct
        # answer for any bundle whose config matches its weights, and
        # also covers the "ambiguous shape ratio + correct config" case.
        claim = _config_claim_for_module(qcfg, mod)
        if claim is not None and claim in candidates:
            continue

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

        # No-op if our inference happens to match a (silent) claim.
        if claim == inferred:
            continue

        patched[mod] = inferred

    if not patched:
        return config  # No overrides needed — config was good for every module

    # Build the patched config (deep copy so we don't mutate the caller's dict)
    new_cfg = json.loads(json.dumps(config))  # safe deep copy via JSON roundtrip
    new_qcfg = new_cfg.setdefault("quantization", {})
    for mod, (bits, gsz) in patched.items():
        new_qcfg[mod] = {"bits": bits, "group_size": gsz}

    logger.warning(
        "quant_shape_inference: patched %d module(s) in %s — config "
        "disagreed with safetensors shapes (top-level config claimed "
        "bits=%s group_size=%s; per-module overrides corrected from shape). "
        "Most common cause: an older JANG/JANGTQ converter wrote uniform "
        "bits=%s into config.json while actual weights were stored with "
        "mixed precision per layer.",
        len(patched),
        bp.name,
        qcfg.get("bits"),
        qcfg.get("group_size"),
        qcfg.get("bits"),
    )
    if skipped_count:
        logger.debug(
            "quant_shape_inference: skipped %d module(s) (couldn't infer "
            "from shape — likely non-standard quant scheme like MXTQ)",
            skipped_count,
        )

    return new_cfg
