"""
JANG Model Loader — Load JANG quantized models into MLX for inference.
Created by Jinho Jang (eric@jangq.ai)

v2 models: MLX-native safetensors — load via mx.load() mmap in seconds.
v1 models: Legacy format — repacks JANG uint8 to MLX uint32 (slow, 5-10 min).

v2 is the default format for new conversions. v1 backward compat is preserved
so existing models on HuggingFace continue to work.
"""

import gc
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Support current "jang_config.json" and legacy names
JANG_CONFIG_FILENAMES = ["jang_config.json", "jjqf_config.json", "jang_cfg.json", "mxq_config.json"]
JANG_FORMAT_VALUES = ["jang", "jjqf", "mxq"]

# Shard flush threshold for v1 streaming repack (~2 GB)
_SHARD_FLUSH_BYTES = 2_000_000_000


def _find_config_path(model_path: Path) -> Optional[Path]:
    for name in JANG_CONFIG_FILENAMES:
        p = model_path / name
        if p.exists():
            return p
    return None


def is_jang_model(model_path: str | Path) -> bool:
    """Check if a directory contains a JANG model."""
    return _find_config_path(Path(model_path)) is not None


def _is_v2_model(model_path: Path) -> bool:
    """Check if a JANG model uses v2 format (MLX-native safetensors).

    MUST only be called on confirmed JANG models (has jang_config.json).
    v2 = has standard safetensors (not .jang.safetensors) + jang_config.json.
    """
    # Must have jang_config.json — without it, this is a standard MLX model
    config_path = _find_config_path(model_path)
    if not config_path:
        return False

    # Check format_version in config first (most reliable)
    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse JANG config {config_path}: {e}")
        return False
    version = cfg.get("format_version", "1.0")
    if str(version).startswith("2"):
        return True

    # Check for v2 index file (standard safetensors index alongside jang_config)
    if (model_path / "model.safetensors.index.json").exists():
        # Only v2 if no .jang.safetensors exist (v1 has .jang.safetensors)
        has_jang = any(model_path.glob("*.jang.safetensors"))
        if not has_jang:
            return True

    return False



# ─── v2 loader (instant) ────────────────────────────────────────────


def _load_jang_v2(path: Path, jang_cfg: dict):
    """
    Load a JANG v2 model — instant via mx.load() mmap.

    v2 models store weights in MLX-native format (uint32 packed weights,
    float16 scales/biases) in standard safetensors. No repacking needed.
    """
    from mlx_lm.utils import load_config, load_model as _load_model_skeleton, load_tokenizer

    start = time.perf_counter()
    config = load_config(path)

    # config.json already has quantization key (written by v2 converter)
    # but ensure it exists for older v2 models
    if "quantization" not in config:
        block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
        bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [4])
        config["quantization"] = {"group_size": block_size, "bits": min(bit_widths)}

    model, config = _load_model_skeleton(
        path, lazy=True, strict=False, model_config=config
    )
    _upgrade_switch_to_quantized(
        model,
        config["quantization"]["bits"],
        config["quantization"]["group_size"],
    )

    # Load weights via mmap — this is instant
    weight_files = _get_v2_weight_files(path)
    logger.info(f"  Loading {len(weight_files)} safetensors shards via mmap")

    # Nemotron-H naming fix: JANG converter uses switch_mlp.up_proj/down_proj
    # but mlx-lm's nemotron_h expects switch_mlp.fc1/fc2. Without this rename,
    # weights are silently dropped (strict=False) and the model runs on random values.
    _nemotron_renames = {
        ".switch_mlp.up_proj.": ".switch_mlp.fc1.",
        ".switch_mlp.down_proj.": ".switch_mlp.fc2.",
    }
    _model_type = config.get("model_type", "")
    _needs_fc_rename = _model_type in ("nemotron_h", "nemotron")

    # Nemotron-H gate: MoEGate is a custom nn.Module (not nn.Linear), so
    # nn.quantize() in _load_model_skeleton does NOT convert it. However,
    # _load_model_skeleton's model.load_weights() loads the raw uint32 gate
    # weight into MoEGate.weight. Our custom weight loading loop below
    # dequantizes the gate weight (uint32 → bfloat16) and overwrites it.

    for sf in weight_files:
        weights = mx.load(str(sf))
        # Nemotron-H: filter mtp/importance, rename fc1/fc2, dequantize gate weights
        if _needs_fc_rename:
            weights = {k: v for k, v in weights.items()
                       if not k.endswith(".importance") and "mtp." not in k}
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        if _needs_fc_rename:
            renamed = {}
            gate_parts = {}  # prefix -> {scales, biases}
            for k, v in weights.items():
                new_k = k
                # Collect gate scales/biases for dequantization
                if ".gate." in k and (k.endswith(".scales") or k.endswith(".biases")):
                    prefix = k.rsplit(".", 1)[0]
                    gate_parts.setdefault(prefix, {})[k.rsplit(".", 1)[1]] = v
                    continue
                # Apply fc1/fc2 rename
                for old, new in _nemotron_renames.items():
                    if old in k:
                        new_k = k.replace(old, new)
                        break
                renamed[new_k] = v
            # Dequantize gate weights (uint32 packed → float for nn.Linear)
            # Gate is 8-bit quantized — try bits high-to-low to find correct shape
            for prefix, parts in gate_parts.items():
                wkey = f"{prefix}.weight"
                if wkey in renamed and "scales" in parts:
                    qw = renamed[wkey]
                    scales = parts["scales"]
                    biases = parts.get("biases", mx.zeros_like(scales))
                    for bits in [8, 6, 4, 3, 2]:
                        elem_per_u32 = 32 // bits
                        real_cols = qw.shape[-1] * elem_per_u32
                        gs = real_cols // scales.shape[-1] if scales.shape[-1] > 0 else 0
                        if gs > 0 and gs * scales.shape[-1] == real_cols:
                            try:
                                dq = mx.dequantize(qw, scales, biases, gs, bits)
                                mx.eval(dq)
                                renamed[wkey] = dq.astype(mx.bfloat16)
                                logger.info(f"  Dequantized gate: {wkey} bits={bits} gs={gs} -> {dq.shape}")
                                break
                            except Exception:
                                continue
            weights = renamed
        model.load_weights(list(weights.items()), strict=False)
        del weights
        gc.collect()

    _fix_quantized_bits(model, {})

    if not hasattr(model, "config"):
        model.config = config

    # bfloat16 compute for 512+ expert models — float16 norm/embedding
    # layers overflow at shared expert down_proj (SiLU*up → 4096-dim dot
    # product exceeds float16 max 65504). bfloat16 has float32 range.
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = _text_cfg.get("num_experts",
                    _text_cfg.get("num_local_experts",
                    _text_cfg.get("n_routed_experts", 0)))
    _hidden = _text_cfg.get("hidden_size", 0)
    if _n_experts >= 512 and _hidden >= 4096:
        model.set_dtype(mx.bfloat16)
        logger.info(
            f"  bfloat16 enabled: {_n_experts} experts, hidden={_hidden} "
            f"(float16 overflow prevention)"
        )

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start

    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", 0)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")
    logger.info(
        f"JANG v2 loaded in {elapsed:.1f}s: {source_model} "
        f"({actual_bits:.1f}-bit avg)"
    )

    tokenizer = load_tokenizer(
        path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, tokenizer


def _load_jang_v2_vlm(path: Path, jang_cfg: dict):
    """Load a JANG v2 Vision-Language model via mmap — instant."""
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args, load_config as vlm_load_config,
        update_module_configs, load_image_processor,
        load_processor, skip_multimodal_module,
    )

    start = time.perf_counter()

    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
    bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [4])
    default_bits = min(bit_widths)

    config = vlm_load_config(path)
    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)
    model = model_class.Model(model_config)

    # Collect all weight keys to determine which layers to quantize
    weight_files = _get_v2_weight_files(path)
    all_weight_keys = set()
    for sf in weight_files:
        data = mx.load(str(sf))
        all_weight_keys.update(data.keys())
        del data
        gc.collect()

    # Build set of quantized module paths from weight keys
    # Weight keys (safetensors): model.language_model.layers.0.mlp.gate_proj.scales
    # Module paths (nn.quantize): language_model.model.layers.0.mlp.gate_proj
    # These don't match — build a suffix set for robust matching
    quantized_suffixes = set()
    for k in all_weight_keys:
        if k.endswith('.scales'):
            qpath = k[:-len('.scales')]
            quantized_suffixes.add(qpath)

    quantization = {"group_size": block_size, "bits": default_bits}

    def get_class_predicate(p, m):
        if skip_multimodal_module(p):
            return False
        if not hasattr(m, "to_quantized"):
            return False
        # Try exact match first
        if p in quantized_suffixes:
            return True
        # Try with model. prefix (safetensors often has model. prefix)
        if f"model.{p}" in quantized_suffixes:
            return True
        # Handle mlx-vlm naming: language_model.model.X → model.language_model.X
        if "language_model.model." in p:
            remapped = p.replace("language_model.model.", "model.language_model.", 1)
            if remapped in quantized_suffixes:
                return True
        # Handle lm_head: module path is language_model.lm_head, weight key is just lm_head
        if p.endswith("lm_head") or "language_model.lm_head" in p:
            if "lm_head" in quantized_suffixes:
                return True
        return False

    nn.quantize(model, group_size=block_size, bits=default_bits,
                class_predicate=get_class_predicate)

    # Load weights via mmap
    # Matches jang-tools 2.1.0 loader: try model.sanitize() first (works for dense models),
    # fall back to minimal sanitize for MoE models where gate_up_proj is already split.
    from mlx_vlm.utils import sanitize_weights
    for sf in weight_files:
        shard_weights = mx.load(str(sf))
        shard_weights = {k: v for k, v in shard_weights.items()
                         if not k.endswith(".importance")}

        # Try model.sanitize() — works for dense VL models.
        # Fails on MoE models because it tries to split gate_up_proj which JANG already split.
        sanitize_ok = False
        if hasattr(model, "sanitize"):
            try:
                shard_weights = model.sanitize(shard_weights)
                sanitize_ok = True
            except (KeyError, ValueError):
                pass

        if not sanitize_ok:
            # Minimal sanitize: rename keys, transpose conv1d, fix norms (skip MoE rename)
            norm_suffixes = (
                ".input_layernorm.weight", ".post_attention_layernorm.weight",
                "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
            )
            fixed = {}
            for k, v in shard_weights.items():
                if "mtp." in k:
                    continue
                if "model.language_model" in k:
                    k = k.replace("model.language_model", "language_model.model")
                elif "model.visual" in k:
                    k = k.replace("model.visual", "vision_tower")
                elif "lm_head" in k and "language_model" not in k:
                    k = k.replace("lm_head", "language_model.lm_head")
                if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                    v = mx.transpose(v, axes=(0, 2, 1))
                if any(k.endswith(s) for s in norm_suffixes) and v.ndim == 1:
                    v = v + 1.0
                fixed[k] = v
            shard_weights = fixed

        # Apply vision/language sanitizers (may not exist for all model classes)
        try:
            shard_weights = sanitize_weights(
                model_class.VisionModel, shard_weights, model_config.vision_config)
            shard_weights = sanitize_weights(
                model_class.LanguageModel, shard_weights, model_config.text_config)
        except (KeyError, ValueError, AttributeError):
            pass

        # Dequantize vision conv weights that were incorrectly quantized
        for k in list(shard_weights.keys()):
            if ("patch_embed" in k or "temporal_embed" in k) and k.endswith(".weight"):
                w = shard_weights[k]
                if w.dtype == mx.uint32:
                    base = k[:-7]
                    s_key, b_key = f"{base}.scales", f"{base}.biases"
                    if s_key in shard_weights and b_key in shard_weights:
                        s, b = shard_weights[s_key], shard_weights[b_key]
                        for try_bits in (2, 3, 4, 6, 8):
                            in_dim = w.shape[-1] * 32 // try_bits
                            if w.shape[-1] * 32 % try_bits != 0 or in_dim % s.shape[-1] != 0:
                                continue
                            try_gs = in_dim // s.shape[-1]
                            if try_gs >= 2:
                                try:
                                    dq = mx.dequantize(w, s, b, group_size=try_gs, bits=try_bits)
                                    shard_weights[k] = dq.astype(mx.float16)
                                    del shard_weights[s_key], shard_weights[b_key]
                                    break
                                except Exception:
                                    continue

        model.load_weights(list(shard_weights.items()), strict=False)
        del shard_weights
        gc.collect()

    _fix_quantized_bits(model, {})

    if not hasattr(model, "config"):
        model.config = model_config

    # bfloat16 for 512+ expert models (same as text loader)
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = _text_cfg.get("num_experts",
                    _text_cfg.get("num_local_experts",
                    _text_cfg.get("n_routed_experts", 0)))
    _hidden = _text_cfg.get("hidden_size", 0)
    if _n_experts >= 512 and _hidden >= 4096:
        model.set_dtype(mx.bfloat16)
        logger.info(f"  bfloat16 enabled: {_n_experts} experts, hidden={_hidden}")

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    logger.info(f"JANG v2 VLM loaded in {elapsed:.1f}s")

    image_processor = load_image_processor(path)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    try:
        processor = load_processor(path, True, eos_token_ids=eos_token_id)
    except (ImportError, ValueError):
        processor = _build_vlm_processor(path, eos_token_id)
    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def _get_v2_weight_files(path: Path) -> list[Path]:
    """Get safetensors weight files for a v2 model."""
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return [path / sf for sf in sorted(set(index["weight_map"].values()))]

    # Fallback: glob for standard safetensors
    files = sorted(path.glob("model-*.safetensors"))
    if not files:
        files = sorted(path.glob("*.safetensors"))
    return files


# ─── Public API ──────────────────────────────────────────────────────


def load_jang_vlm_model(model_path: str | Path):
    """
    Load a JANG Vision-Language model into mlx-vlm for multimodal inference.

    Automatically detects v2 (instant) or v1 (repack) format.

    Returns:
        Tuple of (model, processor) compatible with mlx-vlm.generate()
    """
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    jang_cfg = json.loads(config_path.read_text())
    fmt = jang_cfg.get("format")
    if not fmt or fmt not in JANG_FORMAT_VALUES:
        raise ValueError(f"Not a JANG model: format='{fmt}'")

    # v2: instant load
    if _is_v2_model(path):
        logger.info(f"JANG v2 VLM detected — loading via mmap (instant)")
        return _load_jang_v2_vlm(path, jang_cfg)

    # v1: repack path (legacy)
    logger.info(f"JANG v1 VLM detected — repacking (this takes a few minutes)")
    return _load_jang_v1_vlm(path, jang_cfg, config_path)


def load_jang_model(model_path: str | Path):
    """
    Load a JANG model for inference.

    Automatically detects v2 (instant) or v1 (repack) format.
    v2 loads in seconds via mx.load() mmap.
    v1 repacks JANG uint8 → MLX uint32 (takes 5-10 minutes for large models).

    Returns:
        Tuple of (model, tokenizer) compatible with mlx-lm
    """
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    jang_cfg = json.loads(config_path.read_text())
    fmt = jang_cfg.get("format")
    if not fmt:
        raise ValueError(
            f"JANG config {config_path.name} is missing 'format' field. "
            f"Expected one of: {', '.join(JANG_FORMAT_VALUES)}"
        )
    if fmt not in JANG_FORMAT_VALUES:
        raise ValueError(f"Not a JANG model: format='{fmt}' (expected {', '.join(JANG_FORMAT_VALUES)})")

    version = str(jang_cfg.get("format_version", "1.0"))
    try:
        major = int(version.split(".")[0])
    except ValueError:
        raise ValueError(f"Invalid JANG format_version: '{version}' (expected numeric like '1.0' or '2.0')")
    if major > 2:
        raise ValueError(
            f"Unsupported JANG format version: {version} (this loader supports 1.x and 2.x)"
        )

    # v2: instant load via mmap
    if _is_v2_model(path):
        logger.info(f"JANG v2 detected — loading via mmap (instant)")
        return _load_jang_v2(path, jang_cfg)

    # v1: repack path (legacy)
    logger.info(f"JANG v1 detected — repacking to MLX format (this may take a few minutes)")
    return _load_jang_v1(path, jang_cfg, config_path)


# ─── v1 loader (legacy, repack) ─────────────────────────────────────


def _load_jang_v1(path: Path, jang_cfg: dict, config_path: Path):
    """Load a JANG v1 model by repacking weights from uint8 to uint32."""
    from mlx_lm.utils import load_config, load_model as _load_model_skeleton, load_tokenizer

    start = time.perf_counter()

    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
    target_bits = jang_cfg.get("quantization", {}).get("target_bits", 4)
    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", target_bits)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")

    logger.info(
        f"Loading JANG v1 model: {source_model} "
        f"({actual_bits:.1f}-bit avg, block_size={block_size})"
    )

    config = load_config(path)
    bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [2, 4, 6, 8])
    default_bits = min(bit_widths)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": block_size, "bits": default_bits}

    model, config = _load_model_skeleton(
        path, lazy=True, strict=False, model_config=config
    )
    _upgrade_switch_to_quantized(model, default_bits, block_size)

    result, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        if tmp_dir is not None:
            logger.info(f"  Loading {len(result)} repacked shards via mmap")
            for sf in result:
                shard_weights = mx.load(sf)
                if hasattr(model, "sanitize"):
                    shard_weights = model.sanitize(shard_weights)
                model.load_weights(list(shard_weights.items()), strict=False)
                del shard_weights
                gc.collect()
        else:
            weights = result
            if hasattr(model, "sanitize"):
                weights = model.sanitize(weights)
            model.load_weights(list(weights.items()), strict=False)
            del weights
            gc.collect()

        _fix_quantized_bits(model, {})
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = config

    # bfloat16 for 512+ expert models (same as v2 loader)
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = _text_cfg.get("num_experts",
                    _text_cfg.get("num_local_experts",
                    _text_cfg.get("n_routed_experts", 0)))
    _hidden = _text_cfg.get("hidden_size", 0)
    if _n_experts >= 512 and _hidden >= 4096:
        model.set_dtype(mx.bfloat16)
        logger.info(f"  bfloat16 enabled: {_n_experts} experts, hidden={_hidden}")

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    from mlx.utils import tree_flatten
    n_params = sum(
        p.size for _, p in tree_flatten(model.parameters())
    )
    logger.info(
        f"JANG v1 model loaded in {elapsed:.1f}s: "
        f"{n_params / 1e9:.1f}B params, {actual_bits:.1f}-bit avg"
    )

    tokenizer = load_tokenizer(
        path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, tokenizer


def _load_jang_v1_vlm(path: Path, jang_cfg: dict, config_path: Path):
    """Load a JANG v1 VLM model by repacking (legacy)."""
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args, load_config as vlm_load_config,
        update_module_configs, load_image_processor,
        load_processor, skip_multimodal_module,
    )

    start = time.perf_counter()

    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
    bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [2, 4, 6, 8])
    default_bits = min(bit_widths)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")

    logger.info(f"Loading JANG v1 VLM: {source_model}")

    config = vlm_load_config(path)
    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)
    model = model_class.Model(model_config)

    shard_files, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        all_weight_keys = set()
        for sf in shard_files:
            data = mx.load(sf)
            all_weight_keys.update(data.keys())
            del data
            gc.collect()

        def get_class_predicate(p, m):
            if skip_multimodal_module(p):
                return False
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in all_weight_keys

        nn.quantize(model, group_size=block_size, bits=default_bits,
                    class_predicate=get_class_predicate)

        from mlx_vlm.utils import sanitize_weights
        for sf in shard_files:
            shard_weights = mx.load(sf)
            if hasattr(model, "sanitize"):
                shard_weights = model.sanitize(shard_weights)
            shard_weights = sanitize_weights(
                model_class.VisionModel, shard_weights, model_config.vision_config)
            shard_weights = sanitize_weights(
                model_class.LanguageModel, shard_weights, model_config.text_config)
            model.load_weights(list(shard_weights.items()), strict=False)
            del shard_weights
            gc.collect()

        _fix_quantized_bits(model, {})
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = model_config

    # bfloat16 for 512+ expert models (same as v2 loader)
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = _text_cfg.get("num_experts",
                    _text_cfg.get("num_local_experts",
                    _text_cfg.get("n_routed_experts", 0)))
    _hidden = _text_cfg.get("hidden_size", 0)
    if _n_experts >= 512 and _hidden >= 4096:
        model.set_dtype(mx.bfloat16)
        logger.info(f"  bfloat16 enabled: {_n_experts} experts, hidden={_hidden}")

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    logger.info(f"JANG v1 VLM loaded in {elapsed:.1f}s")

    image_processor = load_image_processor(path)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    try:
        processor = load_processor(path, True, eos_token_ids=eos_token_id)
    except (ImportError, ValueError):
        processor = _build_vlm_processor(path, eos_token_id)
    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


# ─── v1 repack engine (unchanged from original) ─────────────────────


def _repack_jang_to_mlx(
    model_path: Path,
    block_size: int,
    config: dict,
) -> tuple[list[str], str]:
    """
    Load JANG v1 shards and repack quantized tensors into MLX format.
    Returns (shard_file_paths, tmp_dir_path) or (weights_dict, None).
    """
    from safetensors import safe_open

    INDEX_NAMES = ["model.jang.index.json", "model.jjqf.index.json", "model.mxq.index.json"]
    SHARD_GLOBS = ["*.jang.safetensors", "*.jjqf.safetensors", "*.mxq.safetensors"]
    SUFFIXES = (".qweight", ".scales", ".zeros", ".biases", ".bit_map", ".block_offsets", ".shape", ".bits")

    index_path = None
    for name in INDEX_NAMES:
        p = model_path / name
        if p.exists():
            index_path = p
            break

    shard_files = []
    if index_path:
        index = json.loads(index_path.read_text())
        shard_files = [model_path / sf for sf in sorted(set(index["weight_map"].values()))]
    else:
        for pattern in SHARD_GLOBS:
            shard_files.extend(sorted(model_path.glob(pattern)))

    shard_handles = {}
    tensor_to_shard = {}
    all_tensor_names = []

    for sf in shard_files:
        sf_str = str(sf)
        logger.info(f"  Indexing shard: {sf.name if hasattr(sf, 'name') else sf}")
        handle = safe_open(sf_str, framework="numpy")
        shard_handles[sf_str] = handle
        for key in handle.keys():
            tensor_to_shard[key] = sf_str
            all_tensor_names.append(key)

    class LazyTensors:
        def __getitem__(self, key):
            sf_str = tensor_to_shard[key]
            return shard_handles[sf_str].get_tensor(key)
        def __contains__(self, key):
            return key in tensor_to_shard
        def keys(self):
            return all_tensor_names
        def __iter__(self):
            return iter(all_tensor_names)
        def __len__(self):
            return len(all_tensor_names)

    raw_tensors = LazyTensors()

    if not raw_tensors:
        raise FileNotFoundError(f"No JANG weight files found in {model_path}")

    quantized_bases = set()
    non_quantized_names = []

    for name in raw_tensors:
        matched = False
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                quantized_bases.add(name[: -len(suffix)])
                matched = True
                break
        if not matched:
            non_quantized_names.append(name)

    logger.info(
        f"  {len(quantized_bases)} quantized tensors, "
        f"{len(non_quantized_names)} non-quantized tensors"
    )

    import os
    try:
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, AttributeError):
        import subprocess
        total_ram = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())

    model_disk_bytes = sum(sf.stat().st_size for sf in shard_files if sf.exists())
    ram_threshold = int(total_ram * 0.50)
    use_streaming = model_disk_bytes > ram_threshold

    if use_streaming:
        logger.info(f"  Streaming mode: model {model_disk_bytes/1e9:.0f} GB > 50% of {total_ram/1e9:.0f} GB RAM")
    else:
        logger.info(f"  In-memory mode: model {model_disk_bytes/1e9:.0f} GB fits in {total_ram/1e9:.0f} GB RAM")

    tmp_dir = None
    output_shards = []
    current_shard = {}
    current_bytes = 0
    shard_idx = 0
    bit_counts = {}

    if use_streaming:
        for candidate_dir in [str(model_path.parent), str(model_path), None]:
            try:
                tmp_dir = tempfile.mkdtemp(prefix=".jang_repack_", dir=candidate_dir)
                test_f = Path(tmp_dir) / ".write_test"
                test_f.write_text("ok")
                test_f.unlink()
                break
            except (OSError, PermissionError):
                if tmp_dir and Path(tmp_dir).exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir = None
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="jang_repack_")

    import re
    _per_expert_2d_pattern = re.compile(
        r".+\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\."
    )
    expert_buffer = {}

    def _flush_shard():
        nonlocal current_shard, current_bytes, shard_idx
        if not current_shard:
            return
        if not use_streaming:
            return
        shard_path = f"{tmp_dir}/shard_{shard_idx:04d}.safetensors"
        mx.eval(*current_shard.values())
        mx.save_safetensors(shard_path, current_shard)
        output_shards.append(shard_path)
        logger.info(f"  Flushed shard {shard_idx} ({current_bytes / 1e9:.1f} GB, {len(current_shard)} tensors)")
        shard_idx += 1
        current_shard = {}
        current_bytes = 0
        gc.collect()

    def _add_to_shard(key, arr):
        nonlocal current_bytes
        current_shard[key] = arr
        current_bytes += arr.nbytes
        if current_bytes >= _SHARD_FLUSH_BYTES:
            _flush_shard()

    for base in sorted(quantized_bases):
        qweight_raw = raw_tensors[f"{base}.qweight"]
        jang_scales = raw_tensors[f"{base}.scales"].astype(np.float32)
        biases_key = f"{base}.biases"
        zeros_key = f"{base}.zeros"
        if biases_key in raw_tensors:
            jang_biases_raw = raw_tensors[biases_key].astype(np.float32)
        elif zeros_key in raw_tensors:
            jang_zeros = raw_tensors[zeros_key].astype(np.float32)
            jang_biases_raw = -jang_scales * jang_zeros
        else:
            jang_biases_raw = np.zeros_like(jang_scales)

        n_blocks = len(jang_scales)

        bits_key = f"{base}.bits"
        if bits_key in raw_tensors:
            bits = int(raw_tensors[bits_key][0])
        elif f"{base}.bit_map" in raw_tensors:
            bits = int(raw_tensors[f"{base}.bit_map"][0])
        else:
            logger.warning(f"  No bits info for {base}, assuming 4-bit")
            bits = 4

        bit_counts[bits] = bit_counts.get(bits, 0) + n_blocks

        shape_key = f"{base}.shape"
        if shape_key in raw_tensors:
            shape = tuple(int(x) for x in raw_tensors[shape_key])
        else:
            total_weights = n_blocks * block_size
            shape = _infer_weight_shape(base, config, total_weights)

        is_3d = shape is not None and len(shape) >= 3
        if is_3d:
            num_experts = shape[0]
            expert_out = shape[1]
            in_dim = shape[-1]
            out_dim = num_experts * expert_out
        elif shape is not None:
            num_experts = 0
            expert_out = 0
            out_dim, in_dim = shape
        else:
            num_experts = 0
            expert_out = 0
            out_dim = n_blocks
            in_dim = block_size

        packed_bytes = qweight_raw.tobytes()
        pad_needed = (4 - len(packed_bytes) % 4) % 4
        if pad_needed:
            packed_bytes += b'\x00' * pad_needed
        mlx_qweight = np.frombuffer(packed_bytes, dtype=np.uint32)

        packed_per_row = (in_dim * bits + 31) // 32
        expected_len = out_dim * packed_per_row
        if len(mlx_qweight) < expected_len:
            mlx_qweight = np.pad(mlx_qweight, (0, expected_len - len(mlx_qweight)))
        mlx_qweight = mlx_qweight[:expected_len]

        if is_3d:
            mlx_qweight = mlx_qweight.reshape(num_experts, expert_out, packed_per_row)
        else:
            mlx_qweight = mlx_qweight.reshape(out_dim, packed_per_row)

        n_groups_per_row = (in_dim + block_size - 1) // block_size
        expected_groups = out_dim * n_groups_per_row
        jang_biases = jang_biases_raw

        if n_blocks < expected_groups:
            pad = expected_groups - n_blocks
            jang_scales = np.pad(jang_scales, (0, pad), constant_values=1.0)
            jang_biases = np.pad(jang_biases, (0, pad), constant_values=0.0)

        if is_3d:
            mlx_scales = jang_scales[:expected_groups].reshape(num_experts, expert_out, n_groups_per_row)
            mlx_biases = jang_biases[:expected_groups].reshape(num_experts, expert_out, n_groups_per_row)
        else:
            mlx_scales = jang_scales[:expected_groups].reshape(out_dim, n_groups_per_row)
            mlx_biases = jang_biases[:expected_groups].reshape(out_dim, n_groups_per_row)

        if shape is not None and len(shape) >= 3:
            weight_key = base
        else:
            weight_key = f"{base}.weight"

        if is_3d and "gate_up_proj" in base:
            mid = expert_out // 2
            gate_w = mlx_qweight[:, :mid, :]
            up_w = mlx_qweight[:, mid:, :]
            gate_s = mlx_scales[:, :mid, :]
            up_s = mlx_scales[:, mid:, :]
            gate_b = mlx_biases[:, :mid, :]
            up_b = mlx_biases[:, mid:, :]

            sw_prefix = base.replace("experts.gate_up_proj", "switch_mlp")
            _add_to_shard(f"{sw_prefix}.gate_proj.weight", mx.array(gate_w))
            _add_to_shard(f"{sw_prefix}.gate_proj.scales", mx.array(gate_s))
            _add_to_shard(f"{sw_prefix}.gate_proj.biases", mx.array(gate_b))
            _add_to_shard(f"{sw_prefix}.up_proj.weight", mx.array(up_w))
            _add_to_shard(f"{sw_prefix}.up_proj.scales", mx.array(up_s))
            _add_to_shard(f"{sw_prefix}.up_proj.biases", mx.array(up_b))
        elif is_3d and "down_proj" in base:
            sw_prefix = base.replace("experts.down_proj", "switch_mlp")
            _add_to_shard(f"{sw_prefix}.down_proj.weight", mx.array(mlx_qweight))
            _add_to_shard(f"{sw_prefix}.down_proj.scales", mx.array(mlx_scales))
            _add_to_shard(f"{sw_prefix}.down_proj.biases", mx.array(mlx_biases))
        elif not is_3d and "gate_up_proj" in base:
            mid = out_dim // 2
            gate_w = mlx_qweight[:mid, :]
            up_w = mlx_qweight[mid:, :]
            gate_s = mlx_scales[:mid, :]
            up_s = mlx_scales[mid:, :]
            gate_b = mlx_biases[:mid, :]
            up_b = mlx_biases[mid:, :]

            gate_base = base.replace("gate_up_proj", "gate_proj")
            up_base = base.replace("gate_up_proj", "up_proj")
            _add_to_shard(f"{gate_base}.weight", mx.array(gate_w))
            _add_to_shard(f"{gate_base}.scales", mx.array(gate_s))
            _add_to_shard(f"{gate_base}.biases", mx.array(gate_b))
            _add_to_shard(f"{up_base}.weight", mx.array(up_w))
            _add_to_shard(f"{up_base}.scales", mx.array(up_s))
            _add_to_shard(f"{up_base}.biases", mx.array(up_b))
        else:
            if _per_expert_2d_pattern.search(weight_key):
                scale_key = weight_key.replace('.weight', '') if '.weight' in weight_key else weight_key
                expert_buffer[weight_key] = mx.array(mlx_qweight)
                expert_buffer[f"{scale_key}.scales"] = mx.array(mlx_scales)
                expert_buffer[f"{scale_key}.biases"] = mx.array(mlx_biases)
            else:
                _add_to_shard(weight_key, mx.array(mlx_qweight))
                scale_key = weight_key.replace('.weight', '') if '.weight' in weight_key else weight_key
                _add_to_shard(f"{scale_key}.scales", mx.array(mlx_scales))
                _add_to_shard(f"{scale_key}.biases", mx.array(mlx_biases))

        del qweight_raw, jang_scales, jang_biases_raw, jang_biases, packed_bytes
        del mlx_qweight, mlx_scales, mlx_biases

    if expert_buffer:
        _stack_per_expert_weights(expert_buffer, config)
        for k, v in expert_buffer.items():
            _add_to_shard(k, v)
        expert_buffer.clear()
        gc.collect()

    for name in non_quantized_names:
        arr = raw_tensors[name]
        if arr.dtype == np.float32:
            _add_to_shard(name, mx.array(arr))
        elif arr.dtype == np.float16:
            _add_to_shard(name, mx.array(arr))
        else:
            _add_to_shard(name, mx.array(arr.astype(np.float16)))

    for handle in shard_handles.values():
        del handle
    shard_handles.clear()
    gc.collect()

    rename_keys = []
    rename_keys += [(k, "vision_tower" + k[len("model.visual"):]) for k in list(current_shard.keys()) if k.startswith("model.visual")]
    rename_keys += [(k, "language_model.model" + k[len("model.language_model"):]) for k in list(current_shard.keys()) if k.startswith("model.language_model")]
    for old_k, new_k in rename_keys:
        current_shard[new_k] = current_shard.pop(old_k)

    _flush_shard()
    _rename_keys_in_flushed_shards(output_shards, tmp_dir)

    total_blocks = sum(bit_counts.values())
    if total_blocks > 0:
        dist_str = ", ".join(
            f"{b}-bit: {c} ({100 * c // total_blocks}%)"
            for b, c in sorted(bit_counts.items())
        )
        logger.info(f"  Bit distribution: {dist_str}")

    if use_streaming:
        logger.info(f"  Repacked into {len(output_shards)} temp shards in {tmp_dir}")
        return output_shards, tmp_dir
    else:
        logger.info(f"  Repacked {len(current_shard)} tensors in memory")
        return current_shard, None


# ─── Shared helpers ──────────────────────────────────────────────────


def _rename_keys_in_flushed_shards(shard_paths, tmp_dir):
    for shard_path in shard_paths:
        data = mx.load(shard_path)
        needs_rewrite = False
        renamed = {}
        for k, v in data.items():
            if k.startswith("model.visual"):
                new_k = "vision_tower" + k[len("model.visual"):]
                renamed[new_k] = v
                needs_rewrite = True
            elif k.startswith("model.language_model"):
                new_k = "language_model.model" + k[len("model.language_model"):]
                renamed[new_k] = v
                needs_rewrite = True
            else:
                renamed[k] = v
        if needs_rewrite:
            mx.save_safetensors(shard_path, renamed)
        del data, renamed
        gc.collect()


def _stack_per_expert_weights(weights, config):
    import re
    expert_pattern = re.compile(
        r"(.+)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\.weight$"
    )
    expert_groups = {}
    for key in list(weights.keys()):
        m = expert_pattern.match(key)
        if m:
            prefix, expert_id, wtype = m.group(1), int(m.group(2)), m.group(3)
            group_key = (prefix, wtype)
            if group_key not in expert_groups:
                expert_groups[group_key] = {}
            expert_groups[group_key][expert_id] = key

    if not expert_groups:
        return

    name_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    for (prefix, wtype), experts in expert_groups.items():
        if len(experts) < 2:
            continue
        num_experts = max(experts.keys()) + 1
        new_name = name_map.get(wtype, wtype)
        sw_key = f"{prefix}.switch_mlp.{new_name}"

        to_stack = [weights.pop(experts[e]) for e in range(num_experts)]
        weights[f"{sw_key}.weight"] = mx.stack(to_stack)

        for suffix in [".scales", ".biases"]:
            parts = []
            found = True
            for e in range(num_experts):
                sk = experts.get(e, "").replace(".weight", "") + suffix
                if sk in weights:
                    parts.append(weights.pop(sk))
                else:
                    found = False
                    break
            if found and parts:
                weights[f"{sw_key}{suffix}"] = mx.stack(parts)

    if expert_groups:
        logger.info(f"  Stacked {len(expert_groups)} expert groups into QuantizedSwitchLinear format")


def _upgrade_switch_to_quantized(model, bits, group_size):
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    except ImportError:
        return

    for name, module in model.named_modules():
        if not isinstance(module, SwitchLinear):
            continue
        ql = QuantizedSwitchLinear(
            module.input_dims, module.output_dims, module.num_experts,
            bias=hasattr(module, 'bias'),
            group_size=group_size, bits=bits,
        )
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = model
            for p in parts[0].split('.'):
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, parts[1], ql)


def _fix_quantized_bits(model, weights=None):
    """Fix per-layer bits AND group_size for JANG mixed-precision models.

    Matches jang-tools 2.1.0 logic: router/gate tensors prefer gs=64 (precision-critical),
    everything else prefers the module's initialized gs (from config.json).
    """
    import mlx.nn as nn
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        quant_types = (nn.QuantizedLinear, nn.QuantizedEmbedding, QuantizedSwitchLinear)
    except ImportError:
        quant_types = (nn.QuantizedLinear, nn.QuantizedEmbedding)

    for name, module in model.named_modules():
        if not isinstance(module, quant_types):
            continue
        if not hasattr(module, 'scales') or not hasattr(module, 'weight'):
            continue
        try:
            w_cols = module.weight.shape[-1]
            s_cols = module.scales.shape[-1]
            fixed = False

            # Router/gate tensors prefer gs=64 (precision-critical in JANG)
            name_lower = name.lower()
            is_router = (".gate." in name_lower or name_lower.endswith(".gate")
                         or "shared_expert_gate" in name_lower)
            if is_router:
                gs_candidates = [64, module.group_size, 128]
            else:
                gs_candidates = [module.group_size]
                for gs in (64, 128):
                    if gs not in gs_candidates:
                        gs_candidates.append(gs)

            for try_gs in gs_candidates:
                in_dim = s_cols * try_gs
                if in_dim <= 0 or (w_cols * 32) % in_dim != 0:
                    continue
                try_bits = (w_cols * 32) // in_dim
                if try_bits in (2, 3, 4, 5, 6, 8):
                    if try_bits != module.bits:
                        module.bits = try_bits
                    if try_gs != module.group_size:
                        module.group_size = try_gs
                    fixed = True
                    break

            if not fixed:
                # Last resort: try current gs with whatever bits result
                in_dim = s_cols * module.group_size
                if in_dim > 0:
                    actual_bits = (w_cols * 32) // in_dim
                    if actual_bits != module.bits and actual_bits in (2, 3, 4, 5, 6, 8):
                        module.bits = actual_bits
        except Exception:
            pass


def _build_vlm_processor(model_path: Path, eos_token_id=None):
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers.processing_utils import ProcessorMixin
    from mlx_vlm.tokenizer_utils import load_tokenizer as vlm_load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    config = json.loads((model_path / "config.json").read_text())
    model_type = config.get("model_type", "")

    tok_config_path = model_path / "tokenizer_config.json"
    chat_template = None
    if tok_config_path.exists():
        chat_template = json.loads(tok_config_path.read_text()).get("chat_template")

    processor = None
    try:
        from transformers.video_processing_utils import BaseVideoProcessor
        video_stub = BaseVideoProcessor()

        processor_classes = {}
        try:
            from transformers import Qwen3VLProcessor
            processor_classes["qwen3_5"] = Qwen3VLProcessor
            processor_classes["qwen3_5_moe"] = Qwen3VLProcessor
            processor_classes["qwen3_vl"] = Qwen3VLProcessor
        except ImportError:
            pass
        try:
            from transformers import Qwen2VLProcessor
            processor_classes["qwen2_vl"] = Qwen2VLProcessor
            processor_classes["qwen2_5_vl"] = Qwen2VLProcessor
        except ImportError:
            pass

        proc_class = processor_classes.get(model_type)
        if proc_class is not None:
            _orig = ProcessorMixin.check_argument_for_proper_class
            def _permissive(self, name, arg):
                if name == "video_processor":
                    return type(arg)
                return _orig(self, name, arg)
            ProcessorMixin.check_argument_for_proper_class = _permissive
            try:
                processor = proc_class(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    video_processor=video_stub,
                    chat_template=chat_template,
                )
            finally:
                ProcessorMixin.check_argument_for_proper_class = _orig
    except Exception as exc:
        logger.warning(f"Could not construct VL processor: {exc}")

    if processor is None:
        class _SimpleVLMProcessor:
            def __init__(self, tok, ip):
                self.tokenizer = tok
                self.image_processor = ip
            def __call__(self, *a, **kw):
                return self.tokenizer(*a, **kw)
        processor = _SimpleVLMProcessor(tokenizer, image_processor)

    detokenizer_class = vlm_load_tokenizer(model_path, return_tokenizer=False)
    tokenizer_obj = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    processor.detokenizer = detokenizer_class(tokenizer_obj)

    final_eos = eos_token_id if eos_token_id is not None else getattr(tokenizer_obj, "eos_token_ids", None)
    criteria = StoppingCriteria(final_eos, tokenizer_obj)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.stopping_criteria = criteria
    else:
        processor.stopping_criteria = criteria

    return processor


def _infer_weight_shape(base_name, config, n_elements):
    tc = config.get("text_config", {})
    def _get(key, default=0):
        return config.get(key, tc.get(key, default))

    hidden = _get("hidden_size", 0)
    intermediate = _get("intermediate_size", 0)
    moe_intermediate = _get("moe_intermediate_size", intermediate)
    shared_expert_intermediate = _get("shared_expert_intermediate_size", moe_intermediate)
    num_heads = _get("num_attention_heads", 0)
    num_kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim", hidden // num_heads if num_heads else 0)
    vocab_size = _get("vocab_size", 0)

    name = base_name.lower()

    if "qkv_proj" in name:
        out = (num_heads + 2 * num_kv_heads) * head_dim
        return (out, hidden)
    elif "q_proj" in name:
        return (num_heads * head_dim, hidden)
    elif "k_proj" in name:
        return (num_kv_heads * head_dim, hidden)
    elif "v_proj" in name:
        return (num_kv_heads * head_dim, hidden)
    elif "o_proj" in name:
        return (hidden, num_heads * head_dim)
    elif ".experts." in name or ".shared_expert." in name:
        ei = shared_expert_intermediate if ".shared_expert." in name else (moe_intermediate if moe_intermediate else intermediate)
        if "gate_proj" in name or "up_proj" in name or "w1" in name or "w3" in name:
            return (ei, hidden)
        elif "down_proj" in name or "w2" in name:
            return (hidden, ei)
    elif "gate_up_proj" in name:
        return (2 * intermediate, hidden)
    elif "gate_proj" in name or "up_proj" in name or "w1" in name or "w3" in name:
        return (intermediate, hidden)
    elif "down_proj" in name or "w2" in name:
        return (hidden, intermediate)
    elif "embed_tokens" in name:
        return (vocab_size, hidden)
    elif "lm_head" in name:
        return (vocab_size, hidden)

    if n_elements > 0 and hidden > 0 and n_elements % hidden == 0:
        return (n_elements // hidden, hidden)

    logger.warning(f"  Could not infer shape for {base_name} ({n_elements} elements)")
    return None


# ─── Upgrade v1 → v2 ────────────────────────────────────────────────


def upgrade_v1_to_v2(model_path: str | Path) -> None:
    """
    Upgrade a JANG v1 model to v2 format in-place.

    Repacks uint8 qweight → uint32 MLX-native, then replaces
    the .jang.safetensors files with standard .safetensors files.
    After upgrade, the model loads instantly via mx.load() mmap.

    Args:
        model_path: path to JANG v1 model directory
    """
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    if _is_v2_model(path):
        print(f"  Already v2 format: {path}")
        return

    jang_cfg = json.loads(config_path.read_text())
    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)

    # Load model config
    model_config = json.loads((path / "config.json").read_text())
    config = dict(model_config)
    tc = config.get("text_config", {})
    for key in ["hidden_size", "intermediate_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "vocab_size",
                 "moe_intermediate_size", "shared_expert_intermediate_size"]:
        if key in tc and key not in config:
            config[key] = tc[key]

    print(f"  Upgrading JANG v1 → v2: {path}")
    print(f"  This repacks {sum(1 for _ in path.glob('*.jang.safetensors'))} JANG shards to MLX-native format...")

    # Run the v1 repack to get MLX-format tensors
    result, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        # Collect all repacked tensors
        all_tensors = {}
        if tmp_dir is not None:
            for sf in result:
                data = mx.load(sf)
                for k, v in data.items():
                    all_tensors[k] = np.array(v)
                del data
                gc.collect()
        else:
            for k, v in result.items():
                all_tensors[k] = np.array(v)
            del result
            gc.collect()

        print(f"  Repacked {len(all_tensors)} tensors to MLX-native format")

        # Write v2 safetensors
        from safetensors.numpy import save_file

        # Shard into ~5 GB files
        max_shard = 5 * 1024 ** 3
        shards = []
        current_shard = {}
        current_size = 0
        weight_map = {}
        total_size = 0
        shard_idx = 0

        for name in sorted(all_tensors.keys()):
            arr = all_tensors[name]
            arr_bytes = arr.nbytes
            if current_size + arr_bytes > max_shard and current_shard:
                n_shards_est = max(1, sum(a.nbytes for a in all_tensors.values()) // max_shard + 1)
                shard_name = f"model-{shard_idx + 1:05d}-of-{n_shards_est:05d}.safetensors"
                shards.append((shard_name, current_shard))
                shard_idx += 1
                current_shard = {}
                current_size = 0
            current_shard[name] = arr
            current_size += arr_bytes
            total_size += arr_bytes

        if current_shard:
            shard_idx += 1
            shards.append((f"placeholder", current_shard))

        # Fix shard names with correct total count
        n_shards = len(shards)
        final_shards = []
        for i, (_, shard_data) in enumerate(shards):
            shard_name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
            final_shards.append((shard_name, shard_data))
            for tensor_name in shard_data:
                weight_map[tensor_name] = shard_name

        # Write new safetensors files
        for shard_name, shard_data in final_shards:
            save_file(shard_data, str(path / shard_name))
            print(f"  Wrote {shard_name} ({sum(a.nbytes for a in shard_data.values()) / 1e9:.1f} GB)")

        # Write v2 index
        index = {
            "metadata": {
                "format": "jang",
                "format_version": "2.0",
                "total_size": total_size,
            },
            "weight_map": weight_map,
        }
        (path / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )

        # Update config.json with quantization key
        bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [4])
        model_config["quantization"] = {
            "group_size": block_size,
            "bits": min(bit_widths),
        }
        (path / "config.json").write_text(
            json.dumps(model_config, indent=2, ensure_ascii=False) + "\n"
        )

        # Update jang_config.json version
        jang_cfg["format_version"] = "2.0"
        config_path.write_text(
            json.dumps(jang_cfg, indent=2, ensure_ascii=False) + "\n"
        )

        # Remove old v1 files
        old_files = list(path.glob("*.jang.safetensors"))
        old_files += list(path.glob("*.jjqf.safetensors"))
        old_files += list(path.glob("*.mxq.safetensors"))
        for old_idx_name in ["model.jang.index.json", "model.jjqf.index.json", "model.mxq.index.json"]:
            old_idx = path / old_idx_name
            if old_idx.exists():
                old_files.append(old_idx)

        for f in old_files:
            f.unlink()
            print(f"  Removed {f.name}")

        print(f"\n  Upgrade complete! Model now loads instantly via mx.load() mmap.")
        print(f"  v2 shards: {n_shards}, total: {total_size / 1e9:.1f} GB")

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
