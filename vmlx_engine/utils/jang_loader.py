"""
JANG Model Loader — Load JANG quantized models into MLX for inference.
Created by Jinho Jang (eric@jangq.ai)

Repacks JANG weights into MLX native quantized format (QuantizedLinear /
QuantizedSwitchLinear). Models stay quantized in GPU memory — no float16
expansion. Dequantization happens on-the-fly in Metal kernels via
quantized_matmul and gather_qmm, like GGUF stays quantized in llama.cpp.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Support current "jang_config.json" and legacy names
JANG_CONFIG_FILENAMES = ["jang_config.json", "jjqf_config.json", "jang_cfg.json", "mxq_config.json"]
JANG_FORMAT_VALUES = ["jang", "jjqf", "mxq"]


def _find_config_path(model_path: Path) -> Optional[Path]:
    for name in JANG_CONFIG_FILENAMES:
        p = model_path / name
        if p.exists():
            return p
    return None


def is_jang_model(model_path: str | Path) -> bool:
    """Check if a directory contains a JANG model."""
    return _find_config_path(Path(model_path)) is not None


def load_jang_model(model_path: str | Path):
    """
    Load a JANG model by repacking weights into MLX QuantizedLinear format.
    Weights stay quantized in GPU memory — no float16 expansion.

    Returns:
        Tuple of (model, tokenizer) compatible with mlx-lm
    """
    from mlx_lm.utils import load_config, load_model as _load_model_skeleton, load_tokenizer

    path = Path(model_path)
    start = time.perf_counter()

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

    version = jang_cfg.get("format_version", "1.0")
    major = int(version.split(".")[0])
    if major > 1:
        raise ValueError(
            f"Unsupported JANG format version: {version} (this loader supports 1.x)"
        )

    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
    target_bits = jang_cfg.get("quantization", {}).get("target_bits", 4)
    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", target_bits)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")

    logger.info(
        f"Loading JANG model: {source_model} "
        f"({actual_bits:.1f}-bit avg, block_size={block_size})"
    )

    config = load_config(path)

    # Set up quantization config so mlx_lm creates QuantizedLinear layers.
    # Use the COMPRESS tier bits as default — covers 94-99% of layers.
    bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [2, 4, 6, 8])
    default_bits = min(bit_widths)  # COMPRESS tier bits (most common)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config["quantization"] = {
        "group_size": block_size,
        "bits": default_bits,
    }

    model, config = _load_model_skeleton(
        path, lazy=True, strict=False, model_config=config
    )

    # Replace SwitchLinear with QuantizedSwitchLinear BEFORE loading weights
    _upgrade_switch_to_quantized(model, default_bits, block_size)

    # Check if model fits in 75% of system RAM for in-memory loading
    model_bytes = jang_cfg.get("runtime", {}).get("total_weight_bytes", 0)
    try:
        import psutil
        system_ram = psutil.virtual_memory().total
    except ImportError:
        system_ram = 128 * 1024 * 1024 * 1024
    fits_in_ram = model_bytes == 0 or model_bytes < system_ram * 0.75

    if fits_in_ram:
        # Standard path: repack all into one dict, load at once
        weights = _repack_jang_to_mlx(path, block_size, config)
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=False)
        _fix_quantized_bits(model, weights)
    else:
        # Large model path: repack into dict, then load in chunks to reduce peak RAM
        # Uses the SAME proven repack code, just feeds to model in batches
        logger.info(f"  Large model ({model_bytes / 1e9:.1f} GB) — chunked loading to reduce peak RAM")
        weights = _repack_jang_to_mlx(path, block_size, config)
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        # Load in chunks of ~2GB worth of keys
        import gc as _gc
        keys = list(weights.keys())
        chunk_size = 500  # keys per chunk
        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i:i + chunk_size]
            chunk = {k: weights[k] for k in chunk_keys}
            model.load_weights(list(chunk.items()), strict=False)
            # Delete from main dict to free RAM
            for k in chunk_keys:
                del weights[k]
            del chunk
            _gc.collect()
            logger.info(f"  Loaded chunk {i // chunk_size + 1} ({len(chunk_keys)} tensors)")

        _fix_quantized_bits(model, {})

    if not hasattr(model, "config"):
        model.config = config

    mx.eval(model.parameters())  # noqa: S307 — MLX tensor materialization
    elapsed = time.perf_counter() - start
    from mlx.utils import tree_flatten
    n_params = sum(
        p.size for _, p in tree_flatten(model.parameters())
    )
    logger.info(
        f"JANG model loaded in {elapsed:.1f}s: "
        f"{n_params / 1e9:.1f}B params, {actual_bits:.1f}-bit avg"
    )

    tokenizer = load_tokenizer(
        path, eos_token_ids=config.get("eos_token_id", None)
    )

    return model, tokenizer


def _repack_jang_to_mlx(
    model_path: Path,
    block_size: int,
    config: dict,
) -> dict[str, mx.array]:
    """Load JANG shards and repack quantized tensors into MLX QuantizedLinear format."""
    from safetensors.numpy import load_file

    INDEX_NAMES = ["model.jang.index.json", "model.jjqf.index.json", "model.mxq.index.json"]
    SHARD_GLOBS = ["*.jang.safetensors", "*.jjqf.safetensors", "*.mxq.safetensors"]
    SUFFIXES = (".qweight", ".scales", ".zeros", ".biases", ".bit_map", ".block_offsets", ".shape", ".bits")

    index_path = None
    for name in INDEX_NAMES:
        p = model_path / name
        if p.exists():
            index_path = p
            break

    raw_tensors: dict[str, np.ndarray] = {}

    if index_path:
        index = json.loads(index_path.read_text())
        for sf in sorted(set(index["weight_map"].values())):
            logger.info(f"  Loading shard: {sf}")
            raw_tensors.update(load_file(str(model_path / sf)))
    else:
        for pattern in SHARD_GLOBS:
            for sf in sorted(model_path.glob(pattern)):
                logger.info(f"  Loading shard: {sf.name}")
                raw_tensors.update(load_file(str(sf)))

    if not raw_tensors:
        raise FileNotFoundError(f"No JANG weight files found in {model_path}")

    quantized_bases: set[str] = set()
    non_quantized: dict[str, np.ndarray] = {}

    for name in raw_tensors:
        matched = False
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                quantized_bases.add(name[: -len(suffix)])
                matched = True
                break
        if not matched:
            non_quantized[name] = raw_tensors[name]

    logger.info(
        f"  {len(quantized_bases)} quantized tensors, "
        f"{len(non_quantized)} non-quantized tensors"
    )

    weights: dict[str, mx.array] = {}
    bit_counts: dict[int, int] = {}

    for base in sorted(quantized_bases):
        qweight_raw = raw_tensors[f"{base}.qweight"]
        jang_scales = raw_tensors[f"{base}.scales"].astype(np.float32)
        # v1.2+: biases stored directly (no precision loss)
        # v1.0-1.1: zeros stored, need conversion: bias = -scale * zero
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

        # Get bit width
        bits_key = f"{base}.bits"
        if bits_key in raw_tensors:
            bits = int(raw_tensors[bits_key][0])
        elif f"{base}.bit_map" in raw_tensors:
            bits = int(raw_tensors[f"{base}.bit_map"][0])
        else:
            logger.warning(f"  No bits info for {base}, assuming 4-bit")
            bits = 4

        bit_counts[bits] = bit_counts.get(bits, 0) + n_blocks

        # Restore original shape to determine out_dim × in_dim
        shape_key = f"{base}.shape"
        if shape_key in raw_tensors:
            shape = tuple(int(x) for x in raw_tensors[shape_key])
        else:
            total_weights = n_blocks * block_size
            shape = _infer_weight_shape(base, config, total_weights)

        # Determine dimensions
        is_3d = shape is not None and len(shape) >= 3
        if is_3d:
            # 3D expert tensors: [num_experts, out, in] — keep 3D for gather_qmm
            num_experts = shape[0]
            expert_out = shape[1]
            in_dim = shape[-1]
            out_dim = num_experts * expert_out  # for flat scale/qweight math
        elif shape is not None:
            num_experts = 0
            expert_out = 0
            out_dim, in_dim = shape
        else:
            num_experts = 0
            expert_out = 0
            out_dim = n_blocks
            in_dim = block_size

        # Repack JANG uint8 → MLX uint32
        # JANG packs LSB-first into uint8, MLX packs LSB-first into uint32.
        # For bit widths with fast paths (2, 4, 8), the byte layout is compatible —
        # just view the same bytes as uint32.
        # For 3, 5, 6-bit, the packing is also LSB-contiguous, so view-as-uint32 works.
        packed_bytes = qweight_raw.tobytes()
        # Pad to uint32 boundary
        pad_needed = (4 - len(packed_bytes) % 4) % 4
        if pad_needed:
            packed_bytes += b'\x00' * pad_needed
        mlx_qweight = np.frombuffer(packed_bytes, dtype=np.uint32)

        # Reshape packed uint32 weights
        packed_per_row = (in_dim * bits + 31) // 32
        expected_len = out_dim * packed_per_row
        if len(mlx_qweight) < expected_len:
            mlx_qweight = np.pad(mlx_qweight, (0, expected_len - len(mlx_qweight)))
        mlx_qweight = mlx_qweight[:expected_len]

        if is_3d:
            # 3D expert tensors: [num_experts, expert_out, packed_per_row]
            mlx_qweight = mlx_qweight.reshape(num_experts, expert_out, packed_per_row)
        else:
            mlx_qweight = mlx_qweight.reshape(out_dim, packed_per_row)

        # Scales and biases already in MLX format (no conversion needed for v1.2+)
        n_groups_per_row = (in_dim + block_size - 1) // block_size

        # Reshape scales/biases
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

        # Determine weight key name
        # 3D+ expert tensors don't have .weight suffix in HF naming
        if shape is not None and len(shape) >= 3:
            weight_key = base
        else:
            weight_key = f"{base}.weight"

        # For Qwen3.5 MoE: pre-split gate_up_proj into gate_proj + up_proj
        # so sanitize doesn't try to split quantized uint32 data as float
        if is_3d and "gate_up_proj" in base:
            mid = expert_out // 2
            # Split weight: [num_experts, mid, packed] and [num_experts, mid, packed]
            gate_w = mlx_qweight[:, :mid, :]
            up_w = mlx_qweight[:, mid:, :]
            gate_s = mlx_scales[:, :mid, :]
            up_s = mlx_scales[:, mid:, :]
            gate_b = mlx_biases[:, :mid, :]
            up_b = mlx_biases[:, mid:, :]

            # Replace experts.gate_up_proj → switch_mlp.gate_proj / up_proj
            sw_prefix = base.replace("experts.gate_up_proj", "switch_mlp")
            weights[f"{sw_prefix}.gate_proj.weight"] = mx.array(gate_w)
            weights[f"{sw_prefix}.gate_proj.scales"] = mx.array(gate_s)
            weights[f"{sw_prefix}.gate_proj.biases"] = mx.array(gate_b)
            weights[f"{sw_prefix}.up_proj.weight"] = mx.array(up_w)
            weights[f"{sw_prefix}.up_proj.scales"] = mx.array(up_s)
            weights[f"{sw_prefix}.up_proj.biases"] = mx.array(up_b)
        elif is_3d and "down_proj" in base:
            sw_prefix = base.replace("experts.down_proj", "switch_mlp")
            weights[f"{sw_prefix}.down_proj.weight"] = mx.array(mlx_qweight)
            weights[f"{sw_prefix}.down_proj.scales"] = mx.array(mlx_scales)
            weights[f"{sw_prefix}.down_proj.biases"] = mx.array(mlx_biases)
        elif not is_3d and "gate_up_proj" in base:
            # 2D fused gate_up_proj — pre-split to avoid sanitize corrupting uint32 data
            mid = out_dim // 2
            gate_w = mlx_qweight[:mid, :]
            up_w = mlx_qweight[mid:, :]
            gate_s = mlx_scales[:mid, :]
            up_s = mlx_scales[mid:, :]
            gate_b = mlx_biases[:mid, :]
            up_b = mlx_biases[mid:, :]

            gate_base = base.replace("gate_up_proj", "gate_proj")
            up_base = base.replace("gate_up_proj", "up_proj")
            weights[f"{gate_base}.weight"] = mx.array(gate_w)
            weights[f"{gate_base}.scales"] = mx.array(gate_s)
            weights[f"{gate_base}.biases"] = mx.array(gate_b)
            weights[f"{up_base}.weight"] = mx.array(up_w)
            weights[f"{up_base}.scales"] = mx.array(up_s)
            weights[f"{up_base}.biases"] = mx.array(up_b)
        else:
            # Standard tensor
            weights[weight_key] = mx.array(mlx_qweight)
            scale_key = weight_key.replace('.weight', '') if '.weight' in weight_key else weight_key
            weights[f"{scale_key}.scales"] = mx.array(mlx_scales)
            weights[f"{scale_key}.biases"] = mx.array(mlx_biases)

    # Stack per-expert 2D quantized weights into 3D for QuantizedSwitchLinear
    # (MiniMax-style: experts.N.w1.weight → switch_mlp.gate_proj.weight [num_experts, out, packed])
    _stack_per_expert_weights(weights, config)

    # Non-quantized tensors (norms, biases, etc.)
    for name, arr in non_quantized.items():
        if arr.dtype == np.float32:
            weights[name] = mx.array(arr)
        elif arr.dtype == np.float16:
            weights[name] = mx.array(arr)
        else:
            weights[name] = mx.array(arr.astype(np.float16))

    total_blocks = sum(bit_counts.values())
    if total_blocks > 0:
        dist_str = ", ".join(
            f"{b}-bit: {c} ({100 * c // total_blocks}%)"
            for b, c in sorted(bit_counts.items())
        )
        logger.info(f"  Bit distribution: {dist_str}")

    # Rename for mlx-vlm compatibility:
    # model.visual.* → vision_tower.* (VL models)
    # model.language_model.* → language_model.model.* (VL models)
    renamed = {}
    for k, v in weights.items():
        new_k = k
        if new_k.startswith("model.visual"):
            new_k = "vision_tower" + new_k[len("model.visual"):]
        elif new_k.startswith("model.language_model"):
            new_k = "language_model.model" + new_k[len("model.language_model"):]
        renamed[new_k] = v

    return renamed


def _stack_per_expert_weights(weights, config):
    """
    Stack per-expert 2D quantized weights into 3D for QuantizedSwitchLinear.

    MiniMax/Mixtral store experts as: experts.0.w1.weight, experts.1.w1.weight, ...
    MLX's SwitchLinear expects: switch_mlp.gate_proj.weight [num_experts, out, packed]

    Mapping: w1 → gate_proj, w2 → down_proj, w3 → up_proj
    """
    import re

    # Detect per-expert pattern
    expert_pattern = re.compile(
        r"(.+)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\.weight$"
    )

    # Group by layer prefix and weight type
    expert_groups = {}  # (prefix, wtype) → {expert_id: weight_key}
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

    # Mapping for MiniMax/Mixtral naming
    name_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    for (prefix, wtype), experts in expert_groups.items():
        if len(experts) < 2:
            continue

        num_experts = max(experts.keys()) + 1
        new_name = name_map.get(wtype, wtype)
        sw_key = f"{prefix}.switch_mlp.{new_name}"

        # Stack weights
        to_stack = [weights.pop(experts[e]) for e in range(num_experts)]
        weights[f"{sw_key}.weight"] = mx.stack(to_stack)

        # Try to find and stack scales/biases
        base_scale_key = list(experts.values())[0].replace(".weight", "")
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
    """Replace SwitchLinear layers with QuantizedSwitchLinear before weight loading."""
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

        # Replace in parent (handle list indices like layers.39)
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = model
            for p in parts[0].split('.'):
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, parts[1], ql)


def _fix_quantized_bits(model, weights):
    """
    Fix the `bits` attribute on each QuantizedLinear layer to match
    the actual bit width of its JANG-repacked weight.

    MLX QuantizedLinear uses `self.bits` in quantized_matmul. After nn.quantize()
    creates all layers at the default bits, we need to update layers whose JANG
    weights use a different bit width.
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
            in_dim = module.scales.shape[-1] * module.group_size
            actual_bits = (module.weight.shape[-1] * 32) // in_dim
            if actual_bits != module.bits and actual_bits in (2, 3, 4, 5, 6, 8):
                module.bits = actual_bits
        except Exception:
            pass


def _infer_weight_shape(
    base_name: str, config: dict, n_elements: int
) -> Optional[tuple[int, ...]]:
    """Infer original weight shape from config.json and tensor name.

    Handles standard, MoE, and VLM architectures. For MoE/VLM models
    that nest dimensions under text_config, falls back to nested lookup.
    """
    # Support nested configs (MoE/VLM models nest dims in text_config)
    tc = config.get("text_config", {})
    def _get(key: str, default=0):
        return config.get(key, tc.get(key, default))

    hidden = _get("hidden_size", 0)
    intermediate = _get("intermediate_size", 0)
    # MoE models may use moe_intermediate_size for expert MLPs
    moe_intermediate = _get("moe_intermediate_size", intermediate)
    shared_expert_intermediate = _get("shared_expert_intermediate_size", moe_intermediate)
    num_heads = _get("num_attention_heads", 0)
    num_kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim", hidden // num_heads if num_heads else 0)
    vocab_size = _get("vocab_size", 0)

    name = base_name.lower()

    # Attention projections
    if "qkv_proj" in name:
        # Fused QKV: (num_heads + 2*num_kv_heads) * head_dim × hidden
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

    # MoE expert weights — MUST check before general MLP (experts also have gate_proj)
    elif ".experts." in name or ".shared_expert." in name:
        ei = shared_expert_intermediate if ".shared_expert." in name else (moe_intermediate if moe_intermediate else intermediate)
        if "gate_proj" in name or "up_proj" in name or "w1" in name or "w3" in name:
            return (ei, hidden)
        elif "down_proj" in name or "w2" in name:
            return (hidden, ei)

    # MLP projections (standard and Llama-style naming)
    elif "gate_up_proj" in name:
        return (2 * intermediate, hidden)
    elif "gate_proj" in name or "up_proj" in name or "w1" in name or "w3" in name:
        return (intermediate, hidden)
    elif "down_proj" in name or "w2" in name:
        return (hidden, intermediate)

    # Embeddings
    elif "embed_tokens" in name:
        return (vocab_size, hidden)
    elif "lm_head" in name:
        return (vocab_size, hidden)

    # Fallback: try to infer from element count
    if n_elements > 0 and hidden > 0 and n_elements % hidden == 0:
        return (n_elements // hidden, hidden)

    logger.warning(
        f"  Could not infer shape for {base_name} ({n_elements} elements)"
    )
    return None
