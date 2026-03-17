"""
JANG Model Loader — Load JANG quantized models into MLX for inference.
Created by Jinho Jang (eric@jangq.ai)

Repacks JANG weights into MLX native quantized format (QuantizedLinear /
QuantizedSwitchLinear). Models stay quantized in GPU memory — no float16
expansion. Dequantization happens on-the-fly in Metal kernels via
quantized_matmul and gather_qmm, like GGUF stays quantized in llama.cpp.

Memory-efficient: repacked tensors are flushed to temporary safetensors
shards on disk (~5 GB each) instead of accumulating in RAM. The caller
loads them via mx.load() which uses mmap, so peak memory stays close to
1x model size even for 100+ GB models.
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

# Shard flush threshold in bytes (~2 GB).
# Smaller shards reduce peak memory during mx.save_safetensors (which
# must materialize all arrays in the shard before writing to disk).
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


def _is_vlm_config(model_path: Path) -> bool:
    """Check if a model has vision_config in its config.json (i.e., is a VL model)."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())
    return "vision_config" in config


def load_jang_vlm_model(model_path: str | Path):
    """
    Load a JANG Vision-Language model into mlx-vlm for multimodal inference.

    Vision tensors are stored as float16 in JANG format (unquantized).
    Language model tensors are repacked from JANG into MLX QuantizedLinear format.
    The mlx-vlm model skeleton provides the correct architecture with
    vision_tower + language_model submodules.

    Returns:
        Tuple of (model, processor) compatible with mlx-vlm.generate()
    """
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args, load_config as vlm_load_config,
        update_module_configs, load_image_processor,
        load_processor, skip_multimodal_module,
    )

    path = Path(model_path)
    start = time.perf_counter()

    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    jang_cfg = json.loads(config_path.read_text())
    fmt = jang_cfg.get("format")
    if not fmt or fmt not in JANG_FORMAT_VALUES:
        raise ValueError(f"Not a JANG model: format='{fmt}'")

    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", 4)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")
    bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [2, 4, 6, 8])
    default_bits = min(bit_widths)

    logger.info(
        f"Loading JANG VLM: {source_model} "
        f"({actual_bits:.1f}-bit avg, block_size={block_size})"
    )

    # --- Build mlx-vlm model skeleton ---
    config = vlm_load_config(path)
    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)

    model = model_class.Model(model_config)

    # --- Repack JANG weights ---
    shard_files, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        # Collect all repacked weight keys to determine which layers need quantization
        all_weight_keys = set()
        for sf in shard_files:
            data = mx.load(sf)
            all_weight_keys.update(data.keys())
            del data
            gc.collect()

        # Quantize ONLY language model layers (vision stays float16)
        # Use nn.quantize with a predicate that checks for .scales in our weights
        quantization = {"group_size": block_size, "bits": default_bits}

        def get_class_predicate(p, m):
            # Skip vision/audio modules entirely
            if skip_multimodal_module(p):
                return False
            if not hasattr(m, "to_quantized"):
                return False
            # Only quantize if we have scales for this layer
            return f"{p}.scales" in all_weight_keys

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=get_class_predicate,
        )

        # Load repacked weights shard by shard
        logger.info(f"  Loading {len(shard_files)} repacked shards via mmap")
        from mlx_vlm.utils import sanitize_weights
        for sf in shard_files:
            shard_weights = mx.load(sf)

            # Apply model-specific sanitization chain (same as mlx-vlm load_model):
            # 1. Top-level model.sanitize: renames keys, handles conv1d, norm +1 offsets
            # 2. VisionModel.sanitize: transposes patch_embed conv weight to MLX format
            # 3. LanguageModel.sanitize: any language-model-specific transforms
            if hasattr(model, "sanitize"):
                shard_weights = model.sanitize(shard_weights)

            shard_weights = sanitize_weights(
                model_class.VisionModel, shard_weights, model_config.vision_config
            )
            shard_weights = sanitize_weights(
                model_class.LanguageModel, shard_weights, model_config.text_config
            )

            model.load_weights(list(shard_weights.items()), strict=False)
            del shard_weights
            gc.collect()

        _fix_quantized_bits(model, {})
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = model_config

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    logger.info(f"JANG VLM loaded in {elapsed:.1f}s")

    # --- Load processor (tokenizer + image processor) ---
    image_processor = load_image_processor(path)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    try:
        processor = load_processor(path, True, eos_token_ids=eos_token_id)
    except (ImportError, ValueError):
        # Qwen3 VL processor requires torchvision for its video processor
        # component even for image-only inference. Construct it manually
        # with a stub video processor to avoid the PyTorch dependency.
        processor = _build_vlm_processor(path, eos_token_id)
    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def _build_vlm_processor(model_path: Path, eos_token_id=None):
    """
    Build a VLM processor without requiring PyTorch/torchvision.

    Some VL processors (e.g., Qwen3VLProcessor) require torchvision for their
    video processor component, even when doing image-only inference. This
    constructs the processor manually with a stub video processor and adds
    the detokenizer/stopping criteria that mlx-vlm expects.
    """
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers.processing_utils import ProcessorMixin
    from mlx_vlm.tokenizer_utils import load_tokenizer as vlm_load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    # Get the expected processor class from model config
    config = json.loads((model_path / "config.json").read_text())
    model_type = config.get("model_type", "")

    # Load chat template from tokenizer config
    tok_config_path = model_path / "tokenizer_config.json"
    chat_template = None
    if tok_config_path.exists():
        chat_template = json.loads(tok_config_path.read_text()).get("chat_template")

    # Try to construct the correct processor class with a stub video processor.
    # The stub avoids importing torchvision while satisfying the isinstance check.
    processor = None
    try:
        from transformers.video_processing_utils import BaseVideoProcessor
        video_stub = BaseVideoProcessor()

        # Map model_type to processor class
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
            # Patch the class check to accept our stub video processor.
            # transformers resolves BaseVideoProcessor to a dummy class when
            # torchvision is missing, causing isinstance to fail.
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
        # Last resort: use a simple wrapper
        class _SimpleVLMProcessor:
            def __init__(self, tok, ip):
                self.tokenizer = tok
                self.image_processor = ip
            def __call__(self, *a, **kw):
                return self.tokenizer(*a, **kw)
        processor = _SimpleVLMProcessor(tokenizer, image_processor)

    # Add detokenizer + stopping criteria (required by mlx-vlm generate)
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


def load_jang_model(model_path: str | Path):
    """
    Load a JANG model by repacking weights into MLX quantized format.

    Uses disk-backed temporary shards so that even 100+ GB models can
    load on machines with limited unified memory (e.g. 256 GB Mac Studio
    running a 397B model).

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

    # Decide: in-memory fast path vs disk streaming path.
    # If model weight bytes fit in 90% of system RAM, load entirely in-memory (fast).
    # Otherwise use disk-backed temp shards (slower but avoids OOM on 397B+ models).
    model_bytes = jang_cfg.get("runtime", {}).get("total_weight_bytes", 0)
    try:
        import psutil
        system_ram = psutil.virtual_memory().total
    except ImportError:
        system_ram = 128 * 1024 * 1024 * 1024  # assume 128 GB
    use_fast_path = model_bytes > 0 and model_bytes < system_ram * 0.9

    if use_fast_path:
        # Fast in-memory path: repack all weights into a single dict, load at once
        logger.info(f"  Fast path: model {model_bytes / 1e9:.1f} GB fits in RAM ({system_ram / 1e9:.0f} GB)")
        all_weights = _repack_jang_to_mlx_inmemory(path, block_size, config)
        if hasattr(model, "sanitize"):
            all_weights = model.sanitize(all_weights)
        model.load_weights(list(all_weights.items()), strict=False)
        del all_weights
        gc.collect()
        _fix_quantized_bits(model, {})
    else:
        # Disk streaming path for very large models
        logger.info(f"  Streaming path: model {model_bytes / 1e9:.1f} GB exceeds 90% of RAM ({system_ram / 1e9:.0f} GB)")
        shard_files, tmp_dir = _repack_jang_to_mlx(path, block_size, config)
        try:
            logger.info(f"  Loading {len(shard_files)} repacked shards via mmap")
            for i, sf in enumerate(shard_files):
                shard_weights = mx.load(sf)
                if hasattr(model, "sanitize"):
                    shard_weights = model.sanitize(shard_weights)
                model.load_weights(list(shard_weights.items()), strict=False)
                del shard_weights
                gc.collect()
            _fix_quantized_bits(model, {})
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = config

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    from mlx.utils import tree_flatten
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    logger.info(
        f"JANG model loaded in {elapsed:.1f}s: "
        f"{n_params / 1e9:.1f}B params, {actual_bits:.1f}-bit avg"
    )

    tokenizer = load_tokenizer(
        path, eos_token_ids=config.get("eos_token_id", None)
    )

    return model, tokenizer


def _repack_jang_to_mlx_inmemory(
    model_path: Path,
    block_size: int,
    config: dict,
) -> dict[str, mx.array]:
    """Fast in-memory repack for models that fit in system RAM.
    Returns a single dict of all repacked weights (no disk I/O).
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

    # Read all tensors into memory
    weights: dict[str, np.ndarray] = {}
    for sf in shard_files:
        logger.info(f"  Loading shard: {sf.name}")
        handle = safe_open(str(sf), framework="numpy")
        for key in handle.keys():
            weights[key] = handle.get_tensor(key)

    # Classify tensors
    quantized_bases: set[str] = set()
    non_quantized_names: list[str] = []
    for name in weights:
        matched = False
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                quantized_bases.add(name[: -len(suffix)])
                matched = True
                break
        if not matched:
            non_quantized_names.append(name)

    logger.info(f"  {len(quantized_bases)} quantized tensors, {len(non_quantized_names)} non-quantized tensors")

    import re
    _per_expert_2d_pattern = re.compile(
        r".+\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\."
    )
    result: dict[str, mx.array] = {}
    expert_buffer: dict[str, mx.array] = {}
    bit_counts: dict[int, int] = {}

    # Process quantized tensors
    for base in sorted(quantized_bases):
        qweight_raw = weights[f"{base}.qweight"]
        jang_scales = weights[f"{base}.scales"].astype(np.float32)
        biases_key = f"{base}.biases"
        zeros_key = f"{base}.zeros"
        if biases_key in weights:
            jang_biases_raw = weights[biases_key].astype(np.float32)
        elif zeros_key in weights:
            jang_zeros = weights[zeros_key].astype(np.float32)
            jang_biases_raw = -jang_scales * jang_zeros
        else:
            jang_biases_raw = np.zeros_like(jang_scales)

        n_blocks = len(jang_scales)
        bits_key = f"{base}.bits"
        if bits_key in weights:
            bits = int(weights[bits_key].item())
        else:
            bits = int(config.get("quantization", {}).get("bits", 4))

        bit_counts[bits] = bit_counts.get(bits, 0) + qweight_raw.size

        shape_key = f"{base}.shape"
        if shape_key in weights:
            original_shape = tuple(int(d) for d in weights[shape_key])
        else:
            original_shape = None

        if original_shape and len(original_shape) == 2:
            out_dim, in_dim = original_shape
        elif original_shape and len(original_shape) == 3:
            n_experts, out_dim, in_dim = original_shape
        else:
            in_dim = n_blocks * block_size
            out_dim = (qweight_raw.size * 8) // (n_blocks * bits) if n_blocks > 0 else 0

        n_groups_per_row = in_dim // block_size
        if n_groups_per_row <= 0:
            n_groups_per_row = 1

        mlx_qweight = mx.array(np.frombuffer(qweight_raw.tobytes(), dtype=np.uint32).copy())
        if original_shape and len(original_shape) == 3:
            packed_per_row = (in_dim * bits + 31) // 32
            mlx_qweight = mlx_qweight.reshape(n_experts * out_dim, packed_per_row)
        else:
            packed_per_row = (in_dim * bits + 31) // 32
            if packed_per_row > 0 and out_dim > 0:
                mlx_qweight = mlx_qweight.reshape(out_dim, packed_per_row)

        mlx_scales = mx.array(jang_scales.reshape(-1, n_groups_per_row).astype(np.float16))
        mlx_biases = mx.array(jang_biases_raw.reshape(-1, n_groups_per_row).astype(np.float16))

        weight_key = f"{base}.weight"
        is_expert = _per_expert_2d_pattern.match(weight_key)
        if is_expert:
            expert_buffer[weight_key] = mlx_qweight
            expert_buffer[f"{base}.scales"] = mlx_scales
            expert_buffer[f"{base}.biases"] = mlx_biases
        else:
            result[weight_key] = mlx_qweight
            result[f"{base}.scales"] = mlx_scales
            result[f"{base}.biases"] = mlx_biases

    # Process non-quantized tensors
    for name in non_quantized_names:
        result[name] = mx.array(weights[name])

    # Stack per-expert buffers
    if expert_buffer:
        _stack_per_expert_weights(expert_buffer, config)
        result.update(expert_buffer)

    # Log bit distribution
    if bit_counts:
        total_elements = sum(bit_counts.values())
        dist_str = ", ".join(
            f"{bits}-bit: {count} ({count * 100 // total_elements}%)"
            for bits, count in sorted(bit_counts.items())
        )
        logger.info(f"  Bit distribution: {dist_str}")

    return result


def _repack_jang_to_mlx(
    model_path: Path,
    block_size: int,
    config: dict,
) -> tuple[list[str], str]:
    """
    Load JANG shards and repack quantized tensors into MLX format,
    flushing to temporary safetensors shards on disk every ~5 GB.

    Returns:
        (shard_file_paths, tmp_dir_path) — caller loads via mx.load (mmap)
        and must clean up tmp_dir when done.
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

    # Open all shards but don't read tensors yet (lazy access)
    shard_handles: dict[str, safe_open] = {}
    tensor_to_shard: dict[str, str] = {}
    all_tensor_names: list[str] = []

    for sf in shard_files:
        sf_str = str(sf)
        logger.info(f"  Indexing shard: {sf.name if hasattr(sf, 'name') else sf}")
        handle = safe_open(sf_str, framework="numpy")
        shard_handles[sf_str] = handle
        for key in handle.keys():
            tensor_to_shard[key] = sf_str
            all_tensor_names.append(key)

    class LazyTensors:
        """Dict-like access that reads from safetensors on demand."""
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

    # Classify tensors as quantized or non-quantized
    quantized_bases: set[str] = set()
    non_quantized_names: list[str] = []

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

    # Temp directory for repacked shards — use the model's volume to avoid
    # filling the boot drive (Mac Studio may have <1 GB free on /).
    # Try the model directory first, fall back to system temp.
    tmp_dir = None
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
    output_shards: list[str] = []
    current_shard: dict[str, mx.array] = {}
    current_bytes: int = 0
    shard_idx: int = 0
    bit_counts: dict[int, int] = {}
    # Buffer for per-expert 2D tensors that need stacking (MiniMax/Mixtral style).
    # These must all be collected before stacking, so they can't be flushed early.
    import re
    _per_expert_2d_pattern = re.compile(
        r".+\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\."
    )
    expert_buffer: dict[str, mx.array] = {}

    def _flush_shard():
        """Write current_shard to a temp safetensors file and clear it."""
        nonlocal current_shard, current_bytes, shard_idx
        if not current_shard:
            return
        shard_path = f"{tmp_dir}/shard_{shard_idx:04d}.safetensors"
        # Evaluate all arrays before saving to ensure they are materialized
        # in GPU memory (not lazy). This prevents mx.save_safetensors from
        # needing to allocate additional memory during serialization.
        mx.eval(*current_shard.values())
        mx.save_safetensors(shard_path, current_shard)
        output_shards.append(shard_path)
        logger.info(f"  Flushed shard {shard_idx} ({current_bytes / 1e9:.1f} GB, {len(current_shard)} tensors)")
        shard_idx += 1
        current_shard = {}
        current_bytes = 0
        gc.collect()

    def _add_to_shard(key: str, arr: mx.array):
        """Add a tensor to the current shard, flushing if threshold exceeded."""
        nonlocal current_bytes
        current_shard[key] = arr
        current_bytes += arr.nbytes
        if current_bytes >= _SHARD_FLUSH_BYTES:
            _flush_shard()

    # --- Process quantized tensors ---
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

        # Restore original shape to determine out_dim x in_dim
        shape_key = f"{base}.shape"
        if shape_key in raw_tensors:
            shape = tuple(int(x) for x in raw_tensors[shape_key])
        else:
            total_weights = n_blocks * block_size
            shape = _infer_weight_shape(base, config, total_weights)

        # Determine dimensions
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

        # Repack JANG uint8 -> MLX uint32
        packed_bytes = qweight_raw.tobytes()
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
            mlx_qweight = mlx_qweight.reshape(num_experts, expert_out, packed_per_row)
        else:
            mlx_qweight = mlx_qweight.reshape(out_dim, packed_per_row)

        # Scales and biases
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

        # Determine weight key name
        if shape is not None and len(shape) >= 3:
            weight_key = base
        else:
            weight_key = f"{base}.weight"

        # For Qwen3.5 MoE: pre-split gate_up_proj into gate_proj + up_proj
        # so sanitize doesn't try to split quantized uint32 data as float
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
            # Check if this is a per-expert 2D tensor (e.g., experts.5.gate_proj)
            # that needs to be buffered for stacking later.
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

        # Free numpy intermediates for this tensor
        del qweight_raw, jang_scales, jang_biases_raw, jang_biases, packed_bytes
        del mlx_qweight, mlx_scales, mlx_biases

    # --- Stack per-expert 2D weights (MiniMax/Mixtral style) ---
    # Expert buffer holds all per-expert 2D tensors; stack them into 3D
    # then add to the shard system.
    if expert_buffer:
        _stack_per_expert_weights(expert_buffer, config)
        # Add stacked results (and any unstacked leftovers) to shard
        for k, v in expert_buffer.items():
            _add_to_shard(k, v)
        expert_buffer.clear()
        gc.collect()

    # --- Non-quantized tensors (norms, biases, etc.) ---
    for name in non_quantized_names:
        arr = raw_tensors[name]
        if arr.dtype == np.float32:
            _add_to_shard(name, mx.array(arr))
        elif arr.dtype == np.float16:
            _add_to_shard(name, mx.array(arr))
        else:
            _add_to_shard(name, mx.array(arr.astype(np.float16)))

    # Close shard handles — done reading
    for handle in shard_handles.values():
        del handle
    shard_handles.clear()
    gc.collect()

    # --- Rename for mlx-vlm compatibility ---
    # Apply renames to whatever is still in the current (unflushed) shard
    rename_keys = []
    rename_keys += [(k, "vision_tower" + k[len("model.visual"):]) for k in list(current_shard.keys()) if k.startswith("model.visual")]
    rename_keys += [(k, "language_model.model" + k[len("model.language_model"):]) for k in list(current_shard.keys()) if k.startswith("model.language_model")]
    for old_k, new_k in rename_keys:
        current_shard[new_k] = current_shard.pop(old_k)

    # Flush remaining tensors
    _flush_shard()

    # For already-flushed shards, apply renames by rewriting if needed.
    # VLM renames only affect a small number of keys, so check if any
    # flushed shards contain keys that need renaming.
    _rename_keys_in_flushed_shards(output_shards, tmp_dir)

    total_blocks = sum(bit_counts.values())
    if total_blocks > 0:
        dist_str = ", ".join(
            f"{b}-bit: {c} ({100 * c // total_blocks}%)"
            for b, c in sorted(bit_counts.items())
        )
        logger.info(f"  Bit distribution: {dist_str}")

    logger.info(f"  Repacked into {len(output_shards)} temp shards in {tmp_dir}")
    return output_shards, tmp_dir


def _rename_keys_in_flushed_shards(shard_paths: list[str], tmp_dir: str):
    """
    Check already-flushed shards for keys that need VLM renaming.
    Rewrites the shard in-place if any keys match.
    """
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
    """
    Stack per-expert 2D quantized weights into 3D for QuantizedSwitchLinear.

    MiniMax/Mixtral store experts as: experts.0.w1.weight, experts.1.w1.weight, ...
    MLX's SwitchLinear expects: switch_mlp.gate_proj.weight [num_experts, out, packed]

    Mapping: w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj
    """
    import re

    # Detect per-expert pattern
    expert_pattern = re.compile(
        r"(.+)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\.weight$"
    )

    # Group by layer prefix and weight type
    expert_groups = {}  # (prefix, wtype) -> {expert_id: weight_key}
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
        # Fused QKV: (num_heads + 2*num_kv_heads) * head_dim x hidden
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
