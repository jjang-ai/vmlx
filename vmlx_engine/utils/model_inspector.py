# SPDX-License-Identifier: Apache-2.0
"""Model inspection utilities for vmlx-engine.

Provides lightweight model metadata analysis without loading weights.
Used by: convert, info, doctor, and list commands.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("vmlx_engine")


@dataclass
class ModelInfo:
    """Structured metadata about an MLX or HuggingFace model."""

    model_path: str
    model_type: str  # from config.json "model_type"
    architecture: str  # first entry in "architectures" list
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    param_count_billions: float = 0.0
    is_quantized: bool = False
    quant_bits: Optional[int] = None
    quant_group_size: Optional[int] = None
    quant_mode: Optional[str] = None
    needs_latent_moe: bool = False
    moe_latent_size: Optional[int] = None
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    is_mllm: bool = False
    is_hybrid: bool = False  # Has Mamba/SSM layers (e.g., Nemotron-H)
    hybrid_pattern: Optional[str] = None
    is_jang: bool = False  # JANG mixed-precision quantization format
    jang_target_bits: Optional[float] = None
    jang_actual_bits: Optional[float] = None
    jang_block_size: Optional[int] = None
    weight_files: list[str] = field(default_factory=list)
    total_weight_size_gb: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts is not None and self.n_routed_experts > 0

    @property
    def active_params_billions(self) -> float:
        """For MoE models, estimate active parameters per token."""
        if not self.is_moe or not self.num_experts_per_tok:
            return self.param_count_billions
        # Rough: non-expert params + (experts_per_tok / total_experts) * expert_params
        # Expert params are roughly (num_layers * intermediate * hidden * 3 * n_experts)
        # This is an approximation
        if self.n_routed_experts > 0:
            expert_fraction = self.num_experts_per_tok / self.n_routed_experts
            # Assume ~60% of params are in experts for typical MoE
            non_expert = self.param_count_billions * 0.4
            expert = self.param_count_billions * 0.6 * expert_fraction
            return non_expert + expert
        return self.param_count_billions


def inspect_model(model_path: str) -> ModelInfo:
    """
    Inspect a model directory and return structured metadata.

    Works with local directories only. For HuggingFace IDs, resolve
    to a local path first using resolve_model_path().

    Args:
        model_path: Path to model directory containing config.json

    Returns:
        ModelInfo with all available metadata

    Raises:
        FileNotFoundError: If model_path doesn't exist or has no config.json
    """
    path = Path(model_path)

    if not path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"If this is a HuggingFace ID, download it first:\n"
            f"  huggingface-cli download {model_path}"
        )

    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found in {model_path}\n"
            f"This directory may not contain a valid model."
        )

    with open(config_path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise FileNotFoundError(
                f"Invalid JSON in {config_path}: {e}"
            ) from e

    # Collect weight files
    weight_files = sorted(
        [f.name for f in path.glob("*.safetensors")]
    )
    total_size = sum(
        (path / f).stat().st_size for f in weight_files
    )
    total_size_gb = total_size / (1024**3)

    # Extract architecture info
    architectures = config.get("architectures", [])
    architecture = architectures[0] if architectures else config.get("model_type", "unknown")
    model_type = config.get("model_type", "unknown")

    # Dimensions — VLM models often nest text config inside "text_config".
    # Use _cfg() helper to check config first, then text_config, treating
    # None (absent) as "try next source" but preserving explicit 0.
    text_config = config.get("text_config", {})

    def _cfg(key, default=None):
        """Get from config or text_config, distinguishing None from 0."""
        v = config.get(key)
        if v is not None:
            return v
        v = text_config.get(key)
        if v is not None:
            return v
        return default

    hidden_size = _cfg("hidden_size", 0)
    num_layers = _cfg("num_hidden_layers", 0)
    num_heads = _cfg("num_attention_heads", 0)
    num_kv_heads = _cfg("num_key_value_heads", num_heads)
    vocab_size = _cfg("vocab_size", 0)
    intermediate_size = _cfg("intermediate_size", 0)

    # Quantization
    quant_config = config.get("quantization", config.get("quantization_config", {}))
    is_quantized = bool(quant_config)
    quant_bits = quant_config.get("bits") if quant_config else None
    quant_group_size = quant_config.get("group_size") if quant_config else None
    quant_mode = quant_config.get("mode") if quant_config else None

    # MoE — check both top-level and text_config
    n_routed_experts = _cfg("n_routed_experts") or _cfg("num_local_experts")
    num_experts_per_tok = _cfg("num_experts_per_tok")
    moe_latent_size = _cfg("moe_latent_size")
    needs_latent_moe = (
        model_type == "nemotron_h"
        and moe_latent_size is not None
    )

    # Hybrid (Mamba + attention)
    hybrid_pattern = config.get("hybrid_override_pattern")
    is_hybrid = hybrid_pattern is not None

    # Multimodal
    is_mllm = "vision_config" in config or "visual" in config

    # Estimate parameter count
    param_count = _estimate_param_count(config)

    # JANG format detection
    is_jang = False
    jang_target_bits = None
    jang_actual_bits = None
    jang_block_size = None
    for jang_cfg_name in ("jang_config.json", "jjqf_config.json", "mxq_config.json"):
        jang_cfg_path = path / jang_cfg_name
        if jang_cfg_path.exists():
            try:
                jang_meta = json.loads(jang_cfg_path.read_text())
                if jang_meta.get("format") in ("jang", "jjqf", "mxq"):
                    is_jang = True
                    q = jang_meta.get("quantization", {})
                    jang_target_bits = q.get("target_bits")
                    jang_actual_bits = q.get("actual_bits")
                    jang_block_size = q.get("block_size", 64)
                    is_quantized = True
                    break
            except (json.JSONDecodeError, OSError):
                pass

    return ModelInfo(
        model_path=str(path.resolve()),
        model_type=model_type,
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        param_count_billions=param_count,
        is_quantized=is_quantized,
        quant_bits=quant_bits,
        quant_group_size=quant_group_size,
        quant_mode=quant_mode,
        needs_latent_moe=needs_latent_moe,
        moe_latent_size=moe_latent_size,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        is_mllm=is_mllm,
        is_hybrid=is_hybrid,
        hybrid_pattern=hybrid_pattern,
        is_jang=is_jang,
        jang_target_bits=jang_target_bits,
        jang_actual_bits=jang_actual_bits,
        jang_block_size=jang_block_size,
        weight_files=weight_files,
        total_weight_size_gb=total_size_gb,
        config=config,
    )


def _estimate_param_count(config: dict) -> float:
    """
    Estimate total parameter count in billions from config dimensions.

    This is an approximation — errs ~10-15% high, which is conservative
    for memory warnings.
    """

    def _get(key, default=0):
        """Get from config or text_config, skipping None but not 0."""
        v = config.get(key)
        if v is not None:
            return v
        v = tc.get(key)
        if v is not None:
            return v
        return default

    # VLM models nest text dimensions in "text_config"
    tc = config.get("text_config", {})

    hidden = _get("hidden_size")
    layers = _get("num_hidden_layers")
    vocab = _get("vocab_size")
    intermediate = _get("intermediate_size")
    num_heads = _get("num_attention_heads")
    num_kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim") or (hidden // max(num_heads, 1))

    if hidden == 0 or layers == 0:
        return 0.0

    # Embedding + LM head (respect tied embeddings)
    tie_word_embeddings = config.get("tie_word_embeddings", False)
    embed_params = vocab * hidden * (1 if tie_word_embeddings else 2)

    # Per-layer attention
    q_params = hidden * (num_heads * head_dim)
    k_params = hidden * (num_kv_heads * head_dim)
    v_params = hidden * (num_kv_heads * head_dim)
    o_params = (num_heads * head_dim) * hidden
    attn_params = q_params + k_params + v_params + o_params

    # Determine MoE configuration
    n_experts = _get("n_routed_experts") or _get("num_local_experts") or 1
    moe_intermediate = _get("moe_intermediate_size") or intermediate
    latent_size = _get("moe_latent_size")
    model_type = config.get("model_type", "")

    # Determine expert projection count:
    # - Nemotron-H uses SwitchMLP (2 projections: fc1 + fc2, ReLU² activation)
    # - Most other MoE models use SwitchGLU (3 projections: gate + up + down)
    uses_switch_mlp = model_type in ("nemotron_h", "nemotron")
    expert_proj_count = 2 if uses_switch_mlp else 3

    # Dense MLP projection count:
    # - Nemotron-H dense MLP: 2 projections (up + down, ReLU²)
    # - Most models: 3 projections (gate + up + down, SwiGLU)
    dense_proj_count = 2 if uses_switch_mlp else 3

    if n_experts > 1:
        # MoE expert params
        expert_input_dim = hidden
        if latent_size:
            # LatentMoE: experts operate in latent space
            expert_input_dim = latent_size
        expert_params = n_experts * (expert_proj_count * expert_input_dim * moe_intermediate)

        # LatentMoE: add fc1_latent_proj + fc2_latent_proj (shared, not per-expert)
        if latent_size:
            expert_params += 2 * hidden * latent_size

        # Shared expert(s) — operate in full hidden_size
        n_shared = _get("n_shared_experts") or 0
        shared_intermediate = (
            _get("moe_shared_expert_intermediate_size")
            or (intermediate if n_shared > 0 else 0)
        )
        shared_params = n_shared * (dense_proj_count * hidden * shared_intermediate)

        # Gate params (routing weights)
        gate_params = n_experts * hidden
        moe_mlp_params = expert_params + shared_params + gate_params
    else:
        moe_mlp_params = 0

    # Dense MLP params (for non-MoE layers)
    dense_mlp_params = dense_proj_count * hidden * intermediate

    # Count layers by type
    hybrid_pattern = config.get("hybrid_override_pattern", "")
    first_k_dense = _get("first_k_dense_replace")

    if hybrid_pattern:
        # Hybrid model: M=Mamba, E=MoE, *=attention (dense MLP)
        attn_count = hybrid_pattern.count("*")
        moe_count = hybrid_pattern.count("E")
        mamba_count = hybrid_pattern.count("M")

        # Mamba2 params (Nemotron-H specific)
        mamba_num_heads = _get("mamba_num_heads", 128)
        mamba_head_dim = _get("mamba_head_dim", 64)
        ssm_state_size = _get("ssm_state_size", 128)
        n_groups = _get("n_groups", 8)
        conv_kernel = _get("conv_kernel", 4)

        mamba_intermediate = mamba_num_heads * mamba_head_dim
        conv_dim = mamba_intermediate + 2 * n_groups * ssm_state_size
        projection_size = mamba_intermediate + conv_dim + mamba_num_heads

        mamba_params_per_layer = (
            hidden * projection_size  # in_proj
            + mamba_intermediate * hidden  # out_proj
            + conv_dim * conv_kernel  # conv1d (depthwise)
            + mamba_num_heads * 3  # dt_bias + A_log + D
            + mamba_intermediate  # norm
        )

        # Attention layers have dense MLP, MoE layers have MoE MLP
        total_layer_params = (
            attn_count * (attn_params + dense_mlp_params)
            + moe_count * moe_mlp_params
            + mamba_count * mamba_params_per_layer
        )

        # RMSNorm for all layers
        total_layer_params += layers * hidden * 2
    elif n_experts > 1 and first_k_dense:
        # MoE model with some initial dense layers (e.g., GLM-4.7)
        dense_layers = first_k_dense
        moe_layers = layers - dense_layers
        total_layer_params = (
            dense_layers * (attn_params + dense_mlp_params + hidden * 2)
            + moe_layers * (attn_params + moe_mlp_params + hidden * 2)
        )
    elif n_experts > 1:
        # Pure MoE model (all layers have attention + MoE)
        total_layer_params = layers * (attn_params + moe_mlp_params + hidden * 2)
    else:
        # Standard dense model
        total_layer_params = layers * (attn_params + dense_mlp_params + hidden * 2)

    total = embed_params + total_layer_params
    return total / 1e9


def estimate_memory_gb(info: ModelInfo, bits: int = 16) -> float:
    """
    Estimate memory needed to load a model at the given quantization level.

    Includes a 20% overhead factor for activations, KV cache, and framework overhead.

    Args:
        info: ModelInfo from inspect_model()
        bits: Quantization bits (2, 3, 4, 6, 8, 16)

    Returns:
        Estimated memory in GB
    """
    # Base: params * bits / 8 bytes
    base_gb = info.param_count_billions * bits / 8

    # 20% overhead for activations, KV cache bootstrap, framework
    return base_gb * 1.2


def estimate_conversion_memory_gb(info: ModelInfo, target_bits: int) -> float:
    """
    Estimate memory needed during conversion.

    Conversion loads in source dtype (lazy, but still needs memory for
    quantization operations), plus the target model.

    Args:
        info: ModelInfo from inspect_model()
        target_bits: Target quantization bits

    Returns:
        Estimated peak memory during conversion in GB
    """
    # Source is loaded lazily but quantization materializes weights
    source_bits = info.quant_bits if info.is_quantized else 16
    source_gb = info.param_count_billions * source_bits / 8
    target_gb = info.param_count_billions * target_bits / 8

    # Peak: source weights + target weights + overhead
    return (source_gb + target_gb) * 1.1


def available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback: use sysctl on macOS
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            # Total memory * 0.7 as rough available estimate
            total = int(result.stdout.strip()) / (1024**3)
            return total * 0.7
        except Exception:
            return 16.0  # Conservative default


def total_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            return int(result.stdout.strip()) / (1024**3)
        except Exception:
            return 16.0


def list_models_in_dir(search_path: str) -> list[ModelInfo]:
    """
    Scan a directory for MLX/HuggingFace models.

    Looks for subdirectories containing both config.json and at least
    one .safetensors file.

    Args:
        search_path: Directory to scan (non-recursive, checks immediate subdirs)

    Returns:
        List of ModelInfo sorted by model_type
    """
    path = Path(search_path)
    if not path.is_dir():
        return []

    models = []

    # Check if the path itself is a model directory
    if (path / "config.json").exists() and list(path.glob("*.safetensors")):
        try:
            models.append(inspect_model(str(path)))
        except Exception as e:
            logger.debug(f"Failed to inspect {path}: {e}")

    # Check subdirectories
    for subdir in sorted(path.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / "config.json").exists() and list(subdir.glob("*.safetensors")):
            try:
                models.append(inspect_model(str(subdir)))
            except Exception as e:
                logger.debug(f"Failed to inspect {subdir}: {e}")

    return sorted(models, key=lambda m: (m.model_type, m.model_path))


def resolve_model_path(model_name: str) -> str:
    """
    Resolve a model name to a local directory path.

    Checks:
    1. If model_name is already a local directory, return it
    2. If it's a HuggingFace ID, check the local HF cache
    3. If not cached, raise with download instructions

    Args:
        model_name: Local path or HuggingFace model ID

    Returns:
        Absolute path to model directory

    Raises:
        FileNotFoundError: If model not found locally
    """
    # Check local path first
    local = Path(model_name)
    if local.is_dir():
        return str(local.resolve())

    # Check HuggingFace cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                # Find the latest revision
                for revision in sorted(repo.revisions, key=lambda r: r.last_modified or 0, reverse=True):
                    snapshot_path = revision.snapshot_path
                    if (Path(snapshot_path) / "config.json").exists():
                        return str(snapshot_path)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Model not found locally: {model_name}\n"
        f"Download it first:\n"
        f"  huggingface-cli download {model_name}\n"
        f"Or provide a local directory path."
    )


def format_model_info(info: ModelInfo) -> str:
    """Format ModelInfo for human-readable display."""
    lines = []
    lines.append(f"Model: {info.model_path}")
    lines.append(f"Architecture: {info.architecture} ({info.model_type})")
    lines.append(f"Parameters: {info.param_count_billions:.1f}B total")

    if info.is_moe:
        lines.append(
            f"  MoE: {info.n_routed_experts} experts, "
            f"{info.num_experts_per_tok} active per token, "
            f"~{info.active_params_billions:.1f}B active"
        )
    if info.needs_latent_moe:
        lines.append(f"  LatentMoE: latent_size={info.moe_latent_size} (vMLX patched)")
    if info.is_hybrid:
        m = info.hybrid_pattern.count("M") if info.hybrid_pattern else 0
        e = info.hybrid_pattern.count("E") if info.hybrid_pattern else 0
        a = info.hybrid_pattern.count("*") if info.hybrid_pattern else 0
        lines.append(f"  Hybrid: {m} Mamba + {e} MoE + {a} Attention layers")

    lines.append(f"Layers: {info.num_layers}")
    lines.append(f"Hidden size: {info.hidden_size}")
    lines.append(f"Vocab size: {info.vocab_size:,}")

    if info.is_jang:
        bits_str = f"{info.jang_actual_bits:.1f}" if info.jang_actual_bits else f"{info.jang_target_bits}"
        lines.append(
            f"Quantization: JANG {bits_str}-bit mixed-precision "
            f"(block_size={info.jang_block_size})"
        )
    elif info.is_quantized:
        lines.append(
            f"Quantization: {info.quant_bits}-bit "
            f"(group_size={info.quant_group_size}, mode={info.quant_mode})"
        )
    else:
        lines.append("Quantization: None (full precision)")

    if info.is_mllm:
        lines.append("Multimodal: Yes (vision)")

    lines.append(f"Weight files: {len(info.weight_files)} ({info.total_weight_size_gb:.1f} GB)")

    # Memory estimates
    lines.append("")
    lines.append("Estimated memory for inference:")
    total_mem = total_memory_gb()
    for bits in [2, 3, 4, 8, 16]:
        mem = estimate_memory_gb(info, bits)
        fit = "OK" if mem < total_mem * 0.9 else "TIGHT" if mem < total_mem else "TOO LARGE"
        marker = " <--" if info.quant_bits == bits else ""
        lines.append(f"  {bits:2d}-bit: {mem:6.1f} GB  [{fit}]{marker}")
    lines.append(f"  System: {total_mem:.0f} GB unified memory")

    return "\n".join(lines)
