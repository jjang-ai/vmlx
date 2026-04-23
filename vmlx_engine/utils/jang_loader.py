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
from typing import Any, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Support current "jang_config.json" and legacy names
JANG_CONFIG_FILENAMES = [
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
]
JANG_FORMAT_VALUES = ["jang", "jjqf", "mxq"]


def _set_wired_limit_for_model(weight_files):
    """Raise MLX wired memory limit to fit model + headroom.

    MLX's default wired limit is ~75% of the device's recommended max working
    set. Models larger than this default get pages swapped during eval,
    causing Metal command buffer timeouts.

    Sets wired limit to model_size + 8 GB headroom, capped at the OS's
    max working set (which the user can raise via:
        sudo sysctl iogpu.wired_limit_mb=250000
    persisted in /etc/sysctl.conf).

    Official MLX API (mx.set_wired_limit) — not a hack.
    """
    try:
        total_bytes = sum(sf.stat().st_size for sf in weight_files)
        # 8 GB headroom for activations, KV cache, scheduler overhead
        target = total_bytes + 8 * 1024 * 1024 * 1024
        # Cap at OS max working set (sysctl iogpu.wired_limit_mb)
        try:
            max_ws = mx.metal.device_info().get("max_recommended_working_set_size")
            if max_ws and target > max_ws:
                target = max_ws
        except Exception:
            pass
        if hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(target)
        else:
            mx.metal.set_wired_limit(target)
        logger.info(
            f"  Wired limit set to {target / 1e9:.0f} GB "
            f"(model {total_bytes / 1e9:.0f} GB)"
        )
    except Exception as e:
        logger.warning(f"  Could not set wired limit: {e}")


def _chunked_eval_params(model, chunk_size: int = 200):
    """Evaluate model parameters in chunks to avoid Metal GPU timeout on large models (>200GB)."""
    import mlx.utils as _mlx_utils

    _flat = _mlx_utils.tree_flatten(model.parameters())
    for _i in range(0, len(_flat), chunk_size):
        mx.eval(*[v for _, v in _flat[_i : _i + chunk_size]])


def _patch_turboquant_make_cache(model, jang_cfg: dict, model_config: dict):
    """Patch model.make_cache() to return TurboQuantKVCache for JANG models with TQ enabled.

    This is JANG-exclusive — only activates when jang_config.json has turboquant.enabled=true.
    Mirrors the patching done by jang-tools loader.py:226-280.

    Args:
        model: The language model object (has .layers and .make_cache())
        jang_cfg: Parsed jang_config.json dict
        model_config: Parsed config.json dict (or text_config for VLM)
    """
    # MLA models (DeepSeek V2/V3, GLM-5.1, Mistral 4) use CacheList(KVCache, KVCache)
    # per layer. TQ replaces this with flat TurboQuantKVCache which breaks the
    # CacheList structure → BatchGenerator's _make_cache fails → "not subscriptable"
    # error. Skip TQ for MLA models. Centralized via model_inspector.is_mla_model()
    # so the check stays in sync with tokenizer.py and Agent 1's prefix-cache trie
    # (REQ-001 in the 2026-04-07 audit).
    from .model_inspector import _detect_turboquant_layer_types, is_mla_model

    if is_mla_model(model_config):
        logger.info(
            "  TurboQuant skipped: MLA model uses CacheList (incompatible with TQ flat cache)"
        )
        return

    _tq_cfg = jang_cfg.get("turboquant")
    if not _tq_cfg:
        # Auto-enable TQ with defaults for ALL JANG models
        _tq_cfg = {
            "enabled": True,
            "default_key_bits": 3,
            "default_value_bits": 3,
            "critical_key_bits": 4,
            "critical_value_bits": 4,
            "critical_layers": [0, 1, 2, -3, -2, -1],
            "seed": 42,
        }
        logger.info("  TurboQuant auto-enabled (JANG model, no explicit config)")
    if not _tq_cfg.get("enabled", True):
        return

    try:
        from jang_tools.turboquant.config import TurboQuantConfig, make_turboquant_cache
    except ImportError:
        logger.warning("  TurboQuant config found but turboquant module not available")
        return

    n_layers = len(model.layers)
    # Use _tq_cfg (which may be auto-generated defaults) instead of re-reading jang_cfg
    tq_config = TurboQuantConfig.from_jang_config({"turboquant": _tq_cfg}, n_layers)
    if not tq_config:
        return

    # Get text config (may be nested under text_config for VLM wrappers)
    _text_cfg = model_config.get("text_config", model_config)

    _layer_types, _key_dim, _val_dim = _detect_turboquant_layer_types(
        _text_cfg, n_layers, root_cfg=model_config
    )

    _n_attn = sum(1 for t in _layer_types if t == "attention")
    _n_ssm = sum(1 for t in _layer_types if t == "ssm")
    _n_cache = len(_layer_types)
    _n_skip = n_layers - _n_cache
    if _n_ssm > 0 or _n_skip > 0:
        logger.info(
            f"  Hybrid model: {_n_attn} attention + {_n_ssm} SSM"
            + (f" + {_n_skip} no-cache" if _n_skip else "")
            + " layers"
        )

    def _turboquant_make_cache(
        _cfg=tq_config, _n=_n_cache, _kd=_key_dim, _vd=_val_dim, _lt=_layer_types
    ):
        return make_turboquant_cache(_cfg, _n, [_kd] * _n, [_vd] * _n, _lt)

    model.make_cache = _turboquant_make_cache
    logger.info(
        f"  TurboQuant enabled: {tq_config.default_key_bits}-bit keys, "
        f"{tq_config.default_value_bits}-bit values, "
        f"{len(tq_config.critical_layers)} critical layers"
    )


# Shard flush threshold for v1 streaming repack (~2 GB)
_SHARD_FLUSH_BYTES = 2_000_000_000


def _find_config_path(model_path: str | Path) -> Optional[Path]:
    path = Path(model_path)
    for name in JANG_CONFIG_FILENAMES:
        p = path / name
        if p.exists():
            return p
    # Fallback: JANGTQ converter embeds jang_config inside config.json["jang"].
    # Extract it to jang_config.json so the rest of the pipeline works.
    # Falls back to /tmp if the model dir is read-only (HF cache, etc.).
    cfg_path = path / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            if "jang" in cfg and isinstance(cfg["jang"], dict):
                jang_cfg_path = path / "jang_config.json"
                try:
                    jang_cfg_path.write_text(json.dumps(cfg["jang"], indent=2))
                except OSError:
                    import tempfile
                    jang_cfg_path = Path(tempfile.gettempdir()) / f"jang_config_{path.name}.json"
                    jang_cfg_path.write_text(json.dumps(cfg["jang"], indent=2))
                logger.info(f"  Extracted jang_config from config.json['jang'] → {jang_cfg_path}")
                return jang_cfg_path
        except Exception:
            pass
    return None


def _resolve_local_path(model_path: str | Path) -> Path:
    """Resolve a model path or HuggingFace model ID to a local directory.

    If model_path is already a local directory, returns it as-is.
    If it looks like a HF model ID (e.g. 'JANGQ-AI/Qwen3.5-27B-JANG_4S'),
    resolves to the local HF cache snapshot using local_files_only=True.
    Falls back to the original path if resolution fails.
    """
    path = Path(model_path)
    if path.is_dir():
        return path
    model_str = str(model_path)
    if "/" in model_str and not path.is_absolute():
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_str, local_files_only=True)
            return Path(local_dir)
        except Exception:
            pass
    return path


def is_jang_model(model_path: str | Path) -> bool:
    """Check if a directory contains a JANG model."""
    return _find_config_path(_resolve_local_path(model_path)) is not None


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
    # JANGTQ writer: integer `version` field, no `format_version`
    version = cfg.get("format_version", cfg.get("version", "1.0"))
    if str(version).startswith("2"):
        return True

    # JANGTQ: weight_format=mxtq is always v2-shaped (standard safetensors,
    # no .jang.safetensors repack needed, mmap load).
    if cfg.get("weight_format") == "mxtq":
        return True

    # Check for v2 index file (standard safetensors index alongside jang_config)
    if (model_path / "model.safetensors.index.json").exists():
        # Only v2 if no .jang.safetensors exist (v1 has .jang.safetensors)
        has_jang = any(model_path.glob("*.jang.safetensors"))
        if not has_jang:
            return True

    # Fallback: standard `model-NNNNN-of-NNNNN.safetensors` shards without
    # .jang.safetensors → treat as v2 (this covers JANGTQ models that don't
    # ship an index file).
    has_jang = any(model_path.glob("*.jang.safetensors"))
    has_shards = any(model_path.glob("model-*.safetensors"))
    if has_shards and not has_jang:
        return True

    return False


def _is_codebook_vq_model(model_path: str | Path) -> bool:
    """Check if a JANG model uses codebook VQ format.

    Codebook VQ models have:
    - jang_config.json with codebook_vq: true
    - codebook-layer-{NNN}-{type}.safetensors files
    """
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        return False

    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    # Must have codebook_vq flag
    if not cfg.get("quantization", {}).get("codebook_vq", False):
        return False

    # Must have codebook files
    has_codebook_files = any(path.glob("codebook-layer-*.safetensors"))
    return has_codebook_files


# ─── Codebook VQ loader ─────────────────────────────────────────────


def _load_codebook_vq_model(
    path: Path,
    jang_cfg: dict,
    config_manager: Optional[Any] = None,
):
    """
    Load a codebook VQ model - JANG v2 with codebook-compressed expert weights.

    Expert weights are stored as codebook + indices (VQ compressed).
    Non-expert weights (embeddings, norms, attention, shared expert) are standard JANG v2.

    Args:
        path: Model directory path
        jang_cfg: Parsed jang_config.json dict
        config_manager: Optional ConfigManager for settings

    Returns:
        Tuple of (CodebookVQLanguageModel, tokenizer)
    """
    from mlx_lm.utils import (
        load_config,
        load_model as _load_model_skeleton,
        load_tokenizer,
    )
    # Codebook VQ is an experimental JANG format. The `vmlx_engine/models/codebook.py`
    # module (and its `cache/`, `config/`, `metal/` siblings) are not committed to
    # the public `jjang-ai/vmlx` repo — they exist only in local dev installs.
    # Fresh clones that don't carry the experimental stack would otherwise crash
    # with a hard ImportError on the *first* codebook VQ model load. Guard the
    # import here so the error surface is a clean "feature not available" message
    # instead of a traceback leaking internal module layout.
    try:
        from vmlx_engine.models.codebook import CodebookVQLanguageModel
    except ImportError as _cb_err:
        raise RuntimeError(
            "Codebook VQ model format requires the experimental "
            "`vmlx_engine.models.codebook` module, which is not included in "
            "this build. To enable codebook VQ inference, install the "
            "experimental codebook stack (`vmlx_engine/cache/`, `config/`, "
            f"`metal/`, `models/codebook*.py`). Original error: {_cb_err}"
        ) from _cb_err

    start = time.perf_counter()

    # Determine codebook settings from config
    quant_cfg = jang_cfg.get("quantization", {})
    n_codes = quant_cfg.get("n_codes", 16384)
    group_size = quant_cfg.get("codebook_group_size", 8)

    # Count codebook files
    codebook_files = list(path.glob("codebook-layer-*.safetensors"))
    logger.info(f"  Codebook VQ: {len(codebook_files)} codebook files")
    logger.info(f"  Codebook settings: n_codes={n_codes}, group_size={group_size}")

    # Load base model (non-expert weights) via standard JANG v2 loader
    # This loads embeddings, norms, attention layers, shared expert
    base_model, tokenizer = _load_jang_v2(path, jang_cfg)

    # Wrap with codebook VQ wrapper
    model = CodebookVQLanguageModel(
        model_path=path,
        base_model=base_model,
        tokenizer=tokenizer,
        jang_config=jang_cfg,
        config_manager=config_manager,
    )

    _chunked_eval_params(model)

    elapsed = time.perf_counter() - start
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")
    logger.info(f"  Codebook VQ model loaded in {elapsed:.1f}s: {source_model}")

    return model, tokenizer


# ─── v2 loader (instant) ────────────────────────────────────────────


def _is_expert_key(k: str) -> bool:
    """Check if a weight key belongs to MoE experts (switch_mlp/switch_glu).

    Used by smelt mode to filter out expert weights during backbone-only loading.
    Expert weights are loaded separately via ExpertIndex + _load_expert_subset.
    """
    return "switch_mlp" in k or "switch_glu" in k


import re as _re
_LAYER_INDEX_RE = _re.compile(r"(?:layers|backbone\.layers)\.(\d+)\.")


def _filter_by_layer_range(weights: dict, start: int, end: int) -> dict:
    """Filter weights to only include a specific layer range.

    Keeps:
    - Weights for layers in [start, end)
    - Non-layer weights (embed_tokens, lm_head, norm, etc.)

    Used by distributed inference workers to load only their assigned layers.
    """
    filtered = {}
    for k, v in weights.items():
        m = _LAYER_INDEX_RE.search(k)
        if m:
            layer_idx = int(m.group(1))
            if start <= layer_idx < end:
                filtered[k] = v
            # else: skip — not in our range
        else:
            # Non-layer weight (embed_tokens, lm_head, norm, etc.)
            # Always include — coordinator needs embed+lm_head,
            # workers can ignore them (strict=False drops unused)
            filtered[k] = v
    return filtered


def _load_jang_v2(
    path: Path,
    jang_cfg: dict,
    skip_eval: bool = False,
    filter_expert_keys: bool = False,
    layer_range: tuple = None,
):
    """
    Load a JANG v2 model — instant via mx.load() mmap.

    v2 models store weights in MLX-native format (uint32 packed weights,
    float16 scales/biases) in standard safetensors. No repacking needed.

    Args:
        filter_expert_keys: If True, skip expert (switch_mlp/switch_glu) weights
            during loading. Used by smelt mode — experts are filled separately.
            Weights are mmap'd so filtering after load has no RAM penalty.
        layer_range: Optional (start, end) tuple. When set, only loads weights
            for layers in [start, end). Used by distributed inference workers
            to load only their assigned layer range. Embedding and lm_head
            weights are always loaded regardless of layer_range.
    """
    from mlx_lm.utils import (
        load_config,
        load_model as _load_model_skeleton,
        load_tokenizer,
    )

    start = time.perf_counter()
    config = load_config(path)

    # Mistral-Small-4-119B mismatch: HF config.json has top model_type="mistral3"
    # (the VLM wrapper class) but text_config.model_type="mistral4" (the inner
    # MLA language model). When loaded as text-only via mlx_lm, the top-level
    # model_type wins → mistral3 skeleton (standard q_proj/k_proj/v_proj
    # attention) gets instantiated → MLA weights have nowhere to land →
    # model runs on random init → "armanarmanarman" / "Bub Bub Bub" token soup.
    #
    # Fix: when text_config.model_type is mistral4 and top is mistral3, promote
    # text_config to the model config so mlx_lm.load_model picks the proper
    # mistral4.Model class with embed_q / unembed_out MLA structure.
    # Mirrored from the kv_b_proj split fix below — both must run together.
    _tc_for_model_type = config.get("text_config", {}) or {}
    if (
        config.get("model_type") == "mistral3"
        and _tc_for_model_type.get("model_type") == "mistral4"
    ):
        logger.info(
            "  Mistral 4 model_type promotion: top mistral3 + text_config "
            "mistral4 → loading inner text model directly via mlx_lm mistral4 "
            "(VLM wrapper bypassed for text inference)"
        )
        # Build a flat text-only config from text_config + preserve quant
        _flat = dict(_tc_for_model_type)
        _flat.setdefault("model_type", "mistral4")
        if "quantization" in config:
            _flat["quantization"] = config["quantization"]
        # Keep eos/bos from top level if not in text_config
        for _kk in ("eos_token_id", "bos_token_id", "pad_token_id"):
            if _kk in config and _kk not in _flat:
                _flat[_kk] = config[_kk]
        config = _flat

    # Always read block_size from jang_config (needed by _pre_fix_bits_from_shard
    # and other per-shard fixups). config.json's quantization.group_size may differ
    # from the JANG config's block_size for older models.
    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)

    # config.json already has quantization key (written by v2 converter)
    # but ensure it exists for older v2 models
    if "quantization" not in config:
        bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [4])
        config["quantization"] = {"group_size": block_size, "bits": min(bit_widths)}

    # MXTQ / JANGTQ fast path ─────────────────────────────────────────────
    # Detect tq_packed keys in the first shard. If present, delegate loading
    # to jang_tools.load_jangtq.load_jangtq_model() which installs native
    # TurboQuantLinear / TurboQuantSwitchLinear modules and applies all
    # P3/P15/P17/P18 Metal-kernel optimizations (multiblock Hadamard, router
    # mx.compile, thread-tiling OPT=10/20 sweet spot, QKV fusion). The
    # dequant-and-requant fallback below stays in place for environments
    # where jang_tools is unavailable.
    _tq_weight_files = _get_v2_weight_files(path)
    _is_mxtq_v2 = False
    if _tq_weight_files:
        try:
            _tq_first_keys = list(mx.load(str(_tq_weight_files[0])).keys())
            _is_mxtq_v2 = any(k.endswith(".tq_packed") for k in _tq_first_keys)
            del _tq_first_keys
        except Exception:
            _is_mxtq_v2 = False

    if _is_mxtq_v2:
        # Step 1: try to import the fast-path entry point. Only an ImportError
        # here justifies falling back to the dequant path (jang_tools missing).
        try:
            from jang_tools.load_jangtq import load_jangtq_model as _load_jangtq
        except ImportError as _tq_ie:
            logger.warning(
                "  JANGTQ fast path unavailable (%s) — falling back to "
                "dequant-and-requant path",
                _tq_ie,
            )
            _load_jangtq = None

        if _load_jangtq is not None:
            logger.info(
                "MXTQ/JANGTQ detected — using native TurboQuant fast path "
                "(jang_tools.load_jangtq, P3/P15/P17/P18 Metal kernels)"
            )
            if filter_expert_keys:
                logger.warning(
                    "  filter_expert_keys=True ignored on JANGTQ fast path "
                    "(smelt partial-expert loading is not TQ-aware yet)"
                )
            if layer_range is not None:
                logger.warning(
                    "  layer_range=%s ignored on JANGTQ fast path "
                    "(distributed layer-split loading is not TQ-aware yet)",
                    layer_range,
                )
            # Step 2: load. If THIS fails, the model is broken — propagate
            # rather than silently waste 80 s in the fallback path.
            model, tokenizer = _load_jangtq(path, skip_params_eval=skip_eval)

            if not hasattr(model, "config"):
                model.config = config

            # Step 3: vmlx_engine-only post-hooks. Each is wrapped individually
            # so a failure in one (e.g. cache patching on an unsupported model)
            # does NOT discard a successful 60-GB load.
            _model_cfg_tq = json.loads((path / "config.json").read_text())
            if not skip_eval:
                try:
                    _set_wired_limit_for_model(_tq_weight_files)
                except Exception as _wl_e:
                    logger.warning(f"  set_wired_limit skipped: {_wl_e}")
            try:
                _patch_turboquant_make_cache(model, jang_cfg, _model_cfg_tq)
            except Exception as _pt_e:
                logger.warning(
                    f"  TurboQuant cache patching failed ({_pt_e}); "
                    f"model loaded but KV cache will be dense"
                )
                import traceback
                logger.debug(traceback.format_exc())

            elapsed = time.perf_counter() - start
            actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", 0)
            source_model = jang_cfg.get("source_model", {}).get("name", "unknown")
            logger.info(
                f"JANGTQ v2 loaded in {elapsed:.1f}s: {source_model} "
                f"({actual_bits:.1f}-bit avg, native TQ, no dequant)"
            )
            return model, tokenizer

    # Gemma 4 native text MoE must be registered in mlx_lm.models BEFORE the
    # skeleton load runs — otherwise gemma4 JANG models raise
    # `ValueError: Model type gemma4 not supported` from mlx_lm.utils.
    # load_model_with_fallback() registers it too, so this is redundant when
    # called via the CLI path but critical for any direct caller of
    # load_jang_model (benchmark scripts, test harnesses, distributed worker).
    # The register function is idempotent and a no-op when mlx-lm ships
    # gemma4 natively (0.31.2+).
    try:
        from ..models.gemma4_native_register import register_gemma4_native
        register_gemma4_native()
    except Exception as _g4_e:
        logger.debug(f"gemma4 native register skipped: {_g4_e}")

    # Nemotron-H LatentMoE patch: must run BEFORE _load_model_skeleton creates
    # NemotronHBlock instances. For models with moe_latent_size set (e.g.,
    # Nemotron-3-Super-120B), experts operate on a latent dim (1024) rather than
    # hidden_size (4096), with fc1_latent_proj/fc2_latent_proj wrapping the switch_mlp.
    # mlx-lm 0.31.2+ has native support — ensure_latent_moe_support() is a no-op in
    # that case. For older mlx-lm (vmlx pins >=0.30.2) the patch monkey-patches
    # nemotron_h to add LatentMoE. Without this, JANG Nemotron Super models crash
    # with "[gather_qmm] Last dimension of first input with shape (..., 4096) does
    # not match the expanded quantized matrix" at first inference.
    try:
        from .nemotron_latent_moe import ensure_latent_moe_support
        ensure_latent_moe_support(str(path))
    except Exception as _lmoe_e:
        logger.debug(f"LatentMoE patch skipped: {_lmoe_e}")

    model, config = _load_model_skeleton(
        path, lazy=True, strict=False, model_config=config
    )
    _upgrade_switch_to_quantized(
        model,
        config["quantization"]["bits"],
        config["quantization"]["group_size"],
    )

    # Mistral-Small-4-119B (and any future model_type-promoted text load):
    # mlx_lm.utils.load_model's nn.quantize predicate `f"{p}.scales" in
    # weights` cannot match the file's `language_model.model.X.scales` keys
    # against the post-promotion module paths `model.X` — so embed_tokens,
    # q_proj, k_proj, etc. stay as plain nn.Linear / nn.Embedding holding
    # uint32 packed weights → forward pass crashes with rms_norm shape
    # mismatches. Re-run nn.quantize here with a predicate that scans the
    # safetensors HEADERS (no data load) and applies the LM-strip rename to
    # the keys before checking. Cheap (~10ms per shard).
    if (
        _is_mistral4_promoted := (
            getattr(_load_model_skeleton, "__name__", "") == "load_model"
            and (jang_cfg.get("architecture", {}).get("attention", "") == "mla"
                 or "mistral4" in str(config.get("model_type", "")))
        )
    ):
        try:
            from safetensors import safe_open
            _renamed_quant_paths = set()
            _wf_for_scan = _get_v2_weight_files(path)
            for _wf in _wf_for_scan:
                with safe_open(str(_wf), framework="numpy") as _t:
                    for _k in _t.keys():
                        if not _k.endswith(".scales"):
                            continue
                        _base = _k[: -len(".scales")]
                        # Apply the same LM-strip the per-shard loop will
                        if _base.startswith("language_model.model."):
                            _base = "model." + _base[len("language_model.model."):]
                        elif _base.startswith("language_model.lm_head."):
                            _base = "lm_head." + _base[len("language_model.lm_head."):]
                        elif _base.startswith("language_model."):
                            _base = _base[len("language_model."):]
                        _renamed_quant_paths.add(_base)
                        # mlx_lm/models/mistral4.py:sanitize splits kv_b_proj
                        # into embed_q + unembed_out (with re-quantization).
                        # The split happens AFTER nn.quantize, so we need to
                        # pre-register the resulting embed_q / unembed_out
                        # paths in the predicate set so they ALSO get the
                        # QuantizedMultiLinear treatment.
                        if _base.endswith(".kv_b_proj"):
                            _self_attn = _base[: -len(".kv_b_proj")]
                            _renamed_quant_paths.add(f"{_self_attn}.embed_q")
                            _renamed_quant_paths.add(f"{_self_attn}.unembed_out")
            if _renamed_quant_paths:
                import mlx.nn as _nn
                def _post_promo_predicate(p, m):
                    if not hasattr(m, "to_quantized"):
                        return False
                    return p in _renamed_quant_paths
                _nn.quantize(
                    model,
                    group_size=config["quantization"]["group_size"],
                    bits=config["quantization"]["bits"],
                    class_predicate=_post_promo_predicate,
                )
                logger.info(
                    f"  Re-quantized {len(_renamed_quant_paths)} modules via "
                    f"renamed-key predicate (model_type promotion path)"
                )
        except Exception as _rq_err:
            logger.debug(f"  Post-promotion re-quantize skipped: {_rq_err}")

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
    # Gate dequant needed for any MoE model with quantized gate weights (MoEGate is
    # nn.Module not nn.Linear, so nn.quantize skips it but JANG still quantizes raw weights).
    # Applies to: nemotron_h, nemotron, mistral4, deepseek_v3, deepseek_v2, etc.
    # Check both top-level and text_config for n_routed_experts (VLM wrappers nest it)
    _text_cfg = config.get("text_config", config)
    _n_experts = config.get("n_routed_experts", 0) or _text_cfg.get(
        "n_routed_experts", 0
    )
    _needs_gate_dequant = _needs_fc_rename or _n_experts > 0

    # Nemotron-H gate: MoEGate is a custom nn.Module (not nn.Linear), so
    # nn.quantize() in _load_model_skeleton does NOT convert it. However,
    # _load_model_skeleton's model.load_weights() loads the raw uint32 gate
    # weight into MoEGate.weight. Our custom weight loading loop below
    # dequantizes the gate weight (uint32 → bfloat16) and overwrites it.

    # Detect if model has VLM-style key naming (model.language_model.layers)
    # but text model param paths (language_model.model.layers). This happens when
    # qwen3_5_moe (VLM wrapper) is loaded as text — JANG converter stores VLM-style
    # keys but mlx-lm creates language_model.model.* params. Without remapping,
    # ALL weights are silently dropped (strict=False) → model runs on zeros.
    _needs_vlm_key_remap = hasattr(model, "language_model") and "text_config" in config

    # Mistral 4 119B mismatch (companion to the model_type promotion above):
    # the JANG file has VLM-style `language_model.model.X` weight keys, but the
    # promoted mistral4 text model has `model.X` parameter paths. Remap by
    # stripping the `language_model.` prefix so weights actually land in the
    # mistral4 modules. Without this every weight is silently dropped by
    # strict=False and the model runs on init noise → "armanarmanarman" /
    # "ఉ из yılındaaltar" multilingual token soup. NO-REGRESSION-CHECKLIST §11.
    _needs_mistral4_lm_strip = (
        not _needs_vlm_key_remap
        and _model_type == "mistral4"
        and not hasattr(model, "language_model")
    )

    # Gemma 4: JANG stores expert keys as switch_mlp.{gate,up,down}_proj but
    # mlx-lm gemma4/gemma4_text model uses experts.switch_glu.{gate,up,down}_proj.
    # Without this remap, expert weights are silently dropped (strict=False)
    # and the model runs on uninitialized random experts → garbage output.
    _needs_gemma4_switch_remap = _text_cfg.get("model_type", "") == "gemma4_text" or _model_type == "gemma4"

    # MXTQ detection: check first shard for tq_packed keys
    _is_mxtq = False
    _mxtq_seed = jang_cfg.get("mxtq_seed", 42)
    _mxtq_bits_map = jang_cfg.get("mxtq_bits", {})
    if weight_files:
        try:
            _first_keys = list(mx.load(str(weight_files[0])).keys())
        except Exception:
            _first_keys = []
        _is_mxtq = any(k.endswith(".tq_packed") for k in _first_keys)
        if _is_mxtq:
            logger.info("  MXTQ/JANGTQ format detected — will dequant tq_packed weights to fp16")

    for sf in weight_files:
        weights = mx.load(str(sf))

        # MXTQ dequant: detect tq_packed/tq_norms pairs, dequant to fp16,
        # then re-quantize to affine (uint32 .weight + .scales + .biases)
        # so the model's QuantizedLinear modules accept them. Per-expert 2D
        # tensors are stored individually — sanitize() stacks them later.
        if _is_mxtq:
            tq_groups = {}
            regular = {}
            for k, v in weights.items():
                if k.endswith(".tq_packed"):
                    tq_groups.setdefault(k[:-10], {})["packed"] = v
                elif k.endswith(".tq_norms"):
                    tq_groups.setdefault(k[:-9], {})["norms"] = v
                elif k.endswith(".tq_bits"):
                    pass
                else:
                    regular[k] = v

            if tq_groups:
                try:
                    from jang_tools.turboquant.codebook import compute_codebook
                    from jang_tools.turboquant.rotation import generate_random_signs, hadamard_inverse
                    from jang_tools.turboquant.pipeline import unpack_bits

                    _tq_count = 0
                    _q_bits = config.get("quantization", {}).get("bits", 2)
                    _q_gs = block_size
                    for base, parts in tq_groups.items():
                        if "packed" not in parts or "norms" not in parts:
                            continue
                        packed = parts["packed"]
                        norms = parts["norms"]
                        bl = base.lower()
                        if "shared_expert" in bl:
                            bits = _mxtq_bits_map.get("shared_expert", 3)
                        elif "expert" in bl:
                            bits = _mxtq_bits_map.get("routed_expert", 2)
                        else:
                            bits = 2
                        vals_per_u32 = 32 // bits

                        # Dequant: tq_packed → fp16
                        out_feat, packed_cols = packed.shape
                        in_features = packed_cols * vals_per_u32
                        cb = mx.array(compute_codebook(in_features, bits))
                        signs = mx.array(generate_random_signs(in_features, _mxtq_seed))
                        rows = []
                        for r in range(out_feat):
                            idx = unpack_bits(packed[r], bits, in_features)
                            row = mx.take(cb, idx.astype(mx.uint32))
                            rows.append(row)
                        w = mx.stack(rows)
                        w = w * norms[:, None].astype(w.dtype)
                        dq = hadamard_inverse(w, signs).astype(mx.float16)
                        mx.eval(dq)

                        # Re-quantize to affine: fp16 → (uint32 packed, scales, biases)
                        # This produces the standard triplet that QuantizedLinear expects.
                        q_w, q_s, q_b = mx.quantize(dq, group_size=_q_gs, bits=_q_bits)
                        mx.eval(q_w, q_s, q_b)
                        regular[f"{base}.weight"] = q_w
                        regular[f"{base}.scales"] = q_s
                        regular[f"{base}.biases"] = q_b
                        del dq, w, rows
                        _tq_count += 1

                    if _tq_count > 0:
                        logger.info(f"  Dequanted+requanted {_tq_count} MXTQ tensors in shard {sf.name}")
                except ImportError as ie:
                    logger.warning(f"  MXTQ dequant failed (missing jang_tools): {ie}")
                except Exception as e:
                    logger.warning(f"  MXTQ dequant failed: {e}")

            weights = regular

        # Nemotron-H: filter mtp/importance weights
        if _needs_fc_rename:
            weights = {
                k: v
                for k, v in weights.items()
                if not k.endswith(".importance") and "mtp." not in k
            }
        # Mistral 4 LM-prefix strip: weights are `language_model.model.X` and
        # `language_model.lm_head.X` and `lm_head.X`, but the promoted mistral4
        # text model expects `model.X` and `lm_head.X`. Strip `language_model.`
        # so weights actually land. Mirror the audit-2026-04-07 §6.3 hardening
        # pattern (count source/dst, refuse to proceed on silent loss).
        if _needs_mistral4_lm_strip:
            _src_count_lm_model = sum(
                1 for k in weights.keys() if k.startswith("language_model.model.")
            )
            _src_count_lm_head = sum(
                1 for k in weights.keys() if k.startswith("language_model.lm_head.")
            )
            _src_count_top_lm_head = sum(
                1 for k in weights.keys() if k.startswith("lm_head.") and not k.startswith("lm_head.lm_head.")
            )
            stripped = {}
            for k, v in weights.items():
                if k.startswith("language_model.model."):
                    stripped["model." + k[len("language_model.model."):]] = v
                elif k.startswith("language_model.lm_head."):
                    stripped["lm_head." + k[len("language_model.lm_head."):]] = v
                elif k.startswith("language_model."):
                    # other VLM wrapper attrs (e.g. norm, embed_tokens) — strip prefix
                    stripped[k[len("language_model."):]] = v
                else:
                    stripped[k] = v
            _dst_count_model = sum(1 for k in stripped.keys() if k.startswith("model."))
            _dst_count_lm_head = sum(1 for k in stripped.keys() if k.startswith("lm_head."))
            _expected_lm_head = _src_count_lm_head + _src_count_top_lm_head
            if _src_count_lm_model > 0 and _dst_count_model < _src_count_lm_model:
                logger.warning(
                    f"Mistral 4 LM-strip silent loss: src model.* in language_model={_src_count_lm_model} "
                    f"→ dst model.*={_dst_count_model}"
                )
            weights = stripped

        # Remap VLM-style keys for models loaded as text but with VLM key structure.
        # model.language_model.X → language_model.model.X  (layers, embed, norm)
        # lm_head.X → language_model.lm_head.X  (bare top-level in safetensors)
        if _needs_vlm_key_remap:
            # Audit-2026-04-07 risk §6.3 hardening: count source `model.language_model.*`
            # keys before remap, count `language_model.model.*` keys after, and refuse to
            # proceed if any source key was silently lost. `model.load_weights(strict=False)`
            # masks silent drops downstream — without this guard, a regression in the remap
            # logic would produce a model running on partial/zero weights with no error.
            _src_lm_count = sum(
                1 for k in weights.keys() if k.startswith("model.language_model.")
            )
            _src_lm_head = sum(1 for k in weights.keys() if k.startswith("lm_head."))
            remapped = {}
            for k, v in weights.items():
                if k.startswith("model.language_model."):
                    remapped[
                        k.replace("model.language_model.", "language_model.model.", 1)
                    ] = v
                elif k.startswith("lm_head."):
                    remapped["language_model." + k] = v
                else:
                    remapped[k] = v
            _dst_lm_count = sum(
                1 for k in remapped.keys() if k.startswith("language_model.model.")
            )
            _dst_lm_head = sum(
                1 for k in remapped.keys() if k.startswith("language_model.lm_head.")
            )
            if (_src_lm_count > 0 and _dst_lm_count != _src_lm_count) or (
                _src_lm_head > 0 and _dst_lm_head != _src_lm_head
            ):
                raise RuntimeError(
                    f"jang_loader VLM key remap dropped weights: "
                    f"source language_model.*={_src_lm_count} → remapped={_dst_lm_count}, "
                    f"source lm_head.*={_src_lm_head} → remapped={_dst_lm_head}. "
                    f"This means a regression in the remap logic — refusing to load a "
                    f"silently-incomplete VLM-as-text model. File={getattr(sf, 'name', sf)}"
                )
            weights = remapped
        # Gemma 4: remap JANG switch_mlp → experts.switch_glu BEFORE sanitize
        # so that model.load_weights matches the actual model parameter paths.
        if _needs_gemma4_switch_remap:
            g4_remapped = {}
            for k, v in weights.items():
                if ".switch_mlp." in k:
                    k = k.replace(".switch_mlp.", ".experts.switch_glu.")
                g4_remapped[k] = v
            weights = g4_remapped
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        # MoE gate dequant + optional Nemotron fc rename
        if _needs_gate_dequant:
            renamed = {}
            gate_parts = {}  # prefix -> {scales, biases}
            for k, v in weights.items():
                new_k = k
                # Collect gate scales/biases for dequantization
                if ".gate." in k and (k.endswith(".scales") or k.endswith(".biases")):
                    prefix = k.rsplit(".", 1)[0]
                    gate_parts.setdefault(prefix, {})[k.rsplit(".", 1)[1]] = v
                    continue
                # Apply fc1/fc2 rename (Nemotron only)
                if _needs_fc_rename:
                    for old, new in _nemotron_renames.items():
                        if old in k:
                            new_k = k.replace(old, new)
                            break
                renamed[new_k] = v
            # Dequantize gate weights (uint32 packed → float for MoEGate)
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
                        gs = (
                            real_cols // scales.shape[-1] if scales.shape[-1] > 0 else 0
                        )
                        if gs > 0 and gs * scales.shape[-1] == real_cols:
                            try:
                                dq = mx.dequantize(qw, scales, biases, gs, bits)
                                mx.eval(dq)
                                renamed[wkey] = dq.astype(mx.bfloat16)
                                logger.info(
                                    f"  Dequantized gate: {wkey} bits={bits} gs={gs} -> {dq.shape}"
                                )
                                break
                            except Exception:
                                continue
            weights = renamed

        # Mistral4 MLA: split kv_b_proj → embed_q + unembed_out.
        # CRITICAL REGRESSION FIX (2026-04-11): the v2 LLM loader was missing
        # this split (only the v2 VLM loader had it). Mistral-Small-4-119B
        # has `vision_config` in config.json BUT `jang_config.architecture
        # .has_vision: false`, so is_mllm_model() returns False and the model
        # routes to _load_jang_v2 (this function) — which never split
        # kv_b_proj. Result: embed_q / unembed_out modules kept their random
        # init weights, every attention head produced noise, and decode
        # output came out as "armanarmanarman" / "Bub Bub Bub" token soup.
        #
        # MLA stores compressed KV latents — the HF kv_b_proj weight must be
        # dequantized, reshaped (nheads, head_dim, kv_rank), and split into
        # embed_q (nheads, kv_rank, qk_nope) and unembed_out (nheads, v_head,
        # kv_rank). Original split implementation by Jinho Jang (eric@jangq.ai)
        # for vMLX, mirrored from _load_jang_v2_vlm to keep the two paths in
        # sync. NO-REGRESSION-CHECKLIST §11 row for Mistral 4 MLA family.
        _t_cfg_for_mla = config.get("text_config", config)
        _text_mt_for_mla = _t_cfg_for_mla.get("model_type", config.get("model_type", ""))
        if _text_mt_for_mla == "mistral4":
            _nheads = _t_cfg_for_mla.get("num_attention_heads", 32)
            _qk_nope = _t_cfg_for_mla.get("qk_nope_head_dim", 64)
            _v_head = _t_cfg_for_mla.get("v_head_dim", 128)
            _kv_rank = _t_cfg_for_mla.get("kv_lora_rank", 256)
            _head_dim = _qk_nope + _v_head
            _nlayers = _t_cfg_for_mla.get("num_hidden_layers", 36)
            _split_count = 0
            for _l in range(_nlayers):
                for _pfx in [
                    f"language_model.model.layers.{_l}.self_attn",
                    f"model.language_model.layers.{_l}.self_attn",
                    f"model.layers.{_l}.self_attn",
                ]:
                    _kb_key = f"{_pfx}.kv_b_proj.weight"
                    if _kb_key not in weights:
                        continue
                    _v = weights.pop(_kb_key)
                    _s_key = f"{_pfx}.kv_b_proj.scales"
                    _b_key = f"{_pfx}.kv_b_proj.biases"
                    if _s_key in weights:
                        _s = weights.pop(_s_key)
                        _b = weights.pop(_b_key, mx.zeros_like(_s))
                        for _try_bits in [8, 6, 4, 3, 2]:
                            _elem = 32 // _try_bits
                            _real = _v.shape[-1] * _elem
                            _gs = _real // _s.shape[-1] if _s.shape[-1] > 0 else 0
                            if _gs > 0 and _gs * _s.shape[-1] == _real:
                                try:
                                    _v = mx.dequantize(_v, _s, _b, _gs, _try_bits)
                                    break
                                except Exception:
                                    continue
                    _v = _v.reshape(_nheads, _head_dim, _kv_rank)
                    _wk = mx.contiguous(_v[:, :_qk_nope, :].swapaxes(-1, -2))
                    _wv = mx.contiguous(_v[:, _qk_nope:, :])
                    weights[f"{_pfx}.embed_q.weight"] = _wk.astype(mx.float16)
                    weights[f"{_pfx}.unembed_out.weight"] = _wv.astype(mx.float16)
                    _split_count += 1
                    break
            if _split_count > 0:
                logger.info(
                    f"  Mistral 4 MLA: split kv_b_proj → embed_q + unembed_out "
                    f"on {_split_count} layers (LLM v2 loader)"
                )

        # Smelt mode: filter expert weights (loaded separately via ExpertIndex)
        if filter_expert_keys:
            weights = {k: v for k, v in weights.items() if not _is_expert_key(k)}
        # Distributed: only load weights for assigned layer range
        if layer_range is not None:
            weights = _filter_by_layer_range(weights, layer_range[0], layer_range[1])
        # Pre-fix per-layer bits before load to prevent shape mismatch
        # ValueError on JANG mixed-precision models (fixes #62, #63).
        _pre_fix_bits_from_shard(model, weights, block_size)
        model.load_weights(list(weights.items()), strict=False)
        del weights
        gc.collect()

    # Mistral-Small-4-119B + any future model_type-promoted text load: the
    # internal nn.quantize predicate in mlx_lm.utils.load_model could not see
    # the renamed keys (it checks `f"{p}.scales" in weights` BEFORE our
    # LM-strip), so embed_tokens / q_proj / k_proj / etc. ended up as plain
    # nn.Linear / nn.Embedding holding uint32 packed weights → forward pass
    # produced rms_norm 4096-vs-? shape mismatches and "armanarmanarman"
    # token soup. Walk the loaded model and upgrade every Linear/Embedding
    # whose weight is uint32 to its Quantized variant in place. Safe no-op
    # for models that nn.quantize already converted.
    _q_cfg = config.get("quantization", {}) if isinstance(config, dict) else {}
    _q_bits = _q_cfg.get("bits", min(jang_cfg.get("quantization", {}).get("bit_widths_used", [4])))
    _q_gs = _q_cfg.get("group_size", block_size)
    _upg = _upgrade_modules_with_uint32_weights(model, _q_bits, _q_gs)
    if _upg > 0:
        logger.info(
            f"  Upgraded {_upg} modules to Quantized variants (post-load fixup)"
        )

    _fix_quantized_bits(model)

    if not hasattr(model, "config"):
        model.config = config

    # bfloat16 compute for 512+ expert models — float16 norm/embedding
    # layers overflow at shared expert down_proj (SiLU*up → 4096-dim dot
    # product exceeds float16 max 65504). bfloat16 has float32 range.
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = (
        _text_cfg.get("num_experts")
        or _text_cfg.get("num_local_experts")
        or _text_cfg.get("n_routed_experts")
        or 0
    )
    _hidden = _text_cfg.get("hidden_size") or 0
    _text_mt = _text_cfg.get("model_type", _model_cfg.get("model_type", ""))
    _is_mla = (_text_cfg.get("kv_lora_rank") or 0) > 0
    if (_n_experts >= 512 and _hidden >= 4096) or _text_mt == "mistral4" or _is_mla:
        model.set_dtype(mx.bfloat16)
        _reason = "MLA" if _is_mla else f"{_n_experts} experts"
        logger.info(
            f"  bfloat16 enabled: {_reason}, hidden={_hidden} "
            f"(float16 overflow prevention)"
        )

    if not skip_eval:
        _set_wired_limit_for_model(_get_v2_weight_files(path))
        _chunked_eval_params(model)

    # TurboQuant: patch make_cache for JANG models with TQ enabled
    _patch_turboquant_make_cache(model, jang_cfg, _model_cfg)

    elapsed = time.perf_counter() - start

    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", 0)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")
    logger.info(
        f"JANG v2 loaded in {elapsed:.1f}s: {source_model} ({actual_bits:.1f}-bit avg)"
    )

    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def _load_jang_v2_vlm(
    path: Path,
    jang_cfg: dict,
    skip_eval: bool = False,
    filter_expert_keys: bool = False,
):
    """Load a JANG v2 Vision-Language model via mmap — instant."""
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args,
        load_config as vlm_load_config,
        update_module_configs,
        load_image_processor,
        load_processor,
        skip_multimodal_module,
    )

    start = time.perf_counter()

    block_size = jang_cfg.get("quantization", {}).get("block_size", 64)
    bit_widths = jang_cfg.get("quantization", {}).get("bit_widths_used", [4])
    default_bits = min(bit_widths)

    # Nemotron-H LatentMoE patch — see _load_jang_v2 for rationale. Must run
    # BEFORE model_class.Model(model_config) instantiates any NemotronHBlock.
    # No-op on mlx-lm 0.31.2+ (native support). Defensive: covers any future
    # VLM wrapper whose text_config is nemotron_h.
    try:
        from .nemotron_latent_moe import ensure_latent_moe_support
        ensure_latent_moe_support(str(path))
    except Exception as _lmoe_e:
        logger.debug(f"LatentMoE patch skipped: {_lmoe_e}")

    config = vlm_load_config(path)

    # Mistral 4 VLM fallback: the outer config has model_type=mistral3 (the VLM
    # wrapper class name in HuggingFace) but text_config.model_type=mistral4
    # (the inner MLA language model). `get_model_and_args` picks mlx_vlm's
    # mistral3 class which uses standard attention (q_proj/k_proj/v_proj); our
    # weights are Mistral 4 MLA (q_a_proj/q_b_proj/kv_a_proj/kv_b_proj/embed_q/
    # unembed_out) so weights have nowhere to land and the model outputs
    # repetitive token soup ('.;.;.;' / 'SuppAddAdd' / 'cevcev-top-top') —
    # live-observed against Mistral-Small-4-119B-JANG_2L on 2026-04-19.
    #
    # mlx_vlm does not (yet) ship a mistral4 VLM class, only a mistral4
    # language-model. Fall through to the text-only `_load_jang_v2` path which
    # already has the mistral3→mistral4 promotion at line ~460 and produces
    # coherent output. Cost: no image input for Mistral 4 until mlx_vlm adds
    # a VLM class. (Image input was already broken pre-rerouting — the class
    # mismatch meant weights were corrupt; no regression.)
    _tc = config.get("text_config") or {}
    if config.get("model_type") == "mistral3" and _tc.get("model_type") == "mistral4":
        logger.warning(
            "  Mistral 4 VLM not supported by mlx_vlm (no mistral4 VLM class); "
            "falling back to text-only load. Vision input will be ignored."
        )
        return _load_jang_v2(path, jang_cfg, skip_eval=skip_eval, filter_expert_keys=filter_expert_keys)

    # Kimi K2.6 (model_type="kimi_k25") — route through
    # jang_tools.load_jangtq_kimi_vlm so the kimi_k25 → kimi_vl remap is
    # installed in mlx_vlm.MODEL_REMAPPING + MODEL_CONFIG before dispatch,
    # plus apply the VL-specific lower wired_limit (52% vs 70%) and the
    # vision/language command-buffer split that keeps Metal's ~60 s
    # watchdog from killing the first VL forward on 191 GB MoE bundles.
    # See research/KIMI-K2.6-VMLX-INTEGRATION.md §1 for the runtime contract.
    if config.get("model_type") == "kimi_k25":
        try:
            from jang_tools.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model
        except ImportError as _ie:
            raise RuntimeError(
                "Kimi K2.6 VLM requires jang_tools.load_jangtq_kimi_vlm but "
                f"import failed: {_ie}. The bundled Python must include "
                "jang_tools ≥ the release shipping load_jangtq_kimi_vlm.py."
            ) from _ie
        logger.info(
            "Kimi K2.6 JANGTQ VLM detected — using Kimi-specific fast path "
            "(jang_tools.load_jangtq_kimi_vlm: kimi_k25 remap + VL wired_limit "
            "+ vision/language command-buffer split)"
        )
        _kimi_model, _kimi_processor = load_jangtq_kimi_vlm_model(path)
        if not hasattr(_kimi_model, "config"):
            _kimi_model.config = config
        try:
            _lang = getattr(_kimi_model, "language_model", None)
            if _lang is not None:
                _patch_turboquant_make_cache(_lang, jang_cfg, config)
        except Exception as _pe:
            logger.warning(f"  TurboQuant make_cache patch skipped: {_pe}")
        elapsed = time.perf_counter() - start
        logger.info(f"Kimi K2.6 JANGTQ VLM loaded in {elapsed:.1f}s (fast path)")
        return _kimi_model, _kimi_processor

    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    # audio_config: None means no audio — remove so update_module_configs
    # doesn't call from_dict(None) which crashes (Gemma 4 has audio_config: null)
    if config.get("audio_config") is None:
        config.pop("audio_config", None)
    else:
        config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    # Only include modules whose config key exists and is not None
    modules = [
        m
        for m in ["text", "vision", "perceiver", "projector", "audio"]
        if config.get(f"{m}_config") is not None
    ]
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

    # MXTQ detection: check first shard for tq_packed keys (JANGTQ VLM support).
    # JANGTQ emits {"version":2, "weight_format":"mxtq"} and stores weights as
    # `.tq_packed` + `.tq_norms` triplets instead of affine `.scales` + `.biases`.
    # Text loader at line ~730 has two paths: (a) fast path via jang_tools (mlx_lm
    # only, no vision tower), (b) dequant+requant fallback. VLM wrapper MUST go
    # through the fallback because jang_tools.load_jangtq doesn't build a vision
    # tower. Without this detection, quantized_suffixes stays empty, nn.quantize
    # doesn't quantize anything, weights load as zeros, and the first SSM layer
    # crashes with `[reshape] Cannot infer the shape of an empty array` (Qwen 3.6
    # JANGTQ_2L / Qwen3.5-VL-*-JANGTQ* path).
    _vlm_is_mxtq = any(k.endswith(".tq_packed") for k in all_weight_keys)
    _vlm_mxtq_seed = jang_cfg.get("mxtq_seed", 42)
    _vlm_mxtq_bits_map = jang_cfg.get("mxtq_bits", {})
    if _vlm_is_mxtq:
        # JANGTQ VLM fast path via jang_tools.load_jangtq_vlm — mirrors the
        # text-side fast path at line ~509. Uses the same P3/P15/P17/P18
        # Metal kernels (TurboQuantLinear / TurboQuantSwitchLinear) for the
        # language_model's quantized modules while wiring up mlx_vlm's
        # vision_tower + processor. No dequant, no requant — preserves
        # output quality. Replaces the earlier lossy dequant-and-requant
        # fallback that produced gibberish on Qwen3.6-35B-A3B-JANGTQ_2L.
        try:
            from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model as _load_vlm
        except ImportError as _ie:
            raise RuntimeError(
                f"JANGTQ VLM requires jang_tools.load_jangtq_vlm but import failed: {_ie}\n"
                f"Make sure jang_tools ≥ the one including load_jangtq_vlm.py is installed "
                f"into this Python environment."
            ) from _ie
        logger.info(
            "MXTQ/JANGTQ VLM detected — using native TurboQuant fast path "
            "(jang_tools.load_jangtq_vlm, P3/P15/P17/P18 Metal kernels)"
        )
        if filter_expert_keys:
            logger.warning(
                "  filter_expert_keys=True ignored on JANGTQ VLM fast path "
                "(smelt partial-expert loading is not TQ-aware yet)"
            )
        _vlm_model, _vlm_processor = _load_vlm(path)

        # Match the rest of the VLM-path post-processing that would normally
        # fire at the bottom of this function: attach config if missing and
        # return. Also patch TurboQuant make_cache for the language model so
        # the KV cache knows about the TQ layers.
        if not hasattr(_vlm_model, "config"):
            _vlm_model.config = config
        try:
            _lang = getattr(_vlm_model, "language_model", None)
            if _lang is not None:
                _patch_turboquant_make_cache(_lang, jang_cfg, config)
        except Exception as _pe:
            logger.warning(f"  TurboQuant make_cache patch skipped: {_pe}")
        elapsed = time.perf_counter() - start
        actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", 0)
        logger.info(
            f"JANGTQ VLM loaded in {elapsed:.1f}s (fast path) — "
            f"{actual_bits:.1f}-bit avg" if actual_bits else
            f"JANGTQ VLM loaded in {elapsed:.1f}s (fast path)"
        )
        return _vlm_model, _vlm_processor

    # Build set of quantized module paths from weight keys
    # Weight keys (safetensors): model.language_model.layers.0.mlp.gate_proj.scales
    # Module paths (nn.quantize): language_model.model.layers.0.mlp.gate_proj
    # These don't match — build a suffix set for robust matching
    quantized_suffixes = set()
    for k in all_weight_keys:
        _qpath = None
        if k.endswith(".scales"):
            _qpath = k[: -len(".scales")]
        elif _vlm_is_mxtq and k.endswith(".tq_packed"):
            # MXTQ: the `base` of tq_packed/tq_norms becomes the quantized module
            # suffix once we dequant+requant below into scales/biases.
            _qpath = k[: -len(".tq_packed")]
        if _qpath is not None:
            quantized_suffixes.add(_qpath)
            # Also add sanitize-remapped paths so nn.quantize() can match
            # module paths that differ from raw weight keys (e.g., Gemma 4
            # JANG uses switch_mlp.* but model expects experts.switch_glu.*)
            if ".switch_mlp." in _qpath:
                quantized_suffixes.add(
                    _qpath.replace(".switch_mlp.", ".experts.switch_glu.")
                )

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

    nn.quantize(
        model,
        group_size=block_size,
        bits=default_bits,
        class_predicate=get_class_predicate,
    )

    # Load weights via mmap
    # Matches jang-tools 2.1.0 loader: try model.sanitize() first (works for dense models),
    # fall back to minimal sanitize for MoE models where gate_up_proj is already split.
    from mlx_vlm.utils import sanitize_weights

    # Gemma 4: JANG stores expert keys as switch_mlp but model uses experts.switch_glu.
    # Fall through to top-level model_type if text_config.model_type is missing,
    # and accept both "gemma4" and "gemma4_text" — some JANG variants have the top
    # value in text_config, others leave it in the outer config (issue #71).
    _vlm_text_mt = config.get("text_config", {}).get(
        "model_type", config.get("model_type", "")
    )
    _vlm_needs_gemma4_switch_remap = _vlm_text_mt in ("gemma4", "gemma4_text")

    for sf in weight_files:
        shard_weights = mx.load(str(sf))
        shard_weights = {
            k: v for k, v in shard_weights.items() if not k.endswith(".importance")
        }
        # MXTQ dequant+requant for VLM path. Mirrors _load_jang_v2 text
        # loader at line ~745. Detect .tq_packed + .tq_norms pairs, dequant
        # to fp16 via codebook+hadamard math, then re-quantize to affine
        # uint32 / scales / biases so QuantizedLinear modules accept them.
        # Per-expert 2D tensors are stored individually — sanitize() stacks
        # them later. Fixes Qwen 3.6 JANGTQ+VL empty-tensor crash.
        if _vlm_is_mxtq:
            tq_groups = {}
            regular = {}
            for k, v in shard_weights.items():
                if k.endswith(".tq_packed"):
                    tq_groups.setdefault(k[:-10], {})["packed"] = v
                elif k.endswith(".tq_norms"):
                    tq_groups.setdefault(k[:-9], {})["norms"] = v
                elif k.endswith(".tq_bits"):
                    pass
                else:
                    regular[k] = v

            if tq_groups:
                try:
                    from jang_tools.turboquant.codebook import compute_codebook
                    from jang_tools.turboquant.rotation import (
                        generate_random_signs,
                        hadamard_inverse,
                    )
                    from jang_tools.turboquant.pipeline import unpack_bits

                    _tq_count = 0
                    _q_bits = default_bits
                    _q_gs = block_size
                    for base, parts in tq_groups.items():
                        if "packed" not in parts or "norms" not in parts:
                            continue
                        packed = parts["packed"]
                        norms = parts["norms"]
                        bl = base.lower()
                        if "shared_expert" in bl:
                            bits = _vlm_mxtq_bits_map.get("shared_expert", 3)
                        elif "expert" in bl:
                            bits = _vlm_mxtq_bits_map.get("routed_expert", 2)
                        else:
                            bits = 2
                        vals_per_u32 = 32 // bits
                        # VLM JANGTQ writers stack MoE experts as 3D tensors
                        # (num_experts, out_feat, packed_cols). Text loader assumed
                        # 2D per-expert keys; for VLM we must dequant+requant per
                        # expert and stack back to 3D so downstream sanitize() can
                        # feed SwitchGLU.
                        is_3d = packed.ndim == 3
                        if is_3d:
                            num_experts, out_feat, packed_cols = packed.shape
                        else:
                            out_feat, packed_cols = packed.shape
                        in_features = packed_cols * vals_per_u32
                        cb = mx.array(compute_codebook(in_features, bits))
                        signs = mx.array(
                            generate_random_signs(in_features, _vlm_mxtq_seed)
                        )

                        def _dequant_2d(packed_2d, norms_1d):
                            rows = []
                            for r in range(packed_2d.shape[0]):
                                idx = unpack_bits(packed_2d[r], bits, in_features)
                                row = mx.take(cb, idx.astype(mx.uint32))
                                rows.append(row)
                            w_ = mx.stack(rows)
                            w_ = w_ * norms_1d[:, None].astype(w_.dtype)
                            return hadamard_inverse(w_, signs).astype(mx.float16)

                        if is_3d:
                            # Batch all experts: build lazy ops per expert and
                            # stack results before ONE mx.eval. Per-expert
                            # mx.eval kills lazy fusion and makes load
                            # ~sequential — see _load_jang_v2 text path which
                            # also avoids per-row eval.
                            per_expert_qw = []
                            per_expert_qs = []
                            per_expert_qb = []
                            for e in range(num_experts):
                                dq_e = _dequant_2d(packed[e], norms[e])
                                qw_e, qs_e, qb_e = mx.quantize(
                                    dq_e, group_size=_q_gs, bits=_q_bits
                                )
                                per_expert_qw.append(qw_e)
                                per_expert_qs.append(qs_e)
                                per_expert_qb.append(qb_e)
                            stacked_w = mx.stack(per_expert_qw)
                            stacked_s = mx.stack(per_expert_qs)
                            stacked_b = mx.stack(per_expert_qb)
                            mx.eval(stacked_w, stacked_s, stacked_b)
                            regular[f"{base}.weight"] = stacked_w
                            regular[f"{base}.scales"] = stacked_s
                            regular[f"{base}.biases"] = stacked_b
                            del per_expert_qw, per_expert_qs, per_expert_qb
                        else:
                            dq = _dequant_2d(packed, norms)
                            mx.eval(dq)
                            q_w, q_s, q_b = mx.quantize(
                                dq, group_size=_q_gs, bits=_q_bits
                            )
                            mx.eval(q_w, q_s, q_b)
                            regular[f"{base}.weight"] = q_w
                            regular[f"{base}.scales"] = q_s
                            regular[f"{base}.biases"] = q_b
                            del dq
                        _tq_count += 1

                    if _tq_count > 0:
                        logger.info(
                            f"  Dequanted+requanted {_tq_count} MXTQ VLM tensors in shard {sf.name}"
                        )
                except ImportError as _ie:
                    logger.warning(
                        f"  MXTQ VLM dequant failed (missing jang_tools): {_ie}"
                    )
                except Exception as _e:
                    logger.warning(f"  MXTQ VLM dequant failed: {_e}")

            shard_weights = regular

        # Gemma 4 switch_mlp → experts.switch_glu remap (before sanitize)
        if _vlm_needs_gemma4_switch_remap:
            shard_weights = {
                (k.replace(".switch_mlp.", ".experts.switch_glu.") if ".switch_mlp." in k else k): v
                for k, v in shard_weights.items()
            }

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
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                "model.norm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
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
                model_class.VisionModel, shard_weights, model_config.vision_config
            )
            shard_weights = sanitize_weights(
                model_class.LanguageModel, shard_weights, model_config.text_config
            )
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
                            if (
                                w.shape[-1] * 32 % try_bits != 0
                                or in_dim % s.shape[-1] != 0
                            ):
                                continue
                            try_gs = in_dim // s.shape[-1]
                            if try_gs >= 2:
                                try:
                                    dq = mx.dequantize(
                                        w, s, b, group_size=try_gs, bits=try_bits
                                    )
                                    shard_weights[k] = dq.astype(mx.float16)
                                    del shard_weights[s_key], shard_weights[b_key]
                                    break
                                except Exception:
                                    continue

        # Gemma 3n / 4 PLE: ScaledLinear (per_layer_model_projection) and
        # nn.Embedding (embed_tokens_per_layer) lack to_quantized(), so
        # nn.quantize() skips them. JANG packs their weights as uint32 anyway.
        # Without dequantization, forward pass does matmul/take on uint32 →
        # garbage → all <pad> output (#52 / #87).
        #
        # Previously gated on `gemma4_text` only — broadened to cover Gemma 3n
        # (same PLE architecture) AND the "no-PLE" variant (Gemma 4 2B/4B-only
        # gate: if `hidden_size_per_layer_input` is 0/null, the model has
        # `per_layer_model_projection = None` and the safetensors' scales/
        # biases orphan → strict load fails with "Received 2 parameters not in
        # model" (vmlx#87 gyula-coder 2026-04-17). When the module is
        # disabled in the model, drop the orphan keys instead of dequanting.
        _text_mt = config.get("text_config", {}).get("model_type", "")
        _text_cfg_for_ple = config.get("text_config", config)
        _has_ple_module = bool(_text_cfg_for_ple.get("hidden_size_per_layer_input"))
        _ple_eligible_types = {"gemma4_text", "gemma3n", "gemma3n_text", "gemma4"}
        _gemma_family_by_name = _text_mt in _ple_eligible_types or config.get(
            "model_type", ""
        ) in _ple_eligible_types
        # Case 1 — PLE keys exist AND model has the module: dequant to fp16
        # Case 2 — PLE keys exist AND model does NOT have the module: drop orphans
        # Case 3 — non-Gemma models: skip entirely (original behavior)
        if _gemma_family_by_name and not _has_ple_module:
            # Drop orphan PLE quant keys so strict weight load succeeds.
            # Model doesn't instantiate per_layer_model_projection in this config.
            for _orphan_pfx in (
                "language_model.model.per_layer_model_projection",
                "model.language_model.per_layer_model_projection",
                "language_model.model.embed_tokens_per_layer",
                "model.language_model.embed_tokens_per_layer",
            ):
                for _suffix in (".weight", ".scales", ".biases"):
                    _k = _orphan_pfx + _suffix
                    if _k in shard_weights:
                        del shard_weights[_k]
                        logger.info(
                            f"  Dropped orphan Gemma PLE key (model has no PLE "
                            f"module due to hidden_size_per_layer_input=0): {_k}"
                        )
        elif _gemma_family_by_name and _has_ple_module:
            for _ple_name in (
                "per_layer_model_projection",
                "embed_tokens_per_layer",
            ):
                # Try both mlx_vlm naming conventions
                for _pfx in (
                    f"language_model.model.{_ple_name}",
                    f"model.language_model.{_ple_name}",
                ):
                    _w_key = f"{_pfx}.weight"
                    if _w_key not in shard_weights:
                        continue
                    _w = shard_weights[_w_key]
                    if _w.dtype != mx.uint32:
                        continue
                    _s_key = f"{_pfx}.scales"
                    _b_key = f"{_pfx}.biases"
                    if _s_key not in shard_weights:
                        continue
                    _s = shard_weights[_s_key]
                    _b = shard_weights.get(_b_key, mx.zeros_like(_s))
                    _ple_dequantized = False
                    for _try_bits in (8, 6, 4, 3, 2):
                        _elem = 32 // _try_bits
                        _real_cols = _w.shape[-1] * _elem
                        if _s.shape[-1] == 0:
                            continue
                        _gs = _real_cols // _s.shape[-1]
                        if _gs >= 2 and _gs * _s.shape[-1] == _real_cols:
                            try:
                                _dq = mx.dequantize(
                                    _w, _s, _b, group_size=_gs, bits=_try_bits
                                )
                                mx.eval(_dq)
                                shard_weights[_w_key] = _dq.astype(mx.float16)
                                del shard_weights[_s_key]
                                if _b_key in shard_weights:
                                    del shard_weights[_b_key]
                                logger.info(
                                    f"  Dequantized Gemma4 PLE: {_w_key} "
                                    f"(bits={_try_bits}, gs={_gs})"
                                )
                                _ple_dequantized = True
                                break
                            except Exception:
                                continue
                    # Audit-2026-04-07 risk §6.4 hardening: bit-width search exhaustion
                    # used to silently leave the uint32 weight in place — forward pass
                    # would then matmul/take on uint32 and produce garbage / all-pad
                    # output (#52). Fail loud instead so the user gets a clear error.
                    if not _ple_dequantized:
                        raise RuntimeError(
                            f"jang_loader Gemma4 PLE dequant: failed to find a valid "
                            f"bit-width for {_w_key} (shape={_w.shape}, scales="
                            f"{_s.shape}). Tried bits=[8,6,4,3,2]. Without dequant, "
                            f"forward pass produces garbage output (#52). Please "
                            f"verify the JANG file integrity and quant format."
                        )

        # Mistral4 MLA: split kv_b_proj → embed_q + unembed_out.
        # Original implementation by Jinho Jang (eric@jangq.ai) for vMLX.
        # MLA stores compressed KV latents — the HF kv_b_proj weight must be
        # dequantized, reshaped (nheads, head_dim, kv_rank), and split into
        # embed_q (nheads, kv_rank, qk_nope) and unembed_out (nheads, v_head, kv_rank).
        # JANG safetensors store the unsplit kv_b_proj weight, but the model
        # has MultiLinear modules for embed_q (k projection) and unembed_out
        # (v projection). Without this split, they keep random init → garbage.
        _text_mt = config.get("text_config", {}).get("model_type", "")
        if _text_mt == "mistral4":
            _t_cfg = config.get("text_config", config)
            _nheads = _t_cfg.get("num_attention_heads", 32)
            _qk_nope = _t_cfg.get("qk_nope_head_dim", 64)
            _v_head = _t_cfg.get("v_head_dim", 128)
            _kv_rank = _t_cfg.get("kv_lora_rank", 256)
            _head_dim = _qk_nope + _v_head
            _nlayers = _t_cfg.get("num_hidden_layers", 36)
            for _l in range(_nlayers):
                for _pfx in [
                    f"language_model.model.layers.{_l}.self_attn",
                    f"model.language_model.layers.{_l}.self_attn",
                ]:
                    _kb_key = f"{_pfx}.kv_b_proj.weight"
                    if _kb_key not in shard_weights:
                        continue
                    _v = shard_weights.pop(_kb_key)
                    # Dequantize if quantized (JANG stores attention at 8-bit)
                    _s_key = f"{_pfx}.kv_b_proj.scales"
                    _b_key = f"{_pfx}.kv_b_proj.biases"
                    if _s_key in shard_weights:
                        _s = shard_weights.pop(_s_key)
                        _b = shard_weights.pop(_b_key, mx.zeros_like(_s))
                        for _try_bits in [8, 6, 4, 3, 2]:
                            _elem = 32 // _try_bits
                            _real = _v.shape[-1] * _elem
                            _gs = _real // _s.shape[-1] if _s.shape[-1] > 0 else 0
                            if _gs > 0 and _gs * _s.shape[-1] == _real:
                                try:
                                    _v = mx.dequantize(_v, _s, _b, _gs, _try_bits)
                                    break
                                except Exception:
                                    continue
                    # (nheads*head_dim, kv_rank) → (nheads, head_dim, kv_rank)
                    _v = _v.reshape(_nheads, _head_dim, _kv_rank)
                    # embed_q: MultiLinear(qk_nope, kv_rank, nheads) → weight (nheads, kv_rank, qk_nope)
                    _wk = mx.contiguous(_v[:, :_qk_nope, :].swapaxes(-1, -2))
                    # unembed_out: MultiLinear(kv_rank, v_head, nheads) → weight (nheads, v_head, kv_rank)
                    _wv = mx.contiguous(_v[:, _qk_nope:, :])
                    shard_weights[f"{_pfx}.embed_q.weight"] = _wk.astype(mx.float16)
                    shard_weights[f"{_pfx}.unembed_out.weight"] = _wv.astype(mx.float16)
                    logger.debug(
                        f"  Split kv_b_proj layer {_l}: embed_q={_wk.shape}, unembed_out={_wv.shape}"
                    )

        # MoE gate dequant: MoEGate is nn.Module (not nn.Linear), so nn.quantize
        # skips it. But JANG still quantizes the raw gate weight. Dequantize here
        # so MoEGate.__call__ can do float matmul (x @ self.weight.T).
        _n_exp = config.get("text_config", config).get("n_routed_experts", 0)
        if _n_exp > 0:
            _gate_parts = {}
            _gate_keys_to_remove = []
            for k, v in shard_weights.items():
                if ".gate." in k and (k.endswith(".scales") or k.endswith(".biases")):
                    prefix = k.rsplit(".", 1)[0]
                    _gate_parts.setdefault(prefix, {})[k.rsplit(".", 1)[1]] = v
                    _gate_keys_to_remove.append(k)
            for k in _gate_keys_to_remove:
                del shard_weights[k]
            for prefix, parts in _gate_parts.items():
                wkey = f"{prefix}.weight"
                if wkey in shard_weights and "scales" in parts:
                    qw = shard_weights[wkey]
                    scales = parts["scales"]
                    biases = parts.get("biases", mx.zeros_like(scales))
                    for bits in [8, 6, 4, 3, 2]:
                        elem_per_u32 = 32 // bits
                        real_cols = qw.shape[-1] * elem_per_u32
                        gs = (
                            real_cols // scales.shape[-1] if scales.shape[-1] > 0 else 0
                        )
                        if gs > 0 and gs * scales.shape[-1] == real_cols:
                            try:
                                dq = mx.dequantize(qw, scales, biases, gs, bits)
                                mx.eval(dq)
                                shard_weights[wkey] = dq.astype(mx.bfloat16)
                                logger.debug(
                                    f"  Dequantized gate: {wkey} bits={bits} gs={gs}"
                                )
                                break
                            except Exception:
                                continue

        # Smelt mode: filter expert weights (loaded separately via ExpertIndex)
        if filter_expert_keys:
            shard_weights = {
                k: v for k, v in shard_weights.items() if not _is_expert_key(k)
            }
        # Pre-fix per-layer bits before load to prevent shape mismatch
        # ValueError on JANG mixed-precision models (fixes #62, #63).
        _pre_fix_bits_from_shard(model, shard_weights, block_size)
        model.load_weights(list(shard_weights.items()), strict=False)
        del shard_weights
        gc.collect()

    _fix_quantized_bits(model)

    if not hasattr(model, "config"):
        model.config = model_config

    # bfloat16 for MLA models and 512+ expert models
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = (
        _text_cfg.get("num_experts")
        or _text_cfg.get("num_local_experts")
        or _text_cfg.get("n_routed_experts")
        or 0
    )
    _hidden = _text_cfg.get("hidden_size") or 0
    _text_mt = _text_cfg.get("model_type", _model_cfg.get("model_type", ""))
    _is_mla = (_text_cfg.get("kv_lora_rank") or 0) > 0
    if (_n_experts >= 512 and _hidden >= 4096) or _text_mt == "mistral4" or _is_mla:
        model.set_dtype(mx.bfloat16)
        _reason = "MLA" if _is_mla else f"{_n_experts} experts"
        logger.info(f"  bfloat16 enabled: {_reason}, hidden={_hidden}")

    # Vision tower float16 overflow guard (Gemma 4 VLM JANG 2L/4M).
    #
    # Low-bit JANG profiles (2L = 2-bit, 4M = 4-bit) on the ~400M-param Gemma 4
    # vision tower accumulate rounding error through the 27 SigLIP encoder
    # layers. In float16 the absolute magnitudes blow past ±65504 in a middle
    # layer norm → intermediate activations become ±inf → `embed_vision`
    # projection flips every entry to NaN → the language model samples token
    # id 0 (`<pad>`) every step and the user sees "no image" output even
    # though the prompt correctly contains 256 image tokens + pixel_values.
    #
    # Upcasting the vision tower + multimodal projector to bfloat16 keeps the
    # same memory footprint as float16 (16 bits / param) but doubles the
    # dynamic range so the same activations land in the representable region.
    # Verified live on Gemma-4-26B-A4B-it-JANG_2L-CRACK 2026-04-09:
    #   float16  → vision_out min=-inf, embed_vision all NaN, output all <pad>
    #   bfloat16 → vision_out min=-7.0, embed_vision clean, correct answer
    #
    # Applied only to Gemma 4 VLM; other VLMs (Qwen3.5-VL, etc.) stay at
    # whatever mlx_vlm.load() produced.
    if _text_mt == "gemma4_text" and hasattr(model, "vision_tower"):
        try:
            model.vision_tower.set_dtype(mx.bfloat16)
            if hasattr(model, "embed_vision"):
                model.embed_vision.set_dtype(mx.bfloat16)
            logger.info(
                "  Vision tower upcast to bfloat16 (Gemma 4 VLM — avoids "
                "float16 overflow in SigLIP encoder with low-bit JANG quant)"
            )
        except Exception as _vt_err:
            logger.warning(
                "  Failed to upcast Gemma 4 vision tower to bfloat16: %s",
                _vt_err,
            )

    if not skip_eval:
        _set_wired_limit_for_model(_get_v2_weight_files(path))
        _chunked_eval_params(model)

    # TurboQuant: patch language_model.make_cache for JANG VLM with TQ enabled
    _lang_model = getattr(model, "language_model", None)
    if _lang_model is not None and hasattr(_lang_model, "layers"):
        _patch_turboquant_make_cache(_lang_model, jang_cfg, _model_cfg)

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

    # transformers' AutoVideoProcessor requires torchvision, which is not
    # in the bundled Python. Without this install, Qwen3VLProcessor's
    # video_processor stays None and the fallback path raises TypeError
    # the first time a caller passes `videos=`. This class-level patch
    # routes videos through image_processor when video_processor is None.
    try:
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        _install_video_fallback(processor)
    except Exception as _vfe:
        logger.debug(f"video fallback not installed: {_vfe}")

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


def load_jang_vlm_model(
    model_path: str | Path, skip_eval: bool = False, filter_expert_keys: bool = False
):
    """
    Load a JANG Vision-Language model into mlx-vlm for multimodal inference.

    Automatically detects v2 (instant) or v1 (repack) format.

    Args:
        model_path: Path to the JANG VLM model directory
        skip_eval: If True, skip _chunked_eval_params (for smelt deferred eval)
        filter_expert_keys: If True, skip expert weights (for smelt mode)

    Returns:
        Tuple of (model, processor) compatible with mlx-vlm.generate()
    """
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    jang_cfg = json.loads(config_path.read_text())
    # JANGTQ writer emits {"version": 2, "weight_format": "mxtq", ...} and
    # omits the legacy `format` field entirely. Accept that shape in addition
    # to the {"format": "jang"|"jjqf"|"mxq"} legacy envelope. Mirrors the text
    # loader at line ~1649 — without this, Qwen 3.6 JANGTQ_2L (VLM wrapper
    # model_type=qwen3_5_moe) hit `ValueError: Not a JANG model: format='None'`.
    fmt = jang_cfg.get("format")
    weight_format = jang_cfg.get("weight_format")
    if not fmt and weight_format == "mxtq":
        fmt = "mxtq"
    if not fmt or (fmt not in JANG_FORMAT_VALUES and fmt != "mxtq"):
        raise ValueError(
            f"Not a JANG VLM: format='{fmt}' weight_format='{weight_format}' "
            f"(expected one of {', '.join(JANG_FORMAT_VALUES)} or weight_format=mxtq)"
        )

    # v2: instant load
    if _is_v2_model(path):
        logger.info(f"JANG v2 VLM detected — loading via mmap (instant)")
        return _load_jang_v2_vlm(
            path, jang_cfg, skip_eval=skip_eval, filter_expert_keys=filter_expert_keys
        )

    # v1: repack path (legacy)
    logger.info(f"JANG v1 VLM detected — repacking (this takes a few minutes)")
    return _load_jang_v1_vlm(path, jang_cfg, config_path)


def load_jang_model(
    model_path: str | Path,
    config_manager: Optional[Any] = None,
    skip_eval: bool = False,
    filter_expert_keys: bool = False,
    layer_range: tuple = None,
):
    """
    Load a JANG model for inference.

    Automatically detects v2 (instant), v1 (repack), or codebook VQ format.
    v2 loads in seconds via mx.load() mmap.
    v1 repacks JANG uint8 → MLX uint32 (takes 5-10 minutes for large models).
    Codebook VQ uses special wrapper with codebook-compressed expert weights.

    Args:
        model_path: Path to the JANG model directory
        config_manager: Optional ConfigManager for codebook/kernel settings

    Returns:
        Tuple of (model, tokenizer) compatible with mlx-lm
    """
    path = _resolve_local_path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    jang_cfg = json.loads(config_path.read_text())
    # JANGTQ writer emits {"version": 2, "weight_format": "mxtq", ...} and
    # omits the legacy `format` field entirely. Accept that shape in addition
    # to the {"format": "jang"|"jjqf"|"mxq"} legacy envelope.
    fmt = jang_cfg.get("format")
    weight_format = jang_cfg.get("weight_format")
    if not fmt and weight_format == "mxtq":
        fmt = "mxtq"
    if not fmt:
        raise ValueError(
            f"JANG config {config_path.name} is missing 'format' / 'weight_format'. "
            f"Expected one of: {', '.join(JANG_FORMAT_VALUES)} or weight_format=mxtq"
        )
    if fmt not in JANG_FORMAT_VALUES and fmt != "mxtq":
        raise ValueError(
            f"Not a JANG model: format='{fmt}' (expected {', '.join(JANG_FORMAT_VALUES)} "
            f"or weight_format=mxtq)"
        )

    # Legacy: format_version string ("1.0"/"2.0"). JANGTQ: int version 2.
    _raw_ver = jang_cfg.get("format_version", jang_cfg.get("version", "1.0"))
    version = str(_raw_ver)
    try:
        major = int(version.split(".")[0])
    except ValueError:
        raise ValueError(
            f"Invalid JANG version: '{version}' (expected numeric)"
        )
    if major > 2:
        raise ValueError(
            f"Unsupported JANG format version: {version} (this loader supports 1.x and 2.x)"
        )

    # Codebook VQ: check before v2 to ensure it routes correctly
    if _is_codebook_vq_model(path):
        logger.info(f"Codebook VQ model detected — loading with codebook support")
        return _load_codebook_vq_model(path, jang_cfg, config_manager=None)

    # v2: instant load via mmap
    if _is_v2_model(path):
        logger.info(f"JANG v2 detected — loading via mmap (instant)")
        return _load_jang_v2(
            path, jang_cfg, skip_eval=skip_eval, filter_expert_keys=filter_expert_keys,
            layer_range=layer_range,
        )

    # v1: repack path (legacy)
    logger.info(
        f"JANG v1 detected — repacking to MLX format (this may take a few minutes)"
    )
    return _load_jang_v1(path, jang_cfg, config_path)


# ─── v1 loader (legacy, repack) ─────────────────────────────────────


def _load_jang_v1(path: Path, jang_cfg: dict, config_path: Path):
    """Load a JANG v1 model by repacking weights from uint8 to uint32."""
    from mlx_lm.utils import (
        load_config,
        load_model as _load_model_skeleton,
        load_tokenizer,
    )

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

    # Nemotron-H LatentMoE patch — see _load_jang_v2 for rationale.
    try:
        from .nemotron_latent_moe import ensure_latent_moe_support
        ensure_latent_moe_support(str(path))
    except Exception as _lmoe_e:
        logger.debug(f"LatentMoE patch skipped: {_lmoe_e}")

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
                _pre_fix_bits_from_shard(model, shard_weights, block_size)
                model.load_weights(list(shard_weights.items()), strict=False)
                del shard_weights
                gc.collect()
        else:
            weights = result
            if hasattr(model, "sanitize"):
                weights = model.sanitize(weights)
            _pre_fix_bits_from_shard(model, weights, block_size)
            model.load_weights(list(weights.items()), strict=False)
            del weights
            gc.collect()

        _fix_quantized_bits(model)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = config

    # bfloat16 for 512+ expert models (same as v2 loader)
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = (
        _text_cfg.get("num_experts")
        or _text_cfg.get("num_local_experts")
        or _text_cfg.get("n_routed_experts")
        or 0
    )
    _hidden = _text_cfg.get("hidden_size") or 0
    _text_mt = _text_cfg.get("model_type", _model_cfg.get("model_type", ""))
    _is_mla = (_text_cfg.get("kv_lora_rank") or 0) > 0
    if (_n_experts >= 512 and _hidden >= 4096) or _text_mt == "mistral4" or _is_mla:
        model.set_dtype(mx.bfloat16)
        _reason = "MLA" if _is_mla else f"{_n_experts} experts"
        logger.info(f"  bfloat16 enabled: {_reason}, hidden={_hidden}")

    _chunked_eval_params(model)

    _patch_turboquant_make_cache(model, jang_cfg, _model_cfg)

    elapsed = time.perf_counter() - start
    from mlx.utils import tree_flatten

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    logger.info(
        f"JANG v1 model loaded in {elapsed:.1f}s: "
        f"{n_params / 1e9:.1f}B params, {actual_bits:.1f}-bit avg"
    )

    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def _load_jang_v1_vlm(
    path: Path,
    jang_cfg: dict,
    config_path: Path,
):
    """Load a JANG v1 VLM model by repacking (legacy)."""
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args,
        load_config as vlm_load_config,
        update_module_configs,
        load_image_processor,
        load_processor,
        skip_multimodal_module,
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

        nn.quantize(
            model,
            group_size=block_size,
            bits=default_bits,
            class_predicate=get_class_predicate,
        )

        from mlx_vlm.utils import sanitize_weights

        for sf in shard_files:
            shard_weights = mx.load(sf)
            if hasattr(model, "sanitize"):
                shard_weights = model.sanitize(shard_weights)
            shard_weights = sanitize_weights(
                model_class.VisionModel, shard_weights, model_config.vision_config
            )
            shard_weights = sanitize_weights(
                model_class.LanguageModel, shard_weights, model_config.text_config
            )
            _pre_fix_bits_from_shard(model, shard_weights, block_size)
            model.load_weights(list(shard_weights.items()), strict=False)
            del shard_weights
            gc.collect()

        _fix_quantized_bits(model)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = model_config

    # bfloat16 for 512+ expert models (same as v2 loader)
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = (
        _text_cfg.get("num_experts")
        or _text_cfg.get("num_local_experts")
        or _text_cfg.get("n_routed_experts")
        or 0
    )
    _hidden = _text_cfg.get("hidden_size") or 0
    _text_mt = _text_cfg.get("model_type", _model_cfg.get("model_type", ""))
    _is_mla = (_text_cfg.get("kv_lora_rank") or 0) > 0
    if (_n_experts >= 512 and _hidden >= 4096) or _text_mt == "mistral4" or _is_mla:
        model.set_dtype(mx.bfloat16)
        _reason = "MLA" if _is_mla else f"{_n_experts} experts"
        logger.info(f"  bfloat16 enabled: {_reason}, hidden={_hidden}")

    _chunked_eval_params(model)

    _lang_model = getattr(model, "language_model", None)
    if _lang_model is not None and hasattr(_lang_model, "layers"):
        _patch_turboquant_make_cache(_lang_model, jang_cfg, _model_cfg)

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

    # See v2 VLM path for rationale — install torchvision-free video fallback.
    try:
        from jang_tools.load_jangtq_vlm import _install_video_fallback
        _install_video_fallback(processor)
    except Exception as _vfe:
        logger.debug(f"video fallback not installed: {_vfe}")

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

    INDEX_NAMES = [
        "model.jang.index.json",
        "model.jjqf.index.json",
        "model.mxq.index.json",
    ]
    SHARD_GLOBS = ["*.jang.safetensors", "*.jjqf.safetensors", "*.mxq.safetensors"]
    SUFFIXES = (
        ".qweight",
        ".scales",
        ".zeros",
        ".biases",
        ".bit_map",
        ".block_offsets",
        ".shape",
        ".bits",
    )

    index_path = None
    for name in INDEX_NAMES:
        p = model_path / name
        if p.exists():
            index_path = p
            break

    shard_files = []
    if index_path:
        index = json.loads(index_path.read_text())
        shard_files = [
            model_path / sf for sf in sorted(set(index["weight_map"].values()))
        ]
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
        total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError):
        import subprocess

        total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())

    model_disk_bytes = sum(sf.stat().st_size for sf in shard_files if sf.exists())
    ram_threshold = int(total_ram * 0.50)
    use_streaming = model_disk_bytes > ram_threshold

    if use_streaming:
        logger.info(
            f"  Streaming mode: model {model_disk_bytes / 1e9:.0f} GB > 50% of {total_ram / 1e9:.0f} GB RAM"
        )
    else:
        logger.info(
            f"  In-memory mode: model {model_disk_bytes / 1e9:.0f} GB fits in {total_ram / 1e9:.0f} GB RAM"
        )

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
        logger.info(
            f"  Flushed shard {shard_idx} ({current_bytes / 1e9:.1f} GB, {len(current_shard)} tensors)"
        )
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
            packed_bytes += b"\x00" * pad_needed
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
            mlx_scales = jang_scales[:expected_groups].reshape(
                num_experts, expert_out, n_groups_per_row
            )
            mlx_biases = jang_biases[:expected_groups].reshape(
                num_experts, expert_out, n_groups_per_row
            )
        else:
            mlx_scales = jang_scales[:expected_groups].reshape(
                out_dim, n_groups_per_row
            )
            mlx_biases = jang_biases[:expected_groups].reshape(
                out_dim, n_groups_per_row
            )

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
                scale_key = (
                    weight_key.replace(".weight", "")
                    if ".weight" in weight_key
                    else weight_key
                )
                expert_buffer[weight_key] = mx.array(mlx_qweight)
                expert_buffer[f"{scale_key}.scales"] = mx.array(mlx_scales)
                expert_buffer[f"{scale_key}.biases"] = mx.array(mlx_biases)
            else:
                _add_to_shard(weight_key, mx.array(mlx_qweight))
                scale_key = (
                    weight_key.replace(".weight", "")
                    if ".weight" in weight_key
                    else weight_key
                )
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
    rename_keys += [
        (k, "vision_tower" + k[len("model.visual") :])
        for k in list(current_shard.keys())
        if k.startswith("model.visual")
    ]
    rename_keys += [
        (k, "language_model.model" + k[len("model.language_model") :])
        for k in list(current_shard.keys())
        if k.startswith("model.language_model")
    ]
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
                new_k = "vision_tower" + k[len("model.visual") :]
                renamed[new_k] = v
                needs_rewrite = True
            elif k.startswith("model.language_model"):
                new_k = "language_model.model" + k[len("model.language_model") :]
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
        logger.info(
            f"  Stacked {len(expert_groups)} expert groups into QuantizedSwitchLinear format"
        )


def _upgrade_switch_to_quantized(model, bits, group_size):
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    except ImportError:
        return

    for name, module in model.named_modules():
        if not isinstance(module, SwitchLinear):
            continue
        ql = QuantizedSwitchLinear(
            module.input_dims,
            module.output_dims,
            module.num_experts,
            bias=hasattr(module, "bias"),
            group_size=group_size,
            bits=bits,
        )
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model
            for p in parts[0].split("."):
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, parts[1], ql)


def _upgrade_modules_with_uint32_weights(model, default_bits: int, default_group_size: int) -> int:
    """Walk the model and replace any nn.Linear / nn.Embedding whose weight is
    uint32 (i.e. JANG-packed quantized) with the matching Quantized variant.

    Why this exists: mlx_lm.utils.load_model's internal `nn.quantize` predicate
    is `f"{p}.scales" in weights`, where `weights` is the dict loaded directly
    from the safetensors file. For Mistral-Small-4-119B JANG (and any other
    JANG VLM loaded as text-only via the model_type promotion path), the file
    keys are `language_model.model.X` but the model module paths are `model.X`.
    The predicate never matches → modules stay as plain Linear/Embedding →
    JANG uint32 weights load into them but the forward pass treats them as
    floats → garbage / shape mismatches / 'rms_norm weight has 4096 elements'
    crashes deep in the layer call.

    This pass runs AFTER `model.load_weights(...)` so each module already has
    its uint32 weight + scales + biases. We replace the module in place with
    QuantizedLinear / QuantizedEmbedding using bits/group_size inferred from
    the actual weight + scales shapes.

    Returns the number of modules upgraded.
    """
    import mlx.nn as nn
    upgraded = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        w = getattr(module, "weight", None)
        if w is None or w.dtype != mx.uint32:
            continue
        s = getattr(module, "scales", None)
        if s is None:
            continue
        # Infer bits + group_size from actual shapes.
        # weight: (..., packed_cols) where packed_cols = real_cols * bits / 32
        # scales: (..., scale_cols) where scale_cols = real_cols / group_size
        try:
            packed_cols = w.shape[-1]
            scale_cols = s.shape[-1]
            inferred = None
            for try_bits in (8, 6, 4, 3, 2):
                real_cols = packed_cols * 32 // try_bits
                if real_cols % scale_cols != 0:
                    continue
                try_gs = real_cols // scale_cols
                if try_gs in (32, 64, 128):
                    inferred = (try_bits, try_gs)
                    break
            if inferred is None:
                inferred = (default_bits, default_group_size)
            bits, gs = inferred
        except Exception:
            bits, gs = default_bits, default_group_size

        # Build the matching Quantized variant.
        try:
            if isinstance(module, nn.Linear):
                in_dim = w.shape[-1] * 32 // bits
                out_dim = w.shape[0]
                qmod = nn.QuantizedLinear(
                    input_dims=in_dim,
                    output_dims=out_dim,
                    bias=hasattr(module, "bias") and getattr(module, "bias", None) is not None,
                    group_size=gs,
                    bits=bits,
                )
            else:  # Embedding
                # Embedding stores (num_embeddings, packed) for QuantizedEmbedding
                num_emb = w.shape[0]
                emb_dim = w.shape[-1] * 32 // bits
                qmod = nn.QuantizedEmbedding(
                    num_embeddings=num_emb,
                    dims=emb_dim,
                    group_size=gs,
                    bits=bits,
                )
            # Move the loaded uint32 weight + scales + biases into the new module
            qmod.weight = w
            qmod.scales = s
            if hasattr(module, "biases"):
                b = getattr(module, "biases", None)
                if b is not None:
                    qmod.biases = b
        except Exception as e:
            logger.debug(f"  Quantized upgrade failed for {name}: {e}")
            continue

        # Splice into the parent module
        parts = name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        parent = model
        try:
            for p in parts[0].split("."):
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, parts[1], qmod)
            upgraded += 1
        except Exception:
            continue
    return upgraded


def _pre_fix_bits_from_shard(model, shard_weights, block_size):
    """Fix QuantizedLinear.bits from actual weight shapes BEFORE load_weights.

    JANG mixed-precision models have per-layer bit widths (e.g. [3, 4, 8]),
    but nn.quantize() applies a uniform bits=min(bit_widths) to ALL modules.
    With strict=False, load_weights() silently overwrites weight shapes even
    when they don't match the module's expected packed size. However, the
    module's .bits attribute stays at the wrong value until _fix_quantized_bits
    runs — any dequantization in that window crashes with "quantized_matmul:
    shapes incompatible". This function eliminates that dangerous window.

    Also required for the doctor command path, which previously used raw
    mlx_lm.load() with strict=True (default) — that DOES crash on shape
    mismatch (ValueError).

    Must be called after nn.quantize() and after sanitize/remap, but before
    model.load_weights(). Safe to call multiple times across shards.

    Fixes GitHub issues #62 (MiniMax-M2.5-JANG_3L) and #63
    (Qwen3.5-122B-A10B-JANG_4K) where embed_tokens is quantized at 4-bit
    but the module was created at 3-bit (min of bit_widths_used).
    """
    # Build module lookup from model tree — paths match sanitized weight keys
    # after stripping the ".weight" suffix (standard MLX convention).
    modules_by_path = {}
    for mod_path, mod in model.named_modules():
        if hasattr(mod, "bits") and hasattr(mod, "group_size"):
            modules_by_path[mod_path] = mod

    if not modules_by_path:
        return

    fixed_count = 0
    for k, v in shard_weights.items():
        try:
            if not k.endswith(".weight"):
                continue
            if not hasattr(v, "dtype") or v.dtype != mx.uint32:
                continue
            s_key = k[:-7] + ".scales"
            if s_key not in shard_weights:
                continue

            w_cols = v.shape[-1]
            s_cols = shard_weights[s_key].shape[-1]
            if s_cols <= 0:
                continue

            mod_path = k[:-7]  # strip ".weight"
            module = modules_by_path.get(mod_path)
            if module is None:
                continue

            # Try block_size candidates — same priority as _fix_quantized_bits:
            # config block_size first, then module's current gs, then common sizes.
            gs_candidates = [block_size]
            if hasattr(module, "group_size") and module.group_size not in gs_candidates:
                gs_candidates.append(module.group_size)
            for gs in (64, 128):
                if gs not in gs_candidates:
                    gs_candidates.append(gs)

            for try_bs in gs_candidates:
                in_dim = s_cols * try_bs
                if in_dim <= 0 or (w_cols * 32) % in_dim != 0:
                    continue
                actual_bits = (w_cols * 32) // in_dim
                if actual_bits not in (2, 3, 4, 5, 6, 8):
                    continue
                changed = False
                if actual_bits != module.bits:
                    module.bits = actual_bits
                    changed = True
                if try_bs != module.group_size:
                    module.group_size = try_bs
                    changed = True
                if changed:
                    fixed_count += 1
                    logger.debug(
                        f"  Pre-fix bits: {mod_path} → {actual_bits}-bit gs={try_bs}"
                    )
                break
        except Exception as e:
            logger.debug(f"  Pre-fix bits: skipped {k}: {e}")

    if fixed_count > 0:
        logger.info(
            f"  Pre-fixed {fixed_count} module(s) with mixed-precision bit widths"
        )


def _fix_quantized_bits(model):
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
    # MLA models (Mistral 4, DeepSeek V3) use QuantizedMultiLinear for embed_q/unembed_out.
    # Without this, _fix_quantized_bits never corrects the bits/group_size mismatch when
    # nn.quantize sets bits=2 but sanitize loads 8-bit kv_b_proj split weights.
    # Original MLA quantization fix by Jinho Jang (eric@jangq.ai) — vMLX/mlxstudio.
    try:
        from mlx_lm.models.mla import QuantizedMultiLinear

        quant_types = quant_types + (QuantizedMultiLinear,)
    except ImportError:
        pass

    for name, module in model.named_modules():
        if not isinstance(module, quant_types):
            continue
        if not hasattr(module, "scales") or not hasattr(module, "weight"):
            continue
        try:
            w_cols = module.weight.shape[-1]
            s_cols = module.scales.shape[-1]
            fixed = False

            # Router/gate tensors prefer gs=64 (precision-critical in JANG)
            name_lower = name.lower()
            is_router = (
                ".gate." in name_lower
                or name_lower.endswith(".gate")
                or "shared_expert_gate" in name_lower
            )
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
    tokenizer_obj = (
        processor.tokenizer if hasattr(processor, "tokenizer") else processor
    )
    processor.detokenizer = detokenizer_class(tokenizer_obj)

    final_eos = (
        eos_token_id
        if eos_token_id is not None
        else getattr(tokenizer_obj, "eos_token_ids", None)
    )
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
    shared_expert_intermediate = _get(
        "shared_expert_intermediate_size", moe_intermediate
    )
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
        ei = (
            shared_expert_intermediate
            if ".shared_expert." in name
            else (moe_intermediate if moe_intermediate else intermediate)
        )
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
