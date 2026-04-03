# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import json
import logging
from pathlib import Path

from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE

logger = logging.getLogger(__name__)


def _apply_turboquant_to_model(model, model_path: str):
    """Apply TurboQuant KV cache compression to any MLX model.

    Patches model.make_cache() to return TurboQuantKVCache objects instead of
    standard KVCache. Works for ALL model types — JANG models get TQ applied
    in their own loader, this handles standard MLX models.

    Safe to call on any model: if jang_tools.turboquant is not installed or
    the model doesn't have .layers, it silently returns without changes.
    """
    try:
        from jang_tools.turboquant.config import TurboQuantConfig, make_turboquant_cache
    except ImportError:
        return  # TQ not available

    # Need model with .layers to determine layer count and dims
    if not hasattr(model, 'layers') or not model.layers:
        return

    try:
        n_layers = len(model.layers)

        # Read config.json for head dimensions and layer types
        config_path = Path(model_path) / 'config.json'
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text())
        text_cfg = config.get('text_config', config)

        # Key/value dimensions
        key_dim = text_cfg.get('head_dim', 128)
        val_dim = text_cfg.get('head_dim', 128)
        if text_cfg.get('kv_lora_rank', 0) > 0:
            key_dim = text_cfg.get('qk_nope_head_dim', 128) + text_cfg.get('qk_rope_head_dim', 64)
            val_dim = text_cfg.get('v_head_dim', 128)

        # Layer types (hybrid detection)
        layer_type_list = text_cfg.get('layer_types', [])
        hybrid_pattern = text_cfg.get('hybrid_override_pattern',
                            config.get('hybrid_override_pattern', ''))
        if layer_type_list:
            layer_types = [
                'attention' if lt == 'full_attention' else 'ssm'
                for lt in layer_type_list[:n_layers]
            ]
            while len(layer_types) < n_layers:
                layer_types.append('attention')
        elif hybrid_pattern:
            layer_types = []
            for ch in hybrid_pattern[:n_layers]:
                if ch == 'M':
                    layer_types.append('ssm')
                elif ch == '*':
                    layer_types.append('attention')
                # 'E' (MoE expert) layers are FFN-only with no cache.
                # Model.make_cache() and forward pass skip them, so we must too
                # to keep cache indices aligned.
        else:
            layer_types = ['attention'] * n_layers

        # Default TQ config
        tq_config = TurboQuantConfig(
            n_layers=n_layers,
            default_key_bits=3,
            default_value_bits=3,
            critical_key_bits=4,
            critical_value_bits=4,
            critical_layers=[0, 1, 2, -3, -2, -1],
            seed=42,
        )

        n_cache = len(layer_types)
        def _tq_make_cache(_cfg=tq_config, _n=n_cache, _kd=key_dim, _vd=val_dim, _lt=layer_types):
            return make_turboquant_cache(_cfg, _n, [_kd]*_n, [_vd]*_n, _lt)

        model.make_cache = _tq_make_cache

        n_attn = sum(1 for t in layer_types if t == 'attention')
        n_ssm = sum(1 for t in layer_types if t == 'ssm')
        logger.info(f"  TurboQuant auto-enabled: 3-bit keys/values, "
                    f"{n_attn} attention" + (f" + {n_ssm} SSM" if n_ssm > 0 else "") + " layers")
    except Exception as e:
        logger.debug(f"TurboQuant auto-enable failed (non-fatal): {e}")

# Models that require tokenizer fallback
FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def _get_model_type_from_config(model_name: str) -> str | None:
    """Read model_type from config.json if model_name is a local directory."""
    model_path = Path(model_name)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                return config.get("model_type", "").lower() or None
            except Exception:
                pass
    return None


# model_type values that do NOT need tokenizer fallback (standard architectures).
# Must stay in sync with MODEL_TYPE_TO_FAMILY in panel/src/main/model-config-registry.ts.
# Notably EXCLUDES nemotron/nemotron_h which DO need tokenizer fallback.
_STANDARD_ARCHITECTURES = {
    # Qwen
    "qwen3_5", "qwen3_5_moe",
    "qwen3", "qwen3_moe", "qwen3_vl", "qwen3_vl_moe", "qwen3_next",
    "qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen", "qwen_mamba",
    # Llama
    "llama", "llama4",
    # Mistral
    "mistral", "mistral4", "mixtral", "pixtral", "codestral", "devstral", "codestral_mamba",
    # DeepSeek
    "deepseek_v2", "deepseek_v3", "deepseek_vl", "deepseek_vl2", "deepseek_vl_v2",
    "deepseek2", "deepseek",
    # GLM / GPT-OSS
    "chatglm", "glm4", "glm4_moe", "glm4_moe_lite", "glm", "gpt_oss",
    # StepFun
    "step3p5", "step", "step1v",
    # Gemma
    "gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text", "paligemma", "paligemma2",
    # Phi
    "phi3", "phi3v", "phi3small", "phi4", "phi4mm", "phi4flash", "phi4_reasoning", "phi",
    # MiniMax
    "minimax", "minimax_m2", "minimax_m2_5",
    # Jamba / Mamba / SSM
    "jamba", "mamba", "mamba2", "falcon_mamba", "rwkv", "rwkv5", "rwkv6",
    # IBM Granite
    "granite", "granite_moe",
    # Cohere
    "cohere", "cohere2",
    # Others
    "hermes", "kimi_k2", "exaone", "exaone3", "olmo", "olmo2",
    "starcoder2", "stablelm", "baichuan",
    "internlm", "internlm2", "internlm3", "internlm_xcomposer2",
    "yi", "orion",
    # MLLM
    "llava", "llava_next", "idefics2", "idefics3", "cogvlm", "cogvlm2",
    "florence2", "molmo", "minicpmv", "smolvlm", "internvl_chat",
}


def _needs_tokenizer_fallback(model_name: str) -> bool:
    """Check if model needs tokenizer fallback.

    Reads config.json first for authoritative architecture detection.
    A Qwen3 fine-tune named "Nemotron-Orchestrator" should NOT get the
    Nemotron tokenizer fallback — its real architecture is Qwen3.
    """
    # 1. Authoritative: read config.json model_type if local directory
    model_type = _get_model_type_from_config(model_name)
    if model_type:
        if model_type in _STANDARD_ARCHITECTURES:
            logger.info(
                f"config.json model_type='{model_type}' is standard — "
                f"skipping tokenizer fallback for {model_name}"
            )
            return False
        if model_type in ("nemotron", "nemotron_h"):
            return True

    # 2. Try registry (name-based pattern matching)
    try:
        from ..model_config_registry import get_model_config_registry

        registry = get_model_config_registry()
        config = registry.lookup(model_name)
        if config.family_name != "unknown":
            return config.tokenizer_fallback
    except Exception:
        pass  # Fall through to pattern matching

    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


def load_model_with_fallback(model_name: str, tokenizer_config: dict = None):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from mlx_lm import load

    tokenizer_config = tokenizer_config or {}

    # Check if local path exists before loading
    model_path = Path(model_name)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_name}. "
            f"Check that the model directory is available."
        )

    # Check if disk streaming mode is active (lazy mmap loading)
    from .. import server as _server_module
    _lazy = getattr(_server_module, '_stream_from_disk', False)

    # Resolve HuggingFace repo IDs to local paths so that JANG detection
    # (which checks for jang_config.json on disk) works for remote models.
    from ..api.utils import resolve_to_local_path
    local_model_path = resolve_to_local_path(model_name)
    if local_model_path != model_name:
        logger.info(f"Resolved HF model to: {local_model_path}")

    # JANG format MUST be checked FIRST — JANG models use their own loader that
    # repacks weights into QuantizedLinear and handles tokenizer internally.
    # Checking tokenizer fallback first would bypass the JANG loader for Nemotron-H.
    from .jang_loader import is_jang_model
    if is_jang_model(local_model_path):
        from .jang_loader import load_jang_model
        logger.info(f"Detected JANG model: {model_name}")
        return load_jang_model(local_model_path, lazy=_lazy)

    # Check if model needs tokenizer fallback (e.g., Nemotron).
    # Pass resolved local path so _get_model_type_from_config can read config.json.
    if _needs_tokenizer_fallback(local_model_path):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        model, tokenizer = _load_with_tokenizer_fallback(local_model_path, lazy=_lazy)
        _apply_turboquant_to_model(model, local_model_path)
        return model, tokenizer

    try:
        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config, lazy=_lazy)
        _apply_turboquant_to_model(model, local_model_path)
        return model, tokenizer
    except ValueError as e:
        # Fallback for models with non-standard tokenizers
        if "TokenizersBackend" in str(e) or "Tokenizer class" in str(e):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            model, tokenizer = _load_with_tokenizer_fallback(model_name, lazy=_lazy)
            _apply_turboquant_to_model(model, local_model_path)
            return model, tokenizer
        else:
            raise


def _load_with_tokenizer_fallback(model_name: str, lazy: bool = False):
    """Load model with fallback tokenizer for non-standard models like Nemotron."""
    from mlx_lm.utils import load_model

    logger.info("Loading with tokenizer fallback...")

    # Get model path: use local directory directly, or download from HuggingFace
    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(model_name))

    # Apply LatentMoE patch before model load (must happen before NemotronHBlock init)
    from .nemotron_latent_moe import ensure_latent_moe_support
    ensure_latent_moe_support(str(model_path))

    # Load model
    model, _ = load_model(model_path, lazy=lazy)

    # Try to load tokenizer from tokenizer.json directly
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading tokenizer from tokenizer.json")
        base_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        # Read tokenizer_config.json for special tokens and chat template
        tokenizer_config_path = model_path / "tokenizer_config.json"
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                unk_token = config.get("unk_token", unk_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token="<pad>",
        )

        # Set chat template: tokenizer_config.json > chat_template.jinja > fallback
        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template loaded from tokenizer_config.json")
        else:
            # Check for chat_template.jinja file (common with community models)
            jinja_path = model_path / "chat_template.jinja"
            if jinja_path.exists():
                try:
                    jinja_template = jinja_path.read_text()
                    tokenizer.chat_template = jinja_template
                    logger.info("Chat template loaded from chat_template.jinja")
                except Exception as e:
                    logger.warning(f"Failed to read chat_template.jinja: {e}")
                    jinja_template = None
            else:
                jinja_template = None

            # Fall back to built-in templates if no model template found
            if not getattr(tokenizer, 'chat_template', None):
                if _needs_tokenizer_fallback(model_name):
                    tokenizer.chat_template = NEMOTRON_CHAT_TEMPLATE
                    logger.info("Using fallback Nemotron chat template")
                else:
                    tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
                    logger.info("Using default ChatML chat template")

        logger.info("Tokenizer loaded via fallback successfully")
        return model, tokenizer
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
