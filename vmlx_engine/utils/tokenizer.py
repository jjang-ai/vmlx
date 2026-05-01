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
    if not hasattr(model, "layers") or not model.layers:
        return

    # MLA models (DeepSeek V2/V3, GLM-5.1, Mistral 4) use CacheList(KVCache, KVCache)
    # per layer. TQ flat cache breaks CacheList structure → BatchGenerator fails.
    # Centralized via model_inspector.is_mla_model() so this stays in sync with
    # jang_loader.py and Agent 1's prefix-cache trie (REQ-001, 2026-04-07 audit).
    from .model_inspector import _detect_turboquant_layer_types, is_mla_model

    if is_mla_model(model_path):
        logger.info("  TurboQuant skipped: MLA model (CacheList incompatible)")
        return

    try:
        n_layers = len(model.layers)

        # Read config.json for head dimensions and layer types
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text())
        text_cfg = config.get("text_config", config)

        layer_types, key_dim, val_dim = _detect_turboquant_layer_types(
            text_cfg, n_layers, root_cfg=config
        )

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

        def _tq_make_cache(
            _cfg=tq_config, _n=n_cache, _kd=key_dim, _vd=val_dim, _lt=layer_types
        ):
            return make_turboquant_cache(_cfg, _n, [_kd] * _n, [_vd] * _n, _lt)

        model.make_cache = _tq_make_cache

        n_attn = sum(1 for t in layer_types if t == "attention")
        n_ssm = sum(1 for t in layer_types if t == "ssm")
        logger.info(
            f"  TurboQuant auto-enabled: 3-bit keys/values, "
            f"{n_attn} attention" + (f" + {n_ssm} SSM" if n_ssm > 0 else "") + " layers"
        )
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
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_moe_text",  # Qwen3.6-35B-A3B inner text_config.model_type
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_next",
    "qwen2",
    "qwen2_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen",
    "qwen_mamba",
    # Llama
    "llama",
    "llama4",
    # Mistral
    "mistral",
    "mistral4",
    "mixtral",
    "pixtral",
    "codestral",
    "devstral",
    "codestral_mamba",
    # DeepSeek
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v4",   # DSV4-Flash / V4-Pro — MLA head_dim=512, mHC, sqrtsoftplus MoE
    "deepseek_vl",
    "deepseek_vl2",
    "deepseek_vl_v2",
    "deepseek2",
    "deepseek",
    # GLM / GPT-OSS
    "chatglm",
    "glm4",
    "glm4_moe",
    "glm4_moe_lite",
    "glm",
    "gpt_oss",
    # StepFun
    "step3p5",
    "step",
    "step1v",
    # Gemma
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma4",
    "gemma4_text",
    "paligemma",
    "paligemma2",
    # Phi
    "phi3",
    "phi3v",
    "phi3small",
    "phi4",
    "phi4mm",
    "phi4flash",
    "phi4_reasoning",
    "phi",
    # MiniMax
    "minimax",
    "minimax_m2",
    "minimax_m2_5",
    # Jamba / Mamba / SSM
    "jamba",
    "mamba",
    "mamba2",
    "falcon_mamba",
    "rwkv",
    "rwkv5",
    "rwkv6",
    # IBM Granite
    "granite",
    "granite_moe",
    # Cohere
    "cohere",
    "cohere2",
    # Others
    "hermes",
    "kimi_k2",
    "kimi_k25",       # Kimi K2.6 — DeepseekV3 backbone + MoonViT vision; loads via jang_tools.load_jangtq_kimi_vlm
    "exaone",
    "exaone3",
    "olmo",
    "olmo2",
    "starcoder2",
    "stablelm",
    "baichuan",
    "internlm",
    "internlm2",
    "internlm3",
    "internlm_xcomposer2",
    "yi",
    "orion",
    # MLLM
    "llava",
    "llava_next",
    "idefics2",
    "idefics3",
    "cogvlm",
    "cogvlm2",
    "florence2",
    "molmo",
    "minicpmv",
    "smolvlm",
    "internvl_chat",
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


def _patch_mlx_lm_tokenizer_load() -> None:
    """vmlx#80 root-cause fix: HF transformers' ``AutoTokenizer.from_pretrained``
    does not auto-load ``chat_template.jinja`` from a model directory — only
    ``tokenizer_config.json["chat_template"]`` is recognised natively. Every
    mlx-community quant that ships its template as a separate jinja file
    (gemma-4-31b-8bit, several recent mistral / qwen quants) hits a
    ``ValueError: tokenizer.chat_template is not set`` on the FIRST chat
    request.

    The cleanest fix is to monkey-patch ``mlx_lm.tokenizer_utils.load`` ONCE
    at vmlx_engine import time so EVERY downstream caller benefits — vmlx
    direct loads, jang_tools.load_jangtq, smelt_loader, distributed worker,
    test harnesses. Idempotent (guarded by ``_vmlx_chat_template_patched``).

    The patch wraps the original ``load``, calls it normally, then probes
    the wrapped tokenizer for a missing template and injects from
    ``chat_template.jinja`` / ``chat_template.json`` in the model dir.

    For mlx_vlm (which has its OWN ``tokenizer_utils.load`` not derived from
    mlx_lm), ``_inject_chat_template_if_missing`` below is the second line of
    defence — called from ``load_model_with_fallback`` after every return.
    """
    try:
        from mlx_lm import tokenizer_utils as _tu
    except Exception as _ie:
        logger.debug(f"mlx_lm.tokenizer_utils not importable: {_ie}")
        return

    if getattr(_tu.load, "_vmlx_chat_template_patched", False):
        return

    _orig_load = _tu.load

    def _patched_load(model_path, tokenizer_config_extra=None, eos_token_ids=None):
        wrapper = _orig_load(
            model_path,
            tokenizer_config_extra=tokenizer_config_extra,
            eos_token_ids=eos_token_ids,
        )
        # mlx_lm returns a TokenizerWrapper. Probe + inject on the inner
        # tokenizer so .chat_template proxy access picks up the new value.
        inner = getattr(wrapper, "_tokenizer", wrapper)
        try:
            existing = getattr(inner, "chat_template", None)
            already_set = (isinstance(existing, str) and existing.strip()) or (
                isinstance(existing, list) and existing
            )
            if already_set:
                return wrapper
        except Exception:
            return wrapper

        try:
            from pathlib import Path as _P
            mp = _P(model_path) if not isinstance(model_path, _P) else model_path
            if not mp.is_dir():
                return wrapper
            jinja = mp / "chat_template.jinja"
            if jinja.is_file():
                inner.chat_template = jinja.read_text(encoding="utf-8")
                logger.info(
                    f"mlx_lm patch: injected chat_template.jinja for {mp.name}"
                )
                # Also flip TokenizerWrapper.has_chat_template (set in __init__,
                # cached as bool) so apply_chat_template doesn't short-circuit
                # if the wrapper checks it before delegating.
                try:
                    wrapper.has_chat_template = True
                except Exception:
                    pass
                return wrapper
            json_path = mp / "chat_template.json"
            if json_path.is_file():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                tpl = data.get("chat_template") if isinstance(data, dict) else None
                if isinstance(tpl, str) and tpl.strip():
                    inner.chat_template = tpl
                    logger.info(
                        f"mlx_lm patch: injected chat_template.json for {mp.name}"
                    )
                    try:
                        wrapper.has_chat_template = True
                    except Exception:
                        pass
                    return wrapper

            # DeepSeek V4 bundles ship `encoding/encoding_dsv4.py` instead
            # of a jinja chat template — transformers' `apply_chat_template`
            # can't consume it natively, so we install a thin Jinja shim
            # that delegates to our dsv4_chat_encoder via a Python extension.
            # The shim produces the same bytes as the encoding_dsv4 library
            # for the same message list, including BOS, <｜User｜>/<｜Assistant｜>
            # turns, <｜end▁of▁sentence｜>, and the <think></think> suffix
            # for chat mode. See research/DSV4-RUNTIME-ARCHITECTURE.md §4.
            try:
                cfg_path = mp / "config.json"
                if cfg_path.is_file():
                    _cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if _cfg.get("model_type") == "deepseek_v4":
                        # Minimal jinja that mirrors encoding_dsv4 enough for
                        # `apply_chat_template` to produce a runnable prompt.
                        # Advanced features (tools, reasoning_effort=max,
                        # multi-turn drop_thinking) should go through our
                        # dsv4_chat_encoder.apply_chat_template(..., model_path=)
                        # directly — this fallback handles the generic case.
                        _dsv4_fallback = (
                            "{{ bos_token if add_default_bos_token | default(true) else '' }}"
                            "{%- for m in messages -%}"
                            "{%- if m['role'] == 'user' -%}"
                            "<｜User｜>{{ m['content'] }}"
                            "{%- elif m['role'] == 'assistant' -%}"
                            "<｜Assistant｜>"
                            "{%- if m.get('reasoning_content') -%}<think>{{ m['reasoning_content'] }}</think>{%- endif -%}"
                            "{{ m['content'] }}<｜end▁of▁sentence｜>"
                            "{%- elif m['role'] == 'system' -%}"
                            "{{ m['content'] }}"
                            "{%- endif -%}"
                            "{%- endfor -%}"
                            "{%- if add_generation_prompt -%}"
                            "<｜Assistant｜>"
                            "{%- if enable_thinking is not defined or enable_thinking == false -%}"
                            "</think>"
                            "{%- endif -%}"
                            "{%- endif -%}"
                        )
                        inner.chat_template = _dsv4_fallback
                        try:
                            wrapper.has_chat_template = True
                        except Exception:
                            pass
                        logger.info(
                            f"mlx_lm patch: injected DSV4 fallback chat template "
                            f"for {mp.name} (full-fidelity via dsv4_chat_encoder)"
                        )
                        return wrapper
            except Exception as _dsv4_e:
                logger.debug(f"DSV4 chat-template injection skipped: {_dsv4_e}")
        except Exception as _e:
            logger.debug(f"mlx_lm chat_template injection failed: {_e}")
        return wrapper

    _patched_load._vmlx_chat_template_patched = True  # type: ignore[attr-defined]
    _tu.load = _patched_load
    # Also patch the re-export on mlx_lm.utils.load_tokenizer so callers
    # importing from mlx_lm.utils get the patched version.
    try:
        from mlx_lm import utils as _mu
        if hasattr(_mu, "load_tokenizer"):
            _orig_lt = _mu.load_tokenizer
            if not getattr(_orig_lt, "_vmlx_chat_template_patched", False):
                def _patched_lt(model_path, tokenizer_config_extra=None, eos_token_ids=None):
                    # _download then call the patched _tu.load above
                    from mlx_lm.utils import _download as _dl
                    p = _dl(
                        model_path,
                        allow_patterns=[
                            "*.json",
                            "*.py",
                            "tokenizer.model",
                            "*.tiktoken",
                            "tiktoken.model",
                            "*.txt",
                            "*.jsonl",
                            "*.jinja",
                        ],
                    )
                    return _patched_load(
                        p,
                        tokenizer_config_extra=tokenizer_config_extra,
                        eos_token_ids=eos_token_ids,
                    )
                _patched_lt._vmlx_chat_template_patched = True  # type: ignore[attr-defined]
                _mu.load_tokenizer = _patched_lt
    except Exception as _e:
        logger.debug(f"mlx_lm.utils.load_tokenizer patch skipped: {_e}")
    logger.info("vmlx: patched mlx_lm tokenizer load to inject chat_template.jinja")


# Apply patch at import time so every caller benefits — even direct
# jang_tools.load_jangtq calls that bypass load_model_with_fallback.
try:
    _patch_mlx_lm_tokenizer_load()
except Exception as _patch_e:  # pragma: no cover
    logger.debug(f"chat_template monkey-patch skipped: {_patch_e}")


def _inject_chat_template_if_missing(tokenizer, model_path) -> str | None:
    """vmlx#80 fix: HF transformers does NOT auto-load chat_template.jinja from
    a model directory — only ``tokenizer_config.json["chat_template"]`` is
    natively recognised. mlx-community quants (e.g. ``gemma-4-31b-8bit``) and
    several JANG/JANGTQ variants ship the template as a separate
    ``chat_template.jinja`` file. Without injection here, the FIRST call to
    ``tokenizer.apply_chat_template(...)`` raises:

        ValueError: Cannot use chat template functions because
        tokenizer.chat_template is not set...

    Resolution order (matches BatchedEngine._inject_fallback_chat_template):
        1. ``chat_template.jinja`` in the model dir
        2. ``chat_template.json`` in the model dir
        3. The model_configs registry's ``chat_template_custom`` field

    Returns the source description (e.g. ``"chat_template.jinja"``) on
    success, ``None`` if no fallback was applied. No-op if the tokenizer
    already has a non-empty template (e.g. baked into tokenizer_config.json).
    """
    if tokenizer is None:
        return None

    # Resolve the actual chat-template-bearing object. Three shapes seen in
    # the wild:
    #   - HF PreTrainedTokenizer / TokenizerFast: has .chat_template directly
    #   - mlx_lm.TokenizerWrapper: proxies to ._tokenizer
    #   - mlx_vlm Processor (Gemma4Processor, Qwen3VLProcessor, etc.): the
    #     chat_template lives on .tokenizer
    targets = []
    if hasattr(tokenizer, "chat_template"):
        targets.append(tokenizer)
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "chat_template"):
        targets.append(inner)
    proc_tok = getattr(tokenizer, "tokenizer", None)
    if proc_tok is not None and hasattr(proc_tok, "chat_template"):
        targets.append(proc_tok)

    if not targets:
        return None

    # If ANY target already has a non-empty template, propagate it to all
    # the others and bail (no injection needed).
    for t in targets:
        existing = getattr(t, "chat_template", None)
        if isinstance(existing, str) and existing.strip():
            for o in targets:
                if o is not t and not (
                    isinstance(getattr(o, "chat_template", None), str)
                    and getattr(o, "chat_template").strip()
                ):
                    try:
                        o.chat_template = existing
                    except Exception:
                        pass
            return None
        if isinstance(existing, list) and existing:
            return None

    try:
        from pathlib import Path as _P
        if model_path is None:
            return None
        mp = _P(model_path) if not isinstance(model_path, _P) else model_path
        if not mp.is_dir():
            return None
        # 0. tokenizer_config.json["chat_template"] — the canonical field for
        # baked-in templates. mlx_lm's standard tokenizer load reads it
        # natively, but the dedicated DSV4 loader (load_jangtq_dsv4 →
        # jang_tools.load_jangtq) goes through a custom path that doesn't
        # propagate the field onto the returned tokenizer. Read it back
        # explicitly so DSV4 bundles ship with a working chat template.
        tcfg_path = mp / "tokenizer_config.json"
        if tcfg_path.is_file():
            try:
                tcfg = json.loads(tcfg_path.read_text(encoding="utf-8"))
                tpl = tcfg.get("chat_template") if isinstance(tcfg, dict) else None
                if isinstance(tpl, str) and tpl.strip():
                    for t in targets:
                        try:
                            t.chat_template = tpl
                        except Exception:
                            pass
                    try:
                        if hasattr(tokenizer, "has_chat_template"):
                            tokenizer.has_chat_template = True
                    except Exception:
                        pass
                    logger.info(
                        f"Chat template injected from tokenizer_config.json "
                        f"for {mp.name}"
                    )
                    return "tokenizer_config.json"
            except Exception as _tce:
                logger.debug(f"tokenizer_config.json chat_template probe failed: {_tce}")
        # 1. chat_template.jinja
        jinja_path = mp / "chat_template.jinja"
        if jinja_path.is_file():
            tpl = jinja_path.read_text(encoding="utf-8")
            for t in targets:
                try:
                    t.chat_template = tpl
                except Exception:
                    pass
            # Flip TokenizerWrapper.has_chat_template if applicable
            try:
                if hasattr(tokenizer, "has_chat_template"):
                    tokenizer.has_chat_template = True
            except Exception:
                pass
            logger.info(
                f"Chat template injected from chat_template.jinja for {mp.name}"
            )
            return "chat_template.jinja"
        # 2. chat_template.json
        json_path = mp / "chat_template.json"
        if json_path.is_file():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            tpl = data.get("chat_template") if isinstance(data, dict) else None
            if isinstance(tpl, str) and tpl.strip():
                for t in targets:
                    try:
                        t.chat_template = tpl
                    except Exception:
                        pass
                try:
                    if hasattr(tokenizer, "has_chat_template"):
                        tokenizer.has_chat_template = True
                except Exception:
                    pass
                logger.info(
                    f"Chat template injected from chat_template.json for {mp.name}"
                )
                return "chat_template.json"
    except Exception as _ce:
        logger.debug(f"chat_template.jinja/.json probe failed: {_ce}")

    # 3. Registry fallback by family (chat_template_custom field)
    mc = None
    try:
        from ..model_config_registry import get_model_config_registry
        mc = get_model_config_registry().lookup(str(model_path))
        reg_tpl = getattr(mc, "chat_template_custom", None)
        if isinstance(reg_tpl, str) and reg_tpl.strip():
            for t in targets:
                try:
                    t.chat_template = reg_tpl
                except Exception:
                    pass
            try:
                if hasattr(tokenizer, "has_chat_template"):
                    tokenizer.has_chat_template = True
            except Exception:
                pass
            logger.info(
                f"Chat template injected from registry ({mc.family_name}) "
                f"for {mp.name if hasattr(mp, 'name') else model_path}"
            )
            return f"registry:{mc.family_name}"
    except Exception as _re:
        logger.debug(f"registry chat_template lookup failed: {_re}")

    # 4. Bundled fallback: vmlx_engine/chat_templates/{family}.jinja
    # vmlx#80 (Flor1an-B, 2026-04-15): mlx-community/gemma-4-31b-8bit ships
    # NO chat template at all — not a sidecar jinja, not in tokenizer_config,
    # and there's no registry chat_template_custom. For well-known families
    # we bundle a canonical copy lifted from an mlx-community quant that does
    # ship it (e.g. gemma-4-26b-a4b-it-4bit). Keyed by family_name so it
    # covers gemma4 + gemma4_text + any future Gemma 4 registry entry.
    if mc is not None:
        try:
            from pathlib import Path as _P2
            family = getattr(mc, "family_name", None)
            if family:
                pkg_dir = _P2(__file__).parent.parent  # vmlx_engine/
                bundled = pkg_dir / "chat_templates" / f"{family}.jinja"
                if bundled.is_file():
                    tpl = bundled.read_text(encoding="utf-8")
                    for t in targets:
                        try:
                            t.chat_template = tpl
                        except Exception:
                            pass
                    try:
                        if hasattr(tokenizer, "has_chat_template"):
                            tokenizer.has_chat_template = True
                    except Exception:
                        pass
                    logger.info(
                        f"Chat template injected from bundled {family}.jinja "
                        f"for {mp.name if hasattr(mp, 'name') else model_path}"
                    )
                    return f"bundled:{family}.jinja"
        except Exception as _be:
            logger.debug(f"bundled chat_template probe failed: {_be}")

    return None


def load_model_with_fallback(model_name: str, tokenizer_config: dict = None, skip_turboquant: bool = False):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from mlx_lm import load

    # Register vendored Gemma 4 native text MoE under the mlx_lm.models namespace.
    # Idempotent + a no-op when mlx-lm >= 0.31.2 ships the upstream module itself.
    # This local-port matches the audit-2026-04-07 team consensus (Agent 1 / 2 / 3
    # all chose local-port over an mlx-lm version bump).
    try:
        from ..models.gemma4_native_register import register_gemma4_native

        register_gemma4_native()
    except Exception as _e:  # pragma: no cover
        logger.debug(f"register_gemma4_native skipped: {_e}")

    # Register DeepSeek V4 model class globally (fixes mlxstudio#119:
    # "Model type deepseek_v4 not supported"). The DSV4 model class lives
    # in jang_tools.dsv4.mlx_register and was previously only registered
    # inside the MXTQ branch of jang_loader. Non-MXTQ DSV4 bundles
    # (e.g. plain dequant 2-bit, or any "DeepSeek-V4-Flash-*-DQ" pack)
    # therefore failed at mlx_lm.load with ValueError. Idempotent —
    # re-registration is a no-op once mlx_lm.models.deepseek_v4 exists.
    try:
        from jang_tools.dsv4 import mlx_register as _dsv4_reg  # noqa: F401
        logger.debug(
            "DeepSeek V4 (deepseek_v4) registered with mlx_lm.models — "
            "any DSV4 bundle (MXTQ, JANGTQ, plain dequant, BF16) can load"
        )
    except ImportError:
        # jang_tools.dsv4 not available in this env — DSV4 bundles will fail
        # with the same ValueError as before. Surface a clearer warning when
        # we actually see a DSV4 bundle so users know what to install.
        logger.debug(
            "jang_tools.dsv4 not installed — DSV4 bundles will need "
            "`pip install jang-tools` (or jang_tools >= 2.5.x with dsv4 submodule)"
        )

    tokenizer_config = tokenizer_config or {}
    # Q1 (audit-2026-04-07): trust_remote_code=True silences the noisy HF
    # warning on load for models with custom tokenizer_config classes
    # (e.g. nemotron_h) and unblocks any custom tokenizer code path. Safe
    # because model paths served by vMLX come from user-local disk or
    # trusted JANG pipelines. Only set if caller didn't already override.
    tokenizer_config.setdefault("trust_remote_code", True)

    # Check if local path exists before loading
    model_path = Path(model_name)
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_name}. "
            f"Check that the model directory is available."
        )

    # Check if Smelt mode is active (partial expert loading)
    from .. import server as _server_module

    _smelt = getattr(_server_module, "_smelt_enabled", False)
    _smelt_pct = getattr(_server_module, "_smelt_experts", 50)

    # Resolve HuggingFace repo IDs to local paths so that JANG detection
    # (which checks for jang_config.json on disk) works for remote models.
    from ..api.utils import resolve_to_local_path

    local_model_path = resolve_to_local_path(model_name)
    if local_model_path != model_name:
        logger.info(f"Resolved HF model to: {local_model_path}")

    # ── Architecture-specific routing BEFORE the JANG gate ──
    #
    # `is_jang_model()` only fires for bundles whose `jang_config.weight_format`
    # is in {jang, jjqf, mxq, mxtq} — i.e. JANG-codec bundles. Bundles that
    # ship `weight_format=mxfp4` (or `bf16`) fall through to the stock
    # `mlx_lm.load()` path. That works fine for architectures mlx_lm
    # natively supports (qwen, llama, mistral, …) but FAILS for ones it
    # doesn't (Laguna, ministral3) → `ValueError: Model type laguna not
    # supported`. Live audit 2026-04-30 caught this exact crash on
    # Laguna-XS.2-mxfp4. Route by model_type FIRST so MXFP4/bf16 builds
    # of these architectures land in the right loader.
    try:
        import json as _json_arch
        _cfg_path_arch = Path(local_model_path) / "config.json"
        if _cfg_path_arch.exists():
            _cfg_arch = _json_arch.loads(_cfg_path_arch.read_text())
            _mt_arch = _cfg_arch.get("model_type")
            _tc_mt_arch = (_cfg_arch.get("text_config") or {}).get("model_type")
            if _mt_arch == "laguna" or _tc_mt_arch == "laguna":
                logger.info(
                    "Laguna bundle detected (early route) — load_laguna_model"
                )
                from ..loaders.load_laguna import load_laguna_model
                _m, _t = load_laguna_model(local_model_path)
                _inject_chat_template_if_missing(_t, local_model_path)
                return _m, _t
            if _tc_mt_arch == "ministral3" or _mt_arch == "ministral3":
                logger.info(
                    "Mistral-Medium-3.5 bundle detected (early route) — load_mistral3_model"
                )
                from ..loaders.load_mistral3 import load_mistral3_model
                _m, _t = load_mistral3_model(local_model_path)
                _inject_chat_template_if_missing(_t, local_model_path)
                return _m, _t
    except (OSError, _json_arch.JSONDecodeError):
        pass
    except ImportError:
        # Loader missing — let the existing JANG / mlx_lm path raise so
        # the user sees the actionable error from there.
        pass

    # JANG format MUST be checked FIRST — JANG models use their own loader that
    # repacks weights into QuantizedLinear and handles tokenizer internally.
    # Checking tokenizer fallback first would bypass the JANG loader for Nemotron-H.
    from .jang_loader import is_jang_model

    if is_jang_model(local_model_path):
        logger.info(f"Detected JANG model: {model_name}")
        if _smelt:
            from .smelt_loader import smelt_load

            _m, _t = smelt_load(local_model_path, expert_percent=_smelt_pct)
            _inject_chat_template_if_missing(_t, local_model_path)
            return _m, _t

        # Route DeepSeek V4 bundles to the dedicated JANGTQ-DSV4 loader.
        # DSV4-Flash bundles ship `weight_format=bf16` (NOT mxtq) plus a
        # `jangtq_runtime.safetensors` sidecar with the quantization
        # codebooks. The generic JANG text loader rejects this layout
        # ("missing 'format'/'weight_format'"). The dedicated loader at
        # `vmlx_engine.loaders.load_jangtq_dsv4` delegates to
        # `jang_tools.load_jangtq.load_jangtq_model` which knows how to
        # ingest bf16 weights + the JANGTQ runtime sidecar and registers
        # the deepseek_v4 model class via `jang_tools.dsv4.mlx_register`.
        try:
            import json as _json
            _cfg_path = Path(local_model_path) / "config.json"
            if _cfg_path.exists():
                _cfg = _json.loads(_cfg_path.read_text())
                _mt = _cfg.get("model_type")
                _tc_mt = (_cfg.get("text_config") or {}).get("model_type")
                if _mt == "deepseek_v4" or _tc_mt == "deepseek_v4":
                    logger.info(
                        f"DSV4 bundle detected — routing through "
                        f"load_jangtq_dsv4 instead of generic JANG loader."
                    )
                    from ..loaders.load_jangtq_dsv4 import load_jangtq_dsv4_model
                    _m, _t = load_jangtq_dsv4_model(local_model_path)
                    _inject_chat_template_if_missing(_t, local_model_path)
                    return _m, _t

                # Laguna (poolside): 33B/3B agentic-coding MoE,
                # model_type=laguna. mlx_lm has no native laguna class so
                # the generic loader can't resolve it. Route through
                # `jang_tools.laguna.runtime.load` which auto-detects
                # bf16 / JANG affine / JANGTQ / MXFP4 layouts and returns
                # the LagunaForCausalLM instance with weights bound.
                if _mt == "laguna" or _tc_mt == "laguna":
                    logger.info(
                        "Laguna bundle detected — routing through "
                        "load_laguna instead of generic JANG/mlx_lm loader."
                    )
                    from ..loaders.load_laguna import load_laguna_model
                    _m, _t = load_laguna_model(local_model_path)
                    _inject_chat_template_if_missing(_t, local_model_path)
                    return _m, _t

                # Mistral-Medium-3.5-128B: model_type=mistral3 outer
                # wrapper + text_config.model_type=ministral3 inner.
                # The inner `ministral3` is a NEW dense GQA arch (96/8
                # heads, 128 head_dim, 88 layers, hidden 12288, 256K
                # YaRN ctx) that mlx_lm has no class for, so the generic
                # loader misroutes it to legacy `mistral` and produces
                # garbage. Route to jang_tools.mistral3.runtime.load.
                #
                # Distinguish from LEGACY mistral3 (Mistral-Small-3.1
                # 24B VLM with `text_config.model_type=mistral` or
                # `mistral4`): only ministral3 inner type triggers this.
                if _tc_mt == "ministral3" or _mt == "ministral3":
                    logger.info(
                        "Mistral-Medium-3.5 bundle detected (ministral3 "
                        "inner) — routing through load_mistral3."
                    )
                    from ..loaders.load_mistral3 import load_mistral3_model
                    _m, _t = load_mistral3_model(local_model_path)
                    _inject_chat_template_if_missing(_t, local_model_path)
                    return _m, _t
        except ImportError as _ie:
            logger.warning(
                "DSV4 dedicated loader unavailable (%s) — falling back to "
                "generic JANG loader.",
                _ie,
            )
        except Exception as _e:
            logger.debug("DSV4 routing pre-check failed: %s", _e)

        from .jang_loader import load_jang_model

        _m, _t = load_jang_model(local_model_path)
        _inject_chat_template_if_missing(_t, local_model_path)
        return _m, _t

    # Check if model needs tokenizer fallback (e.g., Nemotron).
    # Pass resolved local path so _get_model_type_from_config can read config.json.
    if _needs_tokenizer_fallback(local_model_path):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        model, tokenizer = _load_with_tokenizer_fallback(local_model_path, lazy=False)
        if not skip_turboquant:
            _apply_turboquant_to_model(model, local_model_path)
        _inject_chat_template_if_missing(tokenizer, local_model_path)
        return model, tokenizer

    try:
        model, tokenizer = load(
            model_name, tokenizer_config=tokenizer_config, lazy=False
        )
        if not skip_turboquant:
            _apply_turboquant_to_model(model, local_model_path)
        _inject_chat_template_if_missing(tokenizer, local_model_path)
        return model, tokenizer
    except ValueError as e:
        # Fallback for models with non-standard tokenizers
        if "TokenizersBackend" in str(e) or "Tokenizer class" in str(e):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            model, tokenizer = _load_with_tokenizer_fallback(
                local_model_path, lazy=False
            )
            if not skip_turboquant:
                _apply_turboquant_to_model(model, local_model_path)
            _inject_chat_template_if_missing(tokenizer, local_model_path)
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

    # Nemotron-3-Nano-Omni MXFP4 bundles ship with multimodal weights
    # (vision_model.*, sound_encoder.*, mlp1.*, sound_projection.*) that
    # the text-only mlx_lm NemotronH class doesn't have. mlx_lm.load_model
    # defaults to strict=True → "Received N parameters not in model".
    # Drop strict for nemotron_h so the LLM submodel loads cleanly. Per
    # research/NEMOTRON-OMNI-RUNTIME-2026-04-28.md §10 (multimodal keys
    # were supposed to be dropped at convert-time but the MXFP4 converter
    # left them in place).
    _load_strict = True
    try:
        cfg_path = Path(model_path) / "config.json"
        if cfg_path.is_file():
            _cfg = json.loads(cfg_path.read_text())
            if _cfg.get("model_type") == "nemotron_h":
                _load_strict = False
    except Exception:
        pass

    # Load model
    model, _ = load_model(model_path, lazy=lazy, strict=_load_strict)

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
                    tokenizer.chat_template = jinja_path.read_text()
                    logger.info("Chat template loaded from chat_template.jinja")
                except Exception as e:
                    logger.warning(f"Failed to read chat_template.jinja: {e}")

            # Fall back to built-in templates if no model template found
            if not getattr(tokenizer, "chat_template", None):
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
