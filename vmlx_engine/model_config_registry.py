# SPDX-License-Identifier: Apache-2.0
"""
Model configuration registry for vmlx-engine.

Centralizes all model-specific configuration: cache types, EOS tokens,
chat templates, tool parsers, and architecture hints. Replaces scattered
if/elif checks throughout the codebase with a single source of truth.

Usage:
    from vmlx_engine.model_config_registry import get_model_config_registry

    registry = get_model_config_registry()
    config = registry.lookup("mlx-community/Qwen3-8B-Instruct-4bit")
    print(config.eos_tokens)  # ["<|im_end|>"]
    print(config.tool_parser)  # "qwen"
"""

import logging
import re
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Late-bound reference to mlx_lm.utils.load_config (imported on first use).
# Exposed at module level so tests can mock it with unittest.mock.patch.
load_config = None


@dataclass
class ModelConfig:
    """Configuration profile for a model family."""

    # Identity
    family_name: str
    model_types: List[str]  # e.g. ["llama", "qwen2", "mistral"]

    # Cache configuration
    cache_type: str = "kv"  # "kv" | "mamba" | "hybrid" | "rotating_kv"
    cache_subtype: Optional[str] = None  # e.g. "zaya_cca", "deepseek_v4_composite"

    # Tokenizer overrides
    eos_tokens: Optional[List[str]] = None
    special_tokens_to_clean: Optional[List[str]] = None
    tokenizer_fallback: bool = False

    # Chat template
    chat_template_custom: Optional[str] = None
    preserve_native_tool_format: bool = False

    # Tool calling
    tool_parser: Optional[str] = None
    supports_native_tools: bool = False

    # Reasoning
    reasoning_parser: Optional[str] = None
    think_in_template: bool = False  # True if chat template injects <think> in assistant prefix
    # None means derive support from reasoning_parser/think_in_template.
    # False is an explicit family/runtime compatibility verdict: do not let
    # stale JANG stamps or tokenizer probes auto-enable thinking.
    supports_thinking: Optional[bool] = None

    # Multimodal
    is_mllm: bool = False

    # Architecture-specific hints
    architecture_hints: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: Optional[str] = None
    priority: int = 100  # Lower = higher priority (matched first)


# Default config for unknown models
_DEFAULT_CONFIG = ModelConfig(
    family_name="unknown",
    model_types=[],
    cache_type="kv",
    description="Default configuration for unknown models",
    priority=999,
)


def _config_declares_linear_attention(config: Any) -> bool:
    """Return True when config metadata declares hybrid linear-attention layers."""
    if not isinstance(config, dict):
        return False
    containers = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        containers.append(text_config)
    for container in containers:
        for key in ("layer_types", "layer_type"):
            value = container.get(key)
            if isinstance(value, str):
                if value.lower() == "linear_attention":
                    return True
            elif isinstance(value, list):
                if any(str(item).lower() == "linear_attention" for item in value):
                    return True
    return False


def _config_declares_media(config: Any) -> bool:
    """Return True when config.json declares real image/audio/video support."""
    if not isinstance(config, dict):
        return False
    for key in ("vision_config", "audio_config", "video_config"):
        if key in config and config.get(key) is not None:
            return True
    for key in (
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "video_token_index",
        "audio_token_id",
        "audio_token_index",
    ):
        if key in config and config.get(key) is not None:
            return True
    return False


def _is_mxtq_jang_config(jang_config: Any) -> bool:
    if not isinstance(jang_config, dict):
        return False
    quant = jang_config.get("quantization") or {}
    values = (
        jang_config.get("weight_format"),
        jang_config.get("format"),
        quant.get("weight_format") if isinstance(quant, dict) else None,
        quant.get("format") if isinstance(quant, dict) else None,
        quant.get("method") if isinstance(quant, dict) else None,
        quant.get("profile") if isinstance(quant, dict) else None,
    )
    lowered = [str(value or "").lower() for value in values]
    if any("mxtq" in value or "jangtq" in value for value in lowered):
        return True
    return (
        "mxtq_bits" in jang_config
        or (isinstance(quant, dict) and "mxtq_bits" in quant)
    )


def _is_explicit_affine_jang_config(jang_config: Any) -> bool:
    if not isinstance(jang_config, dict):
        return False
    quant = jang_config.get("quantization") or {}
    values = (
        jang_config.get("weight_format"),
        jang_config.get("format"),
        quant.get("weight_format") if isinstance(quant, dict) else None,
        quant.get("format") if isinstance(quant, dict) else None,
        quant.get("method") if isinstance(quant, dict) else None,
        quant.get("profile") if isinstance(quant, dict) else None,
    )
    lowered = [str(value or "").lower() for value in values]
    return any(
        value in {"jang", "jang_v2", "affine", "jang-importance"}
        or value.startswith("jang_")
        for value in lowered
    )


def _is_affine_jang_qwen_hybrid_vlm(
    model_config: dict[str, Any],
    jang_config: Any,
) -> bool:
    if not isinstance(model_config, dict) or not isinstance(jang_config, dict):
        return False
    model_types = {
        str(model_config.get("model_type") or "").lower(),
        str((model_config.get("text_config") or {}).get("model_type") or "").lower(),
    }
    if not model_types.intersection(
        {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}
    ):
        return False
    if not _config_declares_media(model_config):
        return False
    if not _is_explicit_affine_jang_config(jang_config):
        return False
    return not _is_mxtq_jang_config(jang_config)


def _with_linear_attention_cache_override(
    config: ModelConfig,
    model_config: dict[str, Any],
) -> ModelConfig:
    if (
        config.family_name in {"qwen3_5", "qwen3_5_moe"}
        and config.cache_type == "kv"
        and _config_declares_linear_attention(model_config)
    ):
        return replace(config, cache_type="hybrid")
    return config


def _with_config_metadata_overrides(
    config: ModelConfig,
    model_config: dict[str, Any],
) -> ModelConfig:
    config = _with_linear_attention_cache_override(config, model_config)
    if config.family_name in {"qwen3_5", "qwen3_5_moe"} and _config_declares_media(
        model_config
    ):
        return replace(config, is_mllm=True)
    return config


def _looks_like_image_model_dir(model_path: str) -> bool:
    """Return True for diffusers/mflux image dirs that are not text configs."""
    try:
        from pathlib import Path
        path = Path(model_path)
        if not path.is_dir():
            return False
        if (path / "model_index.json").is_file():
            return True
        if (path / "transformer").is_dir():
            text_encoder = (
                (path / "text_encoder").is_dir()
                or (path / "text_encoder_2").is_dir()
            )
            if text_encoder or (path / "vae").is_dir():
                return True
    except Exception:
        return False
    return False


class ModelConfigRegistry:
    """
    Singleton registry mapping model families to configurations.

    Uses regex-based pattern matching with priority ordering.
    Results are cached for performance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._configs: List[ModelConfig] = []
        self._match_cache: Dict[str, ModelConfig] = {}
        self._rlock = threading.RLock()
        self._initialized = True

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        with self._rlock:
            self._configs.append(config)
            # Sort by priority (lower = higher priority)
            self._configs.sort(key=lambda c: c.priority)
            # Invalidate cache
            self._match_cache.clear()

    def _try_jang_stamp(self, model_name: str) -> Optional["ModelConfig"]:
        """Tier-1: read jang_config.json `capabilities` stamp if present.

        All JANG / JANGTQ artifacts produced 2026-04-16+ carry an authoritative
        capabilities block in jang_config.json of the form:

            "capabilities": {
              "reasoning_parser": "qwen3 | deepseek_r1 | mistral | gemma4",
              "tool_parser":      "qwen | minimax | deepseek | nemotron | mistral | gemma4 | glm47",
              "think_in_template": true | false,
              "supports_tools":    true,
              "supports_thinking": true,
              "family":            "qwen3_5_moe | minimax_m2 | glm5 | nemotron_h | mistral4 | gemma4",
              "modality":          "text | vision",
              "cache_type":        "kv | hybrid | mla",
              "cache_subtype":     "optional architecture-specific cache contract"
            }

        When present, this is authoritative — return a ModelConfig built from
        the stamp without consulting the family registry. That matches what
        the converter actually produced, including any custom overrides (e.g.
        Qwen3.6 VLM wrapper stamped "family": "qwen3_5_moe" from the outside
        even though inner text_config.model_type = "qwen3_5_moe_text").

        Returns None if there's no jang_config.json or no capabilities block.
        """
        try:
            from pathlib import Path
            import json
            jcfg_path = Path(model_name) / "jang_config.json"
            if not jcfg_path.is_file():
                return None
            try:
                jcfg = json.loads(jcfg_path.read_text())
            except Exception:
                return None
            local_model_config: dict[str, Any] = {}
            try:
                cfg_path = Path(model_name) / "config.json"
                if cfg_path.is_file():
                    loaded_cfg = json.loads(cfg_path.read_text())
                    if isinstance(loaded_cfg, dict):
                        local_model_config = loaded_cfg
            except Exception:
                local_model_config = {}
            caps = jcfg.get("capabilities")
            if not isinstance(caps, dict):
                # Fallback: some bundles (notably DSV4-Flash) use the
                # `chat` schema from research/DSV4-RUNTIME-ARCHITECTURE.md §4
                # instead of a top-level `capabilities` block. Synthesize a
                # caps-shaped dict from the chat block so the rest of the
                # stamp pipeline works generically:
                #
                #   jang_config.chat.reasoning.parser       → reasoning_parser
                #   jang_config.chat.tool_calling.parser    → tool_parser
                #   jang_config.chat.reasoning.supported    → think_in_template
                #   jang_config.chat.reasoning.modes        → (informational)
                #   jang_config.model_family                → family
                chat_cfg = jcfg.get("chat")
                if isinstance(chat_cfg, dict):
                    _chat_r = chat_cfg.get("reasoning") or {}
                    _chat_t = chat_cfg.get("tool_calling") or {}
                    caps = {
                        "family": jcfg.get("model_family") or "",
                        "reasoning_parser": _chat_r.get("parser") if isinstance(_chat_r, dict) else None,
                        "tool_parser": _chat_t.get("parser") if isinstance(_chat_t, dict) else None,
                        "think_in_template": bool(
                            isinstance(_chat_r, dict) and _chat_r.get("supported", False)
                        ),
                        "cache_type": jcfg.get("cache_type", "kv"),
                    }
                else:
                    return None
            family = caps.get("family") or ""
            if not family:
                return None

            # Start from an existing family config if the stamp's family name
            # matches one we know OR if the stamped family is actually a
            # model_type alias registered under a canonical family. Ling
            # bundles stamp "family": "bailing_hybrid" because that is the
            # HF model_type, but the registry family is "ling" and carries
            # the required dual-EOS list + sampling defaults. Treating the
            # stamp as a brand-new family silently drops those fields.
            base = None
            for c in self._configs:
                if c.family_name == family:
                    base = c
                    break
            if base is None:
                family_l = str(family).lower()
                for c in self._configs:
                    if family_l in {str(mt).lower() for mt in c.model_types}:
                        base = c
                        break
            if base is None:
                # Fall back to a fresh ModelConfig only when the stamp names a
                # genuinely unknown family/model_type.
                # Build from scratch — stamp is authoritative enough that we
                # trust it even when the family isn't in the registry table.
                base = ModelConfig(
                    family_name=family,
                    model_types=[family],
                    cache_type=caps.get("cache_type", "kv") or "kv",
                    priority=0,
                )

            # Override with stamped values. For most families, the converter
            # stamp wins because it describes the emitted artifact. Keep
            # Keep family reasoning policy separate from quantization bit
            # metadata. Bit width is used for kernel/cache selection and
            # observability, not for hidden runtime guards. ZAYA/ZAYA1-VL are
            # reasoning-capable qwen3-parser families, but their template does
            # not start in an open think rail, so stale converter stamps must
            # not resurrect think_in_template=True. Ling/Bailing is the one
            # explicit no-reasoning contract here: stale JANG sidecars
            # advertise an experimental system-message switch, but the app/API
            # should not expose it as a reasoning-capable model.
            from dataclasses import replace
            updates: Dict[str, Any] = {}
            rp = caps.get("reasoning_parser")
            tp = caps.get("tool_parser")
            tin = caps.get("think_in_template")
            sth = caps.get("supports_thinking")
            ct = caps.get("cache_type")
            cst = caps.get("cache_subtype") or jcfg.get("cache_subtype")
            mod = caps.get("modality")
            model_type_from_config = str(local_model_config.get("model_type") or "").lower()
            has_config_vision = isinstance(local_model_config.get("vision_config"), dict)
            has_config_media = _config_declares_media(local_model_config)
            base_supports_thinking = getattr(base, "supports_thinking", None)
            is_ling_family = base.family_name == "ling"
            is_zaya_family = base.family_name in {"zaya", "zaya1_vl"}
            is_hy3_family = base.family_name == "hy_v3"
            preserve_template_metadata_when_no_thinking = False

            if is_zaya_family:
                # ZAYA and ZAYA1-VL are reasoning-capable, but the honest
                # prompt contract is still not "starts inside <think>".
                # enable_thinking=False renders a closed empty block. Do not
                # let stale converter stamps resurrect think_in_template=True.
                updates["supports_thinking"] = True
                updates["reasoning_parser"] = "qwen3"
                updates["think_in_template"] = False
                zaya_hints = dict(getattr(base, "architecture_hints", None) or {})
                zaya_hints["default_enable_thinking"] = False
                updates["architecture_hints"] = zaya_hints
            elif is_ling_family:
                # Ling/Bailing emits plain content. Do not let drifted bundle
                # stamps resurrect a reasoning parser or thinking-capable
                # advertisement for this family.
                updates["supports_thinking"] = False
                updates["reasoning_parser"] = None
                updates["think_in_template"] = False
            elif is_hy3_family:
                # Hy3's template defaults to `reasoning_effort=no_think` and
                # emits a closed `<think></think>` prefill. It only opens
                # `<think>` when vMLX supplies `reasoning_effort=low|high`.
                # Older conversion stamps used think_in_template=True, which
                # makes the parser treat normal visible text as reasoning on
                # default/no-thinking turns. Keep the canonical runtime
                # contract regardless of stale bundle stamps.
                updates["supports_thinking"] = True
                updates["reasoning_parser"] = "qwen3"
                updates["think_in_template"] = False
            elif base_supports_thinking is False:
                updates["supports_thinking"] = False
            elif isinstance(sth, bool):
                updates["supports_thinking"] = sth
            if (
                rp is not None
                and not (is_zaya_family or is_ling_family or is_hy3_family)
                and base_supports_thinking is not False
            ):
                updates["reasoning_parser"] = rp if rp != "none" else None
            if tp is not None:
                updates["tool_parser"] = tp if tp != "none" else None
            if isinstance(tin, bool) and not (is_zaya_family or is_ling_family or is_hy3_family) and (
                base_supports_thinking is not False or preserve_template_metadata_when_no_thinking
            ):
                updates["think_in_template"] = tin
            if ct:
                updates["cache_type"] = ct
            if cst:
                updates["cache_subtype"] = str(cst)
            if _is_affine_jang_qwen_hybrid_vlm(local_model_config, jcfg):
                # Qwen3.6 affine-JANG carries real VL/video metadata, but the
                # current mlx_vlm qwen3_5 language path corrupts text logits
                # through the M-RoPE fallback. Keep registry in text-loader
                # mode until docs/AUDIT-QWEN-AFFINE-JANG-VLM.md is resolved.
                updates["is_mllm"] = False
            elif mod == "vision" or (mod == "omni" and has_config_media):
                # `omni` only becomes MLLM when config.json carries real
                # media metadata. Some Nemotron-H text extracts keep stale
                # Omni stamps or preprocessor_config.json sidecars but do not
                # have image/audio/video weights; routing those through MLLM
                # is a false positive.
                updates["is_mllm"] = True
            elif mod == "omni":
                updates["is_mllm"] = False
            elif mod == "text":
                if base.family_name == "zaya1_vl" and (
                    model_type_from_config == "zaya1_vl" or has_config_vision
                ):
                    updates["is_mllm"] = True
                else:
                    updates["is_mllm"] = False
            if updates:
                base = replace(base, **updates)
            return base
        except Exception as e:
            logger.debug(f"_try_jang_stamp failed for {model_name}: {e}")
            return None

    def lookup(self, model_name: str) -> ModelConfig:
        """
        Look up configuration for a model name by reading its config.json.

        Resolution order:
          Tier 1 — jang_config.json `capabilities` block (authoritative stamp)
          Tier 2 — config.json text_config.model_type (VLM wrapper inner type)
          Tier 3 — config.json model_type + family registry match
          Default — _DEFAULT_CONFIG

        Returns default config if no match found.
        """
        with self._rlock:
            if model_name in self._match_cache:
                return self._match_cache[model_name]

            # vmlx#115: HF repo IDs (e.g. "Org/Model") arrive here unresolved on the
            # first lookup before the engine downloads/snapshots them. Map to the
            # local snapshot path so jang_config.json + config.json file probes can
            # actually find the files. Local paths fall through unchanged.
            try:
                from .api.utils import resolve_to_local_path
                resolved_path = resolve_to_local_path(model_name)
            except Exception:
                resolved_path = model_name

            # Tier 1 — JANG-stamped capabilities (authoritative, never second-guess)
            _stamped = self._try_jang_stamp(resolved_path)
            if _stamped is not None:
                logger.info(
                    f"Model config: detection_source=jang_stamped family={_stamped.family_name} "
                    f"reasoning_parser={_stamped.reasoning_parser} tool_parser={_stamped.tool_parser} "
                    f"think_in_template={_stamped.think_in_template} cache_type={_stamped.cache_type} "
                    f"cache_subtype={_stamped.cache_subtype} is_mllm={_stamped.is_mllm}"
                )
                self._match_cache[model_name] = _stamped
                return _stamped

            if _looks_like_image_model_dir(resolved_path):
                self._match_cache[model_name] = _DEFAULT_CONFIG
                return _DEFAULT_CONFIG

            model_type = None
            model_config: dict[str, Any] = {}
            has_media_config = False
            try:
                global load_config
                if load_config is None:
                    from mlx_lm.utils import load_config as _load_config_fn
                    load_config = _load_config_fn
                from pathlib import Path
                model_config = load_config(Path(resolved_path))
                model_type = model_config.get("model_type", "").lower()
                has_media_config = any(
                    bool(model_config.get(k))
                    for k in ("vision_config", "audio_config", "video_config")
                )
            except Exception as e:
                logger.warning(f"Could not load config.json for {model_name} to check model_type: {e}")

            # Also check text_config.model_type for VLM wrapper models
            # (e.g., Mistral 4 JANG: top-level model_type="mistral3", text_config.model_type="mistral4")
            text_model_type = None
            if model_type:
                try:
                    from pathlib import Path
                    import json
                    cfg_path = Path(resolved_path) / "config.json"
                    if cfg_path.exists():
                        raw = json.loads(cfg_path.read_text())
                        text_model_type = raw.get("text_config", {}).get("model_type", "").lower()
                        has_media_config = has_media_config or _config_declares_media(raw)
                except Exception:
                    pass

            if model_type:
                # Name-based disambiguation for models sharing model_type:
                # GLM-Z1 uses model_type "glm4" but needs openai_gptoss reasoning (Harmony protocol)
                if model_type == "glm4" and re.search(r"glm.?z1", model_name, re.IGNORECASE):
                    for config in self._configs:
                        if config.family_name == "glm_z1":
                            self._match_cache[model_name] = config
                            return config
                    logger.warning(f"GLM-Z1 model '{model_name}' detected but 'glm_z1' config not registered — falling back to generic glm4 config")

                # MedGemma uses gemma2 model_type but is multimodal
                if model_type == "gemma2" and re.search(r"medgemma", model_name, re.IGNORECASE):
                    for config in self._configs:
                        if config.family_name == "medgemma":
                            self._match_cache[model_name] = config
                            return config

                # Mistral Small 4 VLM is a `mistral3` Pixtral-style wrapper
                # around a `mistral4` MLA language model. Keep multimodal
                # routing from the wrapper, but inherit the inner Mistral 4
                # parser metadata so CLI/API/UI agree on reasoning/tool
                # defaults. Do this before the generic multimodal-wrapper
                # branch, which would otherwise return plain mistral3.
                if (
                    model_type == "mistral3"
                    and text_model_type == "mistral4"
                    and has_media_config
                ):
                    for config in self._configs:
                        if "mistral4" in config.model_types:
                            next_config = _with_config_metadata_overrides(
                                config,
                                model_config,
                            )
                            next_config = replace(next_config, is_mllm=True)
                            logger.info(
                                "Model config: matched Mistral 4 VLM wrapper "
                                "model_type='mistral3' text_config.model_type='mistral4' "
                                "→ mistral4 is_mllm=True"
                            )
                            self._match_cache[model_name] = next_config
                            return next_config

                # Some VLM wrappers use a text-only inner `text_config`
                # model_type (Gemma 4: outer gemma4 + inner gemma4_text).
                # If the top-level type is itself a registered multimodal
                # family and config.json has media config, keep the wrapper
                # family so image/audio/video routing and architecture hints
                # remain active.
                if has_media_config:
                    for config in self._configs:
                        if model_type in config.model_types and config.is_mllm:
                            logger.info(
                                f"Model config: matched multimodal wrapper model_type='{model_type}' "
                                f"with media config → {config.family_name}"
                            )
                            self._match_cache[model_name] = config
                            return config

                # VLM wrappers: prefer text_config.model_type (higher priority inner model)
                # e.g., mistral3 wrapper with mistral4 text model → use mistral4 config
                # Original text_config disambiguation by Jinho Jang (eric@jangq.ai) — vMLX.
                if text_model_type and text_model_type != model_type:
                    for config in self._configs:
                        if text_model_type in config.model_types and config.priority > 0:
                            config = _with_config_metadata_overrides(
                                config,
                                model_config,
                            )
                            logger.info(f"Model config: matched text_config.model_type='{text_model_type}' (wrapper='{model_type}') → {config.family_name}")
                            self._match_cache[model_name] = config
                            return config

                for config in self._configs:
                    if model_type in config.model_types:
                        next_config = _with_config_metadata_overrides(
                            config,
                            model_config,
                        )
                        if next_config.cache_type != config.cache_type:
                            logger.info(
                                "Model config: qwen3_5 linear_attention layers detected "
                                "from config.json; using hybrid cache"
                            )
                        if next_config.is_mllm != config.is_mllm:
                            logger.info(
                                "Model config: qwen3_5 media config detected "
                                "from config.json; using multimodal routing"
                            )
                        self._match_cache[model_name] = next_config
                        return next_config

            self._match_cache[model_name] = _DEFAULT_CONFIG
            return _DEFAULT_CONFIG

    def get_cache_type(self, model_name: str) -> str:
        """Get cache type for a model."""
        return self.lookup(model_name).cache_type

    def get_eos_tokens(self, model_name: str) -> Optional[List[str]]:
        """Get EOS token overrides."""
        return self.lookup(model_name).eos_tokens

    def is_mllm(self, model_name: str) -> bool:
        """Check if model is multimodal."""
        return self.lookup(model_name).is_mllm

    def needs_tokenizer_fallback(self, model_name: str) -> bool:
        """Check if model needs tokenizer fallback."""
        return self.lookup(model_name).tokenizer_fallback

    def get_tool_parser(self, model_name: str) -> Optional[str]:
        """Get recommended tool parser name."""
        return self.lookup(model_name).tool_parser

    def get_reasoning_parser(self, model_name: str) -> Optional[str]:
        """Get recommended reasoning parser name."""
        return self.lookup(model_name).reasoning_parser

    def get_architecture_hints(self, model_name: str) -> Dict[str, Any]:
        """Get architecture-specific hints."""
        return self.lookup(model_name).architecture_hints

    def list_registered(self) -> List[str]:
        """List all registered model family names."""
        with self._rlock:
            return [c.family_name for c in self._configs]

    def clear_cache(self) -> None:
        """Clear pattern matching cache."""
        with self._rlock:
            self._match_cache.clear()

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        with self._rlock:
            self._configs.clear()
            self._match_cache.clear()


_configs_loaded = False
_configs_lock = threading.Lock()


def get_model_config_registry() -> ModelConfigRegistry:
    """Get the global model config registry, auto-loading configs on first access."""
    global _configs_loaded
    registry = ModelConfigRegistry()
    # The singleton can be recreated by tests or embedding hosts while the
    # module-level loaded flag survives. Never return an empty global registry.
    if not _configs_loaded or not registry._configs:
        with _configs_lock:
            if not _configs_loaded or not registry._configs:
                try:
                    from .model_configs import register_all
                    register_all(registry)
                    _configs_loaded = True
                except ImportError:
                    _configs_loaded = True  # No module = nothing to register
    return registry
