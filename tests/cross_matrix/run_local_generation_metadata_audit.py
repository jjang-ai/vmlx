#!/usr/bin/env python3
"""Audit local high-risk model metadata against engine/app defaults.

This is intentionally no-heavy: it reads local bundle JSON and the engine
registry/resolvers without loading model weights. It exists for the failure
class where a model loops or emits corrupted-looking text because app/engine
startup silently drifted away from the bundle-owned chat/generation metadata.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-local-generation-metadata-audit.json")

HIGH_RISK_MODEL_CANDIDATES = (
    "/Users/example/models/JANGQ/Hy3-preview-JANG_2L",
    "/Users/example/models/JANGQ/Hy3-preview-JANGTQ2",
    "/Users/example/models/JANGQ/MiniMax-M2.7-JANG_K",
    "/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ_K",
    "/Users/example/models/dealign.ai/MiniMax-M2.7-JANG_K-CRACK",
    "/Users/example/models/dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK",
    "/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
    "/Users/example/models/JANGQ/Qwen3.6-27B-MXFP4-MTP",
    "/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP",
    "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ_K",
    "/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
    "/Users/example/models/JANGQ/Ling-2.6-flash-JANGTQ",
    "/Users/example/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK",
    "/Users/example/models/JANGQ/Gemma-4-31B-it-JANG_4M-MTP",
    "/Users/example/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
    "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K",
    "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG",
)

THINKING_TEMPLATE_MARKERS = (
    "enable_thinking",
    "<think>",
    "</think>",
    "<|think|>",
    "<｜tool▁calls▁begin｜>",
    "thought\n",
)

THINKING_BUDGET_MARKERS = (
    "thinking_budget",
    "max_thinking_tokens",
    "reasoning_budget",
    "budget_tokens",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _number(data: dict[str, Any], *keys: str) -> float | None:
    val = _nested(data, *keys)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _template_sources(model_path: Path, tokenizer_config: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for filename in ("chat_template.jinja", "chat_template.txt"):
        text = _read_text(model_path / filename)
        if text:
            sources[filename] = text
    tokenizer_template = tokenizer_config.get("chat_template")
    if isinstance(tokenizer_template, str) and tokenizer_template:
        sources["tokenizer_config.chat_template"] = tokenizer_template
    return sources


def _thinking_budget_metadata_default(gen: dict[str, Any], jang: dict[str, Any]) -> int | float | None:
    candidates: tuple[Any, ...] = (
        gen.get("max_thinking_tokens"),
        gen.get("thinking_budget"),
        gen.get("reasoning_budget"),
        _nested(gen, "reasoning", "budget_tokens"),
        _nested(jang, "chat", "sampling_defaults", "max_thinking_tokens"),
        _nested(jang, "chat", "sampling_defaults", "thinking_budget"),
        _nested(jang, "chat", "sampling_defaults", "reasoning_budget"),
        _nested(jang, "chat", "reasoning", "max_thinking_tokens"),
        _nested(jang, "chat", "reasoning", "thinking_budget"),
        _nested(jang, "chat", "reasoning", "budget_tokens"),
    )
    for val in candidates:
        if isinstance(val, (int, float)) and val > 0:
            return val
    return None


def _thinking_budget_row(model_path: Path, gen: dict[str, Any], jang: dict[str, Any], tokenizer_config: dict[str, Any]) -> dict[str, Any]:
    sources = _template_sources(model_path, tokenizer_config)
    template_text = "\n".join(sources.values())
    return {
        "metadata_default": _thinking_budget_metadata_default(gen, jang),
        "template_sources": sorted(sources),
        "template_mentions_thinking": any(marker in template_text for marker in THINKING_TEMPLATE_MARKERS),
        "template_mentions_budget": any(marker in template_text for marker in THINKING_BUDGET_MARKERS),
    }


def _sampling_source(jang: dict[str, Any], gen: dict[str, Any], key: str) -> str | None:
    sampling = _nested(jang, "chat", "sampling_defaults")
    if isinstance(sampling, dict) and isinstance(sampling.get(key), (int, float)):
        return "jang_config.chat.sampling_defaults"
    gen_key = {"max_new_tokens": "max_new_tokens"}.get(key, key)
    if isinstance(gen.get(gen_key), (int, float)):
        return "generation_config"
    return None


def _mode_repetition_keys(jang: dict[str, Any], family: str, enable_thinking: bool | None) -> tuple[str, ...]:
    if family == "deepseek_v4":
        if enable_thinking is False:
            return ("repetition_penalty_chat", "repetition_penalty", "repetition_penalty_thinking")
        return ("repetition_penalty_thinking", "repetition_penalty", "repetition_penalty_chat")
    mode = _nested(jang, "chat", "reasoning", "default_mode")
    if mode == "thinking":
        return ("repetition_penalty_thinking", "repetition_penalty_chat", "repetition_penalty")
    return ("repetition_penalty_chat", "repetition_penalty_thinking", "repetition_penalty")


def _resolved_repetition_from_files(
    jang: dict[str, Any],
    gen: dict[str, Any],
    family: str,
    enable_thinking: bool | None,
) -> tuple[float | None, str | None]:
    sampling = _nested(jang, "chat", "sampling_defaults")
    if isinstance(sampling, dict):
        for key in _mode_repetition_keys(jang, family, enable_thinking):
            if isinstance(sampling.get(key), (int, float)):
                return float(sampling[key]), f"jang_config.chat.sampling_defaults.{key}"
    if isinstance(gen.get("repetition_penalty"), (int, float)):
        return float(gen["repetition_penalty"]), "generation_config.repetition_penalty"
    return None, None


def _registry_row(model_path: Path) -> dict[str, Any]:
    from vmlx_engine.model_config_registry import get_model_config_registry

    cfg = get_model_config_registry().lookup(str(model_path))
    return {
        "family": getattr(cfg, "family_name", None),
        "tool_parser": getattr(cfg, "tool_parser", None),
        "reasoning_parser": getattr(cfg, "reasoning_parser", None),
        "supports_thinking": getattr(cfg, "supports_thinking", None),
        "think_in_template": getattr(cfg, "think_in_template", None),
        "cache_type": getattr(cfg, "cache_type", None),
        "is_mllm": getattr(cfg, "is_mllm", None),
        "native_mtp": getattr(cfg, "native_mtp", None),
    }


def _media_metadata_row(cfg: dict[str, Any], jang: dict[str, Any]) -> dict[str, Any]:
    architecture = _nested(jang, "architecture") or {}
    config_has_vision = isinstance(cfg.get("vision_config"), dict)
    config_has_audio = isinstance(cfg.get("audio_config"), dict)
    jang_has_vision = architecture.get("has_vision") if isinstance(architecture, dict) else None
    jang_has_audio = architecture.get("has_audio") if isinstance(architecture, dict) else None
    return {
        "config_has_vision_config": config_has_vision,
        "config_has_audio_config": config_has_audio,
        "config_has_image_token_id": cfg.get("image_token_id") is not None,
        "config_has_video_token_id": cfg.get("video_token_id") is not None,
        "config_has_audio_token_id": cfg.get("audio_token_id") is not None,
        "jang_has_vision": jang_has_vision if isinstance(jang_has_vision, bool) else None,
        "jang_has_audio": jang_has_audio if isinstance(jang_has_audio, bool) else None,
        "advertised_vision": bool(
            config_has_vision
            or cfg.get("image_token_id") is not None
            or cfg.get("video_token_id") is not None
            or jang_has_vision is True
        ),
        "advertised_audio": bool(
            config_has_audio
            or cfg.get("audio_token_id") is not None
            or jang_has_audio is True
        ),
    }


def _runtime_route_row(model_path: Path) -> dict[str, Any]:
    from vmlx_engine.api.utils import is_mllm_model

    return {
        "engine_is_mllm_default": bool(is_mllm_model(str(model_path))),
        "engine_is_mllm_force": bool(is_mllm_model(str(model_path), force_mllm=True)),
    }


def _engine_resolved_row(model_path: Path, enable_thinking: bool | None) -> dict[str, Any]:
    import vmlx_engine.server as server

    old_model_path = server._model_path
    old_defaults = (
        server._default_temperature,
        server._default_top_p,
        server._default_top_k,
        server._default_min_p,
        server._default_repetition_penalty,
        server._default_max_tokens,
        server._default_max_tokens_explicit,
    )
    try:
        server._model_path = str(model_path)
        server._default_temperature = None
        server._default_top_p = None
        server._default_top_k = None
        server._default_min_p = None
        server._default_repetition_penalty = None
        server._default_max_tokens_explicit = False
        return {
            "temperature": server._resolve_temperature(None, str(model_path)),
            "top_p": server._resolve_top_p(None, str(model_path)),
            "top_k": server._resolve_top_k(None, str(model_path)),
            "min_p": server._resolve_min_p(None, str(model_path)),
            "repetition_penalty": server._resolve_repetition_penalty(
                None,
                str(model_path),
                enable_thinking=enable_thinking,
            ),
            "max_tokens": server._resolve_max_tokens(None, str(model_path)),
        }
    finally:
        server._model_path = old_model_path
        (
            server._default_temperature,
            server._default_top_p,
            server._default_top_k,
            server._default_min_p,
            server._default_repetition_penalty,
            server._default_max_tokens,
            server._default_max_tokens_explicit,
        ) = old_defaults


def audit_model(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    cfg = _read_json(path / "config.json")
    gen = _read_json(path / "generation_config.json")
    jang = _read_json(path / "jang_config.json")
    tokenizer_config = _read_json(path / "tokenizer_config.json")
    registry = _registry_row(path)
    media_metadata = _media_metadata_row(cfg, jang)
    runtime_route = _runtime_route_row(path)
    family = str(registry.get("family") or "")
    caps = _nested(jang, "capabilities") or {}
    sampling = _nested(jang, "chat", "sampling_defaults") or {}
    resolved_chat = _engine_resolved_row(path, enable_thinking=False)
    resolved_thinking = _engine_resolved_row(path, enable_thinking=True)
    rep_chat, rep_chat_source = _resolved_repetition_from_files(jang, gen, family, False)
    rep_thinking, rep_thinking_source = _resolved_repetition_from_files(jang, gen, family, True)
    row = {
        "path": str(path),
        "exists": path.is_dir(),
        "model_type": cfg.get("model_type") or _nested(cfg, "text_config", "model_type"),
        "registry": registry,
        "capabilities": {
            "family": caps.get("family") if isinstance(caps, dict) else None,
            "tool_parser": caps.get("tool_parser") if isinstance(caps, dict) else None,
            "reasoning_parser": caps.get("reasoning_parser") if isinstance(caps, dict) else None,
            "supports_thinking": caps.get("supports_thinking") if isinstance(caps, dict) else None,
            "think_in_template": caps.get("think_in_template") if isinstance(caps, dict) else None,
            "cache_type": caps.get("cache_type") if isinstance(caps, dict) else None,
        },
        "metadata": {
            "generation_config": {
                key: gen.get(key)
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "max_new_tokens",
                    "max_thinking_tokens",
                )
                if key in gen
            },
            "jang_sampling_defaults": sampling if isinstance(sampling, dict) else {},
            "jang_reasoning": _nested(jang, "chat", "reasoning") or {},
        },
        "media_metadata": media_metadata,
        "runtime_route": runtime_route,
        "resolved_chat": resolved_chat,
        "resolved_thinking": resolved_thinking,
        "sources": {
            "temperature": _sampling_source(jang, gen, "temperature"),
            "top_p": _sampling_source(jang, gen, "top_p"),
            "top_k": _sampling_source(jang, gen, "top_k"),
            "min_p": _sampling_source(jang, gen, "min_p"),
            "max_new_tokens": _sampling_source(jang, gen, "max_new_tokens"),
            "repetition_penalty_chat": rep_chat_source,
            "repetition_penalty_thinking": rep_thinking_source,
        },
        "thinking_budget": _thinking_budget_row(path, gen, jang, tokenizer_config),
    }
    notes: list[str] = []
    if not path.is_dir():
        notes.append("missing_local_model")
    if row["capabilities"]["reasoning_parser"] and row["registry"]["reasoning_parser"] != row["capabilities"]["reasoning_parser"]:
        notes.append("registry_overrides_stale_capability_reasoning_parser")
    if row["capabilities"]["tool_parser"] and row["registry"]["tool_parser"] != row["capabilities"]["tool_parser"]:
        notes.append("registry_overrides_stale_capability_tool_parser")
    if rep_chat is None:
        notes.append("bundle_declares_no_chat_repetition_penalty_engine_will_not_invent_one")
    if _number(gen, "max_new_tokens") is None and _number(jang, "chat", "sampling_defaults", "max_new_tokens") is None:
        notes.append("bundle_declares_no_max_new_tokens_engine_fallback_applies")
    if (
        row["thinking_budget"]["template_mentions_thinking"]
        and not row["thinking_budget"]["template_mentions_budget"]
    ):
        notes.append("thinking_budget_override_forwarded_but_template_does_not_enforce")
    if (
        row["model_type"] == "step3p7"
        and media_metadata["advertised_vision"]
        and runtime_route["engine_is_mllm_default"] is False
        and runtime_route["engine_is_mllm_force"] is False
    ):
        notes.append("step3p7_advertised_media_guarded_text_only")
    if resolved_chat["max_tokens"] > 8192:
        notes.append("resolved_default_output_cap_above_8192")
    if resolved_chat["top_k"] < 0:
        notes.append("resolved_top_k_negative")
    row["notes"] = notes
    return row


def build_artifact(models: tuple[str, ...]) -> dict[str, Any]:
    rows = [audit_model(path) for path in models if Path(path).is_dir()]
    hard_failures: list[str] = []
    for row in rows:
        name = Path(row["path"]).name
        if row["resolved_chat"]["top_k"] < 0:
            hard_failures.append(f"{name}: negative resolved top_k")
        if row["resolved_chat"]["max_tokens"] <= 0:
            hard_failures.append(f"{name}: non-positive resolved max_tokens")
        if row["registry"]["tool_parser"] in {"", "auto"}:
            hard_failures.append(f"{name}: invalid registry tool parser")
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if not hard_failures else "fail",
        "hard_failures": hard_failures,
        "row_count": len(rows),
        "review_notes": {
            Path(row["path"]).name: row["notes"]
            for row in rows
            if row["notes"]
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    models = tuple(args.model) if args.model else HIGH_RISK_MODEL_CANDIDATES
    artifact = build_artifact(models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"rows={artifact['row_count']}")
    print("hard_failures=" + json.dumps(artifact["hard_failures"]))
    print("review_notes=" + json.dumps(artifact["review_notes"], sort_keys=True))
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
