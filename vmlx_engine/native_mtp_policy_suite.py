from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_PROMPT_CLASSES = [
    "deterministic",
    "factual",
    "coding",
    "creative",
    "high_entropy",
    "reasoning_off",
    "reasoning_on",
    "cache_repeat",
    "vl_image",
    "video",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _first_config_value(config: dict[str, Any], paths: list[tuple[str, ...]]) -> tuple[Any, str | None]:
    for path in paths:
        value = _nested(config, path)
        if value is not None:
            return value, "config." + ".".join(path)
    return None, None


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _dedupe_preserve(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _duplicate_ids(values: list[int]) -> list[int]:
    seen: set[int] = set()
    duplicates: list[int] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _token_id_for_content(tokenizer: dict[str, Any], content: str) -> int | None:
    decoder = tokenizer.get("added_tokens_decoder") or {}
    if not isinstance(decoder, dict):
        return None
    for raw_id, item in decoder.items():
        if not isinstance(item, dict) or item.get("content") != content:
            continue
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return None
    return None


def audit_stop_token_metadata(
    *,
    config: dict[str, Any],
    generation: dict[str, Any],
    tokenizer: dict[str, Any],
) -> dict[str, Any]:
    """Audit stop-token metadata that can break MTP finish parity.

    Qwen chat artifacts commonly need both `<|im_end|>` and `<|endoftext|>` in
    generation EOS handling. If the converter duplicates one or drops the other,
    greedy MTP can still look coherent on short chat prompts while stop/finish
    parity is not proven.
    """

    generation_eos = _as_int_list(generation.get("eos_token_id"))
    deduped_generation_eos = _dedupe_preserve(generation_eos)
    config_eos = _as_int_list(config.get("eos_token_id"))
    text_config = config.get("text_config") or {}
    text_config_eos = _as_int_list(text_config.get("eos_token_id"))

    im_end_id = _token_id_for_content(tokenizer, "<|im_end|>")
    endoftext_id = _token_id_for_content(tokenizer, "<|endoftext|>")
    required = [
        token_id
        for token_id in [im_end_id, endoftext_id]
        if token_id is not None
    ]
    required = _dedupe_preserve(required) if len(required) == 2 else []

    duplicates = _duplicate_ids(generation_eos)
    missing = [
        token_id for token_id in required if token_id not in deduped_generation_eos
    ]
    issues: list[str] = []
    if duplicates:
        issues.append("duplicate_generation_eos_token_id")
    if missing:
        issues.append("missing_required_generation_eos_token_id")
    if required and not generation_eos:
        issues.append("missing_generation_eos_token_id")
    for token_id in config_eos + text_config_eos:
        if generation_eos and token_id not in deduped_generation_eos:
            issues.append("config_eos_not_in_generation_eos")
            break

    return {
        "generation_eos_token_ids": generation_eos,
        "deduplicated_generation_eos_token_ids": deduped_generation_eos,
        "config_eos_token_ids": config_eos,
        "text_config_eos_token_ids": text_config_eos,
        "tokenizer_im_end_id": im_end_id,
        "tokenizer_endoftext_id": endoftext_id,
        "required_token_ids": required,
        "missing_required_token_ids": missing,
        "duplicate_token_ids": duplicates,
        "issues": issues,
        "stop_config_clean": not issues,
    }


def read_model_policy_metadata(model_path: str | Path) -> dict[str, Any]:
    """Read native routing top-k and generation sampler defaults separately."""

    path = Path(model_path).expanduser()
    config = _read_json(path / "config.json")
    generation = _read_json(path / "generation_config.json")
    tokenizer = _read_json(path / "tokenizer_config.json")
    active_experts, active_source = _first_config_value(
        config,
        [
            ("text_config", "num_experts_per_tok"),
            ("num_experts_per_tok",),
            ("text_config", "top_k_experts"),
            ("top_k_experts",),
            ("text_config", "moe_top_k"),
            ("moe_top_k",),
        ],
    )
    n_routed, n_routed_source = _first_config_value(
        config,
        [
            ("text_config", "num_experts"),
            ("num_experts",),
            ("text_config", "n_routed_experts"),
            ("n_routed_experts",),
        ],
    )
    generation_keys = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "do_sample",
    ]
    metadata = {
        "model_path": str(path),
        "model_type": config.get("model_type"),
        "text_model_type": (config.get("text_config") or {}).get("model_type"),
        "trained_routing_top_k": {
            "active_experts": int(active_experts) if active_experts is not None else None,
            "source": active_source,
            "n_routed_experts": int(n_routed) if n_routed is not None else None,
            "n_routed_experts_source": n_routed_source,
        },
        "generation_config": {
            "exists": bool(generation),
            "source": str(path / "generation_config.json") if generation else None,
            "sampling": {key: generation.get(key) for key in generation_keys},
        },
        "stop_token_audit": audit_stop_token_metadata(
            config=config,
            generation=generation,
            tokenizer=tokenizer,
        ),
    }
    metadata["metadata_readiness"] = {
        "ready": bool(metadata["stop_token_audit"]["stop_config_clean"]),
        "reasons": list(metadata["stop_token_audit"]["issues"]),
    }
    return metadata


def _settings_key(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "target": row["target_settings"],
            "draft": row["draft_settings"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _depth_acceptances(row: dict[str, Any], depth_limit: int | None = None) -> list[float]:
    depths = sorted(row.get("depths", []), key=lambda item: int(item["depth"]))
    if depth_limit is not None:
        depths = [item for item in depths if int(item["depth"]) <= int(depth_limit)]
    if not depths:
        raise ValueError("policy row must include at least one depth")
    return [float(item["expected_acceptance_rate"]) for item in depths]


def score_policy_cost(
    row: dict[str, Any],
    *,
    verify_ms: float,
    draft_ms_per_depth: float,
    fixed_ms: float = 0.0,
    depth_limit: int | None = None,
) -> dict[str, float | int]:
    """Estimate speculative-cycle value from per-depth acceptance and costs.

    This is a ranking model for shadow sweeps, not a production throughput
    predictor. Expected output tokens use the standard speculative-decoding
    shape: one token is always emitted, then each accepted prefix depth adds
    one more token.
    """

    acceptances = _depth_acceptances(row, depth_limit)
    depth = len(acceptances)
    prefix_probability = 1.0
    prefix_tokens = []
    for acceptance in acceptances:
        prefix_probability *= max(0.0, min(1.0, float(acceptance)))
        prefix_tokens.append(prefix_probability)

    expected_output_tokens = 1.0 + sum(prefix_tokens)
    cost_ms = float(verify_ms) + float(fixed_ms) + (float(draft_ms_per_depth) * depth)
    if cost_ms <= 0.0:
        raise ValueError("cost_ms must be positive")
    return {
        "depth": depth,
        "expected_output_tokens": expected_output_tokens,
        "expected_accepted_draft_tokens": sum(prefix_tokens),
        "marginal_last_depth_tokens": prefix_tokens[-1],
        "cost_ms": cost_ms,
        "tokens_per_ms": expected_output_tokens / cost_ms,
    }


def _available_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = result.get("real_logit_policy_sweep") or {}
    rows = sweep.get("rows")
    if rows:
        return list(rows)
    raise ValueError(
        "real_logit_policy_sweep.rows missing; rerun probe with --policy-sweep-save-all"
    )


def aggregate_policy_suite(
    prompt_results: list[dict[str, Any]],
    *,
    verify_ms: float,
    draft_ms_per_depth: float,
    fixed_ms: float = 0.0,
    top_n: int = 12,
    required_prompt_classes: list[str] | None = None,
    min_worst_d3_acceptance: float = 0.5,
) -> dict[str, Any]:
    if not prompt_results:
        raise ValueError("prompt_results must be non-empty")

    by_policy: dict[str, dict[str, Any]] = {}
    prompt_count = len(prompt_results)
    for prompt_result in prompt_results:
        prompt_class = str(prompt_result.get("prompt_class") or "unknown")
        artifact = str(prompt_result.get("artifact") or "")
        result = dict(prompt_result.get("result") or {})
        if result.get("all_argmax_equal") is not True:
            continue
        for row in _available_rows(result):
            key = _settings_key(row)
            entry = by_policy.setdefault(
                key,
                {
                    "target_settings": row["target_settings"],
                    "draft_settings": row["draft_settings"],
                    "prompt_rows": [],
                },
            )
            d3_score = score_policy_cost(
                row,
                verify_ms=verify_ms,
                draft_ms_per_depth=draft_ms_per_depth,
                fixed_ms=fixed_ms,
                depth_limit=3,
            )
            d2_score = score_policy_cost(
                row,
                verify_ms=verify_ms,
                draft_ms_per_depth=draft_ms_per_depth,
                fixed_ms=fixed_ms,
                depth_limit=2,
            )
            depths = sorted(row.get("depths", []), key=lambda item: int(item["depth"]))
            depth3 = next(
                (
                    float(item["expected_acceptance_rate"])
                    for item in depths
                    if int(item["depth"]) == 3
                ),
                None,
            )
            entry["prompt_rows"].append(
                {
                    "prompt_class": prompt_class,
                    "artifact": artifact,
                    "mean_acceptance": float(row.get("mean_acceptance", 0.0)),
                    "worst_acceptance": float(row.get("worst_acceptance", 0.0)),
                    "d3_acceptance": depth3,
                    "d2_tokens_per_ms": float(d2_score["tokens_per_ms"]),
                    "d3_tokens_per_ms": float(d3_score["tokens_per_ms"]),
                    "d3_minus_d2_tokens_per_ms": float(d3_score["tokens_per_ms"])
                    - float(d2_score["tokens_per_ms"]),
                    "d3_expected_output_tokens": float(d3_score["expected_output_tokens"]),
                    "d3_marginal_last_depth_tokens": float(
                        d3_score["marginal_last_depth_tokens"]
                    ),
                }
            )

    rows = []
    for entry in by_policy.values():
        prompt_rows = entry["prompt_rows"]
        if not prompt_rows:
            continue
        d3_scores = [float(item["d3_tokens_per_ms"]) for item in prompt_rows]
        d2_scores = [float(item["d2_tokens_per_ms"]) for item in prompt_rows]
        d3_acceptances = [
            float(item["d3_acceptance"])
            for item in prompt_rows
            if item["d3_acceptance"] is not None
        ]
        rows.append(
            {
                "target_settings": entry["target_settings"],
                "draft_settings": entry["draft_settings"],
                "prompt_count": len(prompt_rows),
                "classes": sorted({str(item["prompt_class"]) for item in prompt_rows}),
                "mean_tokens_per_ms_d3": sum(d3_scores) / len(d3_scores),
                "worst_tokens_per_ms_d3": min(d3_scores),
                "mean_tokens_per_ms_d2": sum(d2_scores) / len(d2_scores),
                "mean_d3_minus_d2_tokens_per_ms": (
                    sum(d3_scores) / len(d3_scores)
                )
                - (sum(d2_scores) / len(d2_scores)),
                "worst_d3_acceptance": min(d3_acceptances) if d3_acceptances else None,
                "mean_d3_acceptance": (
                    sum(d3_acceptances) / len(d3_acceptances)
                    if d3_acceptances
                    else None
                ),
                "prompt_rows": prompt_rows,
            }
        )

    rows.sort(
        key=lambda row: (
            row["prompt_count"] != prompt_count,
            -float(row["mean_tokens_per_ms_d3"]),
            -float(row["worst_tokens_per_ms_d3"]),
            float(row["target_settings"]["temperature"]),
            float(row["draft_settings"]["temperature"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    present_classes = sorted(
        {
            str(item.get("prompt_class") or "unknown")
            for item in prompt_results
        }
    )
    required = list(required_prompt_classes or DEFAULT_REQUIRED_PROMPT_CLASSES)
    missing = [item for item in required if item not in present_classes]
    readiness_reasons = []
    if missing:
        readiness_reasons.append("missing_required_classes")
    if not rows:
        readiness_reasons.append("no_eligible_policy_rows")
    elif rows[0].get("prompt_count") != prompt_count:
        readiness_reasons.append("top_policy_not_present_on_all_prompts")
    if rows and rows[0].get("worst_d3_acceptance") is not None:
        if float(rows[0]["worst_d3_acceptance"]) < float(min_worst_d3_acceptance):
            readiness_reasons.append("worst_d3_acceptance_below_threshold")

    return {
        "prompt_count": prompt_count,
        "eligible_policy_count": len(rows),
        "coverage": {
            "present_classes": present_classes,
            "required_classes": required,
            "missing_classes": missing,
            "has_vl_image": "vl_image" in present_classes,
            "has_video": "video" in present_classes,
            "has_cache_repeat": "cache_repeat" in present_classes,
        },
        "readiness": {
            "d3_shadow_ready": not readiness_reasons,
            "reasons": readiness_reasons,
            "min_worst_d3_acceptance": float(min_worst_d3_acceptance),
        },
        "cost_model": {
            "verify_ms": float(verify_ms),
            "draft_ms_per_depth": float(draft_ms_per_depth),
            "fixed_ms": float(fixed_ms),
        },
        "top": rows[: max(0, int(top_n))],
    }


def load_prompt_artifacts(items: list[str]) -> list[dict[str, Any]]:
    loaded = []
    for item in items:
        if "=" not in item:
            raise ValueError("artifact item must be class=path")
        prompt_class, artifact_path = item.split("=", 1)
        path = Path(artifact_path).expanduser()
        loaded.append(
            {
                "prompt_class": prompt_class,
                "artifact": str(path),
                "result": json.loads(path.read_text()),
            }
        )
    return loaded
