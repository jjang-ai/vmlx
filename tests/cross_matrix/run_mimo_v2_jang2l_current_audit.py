#!/usr/bin/env python3
"""No-heavy MiMo V2.5 JANG_2L current artifact/runtime-boundary audit.

This script intentionally does not launch the 106G model. It answers a narrower
release question from current evidence:

* Is the local canonical bundle byte-size aligned with the TB5/HTTP source
  manifest?
* Are stale local MiMo cache/backup copies absent?
* Do the current live/probe artifacts still prove runtime/tool/long-prompt
  blockers?

It must not convert a narrow short-text/cache pass into release clearance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("/Users/example/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L")
DEFAULT_MANIFEST = Path("build/current-mimo-http-tb5-manifest-20260606.tsv")
DEFAULT_OUT = Path(
    "build/current-mimo-v2-jang2l-current-audit-after-source-vs-quant-required-20260606.json"
)

STRUCTURAL_ARTIFACT = Path("build/current-mimo-jang2l-local-structural-verify-20260606.json")
TEXT_CACHE_ARTIFACT = Path("build/current-mimo-jang2l-live-text-cache-smoke-20260606.json")
SWITCHGLU_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-quantized-switchglu-parity-20260606.json"
)
LENGTH_SWEEP_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json"
)
TOOL_FAILURE_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json"
)
ALL_LOCAL_SMOKE_ARTIFACT = Path(
    "build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-metadata-truth-20260606/summary.json"
)
CACHE_VS_NOCACHE_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-cache-vs-nocache-next-token-20260606.json"
)
SINK_MODE_LENGTH_DIAGNOSTIC_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json"
)
DISABLE_SINK_LENGTH_DIAGNOSTIC_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json"
)
PROMPT_SHAPE_SWEEP_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-prompt-shape-sweep-20260606.json"
)
RENDERED_PROMPT_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-rendered-prompt-compare-20260606.json"
)
FIRST_TOKEN_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-first-token-probe-registered-20260606.json"
)
SOURCE_VS_QUANT_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json"
)
CLEANUP_LOG = Path("build/current-mimo-stale-local-cleanup-20260606.txt")

STALE_TARGETS = [
    Path("/Users/example/vmlx-sync-backups/local-installed-mimo-v2-mlx-model-20260527-swa-mask"),
    Path(
        "/Users/example/.cache/huggingface/modules/transformers_modules/"
        "MiMo_hyphen_V2_dot_5_hyphen_JANG_2L"
    ),
    Path("build/mimo-block-cache"),
    Path("build/current-mimo-jang2l-block-cache-20260606"),
    Path("build/current-mimo-jang2l-tool-block-cache-20260606"),
    Path("build/current-mimo-jang2l-block-cache-after-head-fix-20260606"),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact": str(path), "exists": False, "status": "missing"}
    try:
        data = _load_json(path)
    except Exception as exc:  # pragma: no cover - defensive for corrupted artifacts
        return {
            "artifact": str(path),
            "exists": True,
            "status": "invalid_json",
            "error": str(exc),
        }
    status = data.get("status") if isinstance(data, dict) else None
    return {"artifact": str(path), "exists": True, "status": status, "data": data}


def _verify_manifest(model_parent: Path, manifest: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "manifest": str(manifest),
        "model_parent": str(model_parent),
        "status": "missing_manifest",
        "rows": 0,
        "total_bytes": 0,
        "ok_files": 0,
        "missing_files": [],
        "mismatched_files": [],
    }
    if not manifest.exists():
        return result

    rows: list[tuple[int, str]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        size_text, rel = line.split("\t", 1)
        rows.append((int(size_text), rel))

    result["rows"] = len(rows)
    result["total_bytes"] = sum(size for size, _ in rows)
    for expected_size, rel in rows:
        path = model_parent / rel
        if not path.exists():
            result["missing_files"].append(rel)
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            result["mismatched_files"].append(
                {"path": rel, "expected_size": expected_size, "actual_size": actual_size}
            )
            continue
        result["ok_files"] += 1

    result["status"] = (
        "pass"
        if result["ok_files"] == len(rows)
        and not result["missing_files"]
        and not result["mismatched_files"]
        else "fail"
    )
    return result


def _text_cache_narrow_pass(data: dict[str, Any]) -> bool:
    requests = data.get("requests")
    if not isinstance(requests, list):
        return False
    for entry in requests:
        if not isinstance(entry, dict):
            continue
        response = entry.get("response")
        if not isinstance(response, dict):
            continue
        payload = response.get("json")
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if response.get("http_status") == 200 and content == "cache ok":
            return True
    return False


def _cache_vs_nocache_match(data: dict[str, Any]) -> bool:
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return False
    top_lists = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        top10 = row.get("top10")
        if not isinstance(top10, list):
            return False
        top_lists.append([(item.get("id"), item.get("text")) for item in top10])
    return all(items == top_lists[0] for items in top_lists[1:])


def _length_sweep_blocked(data: dict[str, Any]) -> bool:
    cases = data.get("cases")
    if not isinstance(cases, list):
        return data.get("status") == "fail"
    return any(
        isinstance(case, dict) and str(case.get("status", "")).startswith("fail")
        for case in cases
    )


def _tool_protocol_blocked(data: dict[str, Any]) -> bool:
    if data.get("status") == "fail":
        return True
    observations = data.get("runtime_observations")
    if not isinstance(observations, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("result", "")).startswith("fail")
        for item in observations
    )


def _all_local_smoke_evidence(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results")
    result = results[0] if isinstance(results, list) and results else data
    if not isinstance(result, dict):
        return {
            "exists": True,
            "tool_protocol_pass": False,
            "exact_cache_blocked": True,
            "speed_blocked": True,
        }

    requests = result.get("requests")
    failures = result.get("failures")
    if not isinstance(requests, list):
        requests = []
    if not isinstance(failures, list):
        failures = []

    tool_protocol_pass = False
    for request in requests:
        if not isinstance(request, dict) or request.get("label") != "tool_required":
            continue
        tool_calls = request.get("tool_calls")
        validation_failures = request.get("validation_failures")
        if not isinstance(validation_failures, list):
            validation_failures = []
        if isinstance(tool_calls, list) and tool_calls and not validation_failures:
            tool_protocol_pass = True

    exact_cache_blocked = any(
        isinstance(failure, dict)
        and failure.get("label") in {"text_cache_repeat_1", "text_cache_repeat_2"}
        for failure in failures
    )
    generation_tps = (
        result.get("cache_after", {})
        .get("body", {})
        .get("scheduler_stats", {})
        .get("batch_generator", {})
        .get("generation_tps")
    )
    speed_blocked = not (
        isinstance(generation_tps, (int, float)) and generation_tps >= 40.0
    )
    return {
        "exists": True,
        "tool_protocol_pass": tool_protocol_pass,
        "exact_cache_blocked": exact_cache_blocked,
        "speed_blocked": speed_blocked,
        "generation_tps": generation_tps,
        "failures": failures,
    }


def _all_sink_diagnostic_cases_fail(data: dict[str, Any]) -> bool:
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return False
    return all(
        isinstance(case, dict) and case.get("passes_length_ok") is False
        for case in cases
    )


def _mimo_prompt_shape_first_token_evidence(
    prompt_shape: dict[str, Any],
    rendered_prompt: dict[str, Any],
    first_token: dict[str, Any],
) -> dict[str, Any]:
    results = prompt_shape.get("results")
    if not isinstance(results, list):
        results = []
    by_name = {
        str(item.get("name")): item
        for item in results
        if isinstance(item, dict) and item.get("name") is not None
    }

    failing = by_name.get("long_cache_system_exact") or {}
    folded = by_name.get("long_cache_no_system_exact") or {}
    short_system = by_name.get("short_system_exact") or {}

    rendered_rows = rendered_prompt.get("rows")
    if not isinstance(rendered_rows, list):
        rendered_rows = []
    rendered_by_name = {
        str(item.get("name")): item
        for item in rendered_rows
        if isinstance(item, dict) and item.get("name") is not None
    }
    failing_rendered = rendered_by_name.get("long_cache_system_exact") or {}

    first_rows = first_token.get("rows")
    if not isinstance(first_rows, list):
        first_rows = []
    first_by_name = {
        str(item.get("name")): item
        for item in first_rows
        if isinstance(item, dict) and item.get("name") is not None
    }
    failing_top = (first_by_name.get("failing_system_long") or {}).get("top")
    folded_top = (first_by_name.get("working_folded_long") or {}).get("top")
    short_top = (first_by_name.get("working_short_system") or {}).get("top")
    if not isinstance(failing_top, list):
        failing_top = []
    if not isinstance(folded_top, list):
        folded_top = []
    if not isinstance(short_top, list):
        short_top = []

    failing_top0 = failing_top[0] if failing_top and isinstance(failing_top[0], dict) else {}
    folded_top0 = folded_top[0] if folded_top and isinstance(folded_top[0], dict) else {}
    short_top0 = short_top[0] if short_top and isinstance(short_top[0], dict) else {}
    ack_in_failing_rank = None
    for index, item in enumerate(failing_top, start=1):
        if isinstance(item, dict) and item.get("text") == "ACK":
            ack_in_failing_rank = index
            break

    failing_empty_stop = (
        failing.get("content") == ""
        and failing.get("finish_reason") == "stop"
        and (failing.get("completion_tokens") == 1)
    )
    folded_ack = folded.get("content") == "ACK"
    short_ack = short_system.get("content") == "ACK"
    rendered_valid_generation_prefix = (
        failing_rendered.get("contains_empty_think_generation_prefix") is True
    )
    first_token_stop = (
        failing_top0.get("text") == "<|im_end|>"
        and failing_top0.get("is_special") is True
    )
    working_first_tokens_ack = (
        folded_top0.get("text") == "ACK" and short_top0.get("text") == "ACK"
    )

    blocked = (
        failing_empty_stop
        and folded_ack
        and short_ack
        and rendered_valid_generation_prefix
        and first_token_stop
        and working_first_tokens_ack
    )
    return {
        "blocked": bool(blocked),
        "classification": (
            "decode_loop_or_model_artifact_first_token_stop"
            if blocked
            else "insufficient_prompt_shape_evidence"
        ),
        "failing_empty_stop": bool(failing_empty_stop),
        "folded_user_ack": bool(folded_ack),
        "short_system_ack": bool(short_ack),
        "rendered_valid_generation_prefix": bool(rendered_valid_generation_prefix),
        "failing_top_token": failing_top0,
        "working_folded_top_token": folded_top0,
        "working_short_system_top_token": short_top0,
        "ack_rank_in_failing_prompt": ack_in_failing_rank,
        "failing_prompt_tokens": failing.get("prompt_tokens"),
        "folded_prompt_tokens": folded.get("prompt_tokens"),
    }


def _source_vs_quant_first_divergence_pass(data: dict[str, Any]) -> bool:
    if data.get("status") != "pass" or data.get("remote_evidence_only") is True:
        return False
    if not data.get("source_model_path") or not data.get("quant_model_path"):
        return False
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    allowed = {
        "source_and_quant_match",
        "quant_diverges_from_source",
        "source_also_fails",
    }
    for row in rows:
        if not isinstance(row, dict):
            return False
        if not row.get("prompt_name"):
            return False
        if row.get("classification") not in allowed:
            return False
        if "source_output" not in row or "quant_output" not in row:
            return False
    return True


def build_audit(root: Path, model_path: Path, manifest: Path) -> dict[str, Any]:
    model_parent = model_path.parent
    manifest_check = _verify_manifest(model_parent, manifest)
    stale_state = [
        {
            "path": str(path),
            "exists": (root / path).exists() if not path.is_absolute() else path.exists(),
        }
        for path in STALE_TARGETS
    ]

    structural = _artifact(root / STRUCTURAL_ARTIFACT)
    text_cache = _artifact(root / TEXT_CACHE_ARTIFACT)
    switchglu = _artifact(root / SWITCHGLU_ARTIFACT)
    length_sweep = _artifact(root / LENGTH_SWEEP_ARTIFACT)
    tool_failure = _artifact(root / TOOL_FAILURE_ARTIFACT)
    all_local_smoke = _artifact(root / ALL_LOCAL_SMOKE_ARTIFACT)
    cache_vs_nocache = _artifact(root / CACHE_VS_NOCACHE_ARTIFACT)
    sink_mode_length = _artifact(root / SINK_MODE_LENGTH_DIAGNOSTIC_ARTIFACT)
    disable_sink_length = _artifact(root / DISABLE_SINK_LENGTH_DIAGNOSTIC_ARTIFACT)
    prompt_shape = _artifact(root / PROMPT_SHAPE_SWEEP_ARTIFACT)
    rendered_prompt = _artifact(root / RENDERED_PROMPT_ARTIFACT)
    first_token = _artifact(root / FIRST_TOKEN_ARTIFACT)
    source_vs_quant = _artifact(root / SOURCE_VS_QUANT_ARTIFACT)

    structural_pass = structural.get("status") == "pass"
    text_cache_pass = text_cache.get("exists") and _text_cache_narrow_pass(text_cache["data"])
    switchglu_data = switchglu.get("data", {}) if isinstance(switchglu.get("data"), dict) else {}
    switchglu_shape_match = (
        switchglu_data.get("got_shape") == switchglu_data.get("manual_shape")
        and isinstance(switchglu_data.get("got_shape"), list)
    )
    switchglu_numeric_match = (
        isinstance(switchglu_data.get("max_abs_diff"), (int, float))
        and switchglu_data.get("max_abs_diff") <= 0.001
        and isinstance(switchglu_data.get("mean_abs_diff"), (int, float))
        and switchglu_data.get("mean_abs_diff") <= 0.0002
    )
    switchglu_pass = switchglu.get("exists") and (
        switchglu.get("status") == "pass"
        or switchglu_data.get("status") == "pass"
        or switchglu_data.get("ok") is True
        or (switchglu_shape_match and switchglu_numeric_match)
    )
    cache_match = cache_vs_nocache.get("exists") and _cache_vs_nocache_match(
        cache_vs_nocache["data"]
    )
    length_blocked = length_sweep.get("exists") and _length_sweep_blocked(
        length_sweep["data"]
    )
    smoke_evidence = (
        _all_local_smoke_evidence(all_local_smoke["data"])
        if all_local_smoke.get("exists")
        else {"exists": False}
    )
    if smoke_evidence.get("exists"):
        tool_blocked = not bool(smoke_evidence.get("tool_protocol_pass"))
        exact_cache_blocked = bool(smoke_evidence.get("exact_cache_blocked"))
        speed_blocked = bool(smoke_evidence.get("speed_blocked"))
    else:
        tool_blocked = tool_failure.get("exists") and _tool_protocol_blocked(
            tool_failure["data"]
        )
        exact_cache_blocked = False
        speed_blocked = True
    sink_mode_fails = sink_mode_length.get("exists") and _all_sink_diagnostic_cases_fail(
        sink_mode_length["data"]
    )
    disable_sink_fails = disable_sink_length.get("exists") and _all_sink_diagnostic_cases_fail(
        disable_sink_length["data"]
    )
    prompt_shape_evidence = (
        _mimo_prompt_shape_first_token_evidence(
            prompt_shape["data"],
            rendered_prompt["data"],
            first_token["data"],
        )
        if prompt_shape.get("exists")
        and rendered_prompt.get("exists")
        and first_token.get("exists")
        else {
            "blocked": False,
            "classification": "missing_prompt_shape_first_token_artifact",
        }
    )
    prompt_shape_blocked = bool(prompt_shape_evidence.get("blocked"))
    source_vs_quant_pass = source_vs_quant.get("exists") and _source_vs_quant_first_divergence_pass(
        source_vs_quant["data"]
    )

    stale_present = [item for item in stale_state if item["exists"]]
    release_clearance = (
        manifest_check["status"] == "pass"
        and structural_pass
        and text_cache_pass
        and switchglu_pass
        and cache_match
        and not length_blocked
        and not tool_blocked
        and not exact_cache_blocked
        and not speed_blocked
        and not prompt_shape_blocked
        and source_vs_quant_pass
        and not stale_present
    )

    status = "pass" if release_clearance else "open"
    blockers = []
    if manifest_check["status"] != "pass":
        blockers.append("mimo_manifest_integrity_failed")
    if stale_present:
        blockers.append("stale_mimo_local_state_present")
    if not structural_pass:
        blockers.append("mimo_structural_verify_missing_or_failed")
    if not text_cache_pass:
        blockers.append("mimo_text_cache_narrow_proof_missing_or_failed")
    if not switchglu_pass:
        blockers.append("mimo_switchglu_selected_expert_parity_missing_or_failed")
    if not cache_match:
        blockers.append("mimo_cache_vs_nocache_next_token_missing_or_mismatch")
    if length_blocked:
        blockers.append("mimo_long_prompt_coherence_blocked")
    if tool_blocked:
        blockers.append("mimo_tool_protocol_blocked")
    if exact_cache_blocked:
        blockers.append("mimo_exact_cache_prompt_following_blocked")
    if speed_blocked:
        blockers.append("mimo_decode_speed_below_release_target")
    if prompt_shape_blocked:
        blockers.append("mimo_system_prompt_first_token_stop_blocked")
    if not source_vs_quant_pass:
        blockers.append("mimo_source_vs_quant_first_divergence_missing_or_failed")

    return {
        "status": status,
        "local_release_clearance": release_clearance,
        "model_path": str(model_path),
        "manifest_integrity": manifest_check,
        "stale_local_state": {
            "cleanup_log": str(root / CLEANUP_LOG),
            "all_stale_targets_absent": not stale_present,
            "targets": stale_state,
        },
        "component_ok": {
            "manifest_integrity": manifest_check["status"] == "pass",
            "stale_local_state_absent": not stale_present,
            "structural_verify": bool(structural_pass),
            "text_cache_narrow": bool(text_cache_pass),
            "switchglu_selected_expert_parity": bool(switchglu_pass),
            "cache_vs_nocache_next_token": bool(cache_match),
            "long_prompt_coherence": not bool(length_blocked),
            "tool_protocol": not bool(tool_blocked),
            "exact_cache_prompt_following": not bool(exact_cache_blocked),
            "decode_speed_target": not bool(speed_blocked),
            "system_prompt_first_token_stop": not bool(prompt_shape_blocked),
            "source_vs_quant_first_divergence": bool(source_vs_quant_pass),
            "manual_sink_does_not_clear_length_generation": bool(sink_mode_fails),
            "disable_sink_does_not_clear_length_generation": bool(disable_sink_fails),
        },
        "blockers": blockers,
        "artifacts": {
            "structural": str(STRUCTURAL_ARTIFACT),
            "text_cache": str(TEXT_CACHE_ARTIFACT),
            "switchglu": str(SWITCHGLU_ARTIFACT),
            "length_sweep": str(LENGTH_SWEEP_ARTIFACT),
            "tool_failure": str(TOOL_FAILURE_ARTIFACT),
            "all_local_smoke": str(ALL_LOCAL_SMOKE_ARTIFACT),
            "cache_vs_nocache": str(CACHE_VS_NOCACHE_ARTIFACT),
            "sink_mode_length_diagnostic": str(SINK_MODE_LENGTH_DIAGNOSTIC_ARTIFACT),
            "disable_sink_length_diagnostic": str(DISABLE_SINK_LENGTH_DIAGNOSTIC_ARTIFACT),
            "prompt_shape_sweep": str(PROMPT_SHAPE_SWEEP_ARTIFACT),
            "rendered_prompt_compare": str(RENDERED_PROMPT_ARTIFACT),
            "first_token_probe": str(FIRST_TOKEN_ARTIFACT),
            "source_vs_quant_first_divergence": str(SOURCE_VS_QUANT_ARTIFACT),
        },
        "diagnostics": {
            "cache_vs_nocache_next_token_match": bool(cache_match),
            "all_local_smoke": smoke_evidence,
            "prompt_shape_first_token": prompt_shape_evidence,
            "source_vs_quant_first_divergence": (
                source_vs_quant.get("data")
                if source_vs_quant.get("exists")
                else {
                    "status": "missing",
                    "artifact": str(SOURCE_VS_QUANT_ARTIFACT),
                    "classification": "missing_local_source_vs_quant_first_divergence",
                }
            ),
            "manual_sink_sdpa_clears_length_generation": False if sink_mode_fails else None,
            "disable_sink_clears_length_generation": False if disable_sink_fails else None,
            "sink_boundary": (
                "Manual sink SDPA and disabling SWA sink do not clear the current "
                "MiMo length-generation failures; do not classify the blocker as "
                "an MLX sink-kernel-only issue."
            ),
        },
        "release_boundary": (
            "MiMo local artifact integrity is clean, but release remains blocked "
            "until long-prompt coherence, exact cache prompt-following, decode "
            "speed, and local source-vs-quant first-divergence proof pass without "
            "fake parser injection, forced fallback, or cache disabling."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    root = Path.cwd()
    result = build_audit(root, args.model_path, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "open"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
