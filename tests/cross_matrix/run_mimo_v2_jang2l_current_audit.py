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
DEFAULT_MANIFEST = Path("build/current-mimo-http-manifest-20260606.tsv")
DEFAULT_OUT = Path(
    "build/current-mimo-v2-jang2l-current-audit-after-system-fold-cache-proof-20260606.json"
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
    "build/current-all-local-model-smoke-mimo-v25-jang2l-tools-nomedia-after-stop-token-cache-fix-20260606/summary.json"
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
    "build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-after-tool-row-20260607.json"
)
SYNCED_LONG_TOOL_CACHE_ARTIFACT = Path(
    "build/current-mimo-v25-jang2l-synced-long-tool-cache-proof-20260606.json"
)
TEXT_ROUTE_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-text-route-live-proof-20260606.json"
)
CB_ONESHOT_PREFILL_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-cb-cache-after-mimo-oneshot-prefill-20260606.json"
)
CB_NATIVE_THINKING_OFF_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json"
)
TOOL_SOURCE_PREFLIGHT_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-tool-source-preflight-20260606.json"
)
MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT = Path(
    "build/current-mimo-v2-mllm-inputs-embeds-interface-fix-20260606.json"
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


def _synced_long_tool_cache_evidence(data: dict[str, Any]) -> dict[str, Any]:
    checks = data.get("checks")
    if not isinstance(checks, list):
        checks = []
    by_name = {
        str(item.get("name")): item
        for item in checks
        if isinstance(item, dict) and item.get("name") is not None
    }
    failed = data.get("failed_checks")
    if not isinstance(failed, list):
        failed = []
    long_row = by_name.get("long_prompt_recall") or {}
    tool_row = by_name.get("tool_required") or {}
    cache_rows = [by_name.get("cache_repeat_1") or {}, by_name.get("cache_repeat_2") or {}]
    long_failed = any(
        isinstance(item, dict) and item.get("label") == "long_prompt_recall"
        for item in failed
    )
    tool_failed = any(
        isinstance(item, dict) and item.get("label") == "tool_required"
        for item in failed
    ) or not bool(tool_row.get("tool_calls"))
    cache_exact_pass = all(
        isinstance(row, dict) and row.get("text") == "ACK-CACHE-742"
        for row in cache_rows
    )
    return {
        "exists": True,
        "verdict": data.get("verdict"),
        "long_prompt_blocked": bool(long_failed),
        "tool_protocol_blocked": bool(tool_failed),
        "short_cache_exact_pass": bool(cache_exact_pass),
        "long_prompt_elapsed_s": long_row.get("elapsed_s"),
        "tool_elapsed_s": tool_row.get("elapsed_s"),
        "long_prompt_text_head": str(long_row.get("text") or "")[:200],
        "tool_text_head": str(tool_row.get("text") or "")[:200],
        "failed_checks": failed,
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


def _text_route_evidence(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("live_rows")
    rows = rows if isinstance(rows, dict) else {}
    long_row = rows.get("long_prompt_first_request")
    long_row = long_row if isinstance(long_row, dict) else {}
    cache1 = rows.get("cache_repeat_1")
    cache1 = cache1 if isinstance(cache1, dict) else {}
    cache2 = rows.get("cache_repeat_2")
    cache2 = cache2 if isinstance(cache2, dict) else {}
    tool_row = rows.get("tool_required")
    tool_row = tool_row if isinstance(tool_row, dict) else {}
    blockers = data.get("remaining_blockers")
    blockers = blockers if isinstance(blockers, list) else []
    return {
        "exists": True,
        "status": data.get("status"),
        "fix_status": data.get("fix_status"),
        "long_prompt_recall_pass": bool(long_row.get("sentinel_recall_pass")),
        "long_prompt_oom_cleared": bool(long_row.get("oom_cleared")),
        "cache_visible_pass": bool(cache1.get("empty_visible_cleared"))
        and bool(cache2.get("empty_visible_cleared")),
        "cache_release_exact_pass": bool(cache1.get("exact_expected_ack_cache_742"))
        and bool(cache2.get("exact_expected_ack_cache_742")),
        "tool_call_pass": bool(tool_row.get("tool_call_pass")),
        "speed_blocked": "decode_speed_below_release_target" in blockers,
        "cache_l2_not_reproved": "prefix_paged_l2_cache_hit_not_reproved_on_continuous_batching_route"
        in blockers,
        "source_vs_quant_missing": "source_vs_quant_first_divergence_missing" in blockers,
        "media_unwired": "mimo_vl_audio_video_unwired" in blockers,
        "remaining_blockers": blockers,
        "release_boundary": data.get("release_boundary"),
    }


def _cb_oneshot_prefill_evidence(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and row.get("name") is not None
    }

    def _response(name: str) -> dict[str, Any]:
        row = by_name.get(name) or {}
        response = row.get("response")
        return response if isinstance(response, dict) else {}

    def _health(name: str) -> dict[str, Any]:
        row = by_name.get(name) or {}
        health = row.get("health")
        payload = health.get("json") if isinstance(health, dict) else {}
        return payload if isinstance(payload, dict) else {}

    cache1 = _response("cache_repeat_1")
    cache2 = _response("cache_repeat_2")
    tool = _response("tool_call")
    long_prompt = _response("long_prompt")
    cache2_health = _health("cache_repeat_2_health")
    tool_health = _health("tool_health")

    scheduler = cache2_health.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    cache = cache2_health.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    totals = cache.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    bg = scheduler.get("batch_generator")
    bg = bg if isinstance(bg, dict) else {}

    tool_scheduler = tool_health.get("scheduler")
    tool_scheduler = tool_scheduler if isinstance(tool_scheduler, dict) else {}
    tool_bg = tool_scheduler.get("batch_generator")
    tool_bg = tool_bg if isinstance(tool_bg, dict) else {}

    cache_exact_pass = (
        cache1.get("ok") is True
        and cache2.get("ok") is True
        and cache1.get("content") == "ACK-CACHE-742"
        and cache2.get("content") == "ACK-CACHE-742"
    )
    cache_reproved = (
        isinstance(scheduler.get("cache_hit_tokens"), (int, float))
        and scheduler.get("cache_hit_tokens") >= 37
        and isinstance(totals.get("l2_block_tokens_on_disk"), (int, float))
        and totals.get("l2_block_tokens_on_disk") >= 37
    )
    generation_tps = tool_bg.get("generation_tps", bg.get("generation_tps"))
    return {
        "exists": True,
        "cache_exact_pass": bool(cache_exact_pass),
        "prefix_paged_l2_cache_reproved": bool(cache_reproved),
        "tool_protocol_blocked": not bool(tool.get("tool_calls")),
        "tool_content_head": str(tool.get("content") or "")[:200],
        "long_prompt_crashed": long_prompt.get("ok") is False
        and "RemoteDisconnected" in str(long_prompt.get("error") or ""),
        "long_prompt_error": long_prompt.get("error"),
        "speed_blocked": not (
            isinstance(generation_tps, (int, float)) and generation_tps >= 40.0
        ),
        "generation_tps": generation_tps,
        "cache1_elapsed_s": cache1.get("elapsed"),
        "cache2_elapsed_s": cache2.get("elapsed"),
    }


def _cb_native_thinking_off_evidence(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    rows = data.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and row.get("name") is not None
    }

    def _row_speed(name: str) -> float | None:
        row = by_name.get(name) or {}
        usage = row.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        tokens = usage.get("completion_tokens")
        elapsed = row.get("elapsed_s")
        if not isinstance(tokens, (int, float)) or not isinstance(elapsed, (int, float)):
            return None
        if elapsed <= 0:
            return None
        return float(tokens) / float(elapsed)

    speeds = [
        value
        for value in (
            _row_speed("exact_repeat_1"),
            _row_speed("exact_repeat_2"),
        )
        if value is not None
    ]
    speed_max = max(speeds) if speeds else None
    memory_pressure = any(
        isinstance(row, dict)
        and row.get("ok") is False
        and (
            "503" in str(row.get("error") or "")
            or "Service Unavailable" in str(row.get("error") or "")
        )
        for row in rows
    ) or "working-set pressure reject" in str(data.get("log_tail") or "")
    native_cache = summary.get("native_cache")
    native_cache = native_cache if isinstance(native_cache, dict) else {}
    return {
        "exists": True,
        "status": data.get("status"),
        "cache_exact_pass": bool(
            summary.get("exact_repeat_1") and summary.get("exact_repeat_2")
        ),
        "no_think_tags": bool(summary.get("no_think_tags")),
        "prefix_paged_l2_cache_reproved": (
            isinstance(summary.get("cache_hit_tokens"), (int, float))
            and summary.get("cache_hit_tokens") > 0
            and isinstance(summary.get("l2_tokens_on_disk"), (int, float))
            and summary.get("l2_tokens_on_disk") > 0
        ),
        "mixed_swa_native_cache": native_cache.get("cache_type") == "mixed_swa_kv",
        "storage_quantization_q8": (
            isinstance(native_cache.get("storage_quantization"), dict)
            and native_cache["storage_quantization"].get("enabled") is True
            and native_cache["storage_quantization"].get("bits") == 8
        ),
        "memory_pressure_blocked": bool(memory_pressure),
        "speed_blocked": not (
            isinstance(speed_max, (int, float)) and speed_max >= 40.0
        ),
        "speed_max_tps": speed_max,
        "texts": summary.get("texts"),
        "finish_reasons": summary.get("finish_reasons"),
        "cache_hit_tokens": summary.get("cache_hit_tokens"),
        "l2_tokens_on_disk": summary.get("l2_tokens_on_disk"),
    }


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
    synced_long_tool_cache = _artifact(root / SYNCED_LONG_TOOL_CACHE_ARTIFACT)
    text_route = _artifact(root / TEXT_ROUTE_ARTIFACT)
    cb_oneshot_prefill = _artifact(root / CB_ONESHOT_PREFILL_ARTIFACT)
    cb_native_thinking_off = _artifact(root / CB_NATIVE_THINKING_OFF_ARTIFACT)
    tool_source_preflight = _artifact(root / TOOL_SOURCE_PREFLIGHT_ARTIFACT)
    mllm_inputs_embeds_interface = _artifact(root / MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT)

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
    synced_evidence = (
        _synced_long_tool_cache_evidence(synced_long_tool_cache["data"])
        if synced_long_tool_cache.get("exists")
        and isinstance(synced_long_tool_cache.get("data"), dict)
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
    if synced_evidence.get("exists"):
        tool_blocked = tool_blocked or bool(synced_evidence.get("tool_protocol_blocked"))
        length_blocked = length_blocked or bool(synced_evidence.get("long_prompt_blocked"))
        exact_cache_blocked = exact_cache_blocked and not bool(
            synced_evidence.get("short_cache_exact_pass")
        )
    text_route_evidence = (
        _text_route_evidence(text_route["data"])
        if text_route.get("exists") and isinstance(text_route.get("data"), dict)
        else {"exists": False}
    )
    if text_route_evidence.get("exists"):
        if text_route_evidence.get("tool_call_pass"):
            tool_blocked = False
        if text_route_evidence.get("long_prompt_recall_pass") and text_route_evidence.get(
            "long_prompt_oom_cleared"
        ):
            length_blocked = False
        if text_route_evidence.get("cache_visible_pass"):
            exact_cache_blocked = exact_cache_blocked and not bool(
                text_route_evidence.get("cache_release_exact_pass")
            )
        speed_blocked = speed_blocked or bool(text_route_evidence.get("speed_blocked"))
    cb_evidence = (
        _cb_oneshot_prefill_evidence(cb_oneshot_prefill["data"])
        if cb_oneshot_prefill.get("exists")
        and isinstance(cb_oneshot_prefill.get("data"), dict)
        else {"exists": False}
    )
    if cb_evidence.get("exists"):
        if cb_evidence.get("cache_exact_pass"):
            exact_cache_blocked = False
            prompt_shape_blocked = False
        if cb_evidence.get("prefix_paged_l2_cache_reproved"):
            text_route_evidence["cache_l2_not_reproved"] = False
        tool_blocked = bool(cb_evidence.get("tool_protocol_blocked"))
        length_blocked = length_blocked or bool(cb_evidence.get("long_prompt_crashed"))
        speed_blocked = speed_blocked or bool(cb_evidence.get("speed_blocked"))
    cb_native_evidence = (
        _cb_native_thinking_off_evidence(cb_native_thinking_off["data"])
        if cb_native_thinking_off.get("exists")
        and isinstance(cb_native_thinking_off.get("data"), dict)
        else {"exists": False}
    )
    memory_pressure_blocked = False
    if cb_native_evidence.get("exists"):
        if cb_native_evidence.get("cache_exact_pass"):
            exact_cache_blocked = False
        if cb_native_evidence.get("prefix_paged_l2_cache_reproved"):
            text_route_evidence["cache_l2_not_reproved"] = False
        if cb_native_evidence.get("memory_pressure_blocked"):
            prompt_shape_blocked = False
            memory_pressure_blocked = True
        speed_blocked = speed_blocked or bool(cb_native_evidence.get("speed_blocked"))
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
    if memory_pressure_blocked:
        prompt_shape_blocked = False
    source_vs_quant_pass = source_vs_quant.get("exists") and _source_vs_quant_first_divergence_pass(
        source_vs_quant["data"]
    )
    mllm_inputs_embeds_interface_pass = (
        mllm_inputs_embeds_interface.get("exists")
        and isinstance(mllm_inputs_embeds_interface.get("data"), dict)
        and mllm_inputs_embeds_interface["data"].get("status") == "pass"
    )
    if text_route_evidence.get("exists") and text_route_evidence.get("cache_visible_pass"):
        prompt_shape_blocked = False

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
        and not memory_pressure_blocked
        and source_vs_quant_pass
        and mllm_inputs_embeds_interface_pass
        and not bool(text_route_evidence.get("cache_l2_not_reproved"))
        and not bool(text_route_evidence.get("media_unwired"))
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
    if memory_pressure_blocked:
        blockers.append("mimo_cb_system_prompt_working_set_pressure_blocked")
    if not source_vs_quant_pass:
        blockers.append("mimo_source_vs_quant_first_divergence_missing_or_failed")
    if not mllm_inputs_embeds_interface_pass:
        blockers.append("mimo_mllm_inputs_embeds_interface_missing_or_failed")
    if text_route_evidence.get("cache_l2_not_reproved"):
        blockers.append("mimo_prefix_paged_l2_cache_not_reproved")
    if text_route_evidence.get("media_unwired"):
        blockers.append("mimo_vl_audio_video_unwired")

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
            "cb_system_prompt_working_set_pressure": not bool(memory_pressure_blocked),
            "source_vs_quant_first_divergence": bool(source_vs_quant_pass),
            "mllm_inputs_embeds_interface": bool(mllm_inputs_embeds_interface_pass),
            "text_route_long_prompt": bool(
                text_route_evidence.get("long_prompt_recall_pass")
                and text_route_evidence.get("long_prompt_oom_cleared")
            ),
            "text_route_tool_call": bool(text_route_evidence.get("tool_call_pass")),
            "prefix_paged_l2_cache_reproved": not bool(
                text_route_evidence.get("cache_l2_not_reproved")
            ),
            "mimo_media_wired": not bool(text_route_evidence.get("media_unwired")),
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
            "synced_long_tool_cache": str(SYNCED_LONG_TOOL_CACHE_ARTIFACT),
            "text_route_live_proof": str(TEXT_ROUTE_ARTIFACT),
            "cb_oneshot_prefill_live_proof": str(CB_ONESHOT_PREFILL_ARTIFACT),
            "cb_native_thinking_off_live_proof": str(CB_NATIVE_THINKING_OFF_ARTIFACT),
            "tool_source_preflight": str(TOOL_SOURCE_PREFLIGHT_ARTIFACT),
            "mllm_inputs_embeds_interface": str(MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT),
        },
        "diagnostics": {
            "cache_vs_nocache_next_token_match": bool(cache_match),
            "all_local_smoke": smoke_evidence,
            "prompt_shape_first_token": prompt_shape_evidence,
            "synced_long_tool_cache": synced_evidence,
            "text_route": text_route_evidence,
            "cb_oneshot_prefill": cb_evidence,
            "cb_native_thinking_off": cb_native_evidence,
            "tool_source_preflight": (
                tool_source_preflight.get("data")
                if tool_source_preflight.get("exists")
                else {
                    "status": "missing",
                    "artifact": str(TOOL_SOURCE_PREFLIGHT_ARTIFACT),
                }
            ),
            "mllm_inputs_embeds_interface": (
                mllm_inputs_embeds_interface.get("data")
                if mllm_inputs_embeds_interface.get("exists")
                else {
                    "status": "missing",
                    "artifact": str(MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT),
                    "classification": "missing_mimo_mllm_inputs_embeds_interface_proof",
                }
            ),
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
            "until long-prompt coherence, tool protocol, decode speed, CB system "
            "prompt working-set pressure, local source-vs-quant first-divergence, "
            "and media wiring proof pass without fake parser injection, hidden "
            "thinking-mode rewrites, forced fallback, or cache disabling."
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
