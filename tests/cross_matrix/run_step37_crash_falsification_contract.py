#!/usr/bin/env python3
"""Validate the current Step-3.7 JANG_2L bundled serve crash-falsification proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-step37-crash-falsification-contract-after-gemma4-vl-refresh-20260606.json")
BUNDLED_SMOKE_SUMMARY = Path(
    "build/current-step37-crash-falsification-bundled-20260604/summary.json"
)
BUNDLED_ENDPOINT_RESULT = Path(
    "build/current-step37-endpoint-crash-falsification-bundled-20260604/result.json"
)
PACKAGED_TEXTONLY_GUARD_PROOF = Path(
    "build/step3p7-clean-staged-app-proof-20260605/proof.json"
)
REQUIRED_ENDPOINT_ROWS = {
    "chat_off": "/v1/chat/completions",
    "chat_on": "/v1/chat/completions",
    "legacy_completions": "/v1/completions",
    "responses_off": "/v1/responses",
}
REQUIRED_TEXTONLY_GUARD_ASSERTIONS = {
    "chat_text_before_media_http_200",
    "chat_media_rejected_http_400",
    "chat_media_rejection_mentions_text_only",
    "chat_text_after_media_http_200",
    "responses_media_rejected_http_400",
    "responses_media_rejection_mentions_text_only",
    "responses_text_after_media_http_200",
    "server_health_after_http_200",
}
REQUIRED_SMOKE_LABELS = {
    "text_cache_repeat_1",
    "text_cache_repeat_2",
    "text_multiturn_recall",
    "reasoning_on",
    "tool_required",
    "vl_blue_image",
    "text_no_media_after_image",
    "vl_blue_image_repeat",
    "vl_red_image_changed",
}


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - contract reports malformed artifact
        return {}, f"invalid_json:{path}:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return {}, f"not_object:{path}"
    return payload, None


def _native_cache_ok(native_cache: Any) -> bool:
    return (
        isinstance(native_cache, dict)
        and native_cache.get("family") == "step3p7"
        and native_cache.get("schema") == "mixed_swa_kv_v1"
        and native_cache.get("cache_type") == "mixed_swa_kv"
        and native_cache.get("cache_subtype") == "step3p7_full_sliding_kv"
        and native_cache.get("prefix") is True
        and native_cache.get("paged") is True
        and native_cache.get("block_disk_l2") is True
    )


def _validate_smoke_summary(summary: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "status": summary.get("status"),
        "completed": summary.get("completed"),
        "failed": summary.get("failed"),
        "request_labels": [],
        "native_cache": None,
        "cache_hit_observed": False,
    }
    if summary.get("status") != "pass":
        failures.append("bundled_smoke_status_not_pass")
    if summary.get("failed") != 0:
        failures.append("bundled_smoke_failed_not_zero")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 1:
        failures.append("bundled_smoke_result_count_not_one")
        return failures, evidence
    row = results[0] if isinstance(results[0], dict) else {}
    if row.get("status") != "pass":
        failures.append("bundled_smoke_row_not_pass")
    row_model = row.get("row") if isinstance(row.get("row"), dict) else {}
    if row_model.get("name") != "Step-3.7-Flash-JANG_2L":
        failures.append("bundled_smoke_wrong_model")
    native_cache = row.get("health_before", {}).get("native_cache")
    evidence["native_cache"] = native_cache
    if not _native_cache_ok(native_cache):
        failures.append("bundled_smoke_native_cache_not_step37_mixed_swa_l2")
    requests = row.get("requests")
    if not isinstance(requests, list):
        failures.append("bundled_smoke_requests_missing")
        return failures, evidence
    labels = {str(item.get("label")) for item in requests if isinstance(item, dict)}
    evidence["request_labels"] = sorted(labels)
    missing_labels = sorted(REQUIRED_SMOKE_LABELS - labels)
    if missing_labels:
        failures.append("bundled_smoke_missing_labels:" + ",".join(missing_labels))
    for item in requests:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label"))
        if item.get("code") != 200:
            failures.append(f"bundled_smoke_http_not_200:{label}")
        if item.get("validation_failures"):
            failures.append(f"bundled_smoke_validation_failures:{label}")
        cache_summary = item.get("cache_summary")
        if isinstance(cache_summary, dict) and cache_summary.get("has_cache_hit") is True:
            evidence["cache_hit_observed"] = True
    if not evidence["cache_hit_observed"]:
        failures.append("bundled_smoke_cache_hit_missing")
    return failures, evidence


def _validate_endpoint_result(result: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "native_cache": result.get("health_before", {}).get("native_cache"),
        "rows": {},
    }
    if not _native_cache_ok(evidence["native_cache"]):
        failures.append("endpoint_native_cache_not_step37_mixed_swa_l2")
    rows = result.get("rows")
    if not isinstance(rows, list):
        return ["endpoint_rows_missing"], evidence
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label"))
        seen.add(label)
        row_result = row.get("result") if isinstance(row.get("result"), dict) else {}
        evidence["rows"][label] = {
            "path": row.get("path"),
            "code": row_result.get("code"),
            "server_alive_after": row.get("server_alive_after"),
            "server_returncode": row_result.get("server_returncode"),
        }
        expected_path = REQUIRED_ENDPOINT_ROWS.get(label)
        if expected_path is None:
            continue
        if row.get("path") != expected_path:
            failures.append(f"endpoint_wrong_path:{label}")
        if row_result.get("code") != 200:
            failures.append(f"endpoint_http_not_200:{label}")
        if row.get("server_alive_after") is not True:
            failures.append(f"endpoint_server_not_alive_after:{label}")
        if row_result.get("server_returncode") is not None:
            failures.append(f"endpoint_server_exited_during:{label}")
    missing = sorted(set(REQUIRED_ENDPOINT_ROWS) - seen)
    if missing:
        failures.append("endpoint_missing_rows:" + ",".join(missing))
    return failures, evidence


def _validate_textonly_guard_proof(proof: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    assertions = proof.get("assertions")
    evidence: dict[str, Any] = {
        "pass": proof.get("pass"),
        "assertions": assertions if isinstance(assertions, dict) else {},
    }
    failures: list[str] = []
    if proof.get("pass") is not True:
        failures.append("packaged_textonly_guard_status_not_pass")
    if not isinstance(assertions, dict):
        return failures + ["packaged_textonly_guard_assertions_missing"], evidence
    missing = sorted(REQUIRED_TEXTONLY_GUARD_ASSERTIONS - set(assertions))
    if missing:
        failures.append("packaged_textonly_guard_missing_assertions:" + ",".join(missing))
    for key in sorted(REQUIRED_TEXTONLY_GUARD_ASSERTIONS):
        if assertions.get(key) is not True:
            failures.append(f"packaged_textonly_guard_assertion_false:{key}")
    return failures, evidence


def build_contract(root: Path) -> dict[str, Any]:
    summary, summary_error = _read_json(root / BUNDLED_SMOKE_SUMMARY)
    endpoint, endpoint_error = _read_json(root / BUNDLED_ENDPOINT_RESULT)
    guard, guard_error = _read_json(root / PACKAGED_TEXTONLY_GUARD_PROOF)
    legacy_artifacts_present = not summary_error or not endpoint_error
    failures = [error for error in (summary_error, endpoint_error) if error] if legacy_artifacts_present else []
    smoke_evidence: dict[str, Any] = {}
    endpoint_evidence: dict[str, Any] = {}
    guard_evidence: dict[str, Any] = {}
    if not summary_error:
        smoke_failures, smoke_evidence = _validate_smoke_summary(summary)
        failures.extend(smoke_failures)
    if not endpoint_error:
        endpoint_failures, endpoint_evidence = _validate_endpoint_result(endpoint)
        failures.extend(endpoint_failures)
    if not guard_error:
        guard_failures, guard_evidence = _validate_textonly_guard_proof(guard)
        failures.extend(guard_failures)
    elif not legacy_artifacts_present:
        failures.append(guard_error)
    return {
        "status": "pass" if not failures else "fail",
        "claim": "current bundled Step-3.7 JANG_2L serve path does not crash mid-request and unsupported Step3p7 media fails closed",
        "artifacts": {
            "bundled_smoke_summary": str(BUNDLED_SMOKE_SUMMARY),
            "bundled_endpoint_result": str(BUNDLED_ENDPOINT_RESULT),
            "packaged_textonly_guard_proof": str(PACKAGED_TEXTONLY_GUARD_PROOF),
        },
        "evidence": {
            "bundled_smoke": smoke_evidence,
            "bundled_endpoint": endpoint_evidence,
            "packaged_textonly_guard": guard_evidence,
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_contract(root)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out)}))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
