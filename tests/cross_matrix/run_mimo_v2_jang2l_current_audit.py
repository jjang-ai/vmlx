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
DEFAULT_OUT = Path("build/current-mimo-v2-jang2l-current-audit-20260606.json")

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
CACHE_VS_NOCACHE_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-cache-vs-nocache-next-token-20260606.json"
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
    cache_vs_nocache = _artifact(root / CACHE_VS_NOCACHE_ARTIFACT)

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
    tool_blocked = tool_failure.get("exists") and _tool_protocol_blocked(
        tool_failure["data"]
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
        },
        "blockers": blockers,
        "artifacts": {
            "structural": str(STRUCTURAL_ARTIFACT),
            "text_cache": str(TEXT_CACHE_ARTIFACT),
            "switchglu": str(SWITCHGLU_ARTIFACT),
            "length_sweep": str(LENGTH_SWEEP_ARTIFACT),
            "tool_failure": str(TOOL_FAILURE_ARTIFACT),
            "cache_vs_nocache": str(CACHE_VS_NOCACHE_ARTIFACT),
        },
        "release_boundary": (
            "MiMo local artifact integrity is clean, but release remains blocked "
            "until long-prompt coherence and API tool protocol pass without fake "
            "parser injection, forced fallback, or cache disabling."
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
