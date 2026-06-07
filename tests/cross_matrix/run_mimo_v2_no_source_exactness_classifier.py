#!/usr/bin/env python3
"""Classify current MiMo V2.5 exactness failures without source-vs-quant load.

This runner is intentionally no-heavy. It consumes existing MiMo audit/smoke
artifacts and records what the current evidence excludes. It must not turn
wrong model-emitted literals into a pass and must not replace the missing
source-vs-quant proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_AUDIT = Path(
    "build/current-mimo-v2-jang2l-current-audit-after-do-sample-false-rerun-20260607.json"
)
DEFAULT_SMOKE = Path(
    "build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-after-do-sample-false-rerun-20260607/summary.json"
)
DEFAULT_OUT = Path(
    "build/current-mimo-v2-no-source-exactness-classifier-after-do-sample-false-20260607.json"
)

EXACTNESS_REASONS = {
    "expected_tool_argument_missing",
    "json_exact_object_mismatch",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failures(smoke: dict[str, Any]) -> list[dict[str, Any]]:
    rows = smoke.get("results")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("failures"), list):
            out.extend(f for f in row["failures"] if isinstance(f, dict))
    return out


def _server_log_text(smoke: dict[str, Any]) -> str:
    chunks: list[str] = []
    for row in smoke.get("results", []):
        if not isinstance(row, dict):
            continue
        tail = row.get("server_log_tail")
        if isinstance(tail, list):
            chunks.extend(str(item) for item in tail)
        elif isinstance(tail, str):
            chunks.append(tail)
    return "\n".join(chunks)


def build_classification(audit: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    component_ok = audit.get("component_ok") if isinstance(audit.get("component_ok"), dict) else {}
    failures = _failures(smoke)
    exactness_failures = [
        failure for failure in failures if failure.get("reason") in EXACTNESS_REASONS
    ]
    log_text = _server_log_text(smoke)

    parser_structure_valid = bool(component_ok.get("tool_protocol")) and all(
        isinstance(failure.get("actual"), dict) for failure in exactness_failures
    )
    cache_kv_l2_excluded = bool(component_ok.get("exactness_cache_kv_quant_excluded"))
    deterministic_sampling_seen = (
        "temperature': 0.0" in log_text
        or '"temperature": 0.0' in log_text
        or "temperature=0.0" in log_text
    ) and (
        "top_p': 1.0" in log_text
        or '"top_p": 1.0' in log_text
        or "top_p=1.0" in log_text
    )

    excluded_surfaces = {
        "parser_argument_rewrite": parser_structure_valid,
        "prefix_paged_l2_or_kv_quant_primary_cause": cache_kv_l2_excluded,
        "hidden_stochastic_sampling_primary_cause": deterministic_sampling_seen,
        "tool_protocol_structure": bool(component_ok.get("tool_protocol")),
        "api_cache_responses_plumbing": bool(component_ok.get("api_cache_responses_contract")),
        "decode_speed_target": bool(component_ok.get("decode_speed_target")),
    }

    unresolved_surfaces = {
        "artifact_quantization_or_decode_logits_quality": bool(exactness_failures),
        "source_vs_quant_first_divergence": not bool(
            component_ok.get("source_vs_quant_first_divergence")
        ),
        "long_prompt_coherence": not bool(component_ok.get("long_prompt_coherence")),
        "continuous_batching_system_prompt_pressure": not bool(
            component_ok.get("cb_system_prompt_working_set_pressure")
        ),
        "mimo_media_runtime": not bool(component_ok.get("mimo_media_wired")),
    }

    release_ready = False
    status = "open" if exactness_failures or any(unresolved_surfaces.values()) else "pass"

    return {
        "status": status,
        "release_ready": release_ready,
        "classification": (
            "model_generated_literal_mutation_after_valid_parser_structure"
            if exactness_failures and parser_structure_valid
            else "insufficient_evidence"
        ),
        "source_vs_quant_load_performed": False,
        "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
        "audit_artifact": str(DEFAULT_AUDIT),
        "smoke_artifact": str(DEFAULT_SMOKE),
        "exactness_failures": exactness_failures,
        "excluded_surfaces": excluded_surfaces,
        "unresolved_surfaces": unresolved_surfaces,
        "required_next_evidence": [
            "do not repair semantic values in parser or JSON repair layer",
            "either rerun source-vs-quant when RAM is authorized or provide a stronger artifact-only logits/quantization diagnosis",
            "if artifact/quantization is confirmed, rebuild or reupload MiMo with a corrected quantization contract",
            "if runtime decode is confirmed, fix the decode/kernel path and rerun the exact tool/JSON sentinel rows",
            "MiMo media remains unwired until real VL/audio/video runtime and cache proof exist",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_classification(_load(args.audit), _load(args.smoke))
    artifact["audit_artifact"] = str(args.audit)
    artifact["smoke_artifact"] = str(args.smoke)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"classification={artifact['classification']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
