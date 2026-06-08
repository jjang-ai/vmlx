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
    "build/current-mimo-v2-jang2l-current-audit-after-audio-waveform-smoke-wiring-20260608.json"
)
DEFAULT_SMOKE = Path(
    "build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-after-do-sample-false-rerun-20260607/summary.json"
)
DEFAULT_OUT = Path(
    "build/current-mimo-v2-no-source-exactness-classifier-after-audio-waveform-smoke-wiring-20260608.json"
)
DEFAULT_JANGTQ2_EXACTNESS = Path(
    "build/current-mimo-v25-jangtq2-exactness-isolation-20260608/result.json"
)
DEFAULT_JANG2L_EXACTNESS = Path(
    "build/current-mimo-v25-jang2l-exactness-isolation-20260608/result.json"
)
DEFAULT_JANGTQ2_NO_SWITCHGLU_FASTPATH = Path(
    "build/current-mimo-v25-jangtq2-exactness-no-switchglu-fastpath-diagnostic-20260608/result.json"
)
DEFAULT_JANGTQ2_NO_ROUTER_NO_SWITCHGLU_FASTPATH = Path(
    "build/current-mimo-v25-jangtq2-exactness-no-router-no-switchglu-fastpath-diagnostic-20260608/result.json"
)
DEFAULT_JANGTQ2_TQ_KERNEL_PARITY = Path(
    "build/current-mimo-v25-jangtq2-tq-kernel-parity-no-source-20260608.json"
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


def _choice_text(case: dict[str, Any], *, completion: bool = False) -> str:
    body = case.get("body") if isinstance(case, dict) else None
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    if completion:
        return str(choice.get("text") or "")
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return str(message.get("content") or "")


def _first_tool_arguments(case: dict[str, Any]) -> str:
    body = case.get("body") if isinstance(case, dict) else None
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
    function = first.get("function") if isinstance(first.get("function"), dict) else {}
    return str(function.get("arguments") or "")


def _exactness_probe_summary(
    jangtq2: dict[str, Any] | None,
    jang2l: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = "B7-CAT-09"
    tq_cases = jangtq2.get("cases", {}) if isinstance(jangtq2, dict) else {}
    j2_cases = jang2l.get("cases", {}) if isinstance(jang2l, dict) else {}
    tq_copy = _choice_text(tq_cases.get("completion_copy_b7", {}), completion=True)
    j2_copy = _choice_text(j2_cases.get("completion_copy_b7", {}), completion=True)
    tq_tool_args = _first_tool_arguments(tq_cases.get("chat_tool_b7", {}))
    j2_tool_args = _first_tool_arguments(j2_cases.get("chat_tool_b7", {}))
    j2_tool_text = _choice_text(j2_cases.get("chat_tool_b7", {}))
    return {
        "expected_literal": expected,
        "jangtq2_raw_completion_text": tq_copy,
        "jangtq2_raw_completion_literal_preserved": tq_copy.strip() == expected,
        "jangtq2_tool_arguments": tq_tool_args,
        "jangtq2_tool_literal_preserved": expected in tq_tool_args,
        "jang2l_raw_completion_text": j2_copy,
        "jang2l_raw_completion_literal_preserved": j2_copy.strip() == expected,
        "jang2l_tool_arguments": j2_tool_args,
        "jang2l_tool_content_head": j2_tool_text[:240],
        "jang2l_tool_call_parsed": bool(j2_tool_args),
    }


def _diagnostic_copy_preserved(artifact: dict[str, Any] | None) -> bool | None:
    if not isinstance(artifact, dict):
        return None
    cases = artifact.get("cases")
    if not isinstance(cases, dict):
        return None
    text = _choice_text(cases.get("completion_copy_b7", {}), completion=True)
    if not text:
        return None
    return text.strip() == "B7-CAT-09"


def _tq_kernel_parity_passed(artifact: dict[str, Any] | None) -> bool | None:
    if not isinstance(artifact, dict):
        return None
    if artifact.get("status") != "pass":
        return False
    reports = artifact.get("reports")
    if not isinstance(reports, list) or not reports:
        return False
    return all(
        isinstance(report, dict) and float(report.get("max_abs_diff", 1.0)) < 2e-2
        for report in reports
    )


def build_classification(
    audit: dict[str, Any],
    smoke: dict[str, Any],
    *,
    jangtq2: dict[str, Any] | None = None,
    jang2l: dict[str, Any] | None = None,
    no_switchglu_fastpath: dict[str, Any] | None = None,
    no_router_no_switchglu_fastpath: dict[str, Any] | None = None,
    tq_kernel_parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    no_source_exactness = _exactness_probe_summary(jangtq2, jang2l)
    no_switchglu_copy_preserved = _diagnostic_copy_preserved(no_switchglu_fastpath)
    no_router_no_switchglu_copy_preserved = _diagnostic_copy_preserved(
        no_router_no_switchglu_fastpath
    )
    tq_kernel_parity_passed = _tq_kernel_parity_passed(tq_kernel_parity)
    no_source_runtime_diagnostics = {
        "jangtq2_no_switchglu_fastpath_literal_preserved": no_switchglu_copy_preserved,
        "jangtq2_no_router_no_switchglu_fastpath_literal_preserved": (
            no_router_no_switchglu_copy_preserved
        ),
        "jangtq2_tq_kernel_parity_passed": tq_kernel_parity_passed,
        "jangtq2_switchglu_fastpath_primary_cause_excluded": (
            no_switchglu_copy_preserved is False
        ),
        "jangtq2_compiled_router_primary_cause_excluded": (
            no_router_no_switchglu_copy_preserved is False
        ),
        "jangtq2_tq_gather_kernel_primary_cause_excluded": (
            tq_kernel_parity_passed is True
        ),
    }

    unresolved_surfaces = {
        "artifact_quantization_or_decode_logits_quality": bool(exactness_failures),
        "jangtq2_raw_decode_or_artifact_quality": (
            no_source_exactness["jangtq2_raw_completion_text"] != ""
            and not no_source_exactness["jangtq2_raw_completion_literal_preserved"]
        ),
        "jang2l_chat_tool_quality": (
            no_source_exactness["jang2l_raw_completion_text"] != ""
            and no_source_exactness["jang2l_raw_completion_literal_preserved"]
            and not no_source_exactness["jang2l_tool_call_parsed"]
        ),
        "source_vs_quant_first_divergence": not bool(
            component_ok.get("source_vs_quant_first_divergence")
        ),
        "long_prompt_coherence": not bool(component_ok.get("long_prompt_coherence")),
        "continuous_batching_system_prompt_pressure": not bool(
            component_ok.get("cb_system_prompt_working_set_pressure")
        ),
        "mimo_media_runtime": not bool(component_ok.get("mimo_media_wired")),
        "jangtq2_switchglu_fastpath": (
            no_source_runtime_diagnostics[
                "jangtq2_switchglu_fastpath_primary_cause_excluded"
            ]
            is not True
        ),
        "jangtq2_compiled_router": (
            no_source_runtime_diagnostics[
                "jangtq2_compiled_router_primary_cause_excluded"
            ]
            is not True
        ),
        "jangtq2_tq_gather_kernel": (
            no_source_runtime_diagnostics[
                "jangtq2_tq_gather_kernel_primary_cause_excluded"
            ]
            is not True
        ),
    }

    release_ready = False
    status = "open" if exactness_failures or any(unresolved_surfaces.values()) else "pass"

    if (
        unresolved_surfaces["jangtq2_raw_decode_or_artifact_quality"]
        and unresolved_surfaces["jang2l_chat_tool_quality"]
        and no_source_runtime_diagnostics[
            "jangtq2_switchglu_fastpath_primary_cause_excluded"
        ]
        and no_source_runtime_diagnostics[
            "jangtq2_compiled_router_primary_cause_excluded"
        ]
        and no_source_runtime_diagnostics[
            "jangtq2_tq_gather_kernel_primary_cause_excluded"
        ]
    ):
        classification = (
            "jangtq2_literal_corruption_persists_without_cache_fastpath_router_"
            "and_tq_kernel_parity_passes"
        )
    elif (
        unresolved_surfaces["jangtq2_raw_decode_or_artifact_quality"]
        and unresolved_surfaces["jang2l_chat_tool_quality"]
    ):
        classification = "jangtq2_raw_decode_literal_corruption_jang2l_chat_tool_quality_open"
    elif exactness_failures and parser_structure_valid:
        classification = "model_generated_literal_mutation_after_valid_parser_structure"
    else:
        classification = "insufficient_evidence"

    required_next_evidence = [
        "do not repair semantic values in parser or JSON repair layer",
        "either rerun source-vs-quant when RAM is authorized or provide a stronger artifact-only logits/quantization diagnosis",
        "if artifact/quantization is confirmed, rebuild or reupload MiMo with a corrected quantization contract",
        "if runtime decode is confirmed, fix the decode/kernel path and rerun the exact tool/JSON sentinel rows",
    ]
    if unresolved_surfaces["mimo_media_runtime"]:
        required_next_evidence.append(
            "MiMo media remains unwired until real VL/audio/video runtime and cache proof exist"
        )
    else:
        required_next_evidence.append(
            "MiMo media runtime wiring is no longer the exactness blocker; keep live media E2E, audio depth, and cache/UI proof as separate release gates"
        )

    return {
        "status": status,
        "release_ready": release_ready,
        "classification": classification,
        "source_vs_quant_load_performed": False,
        "source_vs_quant_load_skipped_reason": "user_disallowed_source_vs_quant_due_ram",
        "audit_artifact": str(DEFAULT_AUDIT),
        "smoke_artifact": str(DEFAULT_SMOKE),
        "jangtq2_exactness_artifact": str(DEFAULT_JANGTQ2_EXACTNESS),
        "jang2l_exactness_artifact": str(DEFAULT_JANG2L_EXACTNESS),
        "jangtq2_no_switchglu_fastpath_artifact": str(
            DEFAULT_JANGTQ2_NO_SWITCHGLU_FASTPATH
        ),
        "jangtq2_no_router_no_switchglu_fastpath_artifact": str(
            DEFAULT_JANGTQ2_NO_ROUTER_NO_SWITCHGLU_FASTPATH
        ),
        "jangtq2_tq_kernel_parity_artifact": str(DEFAULT_JANGTQ2_TQ_KERNEL_PARITY),
        "no_source_exactness": no_source_exactness,
        "no_source_runtime_diagnostics": no_source_runtime_diagnostics,
        "exactness_failures": exactness_failures,
        "excluded_surfaces": excluded_surfaces,
        "unresolved_surfaces": unresolved_surfaces,
        "required_next_evidence": required_next_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--jangtq2-exactness", type=Path, default=DEFAULT_JANGTQ2_EXACTNESS)
    parser.add_argument("--jang2l-exactness", type=Path, default=DEFAULT_JANG2L_EXACTNESS)
    parser.add_argument(
        "--jangtq2-no-switchglu-fastpath",
        type=Path,
        default=DEFAULT_JANGTQ2_NO_SWITCHGLU_FASTPATH,
    )
    parser.add_argument(
        "--jangtq2-no-router-no-switchglu-fastpath",
        type=Path,
        default=DEFAULT_JANGTQ2_NO_ROUTER_NO_SWITCHGLU_FASTPATH,
    )
    parser.add_argument(
        "--jangtq2-tq-kernel-parity",
        type=Path,
        default=DEFAULT_JANGTQ2_TQ_KERNEL_PARITY,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    jangtq2 = _load(args.jangtq2_exactness) if args.jangtq2_exactness.exists() else None
    jang2l = _load(args.jang2l_exactness) if args.jang2l_exactness.exists() else None
    no_switchglu_fastpath = (
        _load(args.jangtq2_no_switchglu_fastpath)
        if args.jangtq2_no_switchglu_fastpath.exists()
        else None
    )
    no_router_no_switchglu_fastpath = (
        _load(args.jangtq2_no_router_no_switchglu_fastpath)
        if args.jangtq2_no_router_no_switchglu_fastpath.exists()
        else None
    )
    tq_kernel_parity = (
        _load(args.jangtq2_tq_kernel_parity)
        if args.jangtq2_tq_kernel_parity.exists()
        else None
    )
    artifact = build_classification(
        _load(args.audit),
        _load(args.smoke),
        jangtq2=jangtq2,
        jang2l=jang2l,
        no_switchglu_fastpath=no_switchglu_fastpath,
        no_router_no_switchglu_fastpath=no_router_no_switchglu_fastpath,
        tq_kernel_parity=tq_kernel_parity,
    )
    artifact["audit_artifact"] = str(args.audit)
    artifact["smoke_artifact"] = str(args.smoke)
    artifact["jangtq2_exactness_artifact"] = str(args.jangtq2_exactness)
    artifact["jang2l_exactness_artifact"] = str(args.jang2l_exactness)
    artifact["jangtq2_no_switchglu_fastpath_artifact"] = str(
        args.jangtq2_no_switchglu_fastpath
    )
    artifact["jangtq2_no_router_no_switchglu_fastpath_artifact"] = str(
        args.jangtq2_no_router_no_switchglu_fastpath
    )
    artifact["jangtq2_tq_kernel_parity_artifact"] = str(args.jangtq2_tq_kernel_parity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print(f"classification={artifact['classification']}")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
