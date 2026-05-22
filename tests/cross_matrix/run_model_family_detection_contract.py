#!/usr/bin/env python3
"""Run named no-heavy family detection contracts.

This gate exists because broad parser/cache/model-artifact counts can hide
whether the risky release rows actually ran. It pins the specific families and
artifact formats Eric called out: DSV4, ZAYA/ZAYA1-VL, Ling/Bailing,
Nemotron-H, Qwen 3.6 VL/video/hybrid/MXFP/MTP, MiniMax M2, and Hy3 JANGTQ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-model-family-detection-contract-20260522.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/model_config_registry.py",
    "vmlx_engine/model_configs.py",
    "vmlx_engine/native_mtp.py",
    "panel/src/main/model-config-registry.ts",
    "panel/src/shared/reasoningParserAliases.ts",
    "panel/src/shared/toolParserAliases.ts",
    "panel/tests/model-config-registry.test.ts",
    "tests/cross_matrix/run_decode_speed_gate.py",
    "tests/cross_matrix/run_dsv4_long_context_gate.py",
    "tests/cross_matrix/run_dsv4_responses_cache_gate.py",
    "tests/test_model_config_registry.py",
    "tests/test_model_family_detection_contract.py",
)

ENGINE_PATTERN = (
    "deepseek_v4 or zaya or ling or bailing or nemotron or qwen3_5 "
    "or minimax or hy_v3 or decode_speed"
)
PANEL_PATTERN = (
    "ZAYA|zaya|Ling|Bailing|bailing|Nemotron|Qwen|MXFP|mxfp|JANGTQ|"
    "MTP|MiniMax|minimax|Hy3|backend-covered|TurboQuant"
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_family_detection": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_model_config_registry.py",
            "tests/test_model_family_detection_contract.py",
            "-k",
            ENGINE_PATTERN,
        ],
    ),
    "panel_family_detection": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/model-config-registry.test.ts",
            "--reporter",
            "verbose",
            "--testNamePattern",
            PANEL_PATTERN,
        ],
    ),
}

REQUIRED_ROWS = (
    "dsv4_deepseek_v4_native_cache_and_parser",
    "zaya_text_cca_tools_reasoning",
    "zaya1_vl_cca_multimodal",
    "ling_bailing_hybrid_plain_content",
    "nemotron_h_hybrid_text_not_stale_omni",
    "qwen36_vl_video_hybrid",
    "qwen36_moe_vl_video_hybrid",
    "qwen36_mxfp4_mxfp8_vl",
    "qwen36_native_mtp_vl",
    "minimax_m2_parser_alias",
    "hy3_jangtq_hunyuan_qwen3",
    "decode_speed_dsv4_minimax_canonical_parsers",
    "decode_speed_no_legacy_32k_startup_cap",
    "decode_speed_qwen36_mxfp8_native_mtp_rows",
    "decode_speed_nemotron_omni_nano_jangtq4_row",
    "decode_speed_distinct_jang_jangtq_mxfp_speed_rows",
    "decode_speed_artifact_format_coverage_matrix",
    "decode_speed_all_declared_parsers_are_engine_registered",
    "decode_speed_all_declared_parsers_are_cli_choices",
    "decode_speed_existing_rows_match_engine_parser_policy",
    "decode_speed_existing_rows_match_engine_modality_policy",
    "decode_speed_registry_cache_metadata_health",
)

ROW_MARKERS: dict[str, tuple[str, ...]] = {
    "dsv4_deepseek_v4_native_cache_and_parser": (
        "test_deepseek_v4_eos_includes_latest_reminder",
    ),
    "zaya_text_cca_tools_reasoning": (
        "test_zaya_config",
        "detects text ZAYA as CCA hybrid",
    ),
    "zaya1_vl_cca_multimodal": (
        "test_zaya1_vl_registered_with_full_contract",
        "detects ZAYA1-VL as multimodal CCA hybrid",
    ),
    "ling_bailing_hybrid_plain_content": (
        "test_ling_is_not_a_reasoning_model",
        "detects Ling/Bailing hybrid with tools and no reasoning parser",
    ),
    "nemotron_h_hybrid_text_not_stale_omni": (
        "detects backend-covered model_type=nemotron_h_v2",
        "does not route Nemotron-H text extracts through MLLM",
    ),
    "qwen36_vl_video_hybrid": (
        "keeps MXTQ/JANGTQ Qwen hybrid VLM multimodal",
    ),
    "qwen36_moe_vl_video_hybrid": (
        "marks non-JANG Qwen 3.6 MoE bundles with vision/video metadata as multimodal",
    ),
    "qwen36_mxfp4_mxfp8_vl": (
        "keeps mxfp4 Qwen hybrid VLM multimodal",
        "keeps mxfp8 Qwen hybrid VLM multimodal",
    ),
    "qwen36_native_mtp_vl": (
        "marks Qwen3.6 VL JANG bundles with indexed MTP tensors as native MTP capable",
    ),
    "minimax_m2_parser_alias": (
        "uses the registered MiniMax reasoning parser",
    ),
    "hy3_jangtq_hunyuan_qwen3": (
        "detects Hy3 as text-only KV with Hunyuan tools and qwen3 reasoning",
        "keeps Hy3 JANGTQ2 Low/High reasoning contract",
    ),
    "decode_speed_dsv4_minimax_canonical_parsers": (
        "test_decode_speed_gate_uses_canonical_release_parsers_for_dsv4_and_minimax",
    ),
    "decode_speed_no_legacy_32k_startup_cap": (
        "test_decode_speed_gate_does_not_force_legacy_32k_startup_output_cap",
    ),
    "decode_speed_qwen36_mxfp8_native_mtp_rows": (
        "test_decode_speed_gate_has_explicit_qwen36_mxfp8_and_native_mtp_rows",
    ),
    "decode_speed_nemotron_omni_nano_jangtq4_row": (
        "test_decode_speed_gate_has_explicit_nemotron_omni_nano_jangtq4_row",
    ),
    "decode_speed_distinct_jang_jangtq_mxfp_speed_rows": (
        "test_decode_speed_gate_has_distinct_jang_jangtq_mxfp_speed_rows",
    ),
    "decode_speed_artifact_format_coverage_matrix": (
        "test_decode_speed_gate_artifact_format_coverage_matrix",
    ),
    "decode_speed_all_declared_parsers_are_engine_registered": (
        "test_decode_speed_gate_declared_parsers_are_engine_registered",
    ),
    "decode_speed_all_declared_parsers_are_cli_choices": (
        "test_decode_speed_gate_declared_parsers_are_cli_choices",
    ),
    "decode_speed_existing_rows_match_engine_parser_policy": (
        "test_existing_decode_speed_rows_match_engine_registry_parser_policy",
    ),
    "decode_speed_existing_rows_match_engine_modality_policy": (
        "test_existing_decode_speed_rows_match_engine_registry_modality_policy",
    ),
    "decode_speed_registry_cache_metadata_health": (
        "test_decode_speed_gate_records_registry_cache_metadata_for_existing_rows",
        "test_decode_speed_gate_detects_cache_health_mismatches",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    skipped = None
    deselected = None
    match = re.search(r"Tests\s+(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


def _run(root: Path, name: str, cwd_rel: Path, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=root / cwd_rel,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd_rel),
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout": proc.stdout,
        "stdout_tail": proc.stdout.splitlines()[-120:],
    }


def _row_text(results: dict[str, dict[str, Any]]) -> str:
    chunks: list[str] = []
    for result in results.values():
        chunks.append(str(result.get("stdout", "")))
        chunks.extend(str(line) for line in result.get("stdout_tail", []))
    return "\n".join(chunks)


def _build_checks(results: dict[str, dict[str, Any]]) -> dict[str, bool]:
    text = _row_text(results)
    failed = any(result.get("returncode") != 0 for result in results.values())
    checks: dict[str, bool] = {}
    for row in REQUIRED_ROWS:
        markers = ROW_MARKERS[row]
        checks[row] = not failed and all(marker in text for marker in markers)
    return checks


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    checks = _build_checks(results)
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    matched_rows = [name for name, value in checks.items() if value]
    missing_rows = [name for name, value in checks.items() if not value]
    public_results = {
        name: {key: value for key, value in result.items() if key != "stdout"}
        for name, result in results.items()
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": public_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("failed=" + json.dumps(artifact["failed"]))
    print("missing_rows=" + json.dumps(artifact["missing_rows"]))
    for name, result in artifact["results"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
