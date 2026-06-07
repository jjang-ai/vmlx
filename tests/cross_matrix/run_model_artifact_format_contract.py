#!/usr/bin/env python3
"""Run no-heavy model artifact format detection contracts.

This gate protects the release row for JANG/JANGTQ/MXTQ, MXFP4, MXFP8, and
native-MTP artifact classification. It intentionally stays source/static only:
no model is loaded, and no path/name-only inference is accepted as proof.
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


DEFAULT_OUT = Path("build/current-model-artifact-format-contract-after-mllm-tight-memory-guard-20260607.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/model_config_registry.py",
    "vmlx_engine/model_configs.py",
    "vmlx_engine/native_mtp.py",
    "vmlx_engine/server.py",
    "vmlx_engine/utils/jang_loader.py",
    "vmlx_engine/loaders/load_jangtq.py",
    "vmlx_engine/loaders/load_jangtq_vlm.py",
    "vmlx_engine/loaders/load_jangtq_kimi_vlm.py",
    "vmlx_engine/loaders/load_jangtq_dsv4.py",
    "vmlx_engine/loaders/load_zaya.py",
    "vmlx_engine/loaders/load_laguna.py",
    "vmlx_engine/loaders/load_mistral3.py",
    "tests/test_model_config_registry.py",
    "tests/test_jang_loader.py",
    "tests/test_native_mtp_autodetect.py",
    "tests/test_cross_matrix_audit_runner.py",
)

REQUIRED_ARTIFACT_TEST_MARKERS = (
    # Generic affine JANG bundles must continue through the JANG loader rather
    # than being rejected or misclassified as JANGTQ/MXFP/plain MLX.
    "test_load_jang_model_accepts_affine_weight_format",
    # DSV4 must not be treated as generic JANGTQ/MTP. The static audit needs
    # to preserve explicit drop/no-runtime states instead of inferring runtime
    # from path names or stray weights.
    "test_dsv4_static_audit_reports_mtp_drop_contract",
    # JANGTQ_K mixed routed bits: gate/up and down can have different bit
    # widths; the down gather kernel must compile with down's bits.
    "test_gather_dn_uses_dp_bits",
    # Ling/Bailing hybrid artifacts have had loader-level shape and absent-MTP
    # regressions. These rows protect MXFP4 flat switch_mlp repair while also
    # ensuring already-correct JANGTQ 3D tensors stay untouched.
    "test_sanitize_repairs_flat_2d_switch_mlp_to_3d",
    "test_sanitize_no_op_on_correct_3d_shape",
    "test_sanitize_restores_dwq_split_mla_kv_b_proj",
    "test_sanitize_trims_absent_mtp_layer_before_strict_load",
    # Qwen 3.6 VL/video native-MTP artifacts span plain JANG, MXFP4, MXFP8,
    # and real weight-index detection. These are the artifact formats most
    # likely to regress when loader/autodetect code is touched.
    "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready",
    "test_mxfp4_vlm_sanitize_shifts_mtp_norms_only",
    "test_mxfp_vlm_loader_quantizes_with_declared_mode",
    "test_jang_quant_mode_supports_mxfp8_metadata",
    "test_qwen36_plain_mlx_4bit_keeps_hybrid_cache_without_jang_or_mxfp",
    "test_native_mtp_detection_uses_weights_not_path_name",
)

COMMANDS: dict[str, list[str]] = {
    "model_artifact_format_pytest": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_model_config_registry.py",
        "tests/test_jang_loader.py",
        "tests/test_native_mtp_autodetect.py",
        "tests/test_cross_matrix_audit_runner.py",
        "-k",
        "jang or JANG or mxfp or mxp or mlx_4bit or bailing or switch_mlp or mtp or MTP or cache_profile or dp_bits or gather_dn",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    deselected = None
    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "deselected": deselected}


def _run(root: Path, name: str, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout": proc.stdout,
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def build_artifact(root: Path) -> dict[str, Any]:
    results = {name: _run(root, name, cmd) for name, cmd in COMMANDS.items()}
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    stdout = "\n".join(str(result.get("stdout", "")) for result in results.values())
    missing_markers = [
        marker for marker in REQUIRED_ARTIFACT_TEST_MARKERS if marker not in stdout
    ]
    checks = {
        "jang_and_jangtq_detection": (
            not failed
            and "test_load_jang_model_accepts_affine_weight_format" not in missing_markers
            and "test_gather_dn_uses_dp_bits" not in missing_markers
        ),
        "ling_bailing_hybrid_loader_repairs": (
            not failed
            and "test_sanitize_repairs_flat_2d_switch_mlp_to_3d" not in missing_markers
            and "test_sanitize_no_op_on_correct_3d_shape" not in missing_markers
            and "test_sanitize_restores_dwq_split_mla_kv_b_proj" not in missing_markers
            and "test_sanitize_trims_absent_mtp_layer_before_strict_load" not in missing_markers
        ),
        "mxfp4_detection": not failed and "test_mxfp4_vlm_sanitize_shifts_mtp_norms_only" not in missing_markers,
        "mxfp8_detection": (
            not failed
            and "test_jang_quant_mode_supports_mxfp8_metadata" not in missing_markers
            and "test_mxfp_vlm_loader_quantizes_with_declared_mode" not in missing_markers
        ),
        "plain_mlx_4bit_detection": not failed and "test_qwen36_plain_mlx_4bit_keeps_hybrid_cache_without_jang_or_mxfp" not in missing_markers,
        "dropped_mtp_detection": not failed and "test_dsv4_static_audit_reports_mtp_drop_contract" not in missing_markers,
        "preserved_mtp_detection": not failed and "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready" not in missing_markers,
        "cache_profile_detection": not failed,
        "not_path_name_only": not failed and "test_native_mtp_detection_uses_weights_not_path_name" not in missing_markers,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "missing_markers": missing_markers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": {
            name: {key: value for key, value in result.items() if key != "stdout"}
            for name, result in results.items()
        },
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
    print("missing_markers=" + json.dumps(artifact["missing_markers"]))
    for name, result in artifact["results"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
