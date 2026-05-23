#!/usr/bin/env python3
"""Run no-heavy native MTP policy and wiring contracts.

This gate protects the native-MTP release row: D3 policy, model-local tuning
depths, dropped/preserved MTP detection, MLLM decode-loop safety, telemetry, and
panel launch controls. It does not claim live equivalence or speed clearance.
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


DEFAULT_OUT = Path("build/current-native-mtp-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/native_mtp.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/server.py",
    "vmlx_engine/native_mtp_policy_suite.py",
    "panel/src/main/sessions.ts",
    "panel/src/main/model-config-registry.ts",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/model-config-registry.test.ts",
    "tests/cross_matrix/run_native_mtp_contract.py",
    "tests/test_native_mtp_autodetect.py",
    "tests/test_native_mtp_policy_suite.py",
    "tests/test_native_mtp_bench_harness.py",
    "tests/test_mtp_telemetry_edge_cases.py",
    "tests/test_native_mtp_contract.py",
)

REQUIRED_NATIVE_MTP_TEST_MARKERS = (
    # Engine artifact/status detection: do not infer MTP from path names or
    # stale config alone.
    "test_cli_exposes_native_mtp_runtime_flags",
    "test_qwen36_nested_config_and_layered_tensors_are_native_ready",
    "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready",
    "test_native_mtp_detection_uses_weights_not_path_name",
    "test_config_only_mtp_bundle_does_not_activate_native_runtime",
    "test_runtime_metadata_can_explicitly_drop_configured_mtp",
    "test_jang2k_profile_blocks_native_mtp_runtime_but_keeps_vl_artifact_route",
    "test_jang_quant_mode_supports_mxfp8_metadata",
    # Depth/tuning/D3 policy: D3 can be used only from explicit runtime policy
    # and model-local evidence, not hidden sampler forcing.
    "test_native_mtp_depth_defaults_to_three",
    "test_native_mtp_depth_uses_validated_model_tuning_sidecar_by_default",
    "test_mllm_generator_runs_depth3_native_mtp_verify_cycle",
    "test_mllm_native_mtp_only_enables_for_proven_deterministic_sampling",
    "test_native_mtp_adaptive_depth_lowers_d3_after_poor_third_position",
    # Telemetry and metadata edge cases.
    "test_native_mtp_stats_snapshot_exposes_acceptance_depth_and_timings",
    "test_partial_indexed_layers_flagged",
    # Panel launch controls. These are display/CLI-wiring contracts, not hidden
    # generation defaults.
    "defaults native-MTP bundles to deterministic measured-depth launch policy without hidden sampler flags",
    "lets users disable native MTP without leaving deterministic sampling overrides behind",
    "keeps non-MTP models on bundle-owned generation defaults",
    "DSV4 additional args cannot reenable native MTP or deterministic sampling policy",
    "does not expose Native MTP for config-only bundles without indexed mtp tensors",
    "does not expose Native MTP for Ling/Bailing config-only bundles without indexed mtp tensors",
    "does not expose Native MTP for Hy3 config-only bundles without indexed mtp tensors",
    "real session launcher and settings form expose native MTP controls",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_native_mtp_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_native_mtp_autodetect.py",
            "tests/test_native_mtp_policy_suite.py",
            "tests/test_native_mtp_bench_harness.py",
            "tests/test_mtp_telemetry_edge_cases.py",
        ],
    ),
    "panel_native_mtp_controls": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/settings-flow.test.ts",
            "--testNamePattern",
            "native MTP|MTP|D3|DSV4",
            "--reporter=verbose",
        ],
    ),
    "panel_native_mtp_detection": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/model-config-registry.test.ts",
            "--testNamePattern",
            "Native MTP|MTP",
            "--reporter=verbose",
        ],
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
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    stdout = "\n".join(str(result.get("stdout", "")) for result in results.values())
    missing_markers = [
        marker for marker in REQUIRED_NATIVE_MTP_TEST_MARKERS if marker not in stdout
    ]
    engine_passed = results["engine_native_mtp_contracts"]["counts"]["passed"] or 0
    panel_passed = results["panel_native_mtp_controls"]["counts"]["passed"] or 0
    checks = {
        "native_mtp_d3_default_policy": (
            not failed
            and "test_native_mtp_depth_defaults_to_three" not in missing_markers
            and "test_mllm_generator_runs_depth3_native_mtp_verify_cycle" not in missing_markers
            and "defaults native-MTP bundles to deterministic measured-depth launch policy without hidden sampler flags" not in missing_markers
        ),
        "model_tuning_depth_policy": (
            not failed
            and "test_native_mtp_depth_uses_validated_model_tuning_sidecar_by_default" not in missing_markers
            and "test_native_mtp_adaptive_depth_lowers_d3_after_poor_third_position" not in missing_markers
        ),
        "dropped_and_preserved_mtp_detection": (
            not failed
            and "test_runtime_metadata_can_explicitly_drop_configured_mtp" not in missing_markers
            and "test_partial_indexed_layers_flagged" not in missing_markers
            and "test_qwen36_nested_config_and_layered_tensors_are_native_ready" not in missing_markers
        ),
        "config_only_mtp_never_activates": (
            not failed
            and "test_config_only_mtp_bundle_does_not_activate_native_runtime" not in missing_markers
            and "does not expose Native MTP for config-only bundles without indexed mtp tensors" not in missing_markers
        ),
        "mxfp4_mxfp8_mtp_artifact_detection": (
            not failed
            and "test_qwen36_mxfp4_mtp_bundle_is_text_native_ready" not in missing_markers
            and "test_jang_quant_mode_supports_mxfp8_metadata" not in missing_markers
        ),
        "mllm_native_mtp_decode_loop": (
            not failed
            and "test_mllm_native_mtp_only_enables_for_proven_deterministic_sampling" not in missing_markers
        ),
        "native_mtp_telemetry_edge_cases": (
            not failed
            and "test_native_mtp_stats_snapshot_exposes_acceptance_depth_and_timings" not in missing_markers
        ),
        "panel_native_mtp_controls_visible_when_supported": (
            not failed
            and "real session launcher and settings form expose native MTP controls" not in missing_markers
        ),
        "panel_native_mtp_suppressed_for_dsv4_or_unsupported": (
            not failed
            and "lets users disable native MTP without leaving deterministic sampling overrides behind" not in missing_markers
            and "keeps non-MTP models on bundle-owned generation defaults" not in missing_markers
            and "DSV4 additional args cannot reenable native MTP or deterministic sampling policy" not in missing_markers
        ),
        "live_speed_equivalence_not_claimed": True,
        "legacy_count_floor_still_nontrivial": (
            not failed and engine_passed >= 115 and panel_passed >= 11
        ),
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
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
