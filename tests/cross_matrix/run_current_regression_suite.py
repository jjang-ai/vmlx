#!/usr/bin/env python3
"""Run the current no-heavy regression suite for the release-blocker work.

This suite is intentionally explicit about the DSV4 row that is still
open. It should fail for any *new* open objective row, failed no-heavy contract,
or release-gate failure that is not the known objective digest block.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-regression-suite-20260521.json")

EXPECTED_OPEN_REQUIREMENTS = [
    "DSV4 long-output/code/file-generation quality is release-cleared",
]


def _run_step(name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "stdout_tail": output.splitlines()[-80:],
    }


def _load_objective_digest(root: Path) -> dict[str, Any]:
    path = root / "build/current-objective-proof-audit-20260521.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _open_requirements(digest: dict[str, Any]) -> list[str]:
    return [
        str(item.get("requirement"))
        for item in digest.get("requirements", [])
        if item.get("status") != "pass"
    ]


def _release_gate_failure_is_expected(step: dict[str, Any]) -> bool:
    if step["returncode"] == 0:
        return True
    text = "\n".join(step.get("stdout_tail", []))
    fail_lines = [line for line in text.splitlines() if line.startswith("[FAIL]")]
    expected_line = (
        "[FAIL] objective proof digest: "
        + "; ".join(EXPECTED_OPEN_REQUIREMENTS)
    )
    if fail_lines != [expected_line]:
        return False
    forbidden = (
        "bundled python import gate: FAIL",
        "panel typecheck: FAIL",
        "panel request/type tests: FAIL",
        "version triple: FAIL",
    )
    return not any(pattern in text for pattern in forbidden)


def _step_is_ok(name: str, step: dict[str, Any]) -> bool:
    if name == "release_gate_skip_app":
        return _release_gate_failure_is_expected(step)
    return step["returncode"] == 0


def build_suite_artifact(root: Path, *, include_release_gate: bool = True) -> dict[str, Any]:
    steps: dict[str, dict[str, Any]] = {}
    commands: dict[str, list[str]] = {
        "noheavy_api_cache_contract": [
            sys.executable,
            "tests/cross_matrix/run_noheavy_api_cache_contract.py",
            "--out",
            "build/current-api-cache-contract-proof-20260521.json",
        ],
        "cache_architecture_contracts": [
            sys.executable,
            "tests/cross_matrix/run_cache_architecture_contract.py",
            "--out",
            "build/current-cache-architecture-contract-20260521.json",
        ],
        "noheavy_panel_settings_contract": [
            sys.executable,
            "tests/cross_matrix/run_noheavy_panel_settings_contract.py",
            "--out",
            "build/current-panel-settings-contract-proof-20260521.json",
        ],
        "max_output_context_contracts": [
            sys.executable,
            "tests/cross_matrix/run_max_output_context_contract.py",
            "--out",
            "build/current-max-output-context-contract-20260521.json",
        ],
        "parser_registry_contracts": [
            sys.executable,
            "tests/cross_matrix/run_parser_registry_contract.py",
            "--out",
            "build/current-parser-registry-contract-20260521.json",
        ],
        "generation_defaults_contracts": [
            sys.executable,
            "tests/cross_matrix/run_generation_defaults_contract.py",
            "--out",
            "build/current-generation-defaults-contract-20260521.json",
        ],
        "reasoning_template_contracts": [
            sys.executable,
            "tests/cross_matrix/run_reasoning_template_contract.py",
            "--out",
            "build/current-reasoning-template-contract-20260521.json",
        ],
        "api_surface_contracts": [
            sys.executable,
            "tests/cross_matrix/run_api_surface_contract.py",
            "--out",
            "build/current-api-surface-contract-20260521.json",
        ],
        "release_regression_manifest": [
            sys.executable,
            "tests/cross_matrix/run_release_regression_manifest.py",
            "--out",
            "build/current-release-regression-manifest-20260521.json",
        ],
        "panel_tool_security_contracts": [
            sys.executable,
            "tests/cross_matrix/run_panel_tool_security_contract.py",
            "--out",
            "build/current-panel-tool-security-contract-20260521.json",
        ],
        "tool_call_contracts": [
            sys.executable,
            "tests/cross_matrix/run_tool_call_contract.py",
            "--out",
            "build/current-tool-call-contract-20260521.json",
        ],
        "mcp_policy_contracts": [
            sys.executable,
            "tests/cross_matrix/run_mcp_policy_contract.py",
            "--out",
            "build/current-mcp-policy-contract-20260521.json",
        ],
        "packaged_integrity_contracts": [
            sys.executable,
            "tests/cross_matrix/run_packaged_integrity_contract.py",
            "--out",
            "build/current-packaged-integrity-contract-20260521.json",
        ],
        "release_surface_contracts": [
            sys.executable,
            "tests/cross_matrix/run_release_surface_contract.py",
            "--out",
            "build/current-release-surface-contract-20260521.json",
        ],
        "cli_release_contracts": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_cli_commands.py::TestServeCommandSocketBinding",
        ],
        "jang_model_compat_contracts": [
            sys.executable,
            "tests/cross_matrix/run_jang_model_compat_contract.py",
            "--out",
            "build/current-jang-model-compat-contract-20260521.json",
        ],
        "model_artifact_format_contracts": [
            sys.executable,
            "tests/cross_matrix/run_model_artifact_format_contract.py",
            "--out",
            "build/current-model-artifact-format-contract-20260521.json",
        ],
        "model_family_detection_contracts": [
            sys.executable,
            "tests/cross_matrix/run_model_family_detection_contract.py",
            "--out",
            "build/current-model-family-detection-contract-20260521.json",
        ],
        "native_mtp_contracts": [
            sys.executable,
            "tests/cross_matrix/run_native_mtp_contract.py",
            "--out",
            "build/current-native-mtp-contract-20260521.json",
        ],
        "vl_media_cache_contracts": [
            sys.executable,
            "tests/cross_matrix/run_vl_media_cache_contract.py",
            "--out",
            "build/current-vl-media-cache-contract-20260521.json",
        ],
        "focused_regression_pytest": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_objective_proof_digest.py",
            "tests/test_dsv4_default_cache_tool_loop_gate.py",
            "tests/test_release_gate_python_app.py",
            "tests/test_current_regression_suite.py",
            "tests/test_release_regression_manifest.py",
            "tests/test_model_family_detection_contract.py",
            "tests/test_vl_media_cache_contract.py",
            "-k",
            "objective_proof_digest or default_cache_tool_loop or current_regression_suite or release_regression_manifest or model_family_detection or decode_speed_gate or vl_media_cache_contract",
        ],
        "objective_digest": [
            sys.executable,
            "tests/cross_matrix/summarize_objective_proof.py",
            "--out",
            "build/current-objective-proof-audit-20260521.json",
        ],
    }
    if include_release_gate:
        commands["release_gate_skip_app"] = [
            "panel/scripts/release-gate-python-app.py",
            "--skip-app",
            "--skip-gui",
        ]

    for name, cmd in commands.items():
        steps[name] = _run_step(name, cmd, root)

    digest = _load_objective_digest(root)
    open_requirements = _open_requirements(digest)
    unexpected_open = [
        item for item in open_requirements if item not in EXPECTED_OPEN_REQUIREMENTS
    ]
    missing_expected_open = [
        item for item in EXPECTED_OPEN_REQUIREMENTS if item not in open_requirements
    ]
    failed_steps = [
        name for name, step in steps.items() if not _step_is_ok(name, step)
    ]
    status = (
        "pass"
        if not failed_steps and not unexpected_open and not missing_expected_open
        else "open"
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "known_open_requirements": EXPECTED_OPEN_REQUIREMENTS,
        "open_requirements": open_requirements,
        "unexpected_open_requirements": unexpected_open,
        "missing_expected_open_requirements": missing_expected_open,
        "failed_steps": failed_steps,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-release-gate", action="store_true")
    args = parser.parse_args()

    artifact = build_suite_artifact(
        args.root,
        include_release_gate=not args.skip_release_gate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("open_requirements=" + json.dumps(artifact["open_requirements"]))
    print("failed_steps=" + json.dumps(artifact["failed_steps"]))
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
