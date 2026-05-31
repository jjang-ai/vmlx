#!/usr/bin/env python3
"""Run the current no-heavy regression suite for the release-blocker work.

This suite is intentionally explicit about known-open objective rows. It
should fail for any *new* open objective row, failed no-heavy contract, or
release-gate failure that is not the known objective digest block.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-regression-suite-20260531-live-epipe-signing-dsv4-refresh.json"
)

EXPECTED_OPEN_REQUIREMENTS = [
    "Real Electron UI cross-family live model matrix is release-cleared",
    "DSV4 long-output/code/file-generation quality is release-cleared",
]

CURRENT_OBJECTIVE_DIGEST_ARTIFACT = (
    "build/current-objective-proof-audit-20260531-nemotron-exact-finalizer-ledger.json"
)

CURRENT_SUITE_SOURCE_HASH_FILES = (
    "bench/all_local_model_smoke.py",
    "bench/native_mtp_speed_ab.py",
    "panel/scripts/bundle-python.sh",
    "panel/scripts/live-chat-tools-reasoning-proof.mjs",
    "panel/scripts/live-real-ui-model-proof.mjs",
    "panel/scripts/release-gate-python-app.py",
    "panel/scripts/verify-bundled-python.sh",
    "panel/src/main/api-gateway.ts",
    "panel/src/main/engine-manager.ts",
    "panel/src/main/index.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/src/main/ipc/developer.ts",
    "panel/src/main/ipc/image.ts",
    "panel/src/main/ipc/imageGenerationState.ts",
    "panel/src/main/ipc/models.ts",
    "panel/src/main/model-config-registry.ts",
    "panel/src/main/process-manager.ts",
    "panel/src/main/server.ts",
    "panel/src/main/sessions.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/renderer/src/components/chat/MessageBubble.tsx",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/shared/reasoningParserAliases.ts",
    "panel/tests/api-gateway-ollama-behavior.test.ts",
    "panel/tests/api-gateway-ollama.test.ts",
    "panel/tests/api-gateway-single-model.behavior.test.ts",
    "panel/tests/image-system.test.ts",
    "panel/tests/interleaved-reasoning-render.test.ts",
    "panel/tests/model-config-registry.test.ts",
    "panel/tests/generation-defaults.test.ts",
    "panel/tests/settings-flow.test.ts",
    "tests/cross_matrix/release_regression_manifest.py",
    "tests/cross_matrix/run_api_surface_contract.py",
    "tests/cross_matrix/run_cache_architecture_contract.py",
    "tests/cross_matrix/run_current_regression_suite.py",
    "tests/cross_matrix/run_dsv4_default_cache_tool_loop_gate.py",
    "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py",
    "tests/cross_matrix/run_generation_defaults_contract.py",
    "tests/cross_matrix/run_decode_speed_gate.py",
    "tests/cross_matrix/run_installed_app_runtime_parity_audit.py",
    "tests/cross_matrix/run_issue175_177_installed_runtime_audit.py",
    "tests/cross_matrix/run_issue175_177_live_runtime_audit.py",
    "tests/cross_matrix/run_issue175_admin_sleep_probe.py",
    "tests/cross_matrix/run_issue175_179_release_boundary_audit.py",
    "tests/cross_matrix/run_issue179_minimax_k_model_manifest.py",
    "tests/cross_matrix/run_issue179_reporter_parity_metadata.py",
    "tests/cross_matrix/run_issue179_minimax_k_root_cause_audit.py",
    "tests/cross_matrix/run_issue179_responses_cancel_probe.py",
    "tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py",
    "tests/cross_matrix/run_jang_model_compat_contract.py",
    "tests/cross_matrix/run_max_output_context_contract.py",
    "tests/cross_matrix/run_mcp_policy_contract.py",
    "tests/cross_matrix/run_model_artifact_format_contract.py",
    "tests/cross_matrix/run_model_family_detection_contract.py",
    "tests/cross_matrix/run_native_mtp_contract.py",
    "tests/cross_matrix/run_noheavy_api_cache_contract.py",
    "tests/cross_matrix/run_noheavy_panel_settings_contract.py",
    "tests/cross_matrix/run_packaged_integrity_contract.py",
    "tests/cross_matrix/run_parser_registry_contract.py",
    "tests/cross_matrix/run_panel_tool_security_contract.py",
    "tests/cross_matrix/run_production_family_audit.py",
    "tests/cross_matrix/run_reasoning_template_contract.py",
    "tests/cross_matrix/run_release_regression_manifest.py",
    "tests/cross_matrix/run_release_surface_contract.py",
    "tests/cross_matrix/run_runtime_memory_stress_probe.py",
    "tests/cross_matrix/run_tool_call_contract.py",
    "tests/cross_matrix/run_vl_media_cache_contract.py",
    "tests/cross_matrix/summarize_objective_proof.py",
    "tests/test_all_local_model_smoke.py",
    "tests/test_api_surface_contract.py",
    "tests/test_batching.py",
    "tests/test_cache_architecture_contract.py",
    "tests/test_cancellation.py",
    "tests/test_current_regression_suite.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_dsv4_paged_cache.py",
    "tests/test_dsv4_default_cache_tool_loop_gate.py",
    "tests/test_dsv4_batch_generator_speed.py",
    "tests/test_dsv4_route_mode_code_exactness.py",
    "tests/test_engine_audit.py",
    "tests/test_generation_defaults_contract.py",
    "tests/test_image_api.py",
    "tests/test_image_gen.py",
    "tests/test_installed_app_runtime_parity_audit.py",
    "tests/test_issue175_177_installed_runtime_audit.py",
    "tests/test_issue175_177_live_runtime_audit.py",
    "tests/test_issue175_admin_sleep_probe.py",
    "tests/test_issue175_179_release_boundary_audit.py",
    "tests/test_issue179_minimax_k_model_manifest.py",
    "tests/test_issue179_reporter_parity_metadata.py",
    "tests/test_issue179_minimax_k_root_cause_audit.py",
    "tests/test_issue179_responses_cancel_probe.py",
    "tests/test_local_generation_metadata_audit.py",
    "tests/test_mllm_continuous_batching.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_mcp_policy_contract.py",
    "tests/test_model_config_registry.py",
    "tests/test_mlx_memory_cleanup.py",
    "tests/test_model_family_detection_contract.py",
    "tests/test_objective_proof_digest.py",
    "tests/test_packaged_integrity_contract.py",
    "tests/test_paged_cache_unit.py",
    "tests/test_panel_cli_flag_contract.py",
    "tests/test_reasoning_modes.py",
    "tests/test_release_gate_python_app.py",
    "tests/test_release_regression_manifest.py",
    "tests/test_real_ui_dsv4_memory_preflight.py",
    "tests/test_runtime_memory_stress_probe.py",
    "tests/test_scheduler_repetition_context.py",
    "tests/test_server.py",
    "tests/test_tool_format.py",
    "tests/test_tool_parsers.py",
    "tests/test_vl_media_cache_contract.py",
    "vmlx_engine/api/tool_calling.py",
    "vmlx_engine/block_disk_store.py",
    "vmlx_engine/cli.py",
    "vmlx_engine/engine/batched.py",
    "vmlx_engine/engine/simple.py",
    "vmlx_engine/image_gen.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/mllm_scheduler.py",
    "vmlx_engine/model_config_registry.py",
    "vmlx_engine/model_configs.py",
    "vmlx_engine/mlx_memory.py",
    "vmlx_engine/models/mllm.py",
    "vmlx_engine/paged_cache.py",
    "vmlx_engine/prefix_cache.py",
    "vmlx_engine/reasoning/__init__.py",
    "vmlx_engine/reasoning/think_xml_parser.py",
    "vmlx_engine/reranker.py",
    "vmlx_engine/scheduler.py",
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/__init__.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "vmlx_engine/tool_parsers/xml_function_tool_parser.py",
    "vmlx_engine/tool_parsers/zaya_tool_parser.py",
    "vmlx_engine/utils/dsv4_batch_generator.py",
    "vmlx_engine/utils/jang_loader.py",
)


@contextmanager
def _scoped_jang_tools_source(jang_tools_source: Path | None):
    if jang_tools_source is None:
        yield
        return

    keys = ("VMLX_JANG_TOOLS_SOURCE", "VMLINUX_JANG_TOOLS_SOURCE")
    old = {key: os.environ.get(key) for key in keys}
    value = str(jang_tools_source)
    try:
        for key in keys:
            os.environ[key] = value
        yield
    finally:
        for key, previous in old.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        rel: _sha256(root / rel)
        for rel in CURRENT_SUITE_SOURCE_HASH_FILES
        if (root / rel).exists()
    }


def _load_objective_digest(root: Path) -> dict[str, Any]:
    path = root / CURRENT_OBJECTIVE_DIGEST_ARTIFACT
    return json.loads(path.read_text(encoding="utf-8"))


def _open_requirements(digest: dict[str, Any]) -> list[str]:
    return [
        str(item.get("requirement"))
        for item in digest.get("requirements", [])
        if item.get("status") != "pass"
    ]


def _open_requirement_details(digest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for item in digest.get("requirements", []):
        if item.get("status") == "pass":
            continue
        requirement = str(item.get("requirement"))
        details[requirement] = {
            key: item[key]
            for key in ("status", "caveat", "evidence", "details")
            if key in item
        }
    return details


def _release_gate_failure_is_expected(step: dict[str, Any]) -> bool:
    if step["returncode"] == 0:
        return True
    text = "\n".join(step.get("stdout_tail", []))
    fail_lines = [line for line in text.splitlines() if line.startswith("[FAIL]")]
    expected_digest = (
        "[FAIL] objective proof digest: "
        + "; ".join(EXPECTED_OPEN_REQUIREMENTS)
    )
    expected_release_ready_prefix = "[FAIL] release-ready manifest: exit=1;"
    if not (
        len(fail_lines) == 2
        and fail_lines[0] == expected_digest
        and fail_lines[1].startswith(expected_release_ready_prefix)
    ):
        return False
    forbidden = (
        "bundled python import gate: FAIL",
        "panel typecheck: FAIL",
        "panel request/type tests: FAIL",
        "version triple: FAIL",
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "No such file or directory",
    )
    return not any(pattern in text for pattern in forbidden)


def _release_manifest_failure_is_expected(step: dict[str, Any], root: Path) -> bool:
    if step["returncode"] == 0:
        return True
    cmd = step.get("command")
    if not isinstance(cmd, list) or "--out" not in cmd:
        return False
    try:
        out_path = root / Path(str(cmd[cmd.index("--out") + 1]))
    except (IndexError, ValueError):
        return False
    if not out_path.exists():
        return False
    try:
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    release_clearance = manifest.get("release_clearance")
    if not isinstance(release_clearance, dict):
        return False
    open_requirements = [
        str(item) for item in release_clearance.get("open_requirements") or []
    ]
    unexpected_open = [
        item for item in open_requirements if item not in EXPECTED_OPEN_REQUIREMENTS
    ]
    missing_expected = [
        item for item in EXPECTED_OPEN_REQUIREMENTS if item not in open_requirements
    ]
    blockers = [
        item for item in release_clearance.get("blockers") or [] if isinstance(item, dict)
    ]
    blocker_ids = {str(item.get("id")) for item in blockers}
    allowed_blocker_ids = {
        "packaged_app_developer_id_signing_blocked",
        "dsv4_long_output_code_exactness_open",
        "issue175_179_release_boundary_audit",
        "installed_app_runtime_parity_audit",
        "real_ui_dsv4_memory_blocked",
        "real_ui_request_contract_proofs_stale",
        "real_ui_step37_vlm_runtime_missing",
        "real_ui_unblocked_non_mimo_missing",
        "real_ui_unblocked_non_mimo_partial",
    }
    return (
        manifest.get("release_ready") is False
        and release_clearance.get("release_ready") is False
        and not unexpected_open
        and not missing_expected
        and blocker_ids <= allowed_blocker_ids
    )


def _step_is_ok(name: str, step: dict[str, Any], root: Path) -> bool:
    if name == "release_gate_skip_app":
        return _release_gate_failure_is_expected(step)
    if name == "release_regression_manifest":
        return _release_manifest_failure_is_expected(step, root)
    return step["returncode"] == 0


def _build_summary(steps: dict[str, dict[str, Any]], root: Path) -> dict[str, Any]:
    digest = _load_objective_digest(root)
    open_requirements = _open_requirements(digest)
    open_requirement_details = _open_requirement_details(digest)
    unexpected_open = [
        item for item in open_requirements if item not in EXPECTED_OPEN_REQUIREMENTS
    ]
    missing_expected_open = [
        item for item in EXPECTED_OPEN_REQUIREMENTS if item not in open_requirements
    ]
    failed_steps = [
        name for name, step in steps.items() if not _step_is_ok(name, step, root)
    ]
    status = (
        "pass"
        if (
            not failed_steps
            and not open_requirements
            and not unexpected_open
            and not missing_expected_open
        )
        else "open"
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "known_open_requirements": EXPECTED_OPEN_REQUIREMENTS,
        "open_requirements": open_requirements,
        "open_requirement_details": open_requirement_details,
        "unexpected_open_requirements": unexpected_open,
        "missing_expected_open_requirements": missing_expected_open,
        "source_hashes": _source_hashes(root),
        "failed_steps": failed_steps,
        "steps": steps,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


CURRENT_SUITE_COMMANDS: dict[str, list[str]] = {
    "noheavy_api_cache_contract": [
        sys.executable,
        "tests/cross_matrix/run_noheavy_api_cache_contract.py",
        "--out",
        "build/current-api-cache-contract-proof-20260531-post-step-lfm-refresh.json",
    ],
    "cache_architecture_contracts": [
        sys.executable,
        "tests/cross_matrix/run_cache_architecture_contract.py",
        "--out",
        "build/current-cache-architecture-contract-20260531-step37-mixed-swa-runtime.json",
    ],
    "noheavy_panel_settings_contract": [
        sys.executable,
        "tests/cross_matrix/run_noheavy_panel_settings_contract.py",
        "--out",
        "build/current-panel-settings-contract-proof-20260528-cache-ui-matrix.json",
    ],
    "max_output_context_contracts": [
        sys.executable,
        "tests/cross_matrix/run_max_output_context_contract.py",
        "--out",
        "build/current-max-output-context-contract-20260531-post-step-lfm-refresh.json",
    ],
    "parser_registry_contracts": [
        sys.executable,
        "tests/cross_matrix/run_parser_registry_contract.py",
        "--out",
        "build/current-parser-registry-contract-20260531-post-step-lfm-refresh.json",
    ],
    "generation_defaults_contracts": [
        sys.executable,
        "tests/cross_matrix/run_generation_defaults_contract.py",
        "--out",
        "build/current-generation-defaults-contract-20260531-post-step-lfm-refresh.json",
    ],
    "reasoning_template_contracts": [
        sys.executable,
        "tests/cross_matrix/run_reasoning_template_contract.py",
        "--out",
        "build/current-reasoning-template-contract-20260526-settings-audit.json",
    ],
    "api_surface_contracts": [
        sys.executable,
        "tests/cross_matrix/run_api_surface_contract.py",
        "--out",
        "build/current-api-surface-contract-20260531-nested-epipe-childstream-refresh.json",
    ],
    "panel_tool_security_contracts": [
        sys.executable,
        "tests/cross_matrix/run_panel_tool_security_contract.py",
        "--out",
        "build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json",
    ],
    "tool_call_contracts": [
        sys.executable,
        "tests/cross_matrix/run_tool_call_contract.py",
        "--out",
        "build/current-tool-call-contract-20260528-tool-parser-loop-matrix.json",
    ],
    "mcp_policy_contracts": [
        sys.executable,
        "tests/cross_matrix/run_mcp_policy_contract.py",
        "--out",
        "build/current-mcp-policy-contract-20260531-post-step-lfm-refresh.json",
    ],
    "dsv4_source_exactness_memory_preflight": [
        sys.executable,
        "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py",
        "--memory-preflight-only",
        "--out",
        "build/current-dsv4-route-mode-code-exactness-memory-preflight-20260531-release-decision-refresh.json",
    ],
    "real_ui_dsv4_memory_preflight": [
        sys.executable,
        "tests/cross_matrix/run_real_ui_dsv4_memory_preflight.py",
        "--out",
        "build/current-real-ui-dsv4-memory-preflight-20260531-release-decision-refresh.json",
    ],
    "release_surface_contracts": [
        sys.executable,
        "tests/cross_matrix/run_release_surface_contract.py",
        "--out",
        "build/current-release-surface-contract-20260528-release-surface-matrix.json",
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
        "build/current-jang-model-compat-contract-20260528-pr155-runtime-boundary.json",
    ],
    "model_artifact_format_contracts": [
        sys.executable,
        "tests/cross_matrix/run_model_artifact_format_contract.py",
        "--out",
        "build/current-model-artifact-format-contract-20260531-post-step-lfm-refresh.json",
    ],
    "model_family_detection_contracts": [
        sys.executable,
        "tests/cross_matrix/run_model_family_detection_contract.py",
        "--out",
        "build/current-model-family-detection-contract-20260531-post-step-lfm-refresh.json",
    ],
    "native_mtp_contracts": [
        sys.executable,
        "tests/cross_matrix/run_native_mtp_contract.py",
        "--out",
        "build/current-native-mtp-contract-20260531-post-step-lfm-refresh.json",
    ],
    "vl_media_cache_contracts": [
        sys.executable,
        "tests/cross_matrix/run_vl_media_cache_contract.py",
        "--out",
        "build/current-vl-media-cache-contract-20260531-post-step-lfm-refresh.json",
    ],
    "packaged_integrity_contracts": [
        sys.executable,
        "tests/cross_matrix/run_packaged_integrity_contract.py",
        "--out",
        "build/current-packaged-integrity-contract-20260531-step37-mixed-swa-runtime.json",
    ],
    "installed_app_runtime_parity_audit": [
        sys.executable,
        "tests/cross_matrix/run_installed_app_runtime_parity_audit.py",
        "--out",
        "build/current-installed-app-runtime-parity-audit-20260531-live-epipe-refresh.json",
    ],
    "staged_app_runtime_parity_audit": [
        sys.executable,
        "tests/cross_matrix/run_installed_app_runtime_parity_audit.py",
        "--app",
        "panel/release/mac-arm64/vMLX.app",
        "--user-data",
        "build/staged-parity-user-data",
        "--diagnostic-reports",
        "build/staged-parity-diagnostic-reports",
        "--out",
        "build/current-staged-app-runtime-parity-audit-20260531-step37-mixed-swa-runtime.json",
    ],
    "issue175_179_release_boundary_audit": [
        sys.executable,
        "tests/cross_matrix/run_issue175_179_release_boundary_audit.py",
        "--out",
        "build/current-issue175-179-release-boundary-audit-20260531-post-install-sync.json",
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
        "tests/test_issue179_reporter_parity_metadata.py",
        "tests/test_model_family_detection_contract.py",
        "tests/test_mcp_policy_contract.py",
        "tests/test_vl_media_cache_contract.py",
        "tests/test_batching.py",
        "tests/test_dsv4_batch_generator_speed.py",
        "tests/test_scheduler_repetition_context.py",
        "tests/test_dsv4_paged_cache.py",
        "tests/test_dsv4_route_mode_code_exactness.py",
        "-k",
        "objective_proof_digest or default_cache_tool_loop or current_regression_suite or release_regression_manifest or issue179_reporter_parity_metadata or model_family_detection or mcp_policy_contract or decode_speed_gate or vl_media_cache_contract or dsv4_cache_hit_repetition_processor or generated_only_logits_processor or dsv4_repetition_penalty_uses_generated_only_prompt_context or dsv4_warmup or dsv4_code_exactness_probe",
    ],
    "objective_digest": [
        sys.executable,
        "tests/cross_matrix/summarize_objective_proof.py",
        "--out",
        CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
    ],
    "release_regression_manifest": [
        sys.executable,
        "tests/cross_matrix/run_release_regression_manifest.py",
        "--require-current-proof-sweep",
        "--require-release-ready",
        "--out",
        "build/current-release-regression-manifest-20260531-release-decision-refresh.json",
    ],
}


def build_suite_artifact(
    root: Path,
    *,
    include_release_gate: bool = True,
    jang_tools_source: Path | None = None,
    current_suite_artifact_path: Path | None = None,
) -> dict[str, Any]:
    steps: dict[str, dict[str, Any]] = {}
    commands = {name: list(cmd) for name, cmd in CURRENT_SUITE_COMMANDS.items()}
    if include_release_gate:
        commands["release_gate_skip_app"] = [
            "panel/scripts/release-gate-python-app.py",
            "--skip-app",
            "--skip-gui",
        ]

    with _scoped_jang_tools_source(jang_tools_source):
        for name, cmd in commands.items():
            if (
                name
                in {
                    "packaged_integrity_contracts",
                    "focused_regression_pytest",
                    "release_regression_manifest",
                }
                and current_suite_artifact_path
            ):
                current_path = current_suite_artifact_path
                if not current_path.is_absolute():
                    current_path = root / current_path
                _write_json(current_path, _build_summary(steps, root))
            steps[name] = _run_step(name, cmd, root)

    return _build_summary(steps, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-release-gate", action="store_true")
    parser.add_argument(
        "--jang-tools-source",
        type=Path,
        default=None,
        help=(
            "Clean jang-tools source checkout used for child release/bundled "
            "hash checks. Sets both VMLX_JANG_TOOLS_SOURCE and the legacy "
            "VMLINUX_JANG_TOOLS_SOURCE while running the suite."
        ),
    )
    args = parser.parse_args()

    artifact = build_suite_artifact(
        args.root,
        include_release_gate=not args.skip_release_gate,
        jang_tools_source=args.jang_tools_source,
        current_suite_artifact_path=args.out,
    )
    _write_json(args.out, artifact)
    print(args.out)
    print(f"status={artifact['status']}")
    print("open_requirements=" + json.dumps(artifact["open_requirements"]))
    print("failed_steps=" + json.dumps(artifact["failed_steps"]))
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
