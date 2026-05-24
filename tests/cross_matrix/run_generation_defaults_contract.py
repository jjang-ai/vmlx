#!/usr/bin/env python3
"""Run no-heavy generation-defaults/no-hidden-forcing contracts.

This gate protects model-owned sampling defaults across the app/engine
boundary. Bundles may declare generation defaults in ``generation_config.json``
or JANG chat metadata in ``jang_config.json``; the app must surface those
values without copying them into sticky hidden overrides, and request/API
parameters must remain explicit.
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


DEFAULT_OUT = Path("build/current-generation-defaults-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/cli.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/models.ts",
    "panel/src/main/sessions.ts",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/src/main/database.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/src/shared/sessionConfigMigrations.ts",
    "panel/tests/generation-defaults.test.ts",
    "panel/tests/settings-flow.test.ts",
    "tests/cross_matrix/run_generation_defaults_contract.py",
    "tests/cross_matrix/run_local_generation_metadata_audit.py",
    "tests/test_generation_defaults_contract.py",
    "tests/test_local_generation_metadata_audit.py",
)

REQUIRED_GENERATION_DEFAULT_TEST_MARKERS = (
    # Bundle-owned defaults surfaced from metadata.
    "uses standard MLX generation_config.json for non-JANG and VLM bundles",
    "lets JANG chat sampling metadata override generation_config.json",
    "normalizes disabled top_k sentinels to Off/0 for UI and requests",
    "uses chat repetition penalty when bundle default reasoning mode is not thinking",
    # Panel/session must not copy metadata defaults into sticky hidden startup
    # flags or stale DB config.
    "does not synthesize server --default sampling flags from UI/session config",
    "does not copy model max_new_tokens into hidden startup maxTokens config",
    "database clears legacy session maxTokens before settings UI or launch can reuse them",
    "server startup generation defaults are model-owned and not editable sliders",
    # Engine/API precedence and max-output/context separation.
    "test_chat_and_responses_log_and_forward_supported_sampling_kwargs",
    "test_request_output_caps_override_server_default_without_touching_context_cap",
    "test_request_output_caps_can_go_below_or_above_startup_default",
    "test_reasoning_effort_preserves_bundle_max_new_tokens",
    "test_omitted_server_max_tokens_uses_bundle_max_new_tokens",
    "test_omitted_server_max_tokens_without_bundle_default_is_bounded",
    "test_max_tokens_resolution_contract_applies_to_every_registered_family",
    "test_effort_does_not_synthesize_max_tokens_when_unset",
    # No hidden sampler/repetition floor in real launch command or preview.
    "test_session_command_preview_mirrors_runtime_default_flags",
    "test_local_generation_metadata_audit_reports_high_risk_rows",
    "test_local_generation_metadata_audit_flags_thinking_template_without_budget",
    "test_local_generation_metadata_audit_accepts_template_budget_support",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "panel_generation_defaults": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/generation-defaults.test.ts",
            "tests/settings-flow.test.ts",
            "--testNamePattern",
            (
                "generation_config|JANG chat sampling|disabled top_k|"
                "repetition penalty|server --default sampling|"
                "max_new_tokens|legacy session maxTokens|"
                "model-owned|runtime default flags"
            ),
            "--reporter=verbose",
        ],
    ),
    "engine_generation_defaults": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_engine_audit.py",
            "-k",
            (
                "generation_config or max_tokens_resolution_contract or sampling "
                "or output_caps or reasoning_effort_preserves_bundle "
                "or omitted_server_max_tokens or effort_does_not_synthesize "
                "or session_command_preview_mirrors_runtime_default_flags"
            ),
        ],
    ),
    "local_generation_metadata_audit": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_local_generation_metadata_audit.py",
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
        marker
        for marker in REQUIRED_GENERATION_DEFAULT_TEST_MARKERS
        if marker not in stdout
    ]
    panel_passed = results["panel_generation_defaults"]["counts"]["passed"] or 0
    engine_passed = results["engine_generation_defaults"]["counts"]["passed"] or 0
    checks = {
        "generation_config_defaults_are_surfaced": (
            not failed
            and "uses standard MLX generation_config.json for non-JANG and VLM bundles" not in missing_markers
        ),
        "jang_config_sampling_defaults_override_generation_config": (
            not failed
            and "lets JANG chat sampling metadata override generation_config.json" not in missing_markers
        ),
        "disabled_top_k_sentinels_normalize_to_off": (
            not failed
            and "normalizes disabled top_k sentinels to Off/0 for UI and requests" not in missing_markers
        ),
        "mode_specific_jang_repetition_penalty_is_metadata_owned": (
            not failed
            and "uses chat repetition penalty when bundle default reasoning mode is not thinking" not in missing_markers
        ),
        "request_api_overrides_win_over_startup_defaults": (
            not failed
            and "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" not in missing_markers
            and "test_request_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
        ),
        "bundle_max_new_tokens_preserved_when_omitted": (
            not failed
            and "test_reasoning_effort_preserves_bundle_max_new_tokens" not in missing_markers
            and "test_omitted_server_max_tokens_uses_bundle_max_new_tokens" not in missing_markers
        ),
        "omitted_max_tokens_without_bundle_default_is_bounded": (
            not failed
            and "test_omitted_server_max_tokens_without_bundle_default_is_bounded" not in missing_markers
            and "test_max_tokens_resolution_contract_applies_to_every_registered_family" not in missing_markers
        ),
        "server_default_output_cap_is_not_request_ceiling": (
            not failed
            and "test_request_output_caps_can_go_below_or_above_startup_default" not in missing_markers
        ),
        "no_hidden_sampler_forcing_or_repetition_floor": (
            not failed
            and "test_effort_does_not_synthesize_max_tokens_when_unset" not in missing_markers
            and "test_session_command_preview_mirrors_runtime_default_flags" not in missing_markers
        ),
        "local_high_risk_model_metadata_audit": (
            not failed
            and "test_local_generation_metadata_audit_reports_high_risk_rows" not in missing_markers
            and "test_local_generation_metadata_audit_flags_thinking_template_without_budget" not in missing_markers
            and "test_local_generation_metadata_audit_accepts_template_budget_support" not in missing_markers
        ),
        "panel_does_not_emit_default_sampler_cli_flags": (
            not failed
            and "does not synthesize server --default sampling flags from UI/session config" not in missing_markers
            and "does not copy model max_new_tokens into hidden startup maxTokens config" not in missing_markers
            and "database clears legacy session maxTokens before settings UI or launch can reuse them" not in missing_markers
            and "server startup generation defaults are model-owned and not editable sliders" not in missing_markers
        ),
        "legacy_count_floor_still_nontrivial": (
            not failed and panel_passed >= 10 and engine_passed >= 25
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
