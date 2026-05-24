#!/usr/bin/env python3
"""Run the current-source no-heavy panel settings contract proof.

This helper intentionally does not load models. It runs focused Electron panel
tests that pin the settings behavior Eric called out:

- DSV4 has one native composite prefix-cache switch under Prefix Cache;
- explicit DSV4 prefix-cache opt-out disables DSV4/paged/L2 launch flags;
- DSV4 L2 can be disabled without falling back to generic KV codecs;
- Server Default Max Output Tokens maps to --max-tokens;
- Max Context Tokens maps to --max-prompt-tokens;
- Chat Max Output Tokens remains a per-chat/API override;
- non-DSV4 cache toggles keep their normal prefix/paged/disk semantics;
- i18n copy keeps the naming consistent.
- panel and engine model-family detection cover MiniMax reasoning parser,
  JANG/JANGTQ/MTP, hybrid SSM, VLM, and related parser defaults.
- launch-memory admission keeps lazy-mmap JANG/JANGTQ bundles from being
  falsely blocked by macOS cache pressure while preserving real hard blocks.
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


DEFAULT_OUT = Path("build/current-panel-settings-contract-proof-20260521.json")
SOURCE_HASH_FILES = (
    "panel/src/main/sessions.ts",
    "panel/src/main/memory-enforcer.ts",
    "panel/src/main/model-config-registry.ts",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/CreateSession.tsx",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/src/shared/dsv4Env.ts",
    "panel/src/shared/cacheControlPolicy.ts",
    "panel/src/shared/reasoningParserAliases.ts",
    "panel/src/shared/sessionConfigMigrations.ts",
    "panel/src/renderer/src/i18n/locales/en.json",
    "panel/src/renderer/src/i18n/locales/es.json",
    "panel/src/renderer/src/i18n/locales/ja.json",
    "panel/src/renderer/src/i18n/locales/ko.json",
    "panel/src/renderer/src/i18n/locales/zh.json",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/dsv4-env.test.ts",
    "panel/tests/cache-control-policy.test.ts",
    "panel/tests/generation-defaults.test.ts",
    "panel/tests/i18n-consistency.test.ts",
    "panel/tests/load-progress-honesty.test.ts",
    "panel/tests/model-config-registry.test.ts",
    "tests/test_model_config_registry.py",
    "tests/test_panel_cli_flag_contract.py",
)

REQUIRED_PANEL_SETTINGS_SOURCE_MARKERS = (
    "DSV4 pool quant and native prefix controls stay DSV4-only",
    "launch memory admission is warning-only for lazy-mmap bundles",
    "test_panel_serve_flags_are_registered_engine_cli_flags",
)

COMMANDS: dict[str, list[str]] = {
    "panel_settings_contracts": [
        "npx",
        "vitest",
        "run",
        "tests/settings-flow.test.ts",
        "tests/dsv4-env.test.ts",
        "tests/cache-control-policy.test.ts",
        "tests/generation-defaults.test.ts",
        "tests/i18n-consistency.test.ts",
        "tests/load-progress-honesty.test.ts",
    ],
    "panel_typecheck": ["npm", "run", "typecheck"],
    "panel_model_config_registry": [
        "npx",
        "vitest",
        "run",
        "tests/model-config-registry.test.ts",
    ],
    "engine_model_config_registry": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "../tests/test_model_config_registry.py",
    ],
    "engine_panel_cli_flag_contract": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "../tests/test_panel_cli_flag_contract.py",
    ],
}


def _parse_counts(output: str) -> dict[str, int | None]:
    test_files_passed = None
    tests_passed = None
    tests_skipped = None
    match = re.search(r"Test Files\s+(\d+) passed", output)
    if match:
        test_files_passed = int(match.group(1))
    match = re.search(r"Tests\s+(\d+) passed", output)
    if match:
        tests_passed = int(match.group(1))
    if tests_passed is None:
        match = re.search(r"(\d+) passed", output)
        if match:
            tests_passed = int(match.group(1))
    match = re.search(r"Tests\s+.*?(\d+) skipped", output)
    if match:
        tests_skipped = int(match.group(1))
    return {
        "test_files_passed": test_files_passed,
        "tests_passed": tests_passed,
        "tests_skipped": tests_skipped,
    }


def _run_command(name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
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
        "counts": _parse_counts(output),
        "stdout_tail": output.splitlines()[-40:],
    }


def source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in SOURCE_HASH_FILES:
        path = root / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def missing_source_markers(root: Path) -> list[str]:
    text = "\n".join(
        (root / rel).read_text(encoding="utf-8")
        for rel in (
            "panel/tests/settings-flow.test.ts",
            "panel/tests/dsv4-env.test.ts",
            "tests/test_panel_cli_flag_contract.py",
        )
    )
    return [
        marker
        for marker in REQUIRED_PANEL_SETTINGS_SOURCE_MARKERS
        if marker not in text
    ]


def build_artifact(root: Path) -> dict[str, Any]:
    panel_root = root / "panel"
    commands = {
        name: _run_command(name, cmd, panel_root)
        for name, cmd in COMMANDS.items()
    }
    settings_ok = commands["panel_settings_contracts"]["returncode"] == 0
    settings_counts = commands["panel_settings_contracts"]["counts"]
    model_counts = commands["panel_model_config_registry"]["counts"]
    engine_model_ok = commands["engine_model_config_registry"]["returncode"] == 0
    engine_model_counts = commands["engine_model_config_registry"]["counts"]
    cli_flag_ok = commands["engine_panel_cli_flag_contract"]["returncode"] == 0
    settings_coverage_ok = (
        settings_counts["test_files_passed"] == 6
        and (settings_counts["tests_passed"] or 0) >= 260
    )
    panel_model_ok = (
        commands["panel_model_config_registry"]["returncode"] == 0
        and model_counts["test_files_passed"] == 1
        and (model_counts["tests_passed"] or 0) >= 50
    )
    engine_model_coverage_ok = engine_model_ok and (engine_model_counts["tests_passed"] or 0) >= 120
    typecheck_ok = commands["panel_typecheck"]["returncode"] == 0
    missing_markers = missing_source_markers(root)
    checks = {
        "panel_settings_contract_count": settings_ok and settings_coverage_ok and not missing_markers,
        "dsv4_default_native_prefix_on": settings_ok and settings_coverage_ok,
        "dsv4_explicit_prefix_off_disables_native_flags": settings_ok and settings_coverage_ok,
        "dsv4_l2_explicit_off_preserves_prefix": settings_ok and settings_coverage_ok,
        "dsv4_generic_kv_flags_suppressed": settings_ok and settings_coverage_ok,
        "dsv4_pool_quant_controls_are_dsv4_only": (
            settings_ok and settings_coverage_ok and not missing_markers
        ),
        "max_output_context_cli_split": settings_ok and settings_coverage_ok,
        "chat_max_output_is_per_chat_override": settings_ok and settings_coverage_ok,
        "non_dsv4_cache_toggles_preserved": settings_ok and settings_coverage_ok,
        "i18n_max_output_context_copy": settings_ok and settings_coverage_ok,
        "lazy_mmap_launch_memory_admission": settings_ok and settings_coverage_ok,
        "minimax_reasoning_parser_detection": (
            settings_ok and settings_coverage_ok and panel_model_ok and engine_model_coverage_ok
        ),
        "native_mtp_d3_default_policy": panel_model_ok,
        "model_family_parser_registry": panel_model_ok and engine_model_coverage_ok,
        "engine_cache_architecture_registry": engine_model_coverage_ok,
        "panel_emitted_flags_are_registered_engine_cli_flags": cli_flag_ok,
        "panel_typecheck": typecheck_ok,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
        "missing_source_markers": missing_markers,
        "source_hashes": source_hashes(root),
        "commands": commands,
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
    print("missing_source_markers=" + json.dumps(artifact["missing_source_markers"]))
    for name, result in artifact["commands"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"test_files_passed={counts['test_files_passed']} "
            f"tests_passed={counts['tests_passed']} "
            f"tests_skipped={counts['tests_skipped']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
