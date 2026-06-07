#!/usr/bin/env python3
"""Run no-heavy parser registry parity contracts.

This gate protects the app/engine boundary that previously regressed MiniMax:
the panel must not emit tool or reasoning parser ids that the engine CLI cannot
accept, and family autodetection must keep parser ids canonical.
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


DEFAULT_OUT = Path("build/current-parser-registry-contract-after-jangtq2-objective-refresh-20260607.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/cli.py",
    "vmlx_engine/reasoning/__init__.py",
    "vmlx_engine/model_config_registry.py",
    "vmlx_engine/tool_parsers/__init__.py",
    "panel/src/main/model-config-registry.ts",
    "panel/src/shared/reasoningParserAliases.ts",
    "panel/src/shared/toolParserAliases.ts",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/tests/model-config-registry.test.ts",
    "panel/tests/settings-flow.test.ts",
    "tests/cross_matrix/run_decode_speed_gate.py",
    "tests/cross_matrix/run_parser_registry_contract.py",
    "tests/test_engine_audit.py",
    "tests/test_model_config_registry.py",
    "tests/test_parser_registry_contract.py",
)

REQUIRED_PARSER_TEST_MARKERS = (
    # Engine-side registry/CLI parity. These rows caught the MiniMax M2 parser
    # drift where the panel emitted a parser id the engine argparse rejected.
    "test_all_reasoning_parsers_registered",
    "test_all_reasoning_parsers_instantiable",
    "test_all_reasoning_parsers_valid",
    "test_all_tool_parsers_valid",
    "test_cli_tool_parser_choices_cover_family_registry_parsers",
    "test_cli_reasoning_parser_choices_cover_family_registry_parsers",
    # Family-specific parser contracts. These are the rows users actually hit
    # when auto-detection resolves a launch config from local model metadata.
    "test_registry_overrides_stale_minimax_qwen3_sidecar",
    "test_minimax_eos_includes_role_boundary_marker",
    "test_zaya1_vl_registered_with_full_contract",
    "test_ling_is_not_a_reasoning_model_per_eric_2026_05_11",
    "test_qwen2_must_not_have_reasoning",
    "test_qwen2_vl_must_not_have_reasoning",
    "test_gemma3_reasoning_parser",
    "test_glm_flash_vs_base_reasoning_parser_differs",
    "test_deepseek_v4_eos_includes_latest_reminder",
    "test_lfm2_registered",
    "test_lfm2_moe_config",
    # Panel-side launch and UI parity. These protect aliases/dropdowns and the
    # command builder, not only backend Python config.
    "canonicalizes legacy DSV4 and Hy3 parser aliases before launch",
    "tool parser dropdown exposes DSV4 DSML, Hy3, and ZAYA parsers",
    "tool parser dropdown exposes Liquid LFM2 parser",
    "reasoning parser dropdown covers every parser the panel registry can emit",
    "passes MiniMax through the registered minimax_m2 reasoning parser",
    "uses the registered MiniMax reasoning parser even when bundle sidecars say qwen3",
    "detects Hy3 as text-only KV with Hunyuan tools and qwen3 reasoning",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_parser_registry": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_engine_audit.py",
            "tests/test_model_config_registry.py",
            "-k",
            "parser or minimax or reasoning or reasoning_parser or zaya or ling or lfm2 or deepseek_v4_eos or role_boundary",
        ],
    ),
    "panel_parser_registry": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/model-config-registry.test.ts",
            "tests/settings-flow.test.ts",
            "--testNamePattern",
            "parser|Parser|reasoning|Reasoning|minimax|MiniMax|LFM2|lfm2",
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
        marker for marker in REQUIRED_PARSER_TEST_MARKERS if marker not in stdout
    ]
    engine_passed = results["engine_parser_registry"]["counts"]["passed"] or 0
    panel_passed = results["panel_parser_registry"]["counts"]["passed"] or 0
    checks = {
        "engine_accepts_registered_reasoning_parsers": (
            not failed
            and "test_all_reasoning_parsers_registered" not in missing_markers
            and "test_all_reasoning_parsers_valid" not in missing_markers
            and "test_cli_reasoning_parser_choices_cover_family_registry_parsers" not in missing_markers
        ),
        "engine_accepts_registered_tool_parsers": (
            not failed
            and "test_all_tool_parsers_valid" not in missing_markers
            and "test_cli_tool_parser_choices_cover_family_registry_parsers" not in missing_markers
        ),
        "panel_emitted_reasoning_parsers_are_engine_valid": (
            not failed
            and "passes MiniMax through the registered minimax_m2 reasoning parser" not in missing_markers
            and "uses the registered MiniMax reasoning parser even when bundle sidecars say qwen3" not in missing_markers
        ),
        "panel_emitted_tool_parsers_are_engine_valid": (
            not failed
            and "tool parser dropdown exposes DSV4 DSML, Hy3, and ZAYA parsers" not in missing_markers
        ),
        "minimax_m2_reasoning_parser_regression": (
            not failed
            and "test_registry_overrides_stale_minimax_qwen3_sidecar" not in missing_markers
            and "test_minimax_eos_includes_role_boundary_marker" not in missing_markers
            and "passes MiniMax through the registered minimax_m2 reasoning parser" not in missing_markers
        ),
        "parser_aliases_are_canonical_before_cli": (
            not failed
            and "canonicalizes legacy DSV4 and Hy3 parser aliases before launch" not in missing_markers
        ),
        "zaya_hy3_ling_dsv4_parser_rows_are_present": (
            not failed
            and "test_zaya1_vl_registered_with_full_contract" not in missing_markers
            and "test_ling_is_not_a_reasoning_model_per_eric_2026_05_11" not in missing_markers
            and "test_deepseek_v4_eos_includes_latest_reminder" not in missing_markers
            and "test_lfm2_moe_config" not in missing_markers
            and "test_lfm2_registered" not in missing_markers
            and "detects Hy3 as text-only KV with Hunyuan tools and qwen3 reasoning" not in missing_markers
        ),
        "non_reasoning_family_boundaries_are_present": (
            not failed
            and "test_qwen2_must_not_have_reasoning" not in missing_markers
            and "test_qwen2_vl_must_not_have_reasoning" not in missing_markers
            and "test_gemma3_reasoning_parser" not in missing_markers
            and "test_glm_flash_vs_base_reasoning_parser_differs" not in missing_markers
        ),
        "all_required_parser_markers_present": not failed and not missing_markers,
        "legacy_count_floor_still_nontrivial": (
            not failed and engine_passed >= 30 and panel_passed >= 35
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
