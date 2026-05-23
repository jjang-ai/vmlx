#!/usr/bin/env python3
"""Run no-heavy tool-call parser, loop, and panel security contracts.

This gate protects the DSV4/DSML tool-call path and the app-side built-in tool
loop controls. It intentionally does not load a model; live default-cache DSV4
proof is tracked separately by ``run_dsv4_default_cache_tool_loop_gate.py``.
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


DEFAULT_OUT = Path("build/current-tool-call-contract-20260521.json")

DSML_PATTERN = (
    "visible_text_around_invoke_preserved_no_dsml_leak "
    "or tools_called_implies_no_dsml_in_content "
    "or repairs_dsml_invoke_with_plain_param_tags "
    "or repairs_dsml_tool_calls_wrapper_with_truncated_plain_param_invokes "
    "or streaming_buffers_dsml_tool_calls_wrapper_before_invoke "
    "or server_buffers_dsml_wrapper_marker_before_invoke "
    "or dsml_parser_repairs_schema_gated_malformed_old_dsv4_tool_call "
    "or dsml_parser_repairs_partial_canonical_invoke "
    "or dsml_parser_repairs_dsv4_live_degraded_dsml_params "
    "or dsml_parser_rejects_canonical_attr_residue_and_repairs_live_write_file "
    "or dsml_parser_repairs_partial_invoke_with_malformed_value_attr "
    "or dsml_parser_repairs_htmlish_invoke_degradation "
    "or server_repairs_dsv4_partial_tool_intent_from_request_args "
    "or dsv4_encoder_keeps_function_arguments_as_dsml_params "
    "or dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail "
    "or dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw "
    "or dsv4_fallback_tool_prompt_uses_canonical_tool_calls_wrapper "
    "or dsv4_native_schema_prompt_with_only_generic_examples_gets_concrete_fallback "
    "or tool_markup_residue_strips_all_registered_marker_families "
    "or tool_markup_residue_strips_interrupted_non_dsml_fragments "
    "or responses_api_no_fallback_on_suppressed_reasoning "
    "or responses_extracts_suppressed_reasoning_tool_calls_before_finalize"
)

REQUIRED_TOOL_CALL_TEST_MARKERS = (
    "visible_text_around_invoke_preserved_no_dsml_leak",
    "tools_called_implies_no_dsml_in_content",
    "dsml_parser_repairs_dsv4_live_degraded_dsml_params",
    "dsml_parser_rejects_canonical_attr_residue_and_repairs_live_write_file",
    "dsml_parser_repairs_partial_invoke_with_malformed_value_attr",
    "server_repairs_dsv4_partial_tool_intent_from_request_args",
    "dsv4_encoder_keeps_function_arguments_as_dsml_params",
    "dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail",
    "responses_extracts_suppressed_reasoning_tool_calls_before_finalize",
    "tool_markup_residue_strips_all_registered_marker_families",
    "increments the auto-continue counter once per follow-up attempt",
    "resets text-chat tool streaming state before chained follow-up requests",
    "panel max tool iterations caps tool loops",
    "Responses API preserves the augmented custom system prompt when built-in tools are enabled",
)

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_tool_format.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/coding-tools.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/tests/tool-executor-security.test.ts",
    "panel/tests/tool-auto-continue.test.ts",
    "panel/tests/request-builder.test.ts",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_dsv4_dsml_tool_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_dsml_tool_parser.py",
            "tests/test_tool_format.py",
            "tests/test_engine_audit.py",
            "-k",
            DSML_PATTERN,
        ],
    ),
    "panel_tool_loop_security": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/tool-executor-security.test.ts",
            "tests/tool-auto-continue.test.ts",
            "tests/request-builder.test.ts",
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
        marker
        for marker in REQUIRED_TOOL_CALL_TEST_MARKERS
        if marker not in stdout
    ]
    engine_passed = results["engine_dsv4_dsml_tool_contracts"]["counts"]["passed"] or 0
    panel_passed = results["panel_tool_loop_security"]["counts"]["passed"] or 0
    live_default_cache_artifact = root / "build/current-dsv4-default-cache-tool-loop/result.json"
    checks = {
        "tool_parser_residue_rejected_instead_of_executed": not failed and engine_passed >= 21,
        "schema_valid_dsml_tool_call_preserved": not failed and engine_passed >= 21,
        "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid": not failed and engine_passed >= 21,
        "dsv4_tool_preamble_suppressed_and_not_stored_without_call": not failed and engine_passed >= 21,
        "tool_choice_none_does_not_fallback_to_raw_dsml": not failed and engine_passed >= 21,
        "panel_tool_executor_blocks_unsafe_paths_and_commands": not failed and panel_passed >= 10,
        "panel_max_tool_iterations_caps_tool_loops": not failed and panel_passed >= 10,
        "live_default_cache_dsv4_tool_loop_artifact_present": live_default_cache_artifact.exists(),
        "all_required_tool_call_markers_present": not failed and not missing_markers,
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
        "results": results,
        "live_default_cache_artifact": str(live_default_cache_artifact),
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
