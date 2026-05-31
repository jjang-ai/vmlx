#!/usr/bin/env python3
"""Run no-heavy reasoning/template leak contracts.

This gate protects reasoning rail behavior across server parsers and panel
rendering: explicit on/off controls, visible-content extraction, think-tag leak
prevention, interleaved reasoning display, and reasoning/tool interaction.
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


DEFAULT_OUT = Path("build/current-reasoning-template-contract-20260526-settings-audit.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/reasoning/__init__.py",
    "vmlx_engine/reasoning/deepseek_r1_parser.py",
    "vmlx_engine/reasoning/gemma4_parser.py",
    "tests/test_deepseek_r1_reasoning_no_leak.py",
    "tests/test_gemma4_reasoning_no_leak.py",
    "tests/test_reasoning_modes.py",
    "tests/test_reasoning_tool_interaction.py",
    "panel/src/main/ipc/chat.ts",
    "panel/src/renderer/src/components/chat/ChatMessage.tsx",
    "panel/src/renderer/src/components/chat/ReasoningBlock.tsx",
    "panel/tests/reasoning-display.test.ts",
    "panel/tests/interleaved-reasoning-render.test.ts",
    "panel/tests/interleaved-reasoning-segments.test.ts",
    "tests/cross_matrix/run_reasoning_template_contract.py",
    "tests/test_reasoning_template_contract.py",
)

REQUIRED_REASONING_TEMPLATE_TEST_MARKERS = (
    # Public API rails. These prevent unsafe hidden downgrades: DSV4 must
    # preserve requested reasoning-on/effort rails, and must use its audited
    # quality-safe rail when direct/off is known to corrupt identifiers.
    "test_dsv4_reasoning_effort_preserves_requested_rails",
    "test_dsv4_thinking_policy_does_not_force_tool_calls_to_direct_rail",
    "test_dsv4_bundle_defaults_apply_only_when_request_omits_values",
    "test_minimax_m2_preserves_sampling_values_without_family_floor",
    "test_ling_suppresses_reasoning_parser_and_stale_think_in_template",
    # Family parser no-leak rows. These are the model-family edges users see as
    # leaked think/channel markers or wrong visible content after long/tool runs.
    "test_hy_v3_qwen3_reasoning_parser_no_think_does_not_leak_tags",
    "test_deepseek_r1_reasoning_parser_orphan_close_is_not_visible",
    "test_gemma4_reasoning_parser_orphan_channel_close_is_not_visible",
    "test_tools_called_implies_no_channel_marker_in_content",
    "test_used_by_documented_families",
    # Tool/reasoning interaction. These protect chained tool calls from stale
    # reasoning state, tool markup leakage, and parser order regressions.
    "test_implicit_reasoning_on_tool_followup",
    "test_qwen3_reasoning_then_multiple_tool_calls",
    "test_think_tags_stripped_before_parsing",
    # Panel/UI stream behavior. These make the gate prove the actual UI path:
    # server reasoning fields, fallback extraction, token counts, Responses API
    # events, tool-iteration reset, and interleaved reasoning rendering.
    "content with <think> tags but server provides reasoning_content — no double-extraction",
    "response.output_text.delta also triggers reasoningDone if was reasoning",
    "Responses: local Auto omits enable_thinking so engine auto-detects",
    "Tool iteration: reasoning in iteration 1, tool call, clean iteration 2",
    "server-side DeepSeek reasoning_content handles implicit correctly",
    "replaces old reasoning segments during live interleaved streaming, then can show all after completion",
    "live-replaces previous reasoning segments while streaming and shows all after completion",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_reasoning_template": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_deepseek_r1_reasoning_no_leak.py",
            "tests/test_gemma4_reasoning_no_leak.py",
            "tests/test_reasoning_modes.py",
            "tests/test_reasoning_tool_interaction.py",
        ],
    ),
    "panel_reasoning_rendering": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/reasoning-display.test.ts",
            "tests/interleaved-reasoning-render.test.ts",
            "tests/interleaved-reasoning-segments.test.ts",
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
        for marker in REQUIRED_REASONING_TEMPLATE_TEST_MARKERS
        if marker not in stdout
    ]
    engine_passed = results["engine_reasoning_template"]["counts"]["passed"] or 0
    panel_passed = results["panel_reasoning_rendering"]["counts"]["passed"] or 0
    checks = {
        "reasoning_on_off_request_wiring_explicit": (
            not failed
            and "test_dsv4_reasoning_effort_preserves_requested_rails" not in missing_markers
            and "Responses: local Auto omits enable_thinking so engine auto-detects" not in missing_markers
        ),
        "no_hidden_family_sampling_or_reasoning_forcing": (
            not failed
            and "test_dsv4_bundle_defaults_apply_only_when_request_omits_values" not in missing_markers
            and "test_minimax_m2_preserves_sampling_values_without_family_floor" not in missing_markers
            and "test_ling_suppresses_reasoning_parser_and_stale_think_in_template" not in missing_markers
        ),
        "deepseek_r1_visible_content_no_think_tag_leak": (
            not failed
            and "test_deepseek_r1_reasoning_parser_orphan_close_is_not_visible" not in missing_markers
            and "test_used_by_documented_families" not in missing_markers
            and "server-side DeepSeek reasoning_content handles implicit correctly" not in missing_markers
        ),
        "gemma4_visible_content_no_channel_marker_leak": (
            not failed
            and "test_gemma4_reasoning_parser_orphan_channel_close_is_not_visible" not in missing_markers
            and "test_tools_called_implies_no_channel_marker_in_content" not in missing_markers
        ),
        "hy3_zaya_style_no_think_no_visible_tag_leak": (
            not failed
            and "test_hy_v3_qwen3_reasoning_parser_no_think_does_not_leak_tags" not in missing_markers
        ),
        "reasoning_tool_interaction_preserves_tool_calls": (
            not failed
            and "test_dsv4_thinking_policy_does_not_force_tool_calls_to_direct_rail" not in missing_markers
            and "test_implicit_reasoning_on_tool_followup" not in missing_markers
            and "test_qwen3_reasoning_then_multiple_tool_calls" not in missing_markers
            and "test_think_tags_stripped_before_parsing" not in missing_markers
        ),
        "streaming_fallback_does_not_double_extract_reasoning": (
            not failed
            and "content with <think> tags but server provides reasoning_content — no double-extraction" not in missing_markers
        ),
        "interleaved_reasoning_segments_render": (
            not failed
            and "replaces old reasoning segments during live interleaved streaming, then can show all after completion" not in missing_markers
            and "live-replaces previous reasoning segments while streaming and shows all after completion" not in missing_markers
        ),
        "visible_token_counts_not_corrupted_by_reasoning_ui": (
            not failed
            and "response.output_text.delta also triggers reasoningDone if was reasoning" not in missing_markers
            and "Tool iteration: reasoning in iteration 1, tool call, clean iteration 2" not in missing_markers
        ),
        "legacy_count_floor_still_nontrivial": (
            not failed and engine_passed >= 126 and panel_passed >= 108
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
