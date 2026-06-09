#!/usr/bin/env python3
"""Run no-heavy tool-call parser, loop, and panel security contracts.

This gate protects the DSV4/DSML tool-call path and the app-side built-in tool
loop controls. It intentionally does not load a model; live default-cache DSV4
proof is tracked separately by ``run_dsv4_default_cache_tool_loop_gate.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-tool-call-contract-after-cross-model-loop-metrics-20260609.json")

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
    "or server_repairs_required_single_tool_bare_json_arguments "
    "or dsv4_encoder_keeps_function_arguments_as_dsml_params "
    "or dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail "
    "or dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw "
    "or qwen_issue_192_xml_string_arguments_use_request_schema "
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
    "server_repairs_required_single_tool_bare_json_arguments",
    "dsv4_encoder_keeps_function_arguments_as_dsml_params",
    "dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail",
    "responses_extracts_suppressed_reasoning_tool_calls_before_finalize",
    "qwen_issue_192_xml_string_arguments_use_request_schema",
    "tool_markup_residue_strips_all_registered_marker_families",
    "increments the auto-continue counter once per follow-up attempt",
    "resets text-chat tool streaming state before chained follow-up requests",
    "panel max tool iterations caps tool loops",
    "Responses API preserves the augmented custom system prompt when built-in tools are enabled",
    "TestToolParserManager::test_get_tool_parser_by_name",
    "TestGemma3ToolParser::test_tools_called_implies_no_markdown_in_content",
    "TestGemma4ToolParser::test_tools_called_implies_no_marker_in_content",
    "TestGlm47ToolParser::test_tools_called_implies_no_envelope_in_content",
    "TestGraniteToolParser::test_tools_called_implies_content_is_none",
    "TestHunyuanToolParser::test_visible_text_before_and_after_tool_calls_preserved",
    "TestMiniMaxToolParser::test_content_before_tool_call",
    "TestZayaToolParser::test_nested_parameter_start_repairs_missing_parameter_close",
    "TestXMLFunctionToolParser::test_visible_text_around_tool_call_has_no_xml_function_leak",
    "TestXMLFunctionToolParser::test_registry_aliases_resolve",
)

SOURCE_HASH_FILES = (
    "tests/cross_matrix/run_tool_call_contract.py",
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/abstract_tool_parser.py",
    "vmlx_engine/tool_parsers/qwen_tool_parser.py",
    "vmlx_engine/tool_parsers/auto_tool_parser.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_tool_format.py",
    "tests/test_tool_parsers.py",
    "tests/test_gemma3_tool_parser.py",
    "tests/test_gemma4_tool_parser.py",
    "tests/test_glm47_tool_parser.py",
    "tests/test_granite_tool_parser.py",
    "tests/test_hunyuan_tool_parser.py",
    "tests/test_xml_function_tool_parser.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/coding-tools.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/tests/tool-executor-security.test.ts",
    "panel/tests/tool-auto-continue.test.ts",
    "panel/tests/request-builder.test.ts",
)

RAW_TOOL_DIALECT_PATTERNS = (
    re.compile(r"<tool_call>\s*<function=", re.DOTALL),
    re.compile(r"<tool_call>\s*\{", re.DOTALL),
    re.compile(r"<[A-Za-z_][\w.-]*>\s*.*?\s*</[A-Za-z_][\w.-]*>", re.DOTALL),
    re.compile(r"<｜DSML｜tool_calls>", re.DOTALL),
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
    "engine_family_tool_parser_matrix": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_tool_parsers.py",
            "tests/test_gemma3_tool_parser.py",
            "tests/test_gemma4_tool_parser.py",
            "tests/test_glm47_tool_parser.py",
            "tests/test_granite_tool_parser.py",
            "tests/test_hunyuan_tool_parser.py",
            "tests/test_xml_function_tool_parser.py",
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


def detect_raw_tool_dialect_leak(
    *,
    visible_text: str,
    tools_supplied: bool,
    structured_tool_calls: list[dict[str, Any]] | None = None,
) -> bool:
    if not tools_supplied:
        return False
    if structured_tool_calls:
        return False
    return any(pattern.search(visible_text or "") for pattern in RAW_TOOL_DIALECT_PATTERNS)


def build_raw_tool_dialect_cases() -> dict[str, dict[str, Any]]:
    cases = {
        "step_xml_visible_leak": {
            "family": "step3p7",
            "tools_supplied": True,
            "visible_text": (
                "<tool_call><function=write_file>"
                "<parameter=path>index.html</parameter></function></tool_call>"
            ),
            "structured_tool_calls": [],
        },
        "qwen_schema_xml_converted": {
            "family": "qwen3_5_moe",
            "tools_supplied": True,
            "visible_text": "",
            "structured_tool_calls": [
                {"name": "bash", "arguments": '{"command":"echo ok"}'}
            ],
        },
        "plain_text_no_tools": {
            "family": "generic",
            "tools_supplied": False,
            "visible_text": "<tool_call><function=write_file></function></tool_call>",
            "structured_tool_calls": [],
        },
    }
    return {
        name: {
            **case,
            "raw_tool_dialect_leak": detect_raw_tool_dialect_leak(
                visible_text=str(case["visible_text"]),
                tools_supplied=bool(case["tools_supplied"]),
                structured_tool_calls=case["structured_tool_calls"],
            ),
        }
        for name, case in cases.items()
    }


def normalize_tool_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return re.sub(r"\s+", " ", arguments).strip()
    else:
        parsed = arguments
    if isinstance(parsed, dict):
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return re.sub(r"\s+", " ", str(parsed)).strip()


def tool_signature(call: dict[str, Any]) -> tuple[str, str]:
    return (
        str(call.get("name") or ""),
        normalize_tool_args(call.get("arguments") or {}),
    )


def compute_tool_loop_metrics(
    turns: list[dict[str, Any]],
    *,
    max_turns: int,
    observation_markers: tuple[str, ...],
) -> dict[str, Any]:
    signatures: list[tuple[str, str]] = []
    tool_output_seen = False
    final_text = ""
    for turn in turns:
        for call in turn.get("tool_calls") or []:
            if isinstance(call, dict):
                signatures.append(tool_signature(call))
        for output in turn.get("tool_outputs") or []:
            if output:
                tool_output_seen = True
        text = str(turn.get("final_text") or turn.get("content") or "")
        if text.strip():
            final_text = text

    counts = Counter(signatures)
    repeat_count = sum(max(0, count - 1) for count in counts.values())
    consecutive_near_duplicates = sum(
        1
        for previous, current in zip(signatures, signatures[1:])
        if previous == current
    )
    final_after_tool = tool_output_seen and bool(final_text.strip())
    used_output = final_after_tool and any(
        marker in final_text for marker in observation_markers
    )
    exhausted = len(turns) >= max_turns and not final_after_tool
    return {
        "tool_repeat_count": repeat_count,
        "same_tool_same_args_count": repeat_count,
        "near_duplicate_tool_call_count": consecutive_near_duplicates,
        "turn_budget_exhausted": exhausted,
        "final_answer_after_tool_output": final_after_tool,
        "used_tool_output_in_final_answer": used_output,
        "stopped_after_sufficient_observation": final_after_tool and used_output and not exhausted,
    }


def build_tool_loop_metric_cases() -> dict[str, dict[str, Any]]:
    repeated_loop_turns = [
        {
            "tool_calls": [{"name": "terminal", "arguments": {"command": "journalctl -u app"}}],
            "tool_outputs": ["OOM killed worker pid 42"],
        },
        {
            "tool_calls": [{"name": "terminal", "arguments": {"command": "journalctl -u app"}}],
            "tool_outputs": ["OOM killed worker pid 42"],
        },
        {
            "tool_calls": [{"name": "terminal", "arguments": {"command": "journalctl -u app"}}],
            "tool_outputs": ["OOM killed worker pid 42"],
        },
    ]
    final_turns = [
        {
            "tool_calls": [{"name": "terminal", "arguments": {"command": "journalctl -u app"}}],
            "tool_outputs": ["OOM killed worker pid 42"],
        },
        {
            "final_text": "The service failed because OOM killed worker pid 42.",
        },
    ]
    return {
        "repeated_diagnostic_loop": compute_tool_loop_metrics(
            repeated_loop_turns,
            max_turns=3,
            observation_markers=("OOM killed worker pid 42",),
        ),
        "final_answer_after_observation": compute_tool_loop_metrics(
            final_turns,
            max_turns=3,
            observation_markers=("OOM killed worker pid 42",),
        ),
    }


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    raw_tool_dialect_cases = build_raw_tool_dialect_cases()
    tool_loop_metric_cases = build_tool_loop_metric_cases()
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    stdout = "\n".join(str(result.get("stdout", "")) for result in results.values())
    missing_markers = [
        marker
        for marker in REQUIRED_TOOL_CALL_TEST_MARKERS
        if marker not in stdout
    ]
    engine_passed = results["engine_dsv4_dsml_tool_contracts"]["counts"]["passed"] or 0
    panel_passed = results["panel_tool_loop_security"]["counts"]["passed"] or 0
    family_passed = results["engine_family_tool_parser_matrix"]["counts"]["passed"] or 0
    live_default_cache_artifact = root / "build/current-dsv4-default-cache-tool-loop/result.json"
    live_default_cache_status = None
    live_default_cache_runtime_checks: dict[str, Any] = {}
    if live_default_cache_artifact.exists():
        try:
            live_default_cache_payload = json.loads(
                live_default_cache_artifact.read_text(encoding="utf-8")
            )
            live_default_cache_status = live_default_cache_payload.get("status")
            payload_checks = live_default_cache_payload.get("checks")
            if isinstance(payload_checks, dict):
                live_default_cache_runtime_checks = payload_checks
        except (OSError, json.JSONDecodeError, TypeError):
            live_default_cache_status = "unreadable"
    required_live_runtime_checks = (
        "tool_sequence_ordered",
        "final_done",
        "file_written",
        "native_cache",
        "native_prefix",
        "native_paged",
        "native_l2",
        "generic_tq_kv_off",
        "cached_tokens_seen",
        "dsv4_cache_detail_seen",
    )
    live_default_cache_runtime_passed = (
        live_default_cache_status == "pass"
        or (
            live_default_cache_status == "review"
            and all(
                live_default_cache_runtime_checks.get(key) is True
                for key in required_live_runtime_checks
            )
        )
    )
    checks = {
        "tool_parser_residue_rejected_instead_of_executed": not failed and engine_passed >= 21,
        "schema_valid_dsml_tool_call_preserved": not failed and engine_passed >= 21,
        "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid": not failed and engine_passed >= 21,
        "dsv4_tool_preamble_suppressed_and_not_stored_without_call": not failed and engine_passed >= 21,
        "tool_choice_none_does_not_fallback_to_raw_dsml": not failed and engine_passed >= 21,
        "required_single_tool_bare_json_arguments_repaired": (
            not failed
            and "server_repairs_required_single_tool_bare_json_arguments"
            not in missing_markers
        ),
        "panel_tool_executor_blocks_unsafe_paths_and_commands": not failed and panel_passed >= 10,
        "panel_max_tool_iterations_caps_tool_loops": not failed and panel_passed >= 10,
        "family_tool_parser_matrix_covers_no_leak_and_alias_edges": (
            not failed and family_passed >= 125
        ),
        "raw_tool_dialect_leak_metrics_present": (
            raw_tool_dialect_cases["step_xml_visible_leak"]["raw_tool_dialect_leak"]
            is True
            and raw_tool_dialect_cases["qwen_schema_xml_converted"][
                "raw_tool_dialect_leak"
            ]
            is False
        ),
        "tool_loop_control_metrics_present": (
            set(tool_loop_metric_cases["repeated_diagnostic_loop"])
            >= {
                "tool_repeat_count",
                "same_tool_same_args_count",
                "near_duplicate_tool_call_count",
                "turn_budget_exhausted",
                "final_answer_after_tool_output",
                "used_tool_output_in_final_answer",
                "stopped_after_sufficient_observation",
            }
            and tool_loop_metric_cases["repeated_diagnostic_loop"][
                "turn_budget_exhausted"
            ]
            is True
            and tool_loop_metric_cases["final_answer_after_observation"][
                "stopped_after_sufficient_observation"
            ]
            is True
        ),
        "live_default_cache_dsv4_tool_loop_artifact_present": live_default_cache_artifact.exists(),
        "live_default_cache_dsv4_tool_loop_artifact_passed": live_default_cache_runtime_passed,
        "all_required_tool_call_markers_present": not failed and not missing_markers,
    }
    source_checks = {
        key: value
        for key, value in checks.items()
        if key
        not in {
            "live_default_cache_dsv4_tool_loop_artifact_present",
            "live_default_cache_dsv4_tool_loop_artifact_passed",
        }
    }
    live_default_cache_open = not live_default_cache_runtime_passed
    status = (
        "pass"
        if all(checks.values())
        else "open"
        if all(source_checks.values()) and live_default_cache_open
        else "fail"
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "checks": checks,
        "failed": failed,
        "missing_markers": missing_markers,
        "open_proof_gaps": [
            "live_default_cache_dsv4_tool_loop"
        ]
        if status == "open"
        else [],
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": results,
        "raw_tool_dialect_cases": raw_tool_dialect_cases,
        "tool_loop_metric_cases": tool_loop_metric_cases,
        "live_default_cache_artifact": str(live_default_cache_artifact),
        "live_default_cache_artifact_status": live_default_cache_status,
        "live_default_cache_runtime_checks": live_default_cache_runtime_checks,
        "live_default_cache_required_runtime_checks": list(required_live_runtime_checks),
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
