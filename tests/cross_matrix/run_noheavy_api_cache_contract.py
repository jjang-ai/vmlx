#!/usr/bin/env python3
"""Run the current-source no-heavy API/cache contract proof.

This helper intentionally does not load models. It runs focused tests that pin:

- OpenAI Chat Completions and Responses request sampling/default propagation;
- Anthropic and Ollama adapter surfaces;
- DSV4 native SWA+CSA/HCA cache status separate from generic TurboQuant KV;
- ZAYA typed CCA status;
- hybrid SSM partial reuse and cache-detail telemetry;
- TurboQuant KV runtime and disk-cache serialization contracts.
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


DEFAULT_OUT = Path("build/current-api-cache-contract-proof-20260521.json")
SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "tests/test_engine_audit.py",
    "tests/test_batching.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_tq_disk_cache.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_tool_format.py",
)

COMMANDS: dict[str, list[str]] = {
    "api_route_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_engine_audit.py",
        "-k",
        (
            "chat_and_responses_log_and_forward_supported_sampling_kwargs "
            "or anthropic_messages_omitted_max_tokens_uses_bundle_default "
            "or request_output_caps_override_server_default_without_touching_context_cap "
            "or responses_request_has_sampling_fields "
            "or media_diag_hooks_cover_anthropic_and_ollama_streaming_ingress "
            "or responses_nonstreaming_forwards_tc_id "
            "or chat_completions_nonstreaming "
            "or responses_nonstreaming "
            "or chat_completions_streaming "
            "or responses_streaming "
            "or generic_turboquant_patcher_skips_hybrid_ssm "
            "or ollama_streaming_suppresses_duplicate_done_chunks "
            "or cache_stats_endpoint_projects_cache_reuse_skip_telemetry "
            "or native_cache_status_reports_dsv4_separately_from_tq_kv "
            "or native_cache_status_reports_zaya_typed_cca"
        ),
    ],
    "scheduler_cache_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_batching.py",
        "-k",
        "dsv4 or hybrid_ssm or cache_detail",
    ],
    "tq_and_mllm_cache_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_tq_disk_cache.py",
        "-k",
        "TurboQuant or turboquant or hybrid_ssm or TQ",
    ],
    "dsv4_dsml_tool_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_dsml_tool_parser.py",
        "tests/test_tool_format.py",
        "tests/test_engine_audit.py",
        "-k",
        (
            "visible_text_around_invoke_preserved_no_dsml_leak "
            "or tools_called_implies_no_dsml_in_content "
            "or repairs_dsml_invoke_with_plain_param_tags "
            "or repairs_dsml_tool_calls_wrapper_with_truncated_plain_param_invokes "
            "or streaming_buffers_dsml_tool_calls_wrapper_before_invoke "
            "or server_buffers_dsml_wrapper_marker_before_invoke "
            "or dsml_parser_repairs_schema_gated_malformed_old_dsv4_tool_call "
            "or dsml_parser_repairs_partial_canonical_invoke "
            "or dsml_parser_repairs_dsv4_live_degraded_dsml_params "
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
        ),
    ],
}


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


def build_artifact(root: Path) -> dict[str, Any]:
    commands = {
        name: _run_command(name, cmd, root)
        for name, cmd in COMMANDS.items()
    }
    api_ok = commands["api_route_contracts"]["returncode"] == 0
    scheduler_ok = commands["scheduler_cache_contracts"]["returncode"] == 0
    tq_ok = commands["tq_and_mllm_cache_contracts"]["returncode"] == 0
    dsml_ok = commands["dsv4_dsml_tool_contracts"]["returncode"] == 0
    checks = {
        "openai_chat_sampling_kwargs": api_ok,
        "responses_sampling_kwargs": api_ok,
        "request_output_caps_override_server_default": api_ok,
        "prompt_context_caps_stay_separate_from_output_caps": api_ok,
        "anthropic_bundle_defaults": api_ok,
        "ollama_adapter_surface": api_ok,
        "dsv4_native_cache_status": api_ok and scheduler_ok,
        "dsv4_dsml_parser_residue_rejection": dsml_ok,
        "dsv4_dsml_valid_tool_call_preserved": dsml_ok,
        "dsv4_suppressed_tool_markup_not_stored": dsml_ok,
        "zaya_typed_cca_status": api_ok,
        "hybrid_ssm_partial_reuse": scheduler_ok and tq_ok,
        "turboquant_kv_runtime_contract": tq_ok,
        "turboquant_disk_roundtrip": tq_ok,
        "no_generic_tq_on_hybrid_ssm": api_ok,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
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
    for name, result in artifact["commands"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
