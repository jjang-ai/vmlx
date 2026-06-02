#!/usr/bin/env python3
"""Run no-heavy API surface parity contracts.

This gate protects the API surfaces Eric called out directly: OpenAI Chat
Completions, OpenAI Responses, Anthropic messages, Ollama gateway behavior, and
panel request builders. It intentionally reuses the lower-level API/cache
contract for server-side routing while adding a panel-side request translation
slice so the app boundary is covered by a single artifact.
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


DEFAULT_OUT = Path(
    "build/current-api-surface-contract-20260601-cache-ipc-epipe-refresh.json"
)
NESTED_OUT = Path(
    "build/current-api-cache-contract-api-surface-check-20260528-epipe-aggregate-guard.json"
)

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "tests/cross_matrix/run_noheavy_api_cache_contract.py",
    "tests/test_engine_audit.py",
    "panel/src/main/ipc/chat.ts",
    "panel/src/main/ipc/cache.ts",
    "panel/src/main/ipc/image.ts",
    "panel/src/main/ipc/imageGenerationState.ts",
    "panel/src/main/ipc/developer.ts",
    "panel/src/main/ipc/models.ts",
    "panel/src/main/sessions.ts",
    "panel/src/main/server.ts",
    "panel/src/main/api-gateway.ts",
    "panel/src/main/engine-manager.ts",
    "panel/src/main/process-manager.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/shared/sessionConfigMigrations.ts",
    "panel/scripts/live-real-ui-model-proof.mjs",
    "panel/scripts/live-chat-tools-reasoning-proof.mjs",
    "panel/tests/request-builder.test.ts",
    "panel/tests/api-gateway-ollama.test.ts",
    "panel/tests/api-gateway-ollama-behavior.test.ts",
    "panel/tests/api-gateway-single-model.behavior.test.ts",
    "panel/tests/chat-override-policy.test.ts",
    "panel/tests/image-generation-state.test.ts",
    "panel/tests/image-system.test.ts",
    "tests/cross_matrix/run_api_surface_contract.py",
    "tests/test_api_surface_contract.py",
)

REQUIRED_NESTED_API_CHECKS = (
    "openai_chat_sampling_kwargs",
    "responses_sampling_kwargs",
    "legacy_completions_output_caps_override_server_default",
    "request_output_caps_override_server_default",
    "prompt_context_caps_stay_separate_from_output_caps",
    "anthropic_bundle_defaults",
    "ollama_adapter_surface",
    "streaming_cache_detail_usage",
    "responses_previous_response_history",
    "cache_reuse_endpoints",
    "cache_stats_reuse_skip_telemetry",
    "dsv4_native_cache_status",
    "dsv4_dsml_parser_residue_rejection",
    "dsv4_dsml_valid_tool_call_preserved",
    "dsv4_suppressed_tool_markup_not_stored",
    "zaya_typed_cca_status",
    "jangtq_mpp_nax_health_kernel_name",
    "hybrid_ssm_partial_reuse",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "no_generic_tq_on_hybrid_ssm",
    "all_required_named_rows_ran",
)

REQUIRED_PANEL_API_TEST_MARKERS = (
    "omits sampling and token defaults when unset so the engine resolves bundle metadata",
    "does not invent sampler or output-budget values when chat overrides are absent",
    "keeps per-chat maxTokens as output budget only, never prompt context",
    "omits invalid persisted maxTokens values instead of poisoning Chat Completions",
    "keeps Responses maxTokens as output budget only, never prompt context",
    "omits invalid persisted maxTokens values instead of poisoning Responses",
    "preserves DSV4 Responses max_output_tokens for Max thinking",
    "Hy3 local Responses Auto omits enable_thinking and reasoning_effort",
    "omits malformed Ollama context values instead of poisoning max_prompt_tokens",
    "omits unset and disabled sampling sentinels without dropping explicit overrides",
    "omits malformed Ollama num_predict values instead of poisoning max_tokens",
    "applies gateway timeout handling to Ollama embeddings proxy requests",
    "auto-switches by model id in single-model mode before preserving streaming deltas",
    "refuses auto-switch when previous local model cannot unload before starting target",
    "refuses single-model Ollama routes when previous local model cannot unload",
    "auto-switches single-model Ollama chat while emitting incremental content chunks",
    "auto-switches single-model Ollama generate while emitting incremental response chunks",
    "auto-switches single-model Ollama embeddings before proxying embedding data",
    "auto-switches direct OpenAI streaming by model id without mutating payload or deltas",
    "serializes concurrent single-model switches before starting a second target",
    "auto-switches to a standby model by waking it before direct OpenAI streaming",
    "auto-switches Responses API streaming by model id while preserving output text deltas",
    "auto-switches model capability requests by path model before proxying",
    "auto-switches cache endpoints by query model before proxying cache stats",
    "auto-switches cache entries and clear endpoints by query model before proxying cache endpoints",
    "auto-switches cache warm by body model before proxying warm prompts",
    "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap",
    "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults",
    "Auto chat maxTokens omits per-request output caps so server default can apply",
    "guards gateway streaming writes against client disconnect EPIPE errors",
    "guards every Ollama backend response stream error as a disconnect boundary",
    "aborts Ollama backend response streams when the client response closes",
    "writes each streamed gateway response chunk once and treats EPIPE as disconnect",
    "does not write gateway response chunks after the client socket is destroyed",
    "does not write gateway response chunks after Node marks the response closed",
    "does not write proxied request bodies after the backend socket is destroyed",
    "does not write proxied request bodies after Node marks the request closed",
    "treats top-level request handler EPIPE failures as client disconnects",
    "treats nested broken-pipe stream errors as client disconnects",
    "guards child process stdio stream EPIPE across app-managed process lanes",
    "guards live proof script child stdio EPIPE while collecting e2e evidence",
    "does not end proxied requests after the backend socket is destroyed",
    "does not end proxied requests after Node marks the request closed",
    "does not leave raw backend request end calls unguarded after disconnect",
    "does not leave raw chat IPC backend request finalization unguarded",
    "normalizes cache IPC endpoint EPIPE disconnects instead of surfacing raw unexpected errors",
    "does not log expected chat EPIPE disconnects as raw failed-message console errors",
    "routes local image server request writes through EPIPE-aware helpers",
    "image requests disable connection reuse and normalize reset-like socket errors",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "server_api_surface": (
        Path("."),
        [
            sys.executable,
            "tests/cross_matrix/run_noheavy_api_cache_contract.py",
            "--out",
            str(NESTED_OUT),
        ],
    ),
    "panel_api_request_builders": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/request-builder.test.ts",
            "tests/api-gateway-ollama.test.ts",
            "tests/api-gateway-ollama-behavior.test.ts",
            "tests/api-gateway-single-model.behavior.test.ts",
            "tests/chat-override-policy.test.ts",
            "tests/chat-ui.test.ts",
            "tests/image-generation-state.test.ts",
            "tests/image-system.test.ts",
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
    match = re.search(r"passed=(\d+)", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"deselected=(\d+)", output)
    if match:
        deselected = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match and deselected is None:
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


def _load_nested(root: Path) -> dict[str, Any]:
    path = root / NESTED_OUT
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    nested = _load_nested(root)
    nested_checks = nested.get("checks", {})
    nested_missing_markers = nested.get("missing_markers", [])
    panel_stdout = str(results["panel_api_request_builders"].get("stdout", ""))
    missing_panel_markers = [
        marker for marker in REQUIRED_PANEL_API_TEST_MARKERS if marker not in panel_stdout
    ]
    missing_nested_checks = [
        name for name in REQUIRED_NESTED_API_CHECKS if nested_checks.get(name) is not True
    ]
    panel_passed = results["panel_api_request_builders"]["counts"]["passed"] or 0
    checks = {
        "openai_chat_completions_sampling_defaults": (
            not failed and "openai_chat_sampling_kwargs" not in missing_nested_checks
        ),
        "openai_responses_sampling_defaults": (
            not failed and "responses_sampling_kwargs" not in missing_nested_checks
        ),
        "legacy_completions_output_caps_override_server_default": (
            not failed and "legacy_completions_output_caps_override_server_default" not in missing_nested_checks
        ),
        "chat_and_responses_output_caps_override_server_default": (
            not failed
            and "request_output_caps_override_server_default" not in missing_nested_checks
        ),
        "prompt_context_caps_stay_separate_from_output_caps": (
            not failed
            and "prompt_context_caps_stay_separate_from_output_caps" not in missing_nested_checks
        ),
        "anthropic_adapter_bundle_defaults": (
            not failed and "anthropic_bundle_defaults" not in missing_nested_checks
        ),
        "ollama_adapter_streaming_done_behavior": (
            not failed and "ollama_adapter_surface" not in missing_nested_checks
        ),
        "chat_and_responses_streaming_cache_detail_usage": (
            not failed and "streaming_cache_detail_usage" not in missing_nested_checks
        ),
        "responses_previous_response_history": (
            not failed and "responses_previous_response_history" not in missing_nested_checks
        ),
        "server_cache_and_tool_surfaces_named": (
            not failed and not missing_nested_checks and not nested_missing_markers
        ),
        "panel_request_builder_sampling_and_output_overrides": (
            not failed
            and "omits sampling and token defaults when unset so the engine resolves bundle metadata" not in missing_panel_markers
            and "keeps per-chat maxTokens as output budget only, never prompt context" not in missing_panel_markers
            and "keeps Responses maxTokens as output budget only, never prompt context" not in missing_panel_markers
            and panel_passed >= 53
        ),
        "panel_ollama_gateway_omits_disabled_sentinels": (
            not failed
            and "omits unset and disabled sampling sentinels without dropping explicit overrides" not in missing_panel_markers
            and "omits malformed Ollama num_predict values instead of poisoning max_tokens" not in missing_panel_markers
            and "omits malformed Ollama context values instead of poisoning max_prompt_tokens" not in missing_panel_markers
            and panel_passed >= 53
        ),
        "panel_gateway_single_model_auto_switch_streaming": (
            not failed
            and "auto-switches by model id in single-model mode before preserving streaming deltas" not in missing_panel_markers
            and "refuses auto-switch when previous local model cannot unload before starting target" not in missing_panel_markers
            and "auto-switches single-model Ollama chat while emitting incremental content chunks" not in missing_panel_markers
            and "auto-switches single-model Ollama generate while emitting incremental response chunks" not in missing_panel_markers
            and "auto-switches single-model Ollama embeddings before proxying embedding data" not in missing_panel_markers
            and "auto-switches direct OpenAI streaming by model id without mutating payload or deltas" not in missing_panel_markers
            and "serializes concurrent single-model switches before starting a second target" not in missing_panel_markers
            and "auto-switches to a standby model by waking it before direct OpenAI streaming" not in missing_panel_markers
            and "auto-switches Responses API streaming by model id while preserving output text deltas" not in missing_panel_markers
            and "auto-switches model capability requests by path model before proxying" not in missing_panel_markers
            and panel_passed >= 53
        ),
        "panel_gateway_single_model_auto_switch_cache_endpoints": (
            not failed
            and "auto-switches cache endpoints by query model before proxying cache stats" not in missing_panel_markers
            and "auto-switches cache entries and clear endpoints by query model before proxying cache endpoints" not in missing_panel_markers
            and "auto-switches cache warm by body model before proxying warm prompts" not in missing_panel_markers
            and panel_passed >= 53
        ),
        "panel_chat_override_policy_preserves_explicit_values": (
            not failed
            and "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap" not in missing_panel_markers
            and "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults" not in missing_panel_markers
            and panel_passed >= 53
        ),
        "panel_gateway_streaming_disconnect_epipe_guard": (
            not failed
            and "guards gateway streaming writes against client disconnect EPIPE errors" not in missing_panel_markers
            and "writes each streamed gateway response chunk once and treats EPIPE as disconnect" not in missing_panel_markers
            and "does not write gateway response chunks after the client socket is destroyed" not in missing_panel_markers
            and "does not write gateway response chunks after Node marks the response closed" not in missing_panel_markers
            and "does not write proxied request bodies after the backend socket is destroyed" not in missing_panel_markers
            and "does not write proxied request bodies after Node marks the request closed" not in missing_panel_markers
            and "treats top-level request handler EPIPE failures as client disconnects" not in missing_panel_markers
            and "treats nested broken-pipe stream errors as client disconnects" not in missing_panel_markers
            and "does not end proxied requests after the backend socket is destroyed" not in missing_panel_markers
            and "does not end proxied requests after Node marks the request closed" not in missing_panel_markers
            and "does not leave raw backend request end calls unguarded after disconnect" not in missing_panel_markers
            and panel_passed >= 63
        ),
        "panel_child_process_stdio_epipe_guard": (
            not failed
            and "guards child process stdio stream EPIPE across app-managed process lanes" not in missing_panel_markers
            and "guards live proof script child stdio EPIPE while collecting e2e evidence" not in missing_panel_markers
            and panel_passed >= 65
        ),
        "panel_gateway_ollama_proxy_response_error_guard": (
            not failed
            and "guards every Ollama backend response stream error as a disconnect boundary" not in missing_panel_markers
            and "aborts Ollama backend response streams when the client response closes" not in missing_panel_markers
            and panel_passed >= 61
        ),
        "panel_ipc_backend_request_epipe_guard": (
            not failed
            and "does not leave raw chat IPC backend request finalization unguarded" not in missing_panel_markers
            and "does not log expected chat EPIPE disconnects as raw failed-message console errors" not in missing_panel_markers
            and "routes local image server request writes through EPIPE-aware helpers" not in missing_panel_markers
            and "image requests disable connection reuse and normalize reset-like socket errors" not in missing_panel_markers
            and panel_passed >= 73
        ),
        "all_required_panel_api_markers_present": not failed and not missing_panel_markers,
    }
    public_results = {
        name: {key: value for key, value in result.items() if key != "stdout"}
        for name, result in results.items()
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "missing_nested_checks": missing_nested_checks,
        "missing_nested_markers": nested_missing_markers,
        "missing_panel_markers": missing_panel_markers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": public_results,
        "nested_api_cache_status": nested.get("status"),
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
    print("missing_nested_checks=" + json.dumps(artifact["missing_nested_checks"]))
    print("missing_nested_markers=" + json.dumps(artifact["missing_nested_markers"]))
    print("missing_panel_markers=" + json.dumps(artifact["missing_panel_markers"]))
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
