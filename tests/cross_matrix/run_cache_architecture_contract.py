#!/usr/bin/env python3
"""Run no-heavy cache architecture/family classification contracts.

This gate protects the release row that spans cache-family behavior: DSV4
native composite cache, ZAYA CCA, hybrid SSM/Mamba, MLA, MLLM media-salted
caches, TurboQuant KV runtime, and L2/disk serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-cache-architecture-contract-20260521.json")

CACHE_PATTERN = (
    "dsv4 or hybrid_ssm or cache_detail or MLA or mla or TurboQuant "
    "or turboquant or TQ or zaya or media_salt or cache_profile "
    "or prompt_disk_l2 or mllm_stats_include_cache"
)

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/scheduler.py",
    "vmlx_engine/prefix_cache.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/mllm_scheduler.py",
    "vmlx_engine/tq_disk_cache.py",
    "tests/cross_matrix/run_noheavy_api_cache_contract.py",
    "tests/cross_matrix/run_cache_architecture_contract.py",
    "tests/test_engine_audit.py",
    "tests/test_batching.py",
    "tests/test_hybrid_batching.py",
    "tests/test_hybrid_prefix_cache.py",
    "tests/test_dsv4_paged_cache.py",
    "tests/test_kimi_k25_mla_patch.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_turboquant_cache_contract.py",
    "tests/test_tq_disk_cache.py",
    "tests/test_cache_architecture_contract.py",
    "panel/src/main/sessions.ts",
    "panel/src/shared/cacheControlPolicy.ts",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/cache-control-policy.test.ts",
)

REQUIRED_CACHE_TEST_MARKERS = (
    # DSV4 composite cache can only reuse partial prefixes when the terminal
    # SWA+CSA/HCA composite block is present.
    "test_memory_pressure_reuses_shorter_dsv4_terminal_composite_prefix",
    "test_memory_pressure_refuses_dsv4_partial_without_terminal_composite",
    "test_dsv4_pool_quant_appends_only_new_pool_rows",
    # Hybrid SSM cache must reuse only aligned checkpoints and must not apply
    # generic TQ live-cache paths to SSM state.
    "test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint",
    "test_hybrid_ssm_checkpoint_alignment_falls_back_to_exact_aligned_state",
    "test_hybrid_ssm_auto_mode_disables_live_tq_but_keeps_stored_kv_q4",
    # L2/block-disk must backfill paged cache for later partial reuse.
    "test_prompt_disk_l2_hit_backfills_paged_cache_for_partial_reuse",
    # Qwen hybrid rows keep selective live TQ only where the runtime supports it.
    "test_qwen3_5_moe_linear_attention_keeps_selective_live_tq_and_stored_kv_q4",
    # TQ-native disk serialization must preserve mixed hybrid caches and round
    # trip real safetensors metadata instead of dequantizing everything.
    "test_serialize_tq_cache_mixed_hybrid",
    "test_tq_tensors_roundtrip_via_safetensors",
    # MLLM/cache telemetry must keep cache modes visible to the UI/API.
    "test_mllm_stats_include_cache_fields",
)

REQUIRED_API_CACHE_CHECKS = (
    "dsv4_native_cache_status",
    "zaya_typed_cca_status",
    "hybrid_ssm_partial_reuse",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "no_generic_tq_on_hybrid_ssm",
)

REQUIRED_API_CACHE_COMMAND_MARKERS = (
    "native_cache_status_reports_dsv4_separately_from_tq_kv",
    "native_cache_status_reports_zaya_typed_cca",
    "request_output_caps_override_server_default_without_touching_context_cap",
    "generic_turboquant_patcher_skips_hybrid_ssm",
)

REQUIRED_PANEL_CACHE_MARKERS = (
    "deepseek-v4 disables composite prefix cache by default even with stale cache config",
    "deepseek-v4 diagnostic cache opt-in uses DS4 page-sized blocks",
    "deepseek-v4 respects explicit prefix cache disable",
    "DSV4 pool quant and native prefix controls stay DSV4-only",
    "detected Qwen3.6 hybrid cache forces paged cache over stale saved false",
    "detected Mamba cache forces paged cache while regular KV respects saved false",
    "continuous batching off is a real master switch for LLM cache flags",
    "continuous batching off is a real master switch for VLM cache flags",
    "disabling prefix cache disables all dependent features",
    "MCP tools honor explicit prefix cache disable",
    "cacheMemoryPercent default 15 emits 0.15 when legacy memory cache is active",
    "omits memory-aware cache budget flags when paged cache is active",
    "mutual exclusion: disk cache NOT emitted when paged cache is active",
    "mutual exclusion: block disk cache only emitted when paged cache is active",
    "prefix cache disabled suppresses all cache sub-flags",
)


@dataclass(frozen=True)
class CommandSpec:
    cmd: list[str]
    cwd: str = "."


COMMANDS: dict[str, CommandSpec] = {
    "api_cache_status_contracts": CommandSpec([
        sys.executable,
        "tests/cross_matrix/run_noheavy_api_cache_contract.py",
        "--out",
        "build/current-api-cache-contract-cache-architecture-check-20260521.json",
    ]),
    "cache_family_pytest": CommandSpec([
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_batching.py",
        "tests/test_hybrid_batching.py",
        "tests/test_hybrid_prefix_cache.py",
        "tests/test_dsv4_paged_cache.py",
        "tests/test_kimi_k25_mla_patch.py",
        "tests/test_mllm_scheduler_cache.py",
        "tests/test_turboquant_cache_contract.py",
        "tests/test_tq_disk_cache.py",
        "-k",
        CACHE_PATTERN,
    ]),
    "panel_cache_launch_policy": CommandSpec(
        [
            "npm",
            "exec",
            "vitest",
            "--",
            "run",
            "tests/settings-flow.test.ts",
            "tests/cache-control-policy.test.ts",
            "--reporter=verbose",
            "-t",
            "deepseek-v4|DSV4|Qwen3\\.6|Mamba|cache",
        ],
        cwd="panel",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    skipped = None
    deselected = None
    matches = [int(value) for value in re.findall(r"(\d+) passed", output)]
    if matches:
        passed = max(matches)
    matches = [int(value) for value in re.findall(r"(\d+) skipped", output)]
    if matches:
        skipped = max(matches)
    matches = [int(value) for value in re.findall(r"(\d+) deselected", output)]
    if matches:
        deselected = max(matches)
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


def _run(root: Path, name: str, spec: CommandSpec) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        spec.cmd,
        cwd=root / spec.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": spec.cmd,
        "cwd": spec.cwd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout": proc.stdout,
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def _load_api_cache_artifact(root: Path) -> dict[str, Any]:
    path = root / "build/current-api-cache-contract-cache-architecture-check-20260521.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_artifact(root: Path) -> dict[str, Any]:
    results = {name: _run(root, name, cmd) for name, cmd in COMMANDS.items()}
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    api_artifact = _load_api_cache_artifact(root)
    api_checks = api_artifact.get("checks", {})
    cache_stdout = str(results["cache_family_pytest"].get("stdout", ""))
    missing_markers = [
        marker for marker in REQUIRED_CACHE_TEST_MARKERS if marker not in cache_stdout
    ]
    missing_api_checks = [
        check
        for check in REQUIRED_API_CACHE_CHECKS
        if api_checks.get(check) is not True
    ]
    api_command_text = "\n".join(
        " ".join(command.get("command", []))
        for command in api_artifact.get("commands", {}).values()
    )
    missing_api_command_markers = [
        marker
        for marker in REQUIRED_API_CACHE_COMMAND_MARKERS
        if marker not in api_command_text
    ]
    panel_stdout = str(results["panel_cache_launch_policy"].get("stdout", ""))
    missing_panel_markers = [
        marker
        for marker in REQUIRED_PANEL_CACHE_MARKERS
        if marker not in panel_stdout
    ]
    cache_passed = results["cache_family_pytest"]["counts"]["passed"] or 0
    panel_passed = results["panel_cache_launch_policy"]["counts"]["passed"] or 0
    checks = {
        "dsv4_native_composite_cache_status": (
            not failed
            and "dsv4_native_cache_status" not in missing_api_checks
            and "native_cache_status_reports_dsv4_separately_from_tq_kv" not in missing_api_command_markers
        ),
        "zaya_typed_cca_status": (
            not failed
            and "zaya_typed_cca_status" not in missing_api_checks
            and "native_cache_status_reports_zaya_typed_cca" not in missing_api_command_markers
        ),
        "hybrid_ssm_partial_reuse": (
            not failed
            and "hybrid_ssm_partial_reuse" not in missing_api_checks
            and "test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint" not in missing_markers
            and "test_hybrid_ssm_checkpoint_alignment_falls_back_to_exact_aligned_state" not in missing_markers
        ),
        "generic_tq_not_applied_to_hybrid_ssm": (
            not failed
            and "no_generic_tq_on_hybrid_ssm" not in missing_api_checks
            and "generic_turboquant_patcher_skips_hybrid_ssm" not in missing_api_command_markers
            and "test_hybrid_ssm_auto_mode_disables_live_tq_but_keeps_stored_kv_q4" not in missing_markers
        ),
        "turboquant_kv_runtime_contract": (
            not failed
            and "turboquant_kv_runtime_contract" not in missing_api_checks
            and "test_qwen3_5_moe_linear_attention_keeps_selective_live_tq_and_stored_kv_q4" not in missing_markers
        ),
        "turboquant_disk_roundtrip": (
            not failed
            and "turboquant_disk_roundtrip" not in missing_api_checks
            and "test_serialize_tq_cache_mixed_hybrid" not in missing_markers
            and "test_tq_tensors_roundtrip_via_safetensors" not in missing_markers
        ),
        "dsv4_terminal_composite_contracts": (
            not failed
            and "test_memory_pressure_reuses_shorter_dsv4_terminal_composite_prefix" not in missing_markers
            and "test_memory_pressure_refuses_dsv4_partial_without_terminal_composite" not in missing_markers
        ),
        "prompt_disk_l2_backfill_contracts": (
            not failed
            and "test_prompt_disk_l2_hit_backfills_paged_cache_for_partial_reuse" not in missing_markers
        ),
        "cache_detail_telemetry_contracts": (
            not failed
            and "test_mllm_stats_include_cache_fields" not in missing_markers
        ),
        "panel_cache_launch_policy": (
            not failed
            and not missing_panel_markers
            and panel_passed >= len(REQUIRED_PANEL_CACHE_MARKERS)
        ),
        "legacy_count_floor_still_nontrivial": not failed and cache_passed >= 55,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "missing_markers": missing_markers,
        "missing_api_checks": missing_api_checks,
        "missing_api_command_markers": missing_api_command_markers,
        "missing_panel_markers": missing_panel_markers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": {
            name: {key: value for key, value in result.items() if key != "stdout"}
            for name, result in results.items()
        },
        "api_cache_artifact_status": api_artifact.get("status"),
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
    print("missing_api_checks=" + json.dumps(artifact["missing_api_checks"]))
    print("missing_api_command_markers=" + json.dumps(artifact["missing_api_command_markers"]))
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
