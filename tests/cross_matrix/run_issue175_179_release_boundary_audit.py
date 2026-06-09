#!/usr/bin/env python3
"""No-heavy release-boundary audit for GitHub issues #175-#179.

This gate keeps focused source evidence separate from release clearance. It is
not a substitute for live UI/API/model proofs.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-issue175-179-release-boundary-audit-after-issue179-memory-preflight-20260607.json"
)
INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT = Path(
    "build/current-installed-app-runtime-parity-audit-after-installed-app-rebuild-20260606.json"
)
ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT = Path(
    "build/current-issue175-177-installed-runtime-audit-20260602-v1554-installed-tahoe.json"
)
ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT = Path(
    "build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json"
)
ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT = Path(
    "build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json"
)

OPEN_RELEASE_ROWS = [
    "Real Electron UI cross-family live model matrix is release-cleared",
    "DSV4 long-output/code/file-generation quality is release-cleared",
]

REQUIRED_LIVE_PROOF_AXES = [
    "real_app_parameters",
    "multi_turn_chat",
    "openai_responses_streaming",
    "tool_parser_detection_and_execution",
    "reasoning_parser_display_boundary",
    "prefix_cache_reuse",
    "paged_cache_reuse",
    "l2_disk_cache_reconstruct",
    "turboquant_encode_decode",
    "hybrid_ssm_cca_swa_hsa_csa_attention",
    "metal_matmul_runtime_path",
    "jang_and_jangtq_metadata",
    "vl_image_video_paths",
    "wrong_language_and_parser_tag_leak_check",
]

RUNTIME_CACHE_CLEAR_PATHS = [
    "vmlx_engine/engine/simple.py",
    "vmlx_engine/image_gen.py",
    "vmlx_engine/mllm_scheduler.py",
    "vmlx_engine/models/mllm.py",
    "vmlx_engine/reranker.py",
    "vmlx_engine/scheduler.py",
    "vmlx_engine/server.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _raw_clear_memory_cache_offenders(root: Path) -> list[str]:
    offenders: list[str] = []
    for rel in RUNTIME_CACHE_CLEAR_PATHS:
        path = root / rel
        if not path.exists():
            offenders.append(f"{rel}:missing")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "clear_memory_cache":
                offenders.append(f"{rel}:{node.lineno}")
    return offenders


def _issue179_audit(root: Path) -> dict[str, Any]:
    return _json(root / ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT)


def _installed_app_parity_audit(root: Path) -> dict[str, Any]:
    return _json(root / INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT)


def _installed_issue175_177_audit(root: Path) -> dict[str, Any]:
    return _json(root / ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT)


def _live_issue175_177_audit(root: Path) -> dict[str, Any]:
    return _json(root / ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT)


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    mlx_memory = _read(root / "vmlx_engine/mlx_memory.py")
    paged_cache = _read(root / "vmlx_engine/paged_cache.py")
    prefix_cache = _read(root / "vmlx_engine/prefix_cache.py")
    scheduler = _read(root / "vmlx_engine/scheduler.py")
    server = _read(root / "vmlx_engine/server.py")
    cli = _read(root / "vmlx_engine/cli.py")
    image_gen = _read(root / "vmlx_engine/image_gen.py")
    cancellation = _read(root / "tests/test_cancellation.py")
    issue179 = _issue179_audit(root)
    installed_parity = _installed_app_parity_audit(root)
    installed_175_177 = _installed_issue175_177_audit(root)
    live_175_177 = _live_issue175_177_audit(root)
    installed_parity_checks = installed_parity.get("checks")
    if not isinstance(installed_parity_checks, dict):
        installed_parity_checks = {}
    installed_175_177_checks = installed_175_177.get("checks")
    if not isinstance(installed_175_177_checks, dict):
        installed_175_177_checks = {}
    installed_app_lora_surface_proven = (
        installed_parity.get("status") in {"pass", "open"}
        and installed_parity_checks.get("image_lora_cli_flags") is True
        and installed_parity_checks.get("image_lora_load_signature") is True
        and installed_parity_checks.get("image_lora_server_globals") is True
        and installed_parity_checks.get("text_lora_flags_rejected") is True
    )
    installed_app_memory_clear_runtime_proven = (
        installed_175_177.get("status") == "pass"
        and installed_175_177_checks.get("memory_clear_helper_imports_from_installed_app") is True
        and installed_175_177_checks.get("memory_clear_uses_available_mlx_api") is True
        and installed_175_177_checks.get("runtime_paths_avoid_removed_clear_memory_cache") is True
    )
    installed_app_promoted_block_cleanup_proven = (
        installed_175_177.get("status") == "pass"
        and installed_175_177_checks.get("promoted_disk_blocks_drop_parent_mirror") is True
        and installed_175_177_checks.get(
            "l2_readable_write_through_blocks_drop_parent_mirror"
        )
        is True
    )
    installed_app_cache_selection_telemetry_proven = (
        installed_175_177.get("status") == "pass"
        and installed_175_177_checks.get("cache_selection_telemetry_installed") is True
        and installed_175_177_checks.get("cache_execution_timing_telemetry_installed")
        is True
    )
    live_175_177_checks = live_175_177.get("checks")
    if not isinstance(live_175_177_checks, dict):
        live_175_177_checks = {}
    live_issue175_memory_stress_proven = (
        live_175_177.get("status") == "pass"
        and not live_175_177.get("failures")
        and live_175_177_checks.get("installed_app_qwen_live_probe_passed") is True
        and live_175_177_checks.get("metal_memory_metrics_observed") is True
        and live_175_177_checks.get("admin_sleep_lifecycle_probe_passed") is True
        and live_175_177_checks.get("admin_sleep_deep_unload_observed") is True
    )
    live_issue176_memory_pressure_proven = (
        live_175_177.get("status") == "pass"
        and not live_175_177.get("failures")
        and live_175_177_checks.get("l2_block_disk_hits_observed") is True
        and live_175_177_checks.get("minimax_l2_block_disk_hits_observed") is True
        and live_175_177_checks.get("cold_paged_tq_restart_hit_observed") is True
    )
    live_issue177_ttft_proven = (
        live_175_177.get("status") == "pass"
        and not live_175_177.get("failures")
        and live_175_177_checks.get("minimax_cache_hit_request_latency_improved") is True
        and live_175_177_checks.get("scheduler_cache_execution_timing_observed") is True
        and live_175_177_checks.get("cold_paged_tq_restart_hit_observed") is True
        and live_175_177_checks.get("cold_paged_cache_selection_observed") is True
        and live_175_177_checks.get("cold_paged_stream_ttft_observed") is True
    )

    issue175_checks = {
        "helper_exists": "def clear_mlx_memory_cache" in mlx_memory,
        "prefers_metal_clear_cache": "mx.metal.clear_cache" in mlx_memory,
        "falls_back_to_top_level_clear_cache": "mx.clear_cache" in mlx_memory,
        "no_raw_removed_api_in_runtime_paths": not _raw_clear_memory_cache_offenders(root),
        "installed_app_memory_clear_runtime_proven": installed_app_memory_clear_runtime_proven,
        "installed_app_live_memory_stress_proven": live_issue175_memory_stress_proven,
    }
    issue176_checks = {
        "marks_promoted_disk_blocks": "cache_data_from_disk" in paged_cache,
        "drops_disk_mirror_after_reconstruct": "block.cache_data = None" in prefix_cache
        and "block.cache_data_from_disk = False" in prefix_cache,
        "regression_test_present": "test_paged_cache_reconstruct_drops_promoted_disk_block_mirror"
        in _read(root / "tests/test_engine_audit.py"),
        "l2_readable_write_through_regression_test_present": (
            "test_paged_cache_reconstruct_drops_readable_l2_write_through_mirror"
            in _read(root / "tests/test_engine_audit.py")
        ),
        "installed_app_promoted_block_cleanup_proven": installed_app_promoted_block_cleanup_proven,
        "installed_app_live_memory_pressure_proven": live_issue176_memory_pressure_proven,
    }
    issue177_checks = {
        "selection_telemetry_in_scheduler": "last_cache_selection" in scheduler,
        "selection_telemetry_in_health": "last_cache_selection" in server,
        "cold_paged_reason_present": "cold_paged_reconstruction_cost" in scheduler,
        "focused_regression_test_present": "test_cold_paged_hit_can_yield_to_warm_prefix_cache"
        in _read(root / "tests/test_batching.py"),
        "worker_timing_regression_test_present": (
            "test_worker_paged_tq_cache_hit_records_reconstruction_and_dequant_timing"
            in _read(root / "tests/test_batching.py")
        ),
        "installed_app_cache_selection_telemetry_proven": installed_app_cache_selection_telemetry_proven,
        "installed_live_ttft_and_cold_paged_tq_proven": live_issue177_ttft_proven,
    }
    issue175_source_checks = {
        key: value
        for key, value in issue175_checks.items()
        if key
        not in {
            "installed_app_memory_clear_runtime_proven",
            "installed_app_live_memory_stress_proven",
        }
    }
    issue176_source_checks = {
        key: value
        for key, value in issue176_checks.items()
        if key
        not in {
            "installed_app_promoted_block_cleanup_proven",
            "installed_app_live_memory_pressure_proven",
        }
    }
    issue177_source_checks = {
        key: value
        for key, value in issue177_checks.items()
        if key
        not in {
            "installed_app_cache_selection_telemetry_proven",
            "installed_live_ttft_and_cold_paged_tq_proven",
        }
    }
    issue178_checks = {
        "serve_cli_lora_paths": "--lora-paths" in cli,
        "serve_cli_lora_scales": "--lora-scales" in cli,
        "image_load_lora_signature_guard": "inspect.signature" in image_gen
        and "lora_paths" in image_gen
        and "lora_scales" in image_gen,
        "focused_regression_test_present": "test_load_passes_lora_paths_and_scales_when_supported"
        in _read(root / "tests/test_image_gen.py"),
        "empty_lora_lists_noop_regression_test_present": (
            "test_load_treats_empty_lora_lists_as_no_lora_request"
            in _read(root / "tests/test_image_gen.py")
        ),
        "lora_scale_without_path_still_rejected": (
            "test_load_rejects_nonempty_lora_scales_without_paths"
            in _read(root / "tests/test_image_gen.py")
        ),
        "text_lora_flags_rejected": "test_cli_lora_flags_fail_clearly_for_text_models"
        in _read(root / "tests/test_image_api.py")
        and "_validate_lora_args_for_model_type(args, is_image=_is_image)" in cli,
        "installed_app_lora_surface_proven": installed_app_lora_surface_proven,
    }
    issue179_proven = issue179.get("proven")
    if not isinstance(issue179_proven, dict):
        issue179_proven = {}
    issue179_reporter_comparison = issue179.get("reporter_parity_comparison")
    if not isinstance(issue179_reporter_comparison, dict):
        issue179_reporter_comparison = {}
    issue179_local_repro = issue179.get("local_reporter_prompt_reproduction")
    if not isinstance(issue179_local_repro, dict):
        issue179_local_repro = {}

    issue179_checks = {
        "audit_artifact_pass": issue179.get("status") == "pass",
        "not_proven_empty": issue179.get("not_proven") == [],
        "reporter_cancel_404_recorded": (issue179.get("proven") or {}).get(
            "reporter_log_has_responses_cancel_404"
        )
        is True,
        "local_installed_live_cancel_probe_proven": issue179_proven.get(
            "local_installed_responses_cancel_live_probe"
        )
        is True,
        "reporter_parity_proven": (
            issue179_reporter_comparison.get("status") == "pass"
            and not issue179_reporter_comparison.get("failures")
        )
        or (
            issue179.get("status") == "pass"
            and issue179.get("not_proven") == []
            and issue179_reporter_comparison.get("server_route_markers_match") is True
        ),
        "local_reporter_prompt_reproduction_clean": (
            issue179_proven.get("local_reporter_prompt_reproduction_clean") is True
            and issue179_local_repro.get("clean") is True
            and issue179_local_repro.get("request_matches_reporter") is True
        ),
        "responses_cancel_route_tested": "test_cancel_responses_endpoint_calls_engine_with_response_id"
        in cancellation,
    }

    issue178_source_checks = {
        key: value
        for key, value in issue178_checks.items()
        if key != "installed_app_lora_surface_proven"
    }

    issue179_open = (
        issue179_checks.get("responses_cancel_route_tested") is True
        and not all(issue179_checks.values())
    )

    issues = {
        "175": {
            "title": "`mx.clear_memory_cache()` silently fails on MLX 0.31+",
            "focused_source_slice": (
                "pass"
                if all(issue175_checks.values())
                else "open"
                if all(issue175_source_checks.values())
                else "fail"
            ),
            "checks": issue175_checks,
            "release_clearance": (
                "installed_app_memory_clear_runtime_live_stress_proven"
                if (
                    installed_app_memory_clear_runtime_proven
                    and live_issue175_memory_stress_proven
                )
                else "installed_app_memory_clear_runtime_proven_live_stress_open"
                if installed_app_memory_clear_runtime_proven
                else "open_live_app_runtime_memory_proof_required"
            ),
        },
        "176": {
            "title": "Promoted paged-cache blocks can retain parent KV buffers",
            "focused_source_slice": (
                "pass"
                if all(issue176_checks.values())
                else "open"
                if all(issue176_source_checks.values())
                else "fail"
            ),
            "checks": issue176_checks,
            "release_clearance": (
                "installed_app_promoted_block_cleanup_live_pressure_proven"
                if (
                    installed_app_promoted_block_cleanup_proven
                    and live_issue176_memory_pressure_proven
                )
                else "installed_app_promoted_block_cleanup_proven_live_stress_open"
                if installed_app_promoted_block_cleanup_proven
                else "open_live_memory_pressure_proof_required"
            ),
        },
        "177": {
            "title": "Cache path selection can regress TTFT with paged/TurboQuant caches",
            "focused_source_slice": (
                "pass"
                if all(issue177_checks.values())
                else "open"
                if all(issue177_source_checks.values())
                else "fail"
            ),
            "checks": issue177_checks,
            "release_clearance": (
                "installed_app_cache_selection_live_ttft_proven"
                if live_issue177_ttft_proven
                else "installed_app_cache_selection_telemetry_proven_live_ttft_open"
                if installed_app_cache_selection_telemetry_proven
                else "open_live_ttft_proof_required"
            ),
        },
        "178": {
            "title": "Expose LoRA paths through vmlx serve",
            "focused_source_slice": (
                "pass" if all(issue178_source_checks.values()) else "fail"
            ),
            "checks": issue178_checks,
            "release_clearance": (
                "packaged_app_lora_surface_proven"
                if installed_app_lora_surface_proven
                else "open_packaged_image_lora_proof_required"
            ),
        },
        "179": {
            "title": "MiniMax-M2.7-JANGTQ_K produces gabage",
            "focused_source_slice": (
                "pass" if all(issue179_checks.values()) else "open" if issue179_open else "fail"
            ),
            "checks": issue179_checks,
            "release_clearance": (
                "installed_app_reporter_prompt_and_cancel_boundary_proven"
                if all(issue179_checks.values())
                else "open_reporter_parity_required"
            ),
        },
    }

    failures = [
        number
        for number, issue in issues.items()
        if issue["focused_source_slice"] == "fail"
    ]
    open_packaged_proofs = [
        number
        for number, issue in issues.items()
        if issue["focused_source_slice"] == "open"
        or issue.get("release_clearance", "").startswith("open_")
    ]
    status = "fail" if failures else "open" if open_packaged_proofs else "pass"
    release_boundary = (
        "Issue #175-#179 source/runtime guardrails are present, but installed/live "
        "runtime proof gaps remain. Issue #179 is release-cleared only when the "
        "root-cause audit proves the reporter hash is stale/unknown, current/public "
        "route markers are present, local reproduction is clean, and no proof gaps "
        "remain. The broader release still requires the live model matrix, real "
        "Electron UI matrix, and DSV4 exactness rows."
        if status == "open"
        else (
            "Issue #175-#179 focused installed/live runtime boundaries are "
            "proven. Issue #179 is release-cleared when the root-cause audit "
            "proves the reporter hash is stale/unknown, current/public route "
            "markers are present, local reproduction is clean, and no proof "
            "gaps remain. The broader release still requires the live model "
            "matrix, real Electron UI matrix, and DSV4 exactness rows."
        )
    )
    return {
        "artifact": "",
        "status": status,
        "issues": issues,
        "focused_failures": failures,
        "open_packaged_proofs": open_packaged_proofs,
        "open_release_rows": OPEN_RELEASE_ROWS,
        "required_live_proof_axes": REQUIRED_LIVE_PROOF_AXES,
        "release_boundary": release_boundary,
    }


def write_audit(root: Path, out: Path) -> dict[str, Any]:
    audit = build_audit(root)
    audit["artifact"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    audit = write_audit(args.root, args.out)
    print(json.dumps({"status": audit["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if audit["status"] in {"open", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
