#!/usr/bin/env python3
"""Summarize the current DSV4/cache/tool objective proof state.

This is a no-heavy helper: it reads existing JSON proof artifacts and produces
a requirement-by-requirement digest. It deliberately keeps broad DSV4
long-output/code-quality open unless a dedicated clearance artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cross_matrix.run_max_output_context_contract import (
    SOURCE_HASH_FILES as MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_model_artifact_format_contract import (
    SOURCE_HASH_FILES as MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_model_family_detection_contract import (
    REQUIRED_ROWS as MODEL_FAMILY_CONTRACT_CHECKS,
    SOURCE_HASH_FILES as MODEL_FAMILY_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_generation_defaults_contract import (
    SOURCE_HASH_FILES as GENERATION_DEFAULTS_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_native_mtp_contract import (
    SOURCE_HASH_FILES as NATIVE_MTP_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_parser_registry_contract import (
    SOURCE_HASH_FILES as PARSER_REGISTRY_SOURCE_HASH_FILES,
)
from tests.cross_matrix.run_vl_media_cache_contract import (
    SOURCE_HASH_FILES as VL_MEDIA_SOURCE_HASH_FILES,
)


DEFAULT_OUT = Path("build/current-objective-proof-audit-20260521.json")
DSV4_QUALITY_CLEARANCE_REL = "build/current-dsv4-long-output-quality-clearance-20260521.json"
API_CACHE_CONTRACT_REL = "build/current-api-cache-contract-proof-20260521.json"
PANEL_SETTINGS_CONTRACT_REL = "build/current-panel-settings-contract-proof-20260521.json"
MAX_OUTPUT_CONTEXT_CONTRACT_REL = "build/current-max-output-context-contract-20260521.json"
MODEL_FAMILY_CONTRACT_REL = "build/current-model-family-detection-contract-20260521.json"
PARSER_REGISTRY_CONTRACT_REL = "build/current-parser-registry-contract-20260521.json"
MODEL_ARTIFACT_FORMAT_CONTRACT_REL = "build/current-model-artifact-format-contract-20260521.json"
GENERATION_DEFAULTS_CONTRACT_REL = "build/current-generation-defaults-contract-20260521.json"
NATIVE_MTP_CONTRACT_REL = "build/current-native-mtp-contract-20260521.json"
VL_MEDIA_CONTRACT_REL = "build/current-vl-media-cache-contract-20260521.json"
DSV4_DEFAULT_CACHE_TOOL_LOOP_REL = "build/current-dsv4-default-cache-tool-loop/result.json"
DSV4_QUALITY_CLEARANCE_CHECKS = (
    "identifier_integrity",
    "threejs_single_file",
    "no_markdown_fence",
    "no_corrupt_identifiers",
    "non_length_stop",
    "source_or_rebuilt_body_clearance",
    "cached_vs_no_cache_semantic_equivalence",
)
API_CACHE_CONTRACT_CHECKS = (
    "openai_chat_sampling_kwargs",
    "responses_sampling_kwargs",
    "anthropic_bundle_defaults",
    "ollama_adapter_surface",
    "dsv4_native_cache_status",
    "dsv4_dsml_parser_residue_rejection",
    "dsv4_dsml_valid_tool_call_preserved",
    "dsv4_suppressed_tool_markup_not_stored",
    "zaya_typed_cca_status",
    "hybrid_ssm_partial_reuse",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "no_generic_tq_on_hybrid_ssm",
)
API_CACHE_SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "tests/test_engine_audit.py",
    "tests/test_batching.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_tq_disk_cache.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_tool_format.py",
)
PANEL_SETTINGS_CONTRACT_CHECKS = (
    "dsv4_default_native_prefix_on",
    "dsv4_explicit_prefix_off_disables_native_flags",
    "dsv4_l2_explicit_off_preserves_prefix",
    "dsv4_generic_kv_flags_suppressed",
    "max_output_context_cli_split",
    "chat_max_output_is_per_chat_override",
    "non_dsv4_cache_toggles_preserved",
    "i18n_max_output_context_copy",
    "panel_typecheck",
)
MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS = (
    "server_default_output_cap_uses_max_tokens",
    "startup_output_cap_is_default_not_request_ceiling",
    "request_output_caps_can_go_below_or_above_startup_default",
    "request_output_caps_do_not_mutate_server_default",
    "legacy_completions_output_cap_overrides_server_default",
    "chat_max_tokens_overrides_server_default_per_request",
    "responses_max_output_tokens_overrides_server_default_per_request",
    "anthropic_messages_preserves_bundle_and_explicit_output_caps",
    "ollama_num_predict_maps_only_positive_output_caps",
    "prompt_context_caps_do_not_rewrite_output_cap",
    "panel_server_default_output_maps_to_max_tokens",
    "panel_max_context_maps_to_max_prompt_tokens",
    "stale_32768_session_output_caps_are_migrated",
    "chat_output_cap_remains_per_chat_override",
    "request_builders_omit_auto_output_cap",
    "new_chat_output_caps_are_not_inherited_or_made_sticky",
    "all_family_max_token_precedence_stays_uniform",
    "wake_reload_and_cli_preserve_explicitness",
    "all_required_max_output_context_markers_present",
)
PARSER_REGISTRY_CONTRACT_CHECKS = (
    "engine_accepts_registered_reasoning_parsers",
    "engine_accepts_registered_tool_parsers",
    "panel_emitted_reasoning_parsers_are_engine_valid",
    "panel_emitted_tool_parsers_are_engine_valid",
    "minimax_m2_reasoning_parser_regression",
    "parser_aliases_are_canonical_before_cli",
    "zaya_hy3_ling_dsv4_parser_rows_are_present",
    "non_reasoning_family_boundaries_are_present",
    "all_required_parser_markers_present",
)
MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS = (
    "jang_and_jangtq_detection",
    "ling_bailing_hybrid_loader_repairs",
    "mxfp4_detection",
    "mxfp8_detection",
    "plain_mlx_4bit_detection",
    "dropped_mtp_detection",
    "preserved_mtp_detection",
    "cache_profile_detection",
    "not_path_name_only",
)
GENERATION_DEFAULTS_CONTRACT_CHECKS = (
    "generation_config_defaults_are_surfaced",
    "jang_config_sampling_defaults_override_generation_config",
    "disabled_top_k_sentinels_normalize_to_off",
    "mode_specific_jang_repetition_penalty_is_metadata_owned",
    "request_api_overrides_win_over_startup_defaults",
    "bundle_max_new_tokens_preserved_when_omitted",
    "omitted_max_tokens_without_bundle_default_is_bounded",
    "server_default_output_cap_is_not_request_ceiling",
    "no_hidden_sampler_forcing_or_repetition_floor",
    "panel_does_not_emit_default_sampler_cli_flags",
)
NATIVE_MTP_CONTRACT_CHECKS = (
    "native_mtp_d3_default_policy",
    "model_tuning_depth_policy",
    "dropped_and_preserved_mtp_detection",
    "config_only_mtp_never_activates",
    "mxfp4_mxfp8_mtp_artifact_detection",
    "mllm_native_mtp_decode_loop",
    "native_mtp_telemetry_edge_cases",
    "panel_native_mtp_controls_visible_when_supported",
    "panel_native_mtp_suppressed_for_dsv4_or_unsupported",
    "live_speed_equivalence_not_claimed",
)
VL_MEDIA_CONTRACT_CHECKS = (
    "video_url_request_schema",
    "video_fallback_processing",
    "qwen36_vl_video_detection",
    "media_cache_salt_separates_modal_inputs",
    "hybrid_ssm_vlm_cache_contracts",
    "mllm_tool_replay_preserves_effective_tools",
    "panel_read_video_builtin_tool",
    "panel_media_tool_followup_content_parts",
    "panel_image_display_consistency",
    "panel_vlm_launch_settings",
    "all_required_panel_markers_present",
)
PANEL_SETTINGS_SOURCE_HASH_FILES = (
    "panel/src/main/sessions.ts",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/src/shared/dsv4Env.ts",
    "panel/src/shared/cacheControlPolicy.ts",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/dsv4-env.test.ts",
    "panel/tests/cache-control-policy.test.ts",
)


def _load(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic digest should continue
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def _status(ok: bool, *, partial: bool = False) -> str:
    if ok:
        return "pass"
    return "partial" if partial else "open"


def _add(
    requirements: list[dict[str, Any]],
    requirement: str,
    status: str,
    evidence: list[str],
    *,
    caveat: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
    }
    if caveat:
        item["caveat"] = caveat
    if details is not None:
        item["details"] = details
    requirements.append(item)


def _static_rows(static: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in static.get("rows") or []:
        if not isinstance(row, dict):
            continue
        payload = row.get("static") or {}
        row_id = payload.get("id")
        if isinstance(row_id, str):
            rows[row_id] = payload
    return rows


def _function_calls(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response_body.get("output") or []
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _usage_details(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    return details if isinstance(details, dict) else {}


def _cached_tokens_from_payload(payload: dict[str, Any]) -> int:
    details = _usage_details(payload)
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _cache_detail_from_payload(payload: dict[str, Any]) -> str:
    return str(_usage_details(payload).get("cache_detail") or "")


def _command_tokens(payload: dict[str, Any]) -> list[str]:
    cmd = payload.get("cmd") or payload.get("command") or payload.get("server_args") or []
    if isinstance(cmd, str):
        return cmd.split()
    if isinstance(cmd, list):
        return [str(item) for item in cmd]
    return []


def _resolve_artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _path_present(root: Path, value: str) -> bool:
    path = _resolve_artifact_path(root, value)
    return path.is_file() and path.stat().st_size > 0


def _attach_evidence_file_status(requirements: list[dict[str, Any]], root: Path) -> None:
    for item in requirements:
        evidence = item.get("evidence") or []
        if not isinstance(evidence, list):
            continue
        evidence_files_present: dict[str, bool] = {}
        missing_evidence: list[str] = []
        for value in evidence:
            if not isinstance(value, str) or not value:
                continue
            present = _path_present(root, value)
            evidence_files_present[value] = present
            if not present:
                missing_evidence.append(value)
        details = item.setdefault("details", {})
        if isinstance(details, dict):
            details["evidence_files_present"] = evidence_files_present
            details["missing_evidence"] = missing_evidence
        if item.get("status") == "pass" and missing_evidence:
            item["status"] = "open"
            caveat = item.get("caveat")
            missing_note = "Listed evidence files are missing or empty."
            item["caveat"] = f"{caveat} {missing_note}" if caveat else missing_note


def _dsv4_quality_clearance(clearance: dict[str, Any], root: Path) -> tuple[bool, dict[str, Any]]:
    checks = clearance.get("checks") or {}
    required = {key: checks.get(key) is True for key in DSV4_QUALITY_CLEARANCE_CHECKS}
    artifacts = clearance.get("artifacts") or {}
    required_artifacts = ("identifier_gate", "full_output_gate")
    artifact_paths_present: dict[str, bool] = {}
    missing_artifacts: list[str] = []
    non_string_artifacts: list[str] = []

    for key in required_artifacts:
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            non_string_artifacts.append(key)
            artifact_paths_present[key] = False
            missing_artifacts.append(str(value) if value is not None else key)

    for key, value in artifacts.items():
        if not isinstance(value, str) or not value:
            if key not in non_string_artifacts:
                non_string_artifacts.append(str(key))
            artifact_paths_present[str(key)] = False
            continue
        present = _path_present(root, value)
        artifact_paths_present[str(key)] = present
        if not present:
            missing_artifacts.append(value)

    required_artifact_paths_present = all(
        artifact_paths_present.get(key) is True for key in required_artifacts
    )
    ok = (
        clearance.get("status") == "pass"
        and all(required.values())
        and required_artifact_paths_present
        and not missing_artifacts
        and not non_string_artifacts
    )
    return ok, {
        "clearance_status": clearance.get("status"),
        "clearance_checks": required,
        "clearance_artifacts": artifacts,
        "clearance_artifact_paths_present": artifact_paths_present,
        "missing_clearance_artifacts": missing_artifacts,
        "non_string_clearance_artifacts": non_string_artifacts,
    }


def _contract_checks(payload: dict[str, Any], required: tuple[str, ...]) -> tuple[bool, dict[str, bool]]:
    checks = payload.get("checks") or {}
    required_checks = {key: checks.get(key) is True for key in required}
    return payload.get("status") == "pass" and all(required_checks.values()), required_checks


def _source_hash_status(
    root: Path, payload: dict[str, Any], required_files: tuple[str, ...]
) -> tuple[bool, dict[str, Any]]:
    recorded = payload.get("source_hashes") or {}
    current_hashes: dict[str, str | None] = {}
    missing_source_hashes: list[str] = []
    stale_source_hashes: list[str] = []
    missing_source_files: list[str] = []

    for rel in required_files:
        expected = recorded.get(rel)
        if not isinstance(expected, str) or not expected:
            missing_source_hashes.append(rel)
        path = root / rel
        if not path.is_file():
            current_hashes[rel] = None
            missing_source_files.append(rel)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        current_hashes[rel] = actual
        if isinstance(expected, str) and expected and actual != expected:
            stale_source_hashes.append(rel)

    ok = not missing_source_hashes and not stale_source_hashes and not missing_source_files
    return ok, {
        "source_hashes_recorded": recorded,
        "source_hashes_current": current_hashes,
        "missing_source_hashes": missing_source_hashes,
        "stale_source_hashes": stale_source_hashes,
        "missing_source_files": missing_source_files,
    }


def _contract_detail(
    root: Path,
    payload: dict[str, Any],
    required_checks: tuple[str, ...],
    required_files: tuple[str, ...],
) -> tuple[bool, dict[str, Any]]:
    checks_ok, required = _contract_checks(payload, required_checks)
    hashes_ok, hash_details = _source_hash_status(root, payload, required_files)
    missing_rows = payload.get("missing_rows") or []
    missing_markers = payload.get("missing_markers") or []
    failed = payload.get("failed") or []
    ok = (
        checks_ok
        and hashes_ok
        and not missing_rows
        and not missing_markers
        and not failed
    )
    return ok, {
        "status": payload.get("status"),
        "failed": failed,
        "missing_rows": missing_rows,
        "missing_markers": missing_markers,
        "contract_checks": required,
        **hash_details,
    }


def build_digest(root: Path | str = Path(".")) -> dict[str, Any]:
    root = Path(root)
    cache = _load(root, "build/current-dsv4-cache-proof-digest-20260521.json")
    dev_cache = _load(root, "build/dev-ui-dsv4-live-cache-proof-20260521/result.json")
    dev_tool = _load(root, "build/dev-ui-dsv4-live-tool-proof-20260521/result.json")
    two_tool = _load(
        root, "build/v1546-current-bundled-dsv4-two-tool-proof-20260521001426/result.json"
    )
    default_cache_tool_loop = _load(root, DSV4_DEFAULT_CACHE_TOOL_LOOP_REL)
    cap = _load(root, "build/v1546-dsv4-app-tool-cap-nocache-proof-20260521090706/summary.json")
    ui = _load(root, "build/dev-ui-smoke-20260521/summary.json")
    static = _load(root, "build/current-static-cache-architecture-audit-full-qwen-hybrid-20260521.json")
    longctx = _load(root, "build/current-dsv4-long-context-proof-digest-20260521.json")
    quality_clearance = _load(root, DSV4_QUALITY_CLEARANCE_REL)
    api_cache_contract = _load(root, API_CACHE_CONTRACT_REL)
    panel_settings_contract = _load(root, PANEL_SETTINGS_CONTRACT_REL)
    max_output_context_contract = _load(root, MAX_OUTPUT_CONTEXT_CONTRACT_REL)
    model_family_contract = _load(root, MODEL_FAMILY_CONTRACT_REL)
    parser_registry_contract = _load(root, PARSER_REGISTRY_CONTRACT_REL)
    model_artifact_format_contract = _load(root, MODEL_ARTIFACT_FORMAT_CONTRACT_REL)
    generation_defaults_contract = _load(root, GENERATION_DEFAULTS_CONTRACT_REL)
    native_mtp_contract = _load(root, NATIVE_MTP_CONTRACT_REL)
    vl_media_contract = _load(root, VL_MEDIA_CONTRACT_REL)

    requirements: list[dict[str, Any]] = []
    cache_checks = cache.get("checks") or {}
    native = (
        ((dev_cache.get("before") or {}).get("body") or {}).get("native_cache")
        or (dev_tool.get("health") or {}).get("native_cache")
        or {}
    )
    turn1 = dev_cache.get("turn1") or {}
    turn2 = dev_cache.get("turn2") or {}
    round1_body = ((dev_tool.get("round1") or {}).get("body") or {})
    round2_body = ((dev_tool.get("round2") or {}).get("body") or {})
    round1_calls = _function_calls(round1_body)
    round2_calls = _function_calls(round2_body)
    two_tool_rounds = two_tool.get("rounds") or []
    executed_tools = [
        tool
        for round_item in two_tool_rounds
        for tool in (round_item.get("executed_tools") or [])
        if isinstance(tool, dict)
    ]
    final_text = two_tool_rounds[-1].get("output_text") if two_tool_rounds else None
    two_tool_names = [tool.get("name") for tool in executed_tools]
    default_tool_rounds = default_cache_tool_loop.get("rounds") or []
    default_tool_executed = [
        tool
        for round_item in default_tool_rounds
        for tool in (round_item.get("executed_tools") or [])
        if isinstance(tool, dict)
    ]
    default_tool_names = [tool.get("name") for tool in default_tool_executed]
    default_tool_final_text = (
        default_tool_rounds[-1].get("output_text") if default_tool_rounds else None
    )
    default_tool_cmd = _command_tokens(default_cache_tool_loop)
    default_tool_health = default_cache_tool_loop.get("health") or {}
    default_tool_native = default_tool_health.get("native_cache") or {}
    default_tool_cached_tokens = sum(
        _cached_tokens_from_payload(row) for row in default_tool_rounds
    )
    default_tool_cache_details = [
        _cache_detail_from_payload(row)
        for row in default_tool_rounds
        if _cache_detail_from_payload(row)
    ]
    default_tool_cache_detail_has_dsv4 = any(
        "dsv4" in detail for detail in default_tool_cache_details
    )
    default_tool_cache_ok = (
        "--disable-prefix-cache" not in default_tool_cmd
        and "--dsv4-enable-prefix-cache" in default_tool_cmd
        and "--use-paged-cache" in default_tool_cmd
        and "--enable-block-disk-cache" in default_tool_cmd
        and default_tool_native.get("cache_type") == "native_composite"
        and default_tool_native.get("prefix") is True
        and default_tool_native.get("paged") is True
        and default_tool_native.get("block_disk_l2") is True
        and (default_tool_native.get("generic_turboquant_kv") or {}).get("enabled")
        is False
        and default_tool_cached_tokens > 0
        and default_tool_cache_detail_has_dsv4
    )
    cap_checks = cap.get("checks") or {}
    ui_visible = ui.get("visible_assertions") or {}
    ui_cli = ui.get("cli_preview_assertions") or {}
    rows = _static_rows(static)

    _add(
        requirements,
        "DSV4 Flash prefix/paged/L2 cache is enabled by default from app launch",
        _status(
            all(
                cache_checks.get(key)
                for key in (
                    "persistedDefaultOn",
                    "launchHasDsv4EnablePrefix",
                    "launchHasUsePagedCache",
                    "launchHasBlockDisk",
                    "launchNoDisablePrefix",
                )
            )
        ),
        ["build/current-dsv4-cache-proof-digest-20260521.json", "build/dev-ui-smoke-20260521/summary.json"],
        details={
            key: cache_checks.get(key)
            for key in (
                "persistedDefaultOn",
                "launchHasDsv4EnablePrefix",
                "launchHasUsePagedCache",
                "launchHasBlockDisk",
                "launchNoDisablePrefix",
            )
        },
    )
    _add(
        requirements,
        "DSV4 cache is native SWA+CSA/HCA composite, not generic KV/TurboQuant KV",
        _status(
            native.get("cache_type") == "native_composite"
            and native.get("prefix") is True
            and native.get("paged") is True
            and native.get("block_disk_l2") is True
            and (native.get("generic_turboquant_kv") or {}).get("enabled") is False
        ),
        ["build/dev-ui-dsv4-live-cache-proof-20260521/result.json"],
        details={
            "cache_type": native.get("cache_type"),
            "components": native.get("components"),
            "generic_turboquant_kv": native.get("generic_turboquant_kv"),
            "cache_store_policy": native.get("cache_store_policy"),
        },
    )
    cached_details = (((turn2.get("body") or {}).get("usage") or {}).get("prompt_tokens_details") or {})
    _add(
        requirements,
        "DSV4 same-process cache hit improves latency/TTFT and records paged+dsv4 hit",
        _status(cache_checks.get("hotFasterTtft") and cache_checks.get("sameProcessHitDsv4")),
        ["build/current-dsv4-cache-proof-digest-20260521.json", "build/dev-ui-dsv4-live-cache-proof-20260521/result.json"],
        details={
            "cold_ttft_sec": (cache.get("timings") or {}).get("cold_ttft_sec"),
            "hot_ttft_sec": (cache.get("timings") or {}).get("hot_ttft_sec"),
            "dev_cold_elapsed_sec": turn1.get("elapsed_sec"),
            "dev_hot_elapsed_sec": turn2.get("elapsed_sec"),
            "dev_cached_tokens": cached_details.get("cached_tokens"),
            "dev_cache_detail": cached_details.get("cache_detail"),
        },
    )
    _add(
        requirements,
        "DSV4 block disk L2 stores and hits after restart",
        _status(
            cache_checks.get("blockDiskWrite")
            and cache_checks.get("restartL2DiskHit")
            and cache_checks.get("restartDsv4CacheHit")
        ),
        ["build/current-dsv4-cache-proof-digest-20260521.json"],
        details={
            "after_hot_block_disk": (cache.get("stats_after_hot") or {}).get("block_disk_cache"),
            "after_restart_block_disk": (cache.get("stats_after_restart") or {}).get("block_disk_cache"),
        },
    )
    _add(
        requirements,
        "DSV4 Responses one-tool call stops after tool result",
        _status(
            len(round1_calls) == 1
            and round1_calls[0].get("name") == "list_directory"
            and not round2_calls
            and round2_body.get("output_text") == "DONE"
        ),
        ["build/dev-ui-dsv4-live-tool-proof-20260521/result.json"],
        details={
            "round1_calls": round1_calls,
            "round2_output_text": round2_body.get("output_text"),
            "round2_function_calls": round2_calls,
        },
    )
    _add(
        requirements,
        "DSV4 can perform multiple tool iterations then final answer",
        _status(two_tool_names == ["list_directory", "write_file"] and final_text == "DONE"),
        ["build/v1546-current-bundled-dsv4-two-tool-proof-20260521001426/result.json"],
        caveat=(
            "This proof used --disable-prefix-cache, so it proves DSML multi-tool "
            "loop behavior separately from default-on cache."
        ),
        details={"executed_tools": executed_tools, "final_text": final_text},
    )
    _add(
        requirements,
        "DSV4 default-cache multi-tool agent loop is proven",
        _status(
            default_tool_names == ["list_directory", "write_file"]
            and default_tool_final_text == "DONE"
            and default_tool_cache_ok
        ),
        [DSV4_DEFAULT_CACHE_TOOL_LOOP_REL],
        caveat=(
            None
            if default_tool_cache_ok
            else "Existing multi-tool proof does not prove the default DSV4 native prefix/paged/L2 cache path."
        ),
        details={
            "artifact_status": default_cache_tool_loop.get("status"),
            "artifact_reason": default_cache_tool_loop.get("reason"),
            "executed_tool_names": default_tool_names,
            "final_text": default_tool_final_text,
            "launch_has_disable_prefix_cache": "--disable-prefix-cache" in default_tool_cmd,
            "launch_has_dsv4_enable_prefix_cache": "--dsv4-enable-prefix-cache"
            in default_tool_cmd,
            "launch_has_use_paged_cache": "--use-paged-cache" in default_tool_cmd,
            "launch_has_block_disk_cache": "--enable-block-disk-cache"
            in default_tool_cmd,
            "native_cache_type": default_tool_native.get("cache_type"),
            "native_cache_prefix": default_tool_native.get("prefix"),
            "native_cache_paged": default_tool_native.get("paged"),
            "native_cache_block_disk_l2": default_tool_native.get("block_disk_l2"),
            "generic_turboquant_kv": default_tool_native.get("generic_turboquant_kv"),
            "tool_loop_cached_tokens": default_tool_cached_tokens,
            "tool_loop_cache_details": default_tool_cache_details,
            "tool_loop_cache_detail_has_dsv4": default_tool_cache_detail_has_dsv4,
        },
    )
    _add(
        requirements,
        "App maxToolIterations cap is enforced for DSV4 tool loop",
        _status(
            cap.get("allPassed")
            and all(
                cap_checks.get(key)
                for key in (
                    "maxToolIterationsPersisted",
                    "sawLimitMessage",
                    "executedExactlyOneRealTool",
                    "capFileNotWritten",
                )
            )
        ),
        ["build/v1546-dsv4-app-tool-cap-nocache-proof-20260521090706/summary.json"],
        details=cap_checks,
    )
    max_output_context_ok, max_output_context_checks = _contract_checks(
        max_output_context_contract, MAX_OUTPUT_CONTEXT_CONTRACT_CHECKS
    )
    max_output_context_hash_ok, max_output_context_hash_details = _source_hash_status(
        root,
        max_output_context_contract,
        MAX_OUTPUT_CONTEXT_SOURCE_HASH_FILES,
    )
    ui_max_output_context_ok = (
        all(
            ui_visible.get(key)
            for key in (
                "server_default_max_output_visible",
                "max_context_visible",
                "generation_defaults_visible",
            )
        )
        and ui_cli.get("has_max_tokens_flag_after_reset") is False
        and ui_cli.get("has_max_prompt_tokens_flag_after_reset") is False
    )
    _add(
        requirements,
        "Server default max output and max context are distinct and map to correct CLI flags",
        _status(
            ui_max_output_context_ok
            and max_output_context_ok
            and max_output_context_hash_ok
        ),
        [
            "build/dev-ui-smoke-20260521/summary.json",
            MAX_OUTPUT_CONTEXT_CONTRACT_REL,
        ],
        details={
            "contract_status": max_output_context_contract.get("status"),
            "contract_checks": max_output_context_checks,
            "visible_assertions": {
                key: ui_visible.get(key)
                for key in (
                    "server_default_max_output_visible",
                    "max_context_visible",
                    "generation_defaults_visible",
                )
            },
            "cli_preview_assertions": {
                key: ui_cli.get(key)
                for key in (
                    "has_max_tokens_flag_after_reset",
                    "has_max_prompt_tokens_flag_after_reset",
                )
            },
            **max_output_context_hash_details,
        },
    )
    panel_settings_ok, panel_settings_checks = _contract_checks(
        panel_settings_contract, PANEL_SETTINGS_CONTRACT_CHECKS
    )
    panel_settings_hash_ok, panel_settings_hash_details = _source_hash_status(
        root, panel_settings_contract, PANEL_SETTINGS_SOURCE_HASH_FILES
    )
    _add(
        requirements,
        "Panel settings keep DSV4 cache, max output, and max context controls unambiguous",
        _status(panel_settings_ok and panel_settings_hash_ok),
        [PANEL_SETTINGS_CONTRACT_REL],
        caveat=(
            None
            if panel_settings_ok and panel_settings_hash_ok
            else "Run the no-heavy panel settings contract before claiming settings/UI coverage."
        ),
        details={
            "contract_status": panel_settings_contract.get("status"),
            "contract_checks": panel_settings_checks,
            "commands": panel_settings_contract.get("commands"),
            **panel_settings_hash_details,
        },
    )
    expected_profiles = {
        "dsv4_jang_local": "dsv4_composite",
        "minimax_m27_tq_k": "default",
        "qwen36_dense_mxfp4": "hybrid_ssm",
        "qwen36_dense_jang": "hybrid_ssm",
        "zaya_vl_mxfp4": "zaya_cca",
        "nemotron_omni_tq2": "hybrid_ssm",
        "ling_flash_tq": "hybrid_ssm",
    }
    _add(
        requirements,
        "Cross-family cache architecture is classified per family",
        _status(
            all(
                (rows.get(row_id) or {}).get("cache_profile_expected") == expected
                for row_id, expected in expected_profiles.items()
            )
        ),
        ["build/current-static-cache-architecture-audit-full-qwen-hybrid-20260521.json"],
        caveat="Static architecture proof only; broad live generation still requires per-family live rows.",
        details={
            row_id: {
                "exists": (rows.get(row_id) or {}).get("exists"),
                "expected": (rows.get(row_id) or {}).get("cache_profile_expected"),
                "registry_cache": ((rows.get(row_id) or {}).get("registry") or {}).get("cache_type"),
                "issues": (rows.get(row_id) or {}).get("issues"),
            }
            for row_id in expected_profiles
        },
    )
    model_family_ok, model_family_details = _contract_detail(
        root,
        model_family_contract,
        MODEL_FAMILY_CONTRACT_CHECKS,
        MODEL_FAMILY_SOURCE_HASH_FILES,
    )
    parser_registry_ok, parser_registry_details = _contract_detail(
        root,
        parser_registry_contract,
        PARSER_REGISTRY_CONTRACT_CHECKS,
        PARSER_REGISTRY_SOURCE_HASH_FILES,
    )
    model_artifact_format_ok, model_artifact_format_details = _contract_detail(
        root,
        model_artifact_format_contract,
        MODEL_ARTIFACT_FORMAT_CONTRACT_CHECKS,
        MODEL_ARTIFACT_FORMAT_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "High-risk model family parser, artifact, and launch policy gates are current",
        _status(model_family_ok and parser_registry_ok and model_artifact_format_ok),
        [
            MODEL_FAMILY_CONTRACT_REL,
            PARSER_REGISTRY_CONTRACT_REL,
            MODEL_ARTIFACT_FORMAT_CONTRACT_REL,
        ],
        caveat=(
            "This is no-heavy source/static compatibility proof; live multi-turn "
            "output quality and speed rows remain separate."
        ),
        details={
            "model_family": model_family_details,
            "parser_registry": parser_registry_details,
            "model_artifact_format": model_artifact_format_details,
        },
    )
    generation_defaults_ok, generation_defaults_details = _contract_detail(
        root,
        generation_defaults_contract,
        GENERATION_DEFAULTS_CONTRACT_CHECKS,
        GENERATION_DEFAULTS_SOURCE_HASH_FILES,
    )
    native_mtp_ok, native_mtp_details = _contract_detail(
        root,
        native_mtp_contract,
        NATIVE_MTP_CONTRACT_CHECKS,
        NATIVE_MTP_SOURCE_HASH_FILES,
    )
    vl_media_ok, vl_media_details = _contract_detail(
        root,
        vl_media_contract,
        VL_MEDIA_CONTRACT_CHECKS,
        VL_MEDIA_SOURCE_HASH_FILES,
    )
    _add(
        requirements,
        "Generation defaults, Native MTP, and VL media gates are current",
        _status(generation_defaults_ok and native_mtp_ok and vl_media_ok),
        [
            GENERATION_DEFAULTS_CONTRACT_REL,
            NATIVE_MTP_CONTRACT_REL,
            VL_MEDIA_CONTRACT_REL,
        ],
        caveat=(
            "This proves no-heavy wiring, metadata ownership, and media/cache "
            "compatibility. Live MTP speed/equivalence and video/Omni quality "
            "remain separate live rows."
        ),
        details={
            "generation_defaults": generation_defaults_details,
            "native_mtp": native_mtp_details,
            "vl_media": vl_media_details,
        },
    )
    api_cache_ok, api_cache_checks = _contract_checks(
        api_cache_contract, API_CACHE_CONTRACT_CHECKS
    )
    api_cache_hash_ok, api_cache_hash_details = _source_hash_status(
        root, api_cache_contract, API_CACHE_SOURCE_HASH_FILES
    )
    _add(
        requirements,
        "Current-source API adapters and non-DSV4 cache contracts are no-heavy covered",
        _status(api_cache_ok and api_cache_hash_ok),
        [API_CACHE_CONTRACT_REL],
        caveat=(
            None
            if api_cache_ok and api_cache_hash_ok
            else "Run the current-source no-heavy API/cache contract proof before claiming API/cache coverage."
        ),
        details={
            "contract_status": api_cache_contract.get("status"),
            "contract_checks": api_cache_checks,
            "commands": api_cache_contract.get("commands"),
            **api_cache_hash_details,
        },
    )
    quality_ok, quality_details = _dsv4_quality_clearance(quality_clearance, root)
    quality_details.update(
        {
            "long_context_status": longctx.get("status"),
            "long_context_notes": longctx.get("notes"),
        }
    )
    _add(
        requirements,
        "DSV4 long-output/code/file-generation quality is release-cleared",
        _status(quality_ok),
        [
            DSV4_QUALITY_CLEARANCE_REL,
            "build/current-dsv4-long-context-proof-digest-20260521.json",
            "build/current-dsv4-identifier-count-ablation-20260521/result.json",
            "docs/internal/release-gates/20260520_sisyphus_dsv4_identifier_gate_jang_affine_current/result.json",
        ],
        caveat=(
            None
            if quality_ok
            else "Current long-context digest is status=review with caveats; "
            "identifier ablations still show exact-code/identifier corruption. "
            "Do not release-claim this yet."
        ),
        details=quality_details,
    )
    _attach_evidence_file_status(requirements, root)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirements": requirements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    digest = build_digest(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(digest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out)
    for item in digest["requirements"]:
        print(f"{item['status'].upper():7} {item['requirement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
