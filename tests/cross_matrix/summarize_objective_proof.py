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
DSV4_CURRENT_IDENTIFIER_CANARY_REL = (
    "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json"
)
DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_STRICT_REL = (
    "build/current-dsv4-jangtq-k-identifier-canary-strict-nocache-bundled-b3345c29-rerun2-20260524.json"
)
DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_REL = "build/current-dsv4-live-identifier-canary-20260523.json"
DSV4_CURRENT_IDENTIFIER_MATRIX_REL = "build/current-dsv4-live-identifier-matrix-20260523.json"
DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL = "build/current-dsv4-installed-tokenizer-roundtrip-20260523.json"
DSV4_LIVE_LOGPROBS_COPY_REL = "build/current-dsv4-live-logprobs-copy-20260523.json"
DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL = "build/current-dsv4-live-logprob-context-matrix-20260523.json"
DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL = (
    "build/current-dsv4-live-cache-context-identifier-probe-20260523.json"
)
DSV4_SOURCE_NOCACHE_IDENTIFIER_REL = (
    "build/current-dsv4-live-identifier-list-nocache-source-20260523.json"
)
DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL = (
    "build/current-dsv4-live-identifier-sameprompt-nocache-source-20260523.json"
)
DSV4_SOURCE_CACHE_COMPARISON_REL = (
    "build/current-dsv4-live-identifier-cache-source-comparison-20260523.json"
)
DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_REL = (
    "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-thinking-on-responses-controls.json"
)
DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_FALLBACK_REL = (
    "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524-promptdiag-bb5cfe0c.json"
)
DSV4_CURRENT_ROUTE_MODE_DRYRUN_REL = (
    "build/current-dsv4-route-mode-code-exactness-dryrun-20260524-thinking-on-responses-controls.json"
)
DSV4_CURRENT_ROUTE_MODE_DRYRUN_FALLBACK_REL = (
    "build/current-dsv4-route-mode-code-exactness-dryrun-20260524.json"
)
API_CACHE_CONTRACT_REL = "build/current-api-cache-contract-proof-20260521.json"
PANEL_SETTINGS_CONTRACT_REL = "build/current-panel-settings-contract-proof-20260521.json"
MAX_OUTPUT_CONTEXT_CONTRACT_REL = (
    "build/current-max-output-context-contract-20260524-after-strict-dsv4-canary.json"
)
MAX_OUTPUT_CONTEXT_CONTRACT_FALLBACK_REL = "build/current-max-output-context-contract-20260521.json"
MODEL_FAMILY_CONTRACT_REL = "build/current-model-family-detection-contract-20260521.json"
PARSER_REGISTRY_CONTRACT_REL = "build/current-parser-registry-contract-20260521.json"
MODEL_ARTIFACT_FORMAT_CONTRACT_REL = "build/current-model-artifact-format-contract-20260521.json"
GENERATION_DEFAULTS_CONTRACT_REL = "build/current-generation-defaults-contract-20260521.json"
NATIVE_MTP_CONTRACT_REL = "build/current-native-mtp-contract-20260521.json"
VL_MEDIA_CONTRACT_REL = "build/current-vl-media-cache-contract-20260521.json"
QWEN_JANG_SOURCE_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-source-keepalloc-20260522.json"
QWEN_JANG_PACKAGED_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-packaged-tahoe-dmg-20260522.json"
QWEN_NATIVE_MTP_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-20260523.json"
QWEN_NATIVE_MTP_PREFILL_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json"
QWEN_NATIVE_MTP_PACKAGED_PREFILL_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json"
QWEN_NATIVE_MTP_PREFILL_TRACE_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json"
QWEN_JANG_TEXT_BASELINE_SPEED_REL = "build/current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json"
QWEN_NATIVE_MTP_NO_PREFIX_LOGITS_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-no-prefix-logits-20260523.json"
QWEN_NATIVE_MTP_HYBRID_LONG_PREFIX_SPLIT_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-hybrid-long-prefix-split-20260523.json"
QWEN_NATIVE_MTP_KVNONE_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-kvnone-20260523.json"
QWEN_NATIVE_MTP_ROUTE_TRACE_REL = "build/current-decode-speed-live-qwen27-jang4m-mtp-route-trace-20260523.json"
QWEN_RAW_FORWARD_AB_1024_REL = "build/current-qwen-forward-path-ab-1024-vlm-loader-20260523.json"
QWEN_RAW_FORWARD_AB_4096_REL = "build/current-qwen-forward-path-ab-4096-vlm-loader-20260523.json"
QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL = (
    "build/current-decode-speed-live-qwen27-jang4m-mtp-default-after-norm-shift-20260523.json"
)
QWEN_NATIVE_MTP_AB_REL = "build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-20260523/result.json"
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


def _load_first_present(root: Path, values: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    for value in values:
        if _path_present(root, value):
            return value, _load(root, value)
    first = values[0] if values else ""
    return first, _load(root, first) if first else {}


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


def _dsv4_identifier_canary_detail(canary: dict[str, Any], root: Path, rel: str) -> dict[str, Any]:
    path_present = _path_present(root, rel)
    if isinstance(canary.get("cases"), list):
        case_summaries: list[dict[str, Any]] = []
        failed_cases: list[str] = []
        for case in canary.get("cases") or []:
            if not isinstance(case, dict):
                continue
            name = str(case.get("name") or "")
            exact = case.get("exact") is True
            if not exact and name:
                failed_cases.append(name)
            content = case.get("content")
            if not isinstance(content, str):
                content = ""
            case_summaries.append(
                {
                    "name": name,
                    "route": case.get("route"),
                    "http_code": case.get("http_code"),
                    "finish_reason": case.get("finish"),
                    "exact": exact,
                    "corrupt_patterns": case.get("corrupt_patterns") or [],
                    "missing_identifiers": case.get("missing") or [],
                    "decode_tok_s_wall": case.get("decode_tps_wall"),
                    "completion_tokens": case.get("completion_tokens"),
                    "prompt_tokens": case.get("prompt_tokens"),
                    "content": content[:1000],
                }
            )
        return {
            "artifact": rel,
            "present": path_present,
            "status": canary.get("status") if path_present else "missing",
            "model_path": canary.get("model"),
            "env": canary.get("env_overrides") or {},
            "case_count": len(case_summaries),
            "failed_cases": failed_cases,
            "case_summaries": case_summaries,
        }
    if isinstance(canary.get("probes"), list):
        probe_summaries: list[dict[str, Any]] = []
        for probe in canary.get("probes") or []:
            if not isinstance(probe, dict):
                continue
            content = probe.get("content")
            if not isinstance(content, str):
                content = ""
            probe_summaries.append(
                {
                    "name": probe.get("name"),
                    "code": probe.get("code"),
                    "finish_reason": probe.get("finish"),
                    "elapsed_sec": probe.get("elapsed_sec"),
                    "usage": probe.get("usage"),
                    "perf": probe.get("perf"),
                    "analysis": probe.get("analysis") or {},
                    "content": content[:1000],
                }
            )
        return {
            "artifact": rel,
            "present": path_present,
            "status": canary.get("status") if path_present else "missing",
            "health_model_name": (canary.get("health_after_load") or {}).get("model_name")
            if isinstance(canary.get("health_after_load"), dict)
            else None,
            "served_model_name": canary.get("served_model_name"),
            "env": canary.get("env") or {},
            "failures": canary.get("failures") or [],
            "probe_summaries": probe_summaries,
        }
    response = canary.get("response") if isinstance(canary.get("response"), dict) else {}
    choices = response.get("choices") if isinstance(response, dict) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    content = canary.get("content")
    if not isinstance(content, str):
        content = ""
    return {
        "artifact": rel,
        "present": path_present,
        "status": canary.get("status") if path_present else "missing",
        "elapsed_sec": canary.get("elapsed_sec"),
        "required_identifier_counts": canary.get("required_identifier_counts") or {},
        "bad_patterns": canary.get("bad_patterns") or [],
        "has_markdown_fence": canary.get("has_markdown_fence"),
        "finish_reason": first_choice.get("finish_reason"),
        "usage": response.get("usage") if isinstance(response, dict) else None,
        "content": content[:1000],
    }


def _dsv4_prompt_rail_exactness_detail(
    artifact: dict[str, Any],
    dry_run: dict[str, Any],
    root: Path,
    artifact_rel: str,
    dry_run_rel: str,
) -> dict[str, Any]:
    path_present = _path_present(root, artifact_rel)
    dry_run_present = _path_present(root, dry_run_rel)
    cases = artifact.get("cases") if isinstance(artifact.get("cases"), list) else []
    dry_run_cases = dry_run.get("cases") if isinstance(dry_run.get("cases"), list) else []
    case_summaries: list[dict[str, Any]] = []
    failed_thinking_closed_cases: list[str] = []
    thinking_open_cases_without_identifier_corruption: list[str] = []
    rep1_still_corrupt_patterns: dict[str, list[str]] = {}

    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        prompt_diagnostics = (
            case.get("prompt_diagnostics")
            if isinstance(case.get("prompt_diagnostics"), dict)
            else {}
        )
        suffix = str(prompt_diagnostics.get("assistant_suffix_kind") or "")
        corrupt_patterns = [
            str(pattern) for pattern in (case.get("corrupt_patterns") or [])
        ]
        missing = [str(identifier) for identifier in (case.get("missing") or [])]
        exact = case.get("exact") is True
        if suffix == "thinking_closed" and not exact and name:
            failed_thinking_closed_cases.append(name)
        if suffix == "thinking_open" and not corrupt_patterns and not missing and name:
            thinking_open_cases_without_identifier_corruption.append(name)
        if name.endswith("_rep1") and corrupt_patterns:
            rep1_still_corrupt_patterns[name] = corrupt_patterns
        prompt_tail = prompt_diagnostics.get("prompt_tail")
        if not isinstance(prompt_tail, str):
            prompt_tail = ""
        case_summaries.append(
            {
                "name": name,
                "route": case.get("route"),
                "exact": exact,
                "assistant_suffix_kind": suffix,
                "prompt_endswith_assistant_think_open": (
                    prompt_diagnostics.get("prompt_endswith_assistant_think_open")
                ),
                "prompt_endswith_assistant_think_close": (
                    prompt_diagnostics.get("prompt_endswith_assistant_think_close")
                ),
                "missing_identifiers": missing,
                "corrupt_patterns": corrupt_patterns,
                "has_markdown_fence": case.get("has_markdown_fence"),
                "finish_reason": case.get("finish") or case.get("finish_reason"),
                "prompt_tail": prompt_tail[-500:],
            }
        )

    dry_run_suffixes: dict[str, str] = {}
    dry_run_overrides: dict[str, Any] = {}
    for case in dry_run_cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        if not name:
            continue
        diagnostics = (
            case.get("prompt_diagnostics")
            if isinstance(case.get("prompt_diagnostics"), dict)
            else {}
        )
        dry_run_suffixes[name] = str(diagnostics.get("assistant_suffix_kind") or "")
        dry_run_overrides[name] = case.get("request_overrides") or {}

    return {
        "artifact": artifact_rel,
        "present": path_present,
        "status": artifact.get("status") if path_present else "missing",
        "case_count": len(case_summaries),
        "failed_thinking_closed_cases": failed_thinking_closed_cases,
        "thinking_open_cases_without_identifier_corruption": (
            thinking_open_cases_without_identifier_corruption
        ),
        "rep1_still_corrupt_patterns": rep1_still_corrupt_patterns,
        "case_summaries": case_summaries,
        "dry_run_artifact": dry_run_rel,
        "dry_run_present": dry_run_present,
        "dry_run_status": dry_run.get("status") if dry_run_present else "missing",
        "dry_run_case_count": dry_run.get("case_count") if dry_run_present else None,
        "dry_run_suffixes": dry_run_suffixes,
        "dry_run_request_overrides": dry_run_overrides,
    }


def _dsv4_identifier_matrix_detail(matrix: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_CURRENT_IDENTIFIER_MATRIX_REL)
    probes = matrix.get("probes") if isinstance(matrix.get("probes"), list) else []
    probe_summaries: list[dict[str, Any]] = []
    failed_probe_ids: list[str] = []
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        probe_id = probe.get("id")
        if probe.get("status") != "pass" and isinstance(probe_id, str):
            failed_probe_ids.append(probe_id)
        content = probe.get("content")
        if not isinstance(content, str):
            content = ""
        probe_summaries.append(
            {
                "id": probe_id,
                "status": probe.get("status"),
                "elapsed_sec": probe.get("elapsed_sec"),
                "required_identifier_counts": probe.get("required_identifier_counts") or {},
                "has_markdown_fence": probe.get("has_markdown_fence"),
                "content": content[:1000],
            }
        )
    return {
        "artifact": DSV4_CURRENT_IDENTIFIER_MATRIX_REL,
        "present": path_present,
        "status": matrix.get("status") if path_present else "missing",
        "probe_count": len(probe_summaries),
        "failed_probe_ids": failed_probe_ids,
        "copy_task_failed": "copy_identifiers_only" in failed_probe_ids,
        "code_task_failed": "single_line_threejs_code" in failed_probe_ids,
        "probe_summaries": probe_summaries,
    }


def _dsv4_tokenizer_roundtrip_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    row_summaries: list[dict[str, Any]] = []
    failed_inputs: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("input")
        if row.get("roundtrip_exact") is not True and isinstance(text, str):
            failed_inputs.append(text)
        row_summaries.append(
            {
                "input": text,
                "ids": row.get("ids") or [],
                "tokens": row.get("tokens") or [],
                "decoded": row.get("decoded"),
                "roundtrip_exact": row.get("roundtrip_exact") is True,
            }
        )
    status = "missing"
    if path_present:
        status = "pass" if payload.get("all_roundtrip_exact") is True and not failed_inputs else "review"
    return {
        "artifact": DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL,
        "present": path_present,
        "status": status,
        "load_method": payload.get("load_method"),
        "tokenizer_class": payload.get("tokenizer_class"),
        "all_roundtrip_exact": payload.get("all_roundtrip_exact") is True,
        "row_count": len(row_summaries),
        "failed_inputs": failed_inputs,
        "rows": row_summaries,
    }


def _dsv4_logprob_copy_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_LIVE_LOGPROBS_COPY_REL)
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    logprobs = first_choice.get("logprobs") if isinstance(first_choice.get("logprobs"), dict) else {}
    entries = logprobs.get("content") if isinstance(logprobs.get("content"), list) else []
    tokens = [entry.get("token") for entry in entries if isinstance(entry, dict)]
    wrong_entry: dict[str, Any] = {}
    wrong_index: int | None = None
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if entry.get("token") == "c":
            prev_tokens = tokens[max(0, idx - 4) : idx]
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if prev_tokens[-2:] == [".P", "ers"] and next_token == "pective":
                wrong_entry = entry
                wrong_index = idx
                break
    top_logprobs = (
        wrong_entry.get("top_logprobs")
        if isinstance(wrong_entry.get("top_logprobs"), list)
        else []
    )
    top_tokens = [
        item.get("token")
        for item in top_logprobs
        if isinstance(item, dict) and isinstance(item.get("token"), str)
    ]
    wrong_rank = top_tokens.index("c") + 1 if "c" in top_tokens else None
    correct_rank = top_tokens.index("pective") + 1 if "pective" in top_tokens else None
    return {
        "artifact": DSV4_LIVE_LOGPROBS_COPY_REL,
        "present": path_present,
        "status": (
            "review"
            if path_present and wrong_entry
            else ("pass" if path_present and payload.get("status") == "pass" else "missing")
        ),
        "content": (payload.get("content") if isinstance(payload.get("content"), str) else "")[:1000],
        "identifier_counts": payload.get("identifier_counts") or {},
        "logprob_entry_count": payload.get("logprob_entry_count"),
        "tokens": tokens,
        "wrong_token_index": wrong_index,
        "wrong_token": wrong_entry.get("token") if wrong_entry else None,
        "wrong_token_logprob": wrong_entry.get("logprob") if wrong_entry else None,
        "correct_token": "pective" if "pective" in top_tokens else None,
        "wrong_token_rank": wrong_rank,
        "correct_token_rank": correct_rank,
        "model_preferred_wrong_token": (
            wrong_rank == 1 and correct_rank is not None and correct_rank > wrong_rank
        ),
        "top_logprobs_at_wrong_token": top_logprobs,
    }


def _dsv4_logprob_context_matrix_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL)
    probes = payload.get("probes") if isinstance(payload.get("probes"), list) else []
    probe_summaries: dict[str, dict[str, Any]] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        probe_id = probe.get("id")
        if not isinstance(probe_id, str):
            continue
        content = probe.get("content") if isinstance(probe.get("content"), str) else ""
        probe_summaries[probe_id] = {
            "content": content[:1000],
            "target_count": probe.get("target_count"),
            "has_wrong_perscpective": probe.get("has_wrong_perscpective") is True,
            "tokens": probe.get("tokens") or [],
            "wrong_c_after_pers_events": probe.get("wrong_c_after_pers_events") or [],
        }
    isolated = probe_summaries.get("isolated_identifier", {})
    list_copy = probe_summaries.get("list_copy", {})
    constructor = probe_summaries.get("constructor_sentence", {})
    isolated_passed = isolated.get("target_count") == 1 and not isolated.get("has_wrong_perscpective")
    list_failed = list_copy.get("target_count") == 0 and bool(list_copy.get("wrong_c_after_pers_events"))
    constructor_content = constructor.get("content") if isinstance(constructor.get("content"), str) else ""
    constructor_failed = constructor.get("target_count") == 0 and "PerspectiveCamera" not in constructor_content
    return {
        "artifact": DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL,
        "present": path_present,
        "status": payload.get("status") if path_present else "missing",
        "target": payload.get("target"),
        "isolated_identifier_passed": isolated_passed,
        "list_copy_failed": list_failed,
        "constructor_sentence_failed": constructor_failed,
        "context_sensitive_identifier_failure": isolated_passed
        and (list_failed or constructor_failed),
        "probe_summaries": probe_summaries,
    }


def _dsv4_cache_context_identifier_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL)
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    probe_summaries: dict[str, dict[str, Any]] = {}
    corrupt_identifiers_by_probe: dict[str, list[str]] = {}
    missing_identifiers_by_probe: dict[str, list[str]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        name = result.get("name")
        if not isinstance(name, str):
            continue
        corrupt = [
            item
            for item in (result.get("corrupt_identifier_tokens") or [])
            if isinstance(item, str)
        ]
        missing = [
            item
            for item in (result.get("expected_identifiers_missing") or [])
            if isinstance(item, str)
        ]
        content = result.get("content") if isinstance(result.get("content"), str) else ""
        corrupt_identifiers_by_probe[name] = corrupt
        missing_identifiers_by_probe[name] = missing
        probe_summaries[name] = {
            "status": result.get("status"),
            "elapsed_sec": result.get("elapsed_sec"),
            "content": content[:1000],
            "expected_identifiers_missing": missing,
            "corrupt_identifier_tokens": corrupt,
        }
    health_before = payload.get("health_before") if isinstance(payload.get("health_before"), dict) else {}
    native_cache = (
        health_before.get("native_cache") if isinstance(health_before.get("native_cache"), dict) else {}
    )
    pool_quant = (
        native_cache.get("pool_quant") if isinstance(native_cache.get("pool_quant"), dict) else {}
    )
    generic_tq = (
        native_cache.get("generic_turboquant_kv")
        if isinstance(native_cache.get("generic_turboquant_kv"), dict)
        else {}
    )
    return {
        "artifact": DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL,
        "present": path_present,
        "status": payload.get("status") if path_present else "missing",
        "plain_list_failed": bool(
            corrupt_identifiers_by_probe.get("list_plain")
            or missing_identifiers_by_probe.get("list_plain")
        ),
        "unique_prefix_failed": bool(
            corrupt_identifiers_by_probe.get("list_unique_prefix")
            or missing_identifiers_by_probe.get("list_unique_prefix")
        ),
        "constructor_unique_prefix_passed": (
            probe_summaries.get("constructor_unique_prefix", {}).get("status") == "pass"
        ),
        "native_cache_pool_quant_enabled": pool_quant.get("enabled") is True,
        "generic_turboquant_kv_enabled": generic_tq.get("enabled") is True,
        "corrupt_identifiers_by_probe": corrupt_identifiers_by_probe,
        "missing_identifiers_by_probe": missing_identifiers_by_probe,
        "probe_summaries": probe_summaries,
    }


def _dsv4_source_nocache_identifier_detail(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    path_present = _path_present(root, DSV4_SOURCE_NOCACHE_IDENTIFIER_REL)
    probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
    counts = probe.get("identifier_counts") if isinstance(probe.get("identifier_counts"), dict) else {}
    health = payload.get("health_before") if isinstance(payload.get("health_before"), dict) else {}
    native_cache = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    pool_quant = (
        native_cache.get("pool_quant") if isinstance(native_cache.get("pool_quant"), dict) else {}
    )
    generic_tq = (
        native_cache.get("generic_turboquant_kv")
        if isinstance(native_cache.get("generic_turboquant_kv"), dict)
        else {}
    )
    content = probe.get("content") if isinstance(probe.get("content"), str) else ""
    return {
        "artifact": DSV4_SOURCE_NOCACHE_IDENTIFIER_REL,
        "present": path_present,
        "status": payload.get("status") if path_present else "missing",
        "probe_elapsed_sec": payload.get("probe_elapsed_sec"),
        "usage": probe.get("usage"),
        "identifier_counts": counts,
        "all_identifiers_present": bool(counts) and all(
            isinstance(value, int) and value > 0 for value in counts.values()
        ),
        "content": content[:1000],
        "native_cache_prefix": native_cache.get("prefix") is True,
        "native_cache_paged": native_cache.get("paged") is True,
        "native_cache_block_disk_l2": native_cache.get("block_disk_l2") is True,
        "native_cache_pool_quant_enabled": pool_quant.get("enabled") is True,
        "generic_turboquant_kv_enabled": generic_tq.get("enabled") is True,
    }


def _dsv4_source_same_prompt_cache_boundary_detail(
    same_prompt_payload: dict[str, Any],
    cache_payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    same_present = _path_present(root, DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL)
    cache_present = _path_present(root, DSV4_SOURCE_CACHE_COMPARISON_REL)
    probe = same_prompt_payload.get("probe") if isinstance(same_prompt_payload.get("probe"), dict) else {}
    analysis = probe.get("analysis") if isinstance(probe.get("analysis"), dict) else {}
    same_content = analysis.get("content") if isinstance(analysis.get("content"), str) else ""
    missing = [
        item
        for item in (analysis.get("missing_identifiers") or [])
        if isinstance(item, str)
    ]
    same_nocache_failed = same_present and (
        same_prompt_payload.get("status") != "pass"
        or bool(missing)
        or analysis.get("has_common_corruptions") is True
    )

    cases = cache_payload.get("cases") if isinstance(cache_payload.get("cases"), list) else []
    case_summaries: dict[str, dict[str, Any]] = {}
    cache_failures: dict[str, list[dict[str, Any]]] = {}
    cache_hit_proven_by_case: dict[str, bool] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        name = case.get("name")
        if not isinstance(name, str):
            continue
        failures = [
            item
            for item in (case.get("failures") or [])
            if isinstance(item, dict)
        ]
        cache_failures[name] = failures
        cache_hit_proven_by_case[name] = False
        case_summaries[name] = {
            "status": case.get("status"),
            "artifact": case.get("artifact"),
            "failure_count": len(failures),
            "failures": failures,
        }

    pooloff_failed = bool(cache_failures.get("pooloff")) or (
        case_summaries.get("pooloff", {}).get("status") not in {None, "pass"}
    )
    poolon_failed = bool(cache_failures.get("poolon")) or (
        case_summaries.get("poolon", {}).get("status") not in {None, "pass"}
    )
    return {
        "artifact": DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL,
        "cache_comparison_artifact": DSV4_SOURCE_CACHE_COMPARISON_REL,
        "present": same_present and cache_present,
        "same_prompt_nocache": {
            "status": same_prompt_payload.get("status") if same_present else "missing",
            "content": same_content[:1000],
            "missing_identifiers": missing,
            "has_common_corruptions": analysis.get("has_common_corruptions") is True,
            "usage": probe.get("usage"),
        },
        "same_prompt_nocache_failed": same_nocache_failed,
        "cache_enabled_pooloff_failed": pooloff_failed,
        "cache_enabled_poolon_failed": poolon_failed,
        "pool_quant_is_not_differentiator": pooloff_failed and poolon_failed,
        "cache_hit_restore_not_proven_by_short_prompt": (
            same_nocache_failed and pooloff_failed and poolon_failed
        ),
        "cache_case_summaries": case_summaries,
        "cache_hit_proven_by_case": cache_hit_proven_by_case,
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


def _speed_artifact_detail(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    results = payload.get("results") or []
    row = results[0] if results and isinstance(results[0], dict) else {}
    pp_rows = row.get("pp_rows") or []
    pp_values: list[float] = []
    loopish_values: list[bool] = []
    for item in pp_rows:
        if not isinstance(item, dict):
            continue
        try:
            pp_values.append(float(item.get("pp_wall_tok_s")))
        except (TypeError, ValueError):
            pass
        loopish_values.append(bool(item.get("loopish")))
    min_pp = min(pp_values) if pp_values else None
    status = row.get("status")
    notes = row.get("notes") or []
    ok = (
        status == "pass"
        and min_pp is not None
        and min_pp >= 600.0
        and not any(loopish_values)
        and not notes
    )
    return ok, {
        "status": status,
        "notes": notes,
        "runtime_wheels": row.get("runtime_wheels"),
        "pp_wall_tok_s": pp_values,
        "min_pp_wall_tok_s": min_pp,
        "loopish_values": loopish_values,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _native_mtp_ab_detail(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    rows = payload.get("rows") or []
    by_label = {
        row.get("label"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    }
    baseline = by_label.get("baseline_no_mtp") or {}
    native = by_label.get("native_mtp") or {}
    baseline_tps = _float_or_none((baseline.get("summary") or {}).get("mean_wall_tok_s"))
    native_tps = _float_or_none((native.get("summary") or {}).get("mean_wall_tok_s"))
    speedup = _float_or_none(payload.get("speedup_vs_baseline"))
    output_equivalence = payload.get("output_equivalence") or {}
    mtp_totals = ((native.get("mtp_stats") or {}).get("totals") or {})
    acceptance_rate = _float_or_none(mtp_totals.get("acceptance_rate"))
    ok = (
        bool(output_equivalence.get("all_content_equal"))
        and bool(output_equivalence.get("all_full_text_equal"))
        and speedup is not None
        and speedup >= 1.1
        and native_tps is not None
        and native_tps >= 25.0
        and acceptance_rate is not None
        and acceptance_rate >= 0.5
    )
    return ok, {
        "speedup_vs_baseline": speedup,
        "baseline_decode_tps_wall": baseline_tps,
        "native_mtp_decode_tps_wall": native_tps,
        "output_equivalence": {
            "all_content_equal": output_equivalence.get("all_content_equal"),
            "all_full_text_equal": output_equivalence.get("all_full_text_equal"),
        },
        "native_mtp_acceptance_rate": acceptance_rate,
        "native_mtp_accepted_tokens": mtp_totals.get("accepted_tokens"),
        "native_mtp_drafted_tokens": mtp_totals.get("drafted_tokens"),
        "best_native_mtp_depth": payload.get("best_native_mtp_depth"),
    }


def _prefill_trace_detail(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    row = results[0] if results and isinstance(results[0], dict) else {}
    health = row.get("health_after") if isinstance(row.get("health_after"), dict) else {}
    scheduler = health.get("scheduler") if isinstance(health.get("scheduler"), dict) else {}
    batch_generator = (
        scheduler.get("batch_generator")
        if isinstance(scheduler.get("batch_generator"), dict)
        else {}
    )
    trace = batch_generator.get("last_prefill_trace")
    diagnosis: dict[str, Any] = {}
    if isinstance(trace, dict):
        forward_ms = _float_or_none(trace.get("forward_ms"))
        logits_eval_ms = _float_or_none(trace.get("logits_eval_ms"))
        sample_ms = _float_or_none(trace.get("sample_ms"))
        total_ms = _float_or_none(trace.get("total_ms"))
        preprocess_ms = _float_or_none(trace.get("preprocess_ms"))
        diagnosis = {
            "text_only_mllm_path": trace.get("has_images") is False,
            "native_mtp": trace.get("native_mtp") is True,
            "is_hybrid": trace.get("is_hybrid") is True,
            "uses_vlm_language_copy": str(trace.get("language_model_class") or "").startswith(
                "mlx_vlm."
            ),
            "force_text_rope_1d": trace.get("force_text_rope_1d") is True,
            "supports_return_logits": trace.get("supports_return_logits") is True,
            "preprocess_is_not_bottleneck": (
                preprocess_ms is not None
                and total_ms is not None
                and total_ms > 0
                and preprocess_ms / total_ms < 0.01
            ),
            "logits_eval_dominates_forward": (
                logits_eval_ms is not None
                and forward_ms is not None
                and forward_ms > 0
                and logits_eval_ms / forward_ms >= 5.0
            ),
            "sample_dominates_total": (
                sample_ms is not None
                and total_ms is not None
                and total_ms > 0
                and sample_ms / total_ms >= 0.5
            ),
            "logits_eval_to_forward_ratio": (
                round(logits_eval_ms / forward_ms, 3)
                if logits_eval_ms is not None and forward_ms is not None and forward_ms > 0
                else None
            ),
            "sample_to_total_ratio": (
                round(sample_ms / total_ms, 3)
                if sample_ms is not None and total_ms is not None and total_ms > 0
                else None
            ),
            "suspected_bottleneck": "unknown",
        }
        if (
            diagnosis["text_only_mllm_path"]
            and diagnosis["logits_eval_dominates_forward"]
            and diagnosis["sample_dominates_total"]
        ):
            diagnosis["suspected_bottleneck"] = "logits/sample materialization"
    return {
        "status": row.get("status"),
        "notes": row.get("notes") or [],
        "last_prefill_trace": trace,
        "diagnosis": diagnosis,
    }


def _no_prefix_logits_trial_detail(payload: dict[str, Any]) -> dict[str, Any]:
    ok, details = _speed_artifact_detail(payload)
    details["clears_prompt_processing_floor"] = ok
    return details


def _qwen_pp_trial_detail(payload: dict[str, Any]) -> dict[str, Any]:
    ok, details = _speed_artifact_detail(payload)
    details["clears_prompt_processing_floor"] = ok
    return details


def _qwen_raw_forward_ab_detail(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    raw_forward_only = True
    for payload in payloads:
        policy = str(payload.get("generation_defaults_policy") or "")
        raw_forward_only = raw_forward_only and "raw-forward-only" in policy
        for item in payload.get("comparison") or []:
            if not isinstance(item, dict):
                continue
            comparisons.append(
                {
                    "prompt_tokens": item.get("prompt_tokens"),
                    "text_pp_tok_s": _float_or_none(item.get("text_pp_tok_s")),
                    "vlm_pp_tok_s": _float_or_none(item.get("vlm_pp_tok_s")),
                    "text_over_vlm": _float_or_none(item.get("text_over_vlm")),
                }
            )
    ratios = [
        float(item["text_over_vlm"])
        for item in comparisons
        if item.get("text_over_vlm") is not None
    ]
    return {
        "raw_forward_only": raw_forward_only,
        "comparisons": comparisons,
        "min_text_over_vlm": min(ratios) if ratios else None,
        "max_text_over_vlm": max(ratios) if ratios else None,
    }


def _prompt_processing_speed_detail(
    text_payload: dict[str, Any],
    mtp_prefill_payload: dict[str, Any],
    mtp_packaged_payload: dict[str, Any],
    trace_payload: dict[str, Any],
    no_prefix_logits_payload: dict[str, Any],
    hybrid_long_prefix_split_payload: dict[str, Any],
    kvnone_payload: dict[str, Any],
    route_trace_payload: dict[str, Any],
    raw_forward_ab_payloads: list[dict[str, Any]],
    norm_shift_clearance_payload: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    text_ok, text_details = _speed_artifact_detail(text_payload)
    mtp_ok, mtp_details = _speed_artifact_detail(mtp_prefill_payload)
    mtp_packaged_ok, mtp_packaged_details = _speed_artifact_detail(mtp_packaged_payload)
    norm_shift_ok, norm_shift_details = _speed_artifact_detail(norm_shift_clearance_payload)
    trace_details = _prefill_trace_detail(trace_payload)
    no_prefix_logits_details = _no_prefix_logits_trial_detail(no_prefix_logits_payload)
    hybrid_long_prefix_split_details = _qwen_pp_trial_detail(hybrid_long_prefix_split_payload)
    kvnone_details = _qwen_pp_trial_detail(kvnone_payload)
    route_trace_details = _prefill_trace_detail(route_trace_payload)
    raw_forward_ab_details = _qwen_raw_forward_ab_detail(raw_forward_ab_payloads)
    return text_ok and norm_shift_ok, {
        "text_loader": text_details,
        "native_mtp_prefill_source_native_wheels": mtp_details,
        "native_mtp_prefill_packaged_compat_wheels": mtp_packaged_details,
        "native_mtp_after_norm_shift_default_cache": norm_shift_details,
        "prefill_trace": trace_details,
        "native_mtp_no_prefix_logits_trial": no_prefix_logits_details,
        "native_mtp_hybrid_long_prefix_split_trial": hybrid_long_prefix_split_details,
        "native_mtp_kvnone_trial": kvnone_details,
        "native_mtp_vlm_route_trace": route_trace_details,
        "raw_forward_path_ab": raw_forward_ab_details,
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
    dsv4_current_identifier_canary_rel, dsv4_current_identifier_canary = _load_first_present(
        root,
        (
            DSV4_CURRENT_IDENTIFIER_CANARY_REL,
            DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_STRICT_REL,
            DSV4_CURRENT_IDENTIFIER_CANARY_FALLBACK_REL,
        ),
    )
    dsv4_current_identifier_matrix = _load(root, DSV4_CURRENT_IDENTIFIER_MATRIX_REL)
    dsv4_installed_tokenizer_roundtrip = _load(root, DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL)
    dsv4_live_logprobs_copy = _load(root, DSV4_LIVE_LOGPROBS_COPY_REL)
    dsv4_live_logprob_context_matrix = _load(root, DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL)
    dsv4_live_cache_context_identifier = _load(root, DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL)
    dsv4_source_nocache_identifier = _load(root, DSV4_SOURCE_NOCACHE_IDENTIFIER_REL)
    dsv4_source_same_prompt_nocache = _load(root, DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL)
    dsv4_source_cache_comparison = _load(root, DSV4_SOURCE_CACHE_COMPARISON_REL)
    dsv4_prompt_rail_exactness_rel, dsv4_prompt_rail_exactness = _load_first_present(
        root,
        (
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_FALLBACK_REL,
        ),
    )
    dsv4_route_mode_dryrun_rel, dsv4_route_mode_dryrun = _load_first_present(
        root,
        (
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_REL,
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_FALLBACK_REL,
        ),
    )
    api_cache_contract = _load(root, API_CACHE_CONTRACT_REL)
    panel_settings_contract = _load(root, PANEL_SETTINGS_CONTRACT_REL)
    max_output_context_contract_rel, max_output_context_contract = _load_first_present(
        root,
        (
            MAX_OUTPUT_CONTEXT_CONTRACT_REL,
            MAX_OUTPUT_CONTEXT_CONTRACT_FALLBACK_REL,
        ),
    )
    model_family_contract = _load(root, MODEL_FAMILY_CONTRACT_REL)
    parser_registry_contract = _load(root, PARSER_REGISTRY_CONTRACT_REL)
    model_artifact_format_contract = _load(root, MODEL_ARTIFACT_FORMAT_CONTRACT_REL)
    generation_defaults_contract = _load(root, GENERATION_DEFAULTS_CONTRACT_REL)
    native_mtp_contract = _load(root, NATIVE_MTP_CONTRACT_REL)
    vl_media_contract = _load(root, VL_MEDIA_CONTRACT_REL)
    qwen_jang_source_speed = _load(root, QWEN_JANG_SOURCE_SPEED_REL)
    qwen_jang_packaged_speed = _load(root, QWEN_JANG_PACKAGED_SPEED_REL)
    qwen_jang_text_baseline_speed = _load(root, QWEN_JANG_TEXT_BASELINE_SPEED_REL)
    qwen_native_mtp_prefill_speed = _load(root, QWEN_NATIVE_MTP_PREFILL_SPEED_REL)
    qwen_native_mtp_packaged_prefill_speed = _load(root, QWEN_NATIVE_MTP_PACKAGED_PREFILL_SPEED_REL)
    qwen_native_mtp_prefill_trace = _load(root, QWEN_NATIVE_MTP_PREFILL_TRACE_REL)
    qwen_native_mtp_no_prefix_logits = _load(root, QWEN_NATIVE_MTP_NO_PREFIX_LOGITS_REL)
    qwen_native_mtp_hybrid_long_prefix_split = _load(root, QWEN_NATIVE_MTP_HYBRID_LONG_PREFIX_SPLIT_REL)
    qwen_native_mtp_kvnone = _load(root, QWEN_NATIVE_MTP_KVNONE_REL)
    qwen_native_mtp_route_trace = _load(root, QWEN_NATIVE_MTP_ROUTE_TRACE_REL)
    qwen_raw_forward_ab_1024 = _load(root, QWEN_RAW_FORWARD_AB_1024_REL)
    qwen_raw_forward_ab_4096 = _load(root, QWEN_RAW_FORWARD_AB_4096_REL)
    qwen_native_mtp_norm_shift_clearance = _load(root, QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL)
    qwen_native_mtp_ab = _load(root, QWEN_NATIVE_MTP_AB_REL)

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
            max_output_context_contract_rel,
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
    qwen_source_speed_ok, qwen_source_speed_details = _speed_artifact_detail(
        qwen_jang_source_speed
    )
    qwen_packaged_speed_ok, qwen_packaged_speed_details = _speed_artifact_detail(
        qwen_jang_packaged_speed
    )
    _add(
        requirements,
        "Qwen/JANG packaged MX matmul speed is release-cleared",
        _status(qwen_source_speed_ok and qwen_packaged_speed_ok),
        [QWEN_JANG_SOURCE_SPEED_REL, QWEN_JANG_PACKAGED_SPEED_REL],
        caveat=(
            "Source/native-wheel speed and packaged-app speed are separate. "
            "A source pass does not clear a packaged compat-wheel PP review."
        ),
        details={
            "source": qwen_source_speed_details,
            "packaged": qwen_packaged_speed_details,
        },
    )
    qwen_native_mtp_decode_ok, qwen_native_mtp_decode_details = (
        _native_mtp_ab_detail(qwen_native_mtp_ab)
    )
    _add(
        requirements,
        "Qwen native MTP live decode speed and output equivalence are release-cleared",
        _status(qwen_native_mtp_decode_ok),
        [QWEN_NATIVE_MTP_AB_REL],
        caveat=(
            "This row covers decode speed/equivalence only. Prompt-processing "
            "throughput is tracked separately so MTP decode is not blamed for "
            "a broader Qwen/JANG PP floor failure."
        ),
        details=qwen_native_mtp_decode_details,
    )
    qwen_prompt_ok, qwen_prompt_details = _prompt_processing_speed_detail(
        qwen_jang_text_baseline_speed,
        qwen_native_mtp_prefill_speed,
        qwen_native_mtp_packaged_prefill_speed,
        qwen_native_mtp_prefill_trace,
        qwen_native_mtp_no_prefix_logits,
        qwen_native_mtp_hybrid_long_prefix_split,
        qwen_native_mtp_kvnone,
        qwen_native_mtp_route_trace,
        [qwen_raw_forward_ab_1024, qwen_raw_forward_ab_4096],
        qwen_native_mtp_norm_shift_clearance,
    )
    _add(
        requirements,
        "Qwen 27B JANG_4M prompt-processing speed floor is release-cleared",
        _status(qwen_prompt_ok),
        [
            QWEN_JANG_TEXT_BASELINE_SPEED_REL,
            QWEN_NATIVE_MTP_PREFILL_SPEED_REL,
            QWEN_NATIVE_MTP_PACKAGED_PREFILL_SPEED_REL,
            QWEN_NATIVE_MTP_PREFILL_TRACE_REL,
            QWEN_NATIVE_MTP_NO_PREFIX_LOGITS_REL,
            QWEN_NATIVE_MTP_HYBRID_LONG_PREFIX_SPLIT_REL,
            QWEN_NATIVE_MTP_KVNONE_REL,
            QWEN_NATIVE_MTP_ROUTE_TRACE_REL,
            QWEN_RAW_FORWARD_AB_1024_REL,
            QWEN_RAW_FORWARD_AB_4096_REL,
            QWEN_NATIVE_MTP_NORM_SHIFT_CLEARANCE_REL,
        ],
        caveat=(
            "The current clearance row is the post norm-format loader fix live "
            "default MTP/VLM artifact. Older native-MTP/VL prefill diagnostics "
            "remain in the details because they identify the prior slow/corrupt "
            "route; packaged-app parity is still covered by the packaged "
            "integrity gate and separate Qwen/JANG packaged speed row."
        ),
        details=qwen_prompt_details,
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
            "current_installed_identifier_canary": _dsv4_identifier_canary_detail(
                dsv4_current_identifier_canary, root, dsv4_current_identifier_canary_rel
            ),
            "current_installed_identifier_matrix": _dsv4_identifier_matrix_detail(
                dsv4_current_identifier_matrix, root
            ),
            "current_installed_tokenizer_roundtrip": _dsv4_tokenizer_roundtrip_detail(
                dsv4_installed_tokenizer_roundtrip, root
            ),
            "current_installed_logprob_copy_probe": _dsv4_logprob_copy_detail(
                dsv4_live_logprobs_copy, root
            ),
            "current_installed_logprob_context_matrix": _dsv4_logprob_context_matrix_detail(
                dsv4_live_logprob_context_matrix, root
            ),
            "current_installed_unique_prefix_identifier_probe": _dsv4_cache_context_identifier_detail(
                dsv4_live_cache_context_identifier, root
            ),
            "current_source_nocache_identifier_probe": _dsv4_source_nocache_identifier_detail(
                dsv4_source_nocache_identifier, root
            ),
            "current_source_same_prompt_cache_boundary": (
                _dsv4_source_same_prompt_cache_boundary_detail(
                    dsv4_source_same_prompt_nocache,
                    dsv4_source_cache_comparison,
                    root,
                )
            ),
            "current_installed_prompt_rail_exactness_probe": (
                _dsv4_prompt_rail_exactness_detail(
                    dsv4_prompt_rail_exactness,
                    dsv4_route_mode_dryrun,
                    root,
                    dsv4_prompt_rail_exactness_rel,
                    dsv4_route_mode_dryrun_rel,
                )
            ),
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
            DSV4_CURRENT_IDENTIFIER_MATRIX_REL,
            DSV4_INSTALLED_TOKENIZER_ROUNDTRIP_REL,
            DSV4_LIVE_LOGPROBS_COPY_REL,
            DSV4_LIVE_LOGPROB_CONTEXT_MATRIX_REL,
            DSV4_LIVE_CACHE_CONTEXT_IDENTIFIER_REL,
            DSV4_SOURCE_NOCACHE_IDENTIFIER_REL,
            DSV4_SOURCE_SAME_PROMPT_NOCACHE_REL,
            DSV4_SOURCE_CACHE_COMPARISON_REL,
            DSV4_CURRENT_PROMPT_RAIL_EXACTNESS_REL,
            DSV4_CURRENT_ROUTE_MODE_DRYRUN_REL,
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
