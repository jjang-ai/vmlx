#!/usr/bin/env python3
"""Summarize the full vMLX/MLXStudio release objective as a blocker checklist.

This runner is intentionally no-heavy: it does not load models and does not
turn narrow contract evidence into release clearance. Its job is to keep the
full requested release scope visible in one machine-readable artifact while the
real live gates are being completed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(
    "build/current-full-release-objective-checklist-after-mimo-media-route-proof-20260610.json"
)

MIMO_AUDIT = Path(
    "build/current-mimo-v2-jang2l-current-audit-after-media-route-proof-20260610.json"
)
MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER = Path(
    "build/current-mimo-v2-no-source-exactness-classifier-after-devapp-jangtq2-exactness-20260610.json"
)
MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE = Path(
    "build/current-mimo-v25-jangtq2-media-runtime-source-proof-20260610.json"
)
MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE = Path(
    "build/current-mimo-v25-jangtq2-video-audio-source-proof-20260610.json"
)
MIMO_JANGTQ2_RESPONSES_TOOLS_CACHE_UI = Path(
    "build/current-real-ui-dev-app-mimo-v25-jangtq2-responses-tools-cache-20260610.json"
)
NOHEAVY_API_CACHE = Path(
    "build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json"
)
RESPONSES_RAW_SSE_PARITY = Path(
    "build/current-responses-raw-sse-parity-direct-gateway-tunnel-gemma4-12b-mxfp8-crack-20260610.json"
)
API_SURFACE_CONTRACT = Path(
    "build/current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json"
)
PANEL_SETTINGS_CONTRACT = Path(
    "build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json"
)
TOOL_CALL_CONTRACT = Path(
    "build/current-tool-call-contract-after-cross-model-loop-metrics-20260609.json"
)
RELEASE_MANIFEST = Path(
    "build/current-release-regression-manifest-after-pr-intake-matrix-refresh-20260609.json"
)
OBJECTIVE_DIGEST = Path(
    "build/current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json"
)
ISSUE179_AUDIT = Path(
    "build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json"
)
QWEN27_RESPONSES_CANCEL = Path(
    "build/current-qwen27-mxfp4-mtp-responses-cancel-mtp-deterministic-20260609.json"
)
QWEN27_API_PARITY = Path(
    "build/current-qwen27-mxfp4-mtp-api-parity-20260607/summary.json"
)
QWEN27_RESTART_L2_RESTORE = Path(
    "build/current-qwen27-mxfp4-mtp-restart-l2-restore-20260609/summary.json"
)
QWEN27_INSTALLED_VIDEO = Path(
    "docs/internal/agent-notes/current-real-ui-installed-app-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json"
)
QWEN27_LONG_CONTEXT_CACHE_TAIL = Path(
    "build/current-qwen27-jang4m-mtp-installed-long-context-cache-tail-20260607.json"
)
QWEN35_STARTUP_MTP = Path(
    "build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-deterministic-20260607/00_startup.json"
)
QWEN35_LONG_TOOL_CACHE = Path(
    "build/current-qwen35-mxfp8-mtp-responses-long-tool-cache-after-historical-tool-required-20260607/SUMMARY.json"
)
QWEN35_RESTART_L2_RESTORE = Path(
    "build/current-qwen35-mxfp8-mtp-restart-l2-restore-20260607/summary.json"
)
QWEN35_RAW_SSE_PARITY = Path(
    "build/current-responses-raw-sse-parity-qwen35-direct-gateway-tunnel-after-public-recapture-20260610.json"
)
QWEN35_INSTALLED_VIDEO = Path(
    "docs/internal/agent-notes/current-real-ui-installed-app-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json"
)
GEMMA4_12B_JANG4M_SMOKE = Path(
    "build/current-all-local-model-smoke-gemma4-12b-jang4m-tools-nomedia-after-code-column-prompt-20260610/JANGQ_gemma-4-12B-it-JANG_4M/result.json"
)
GEMMA4_12B_JANG4M_AUTOQ4_CACHE = Path(
    "build/current-gemma4-12b-jang4m-autoq4-mixed-swa-cache-live-20260610.json"
)
GEMMA4_12B_ISSUE191_STARTUP_VISIBLE = Path(
    "build/current-gemma4-12b-issue191-source-startup-visible-proof-20260609.json"
)
GEMMA4_12B_JANG4M_MEDIA_SMOKE = Path(
    "build/current-gemma4-12b-mxfp4-jang4m-media-smoke-live-20260610.json"
)
GEMMA_QAT_NATIVE_MXFP4_INVENTORY = Path(
    "build/current-gemma-qat-native-mxfp4-local-inventory-after-12b-installed-app-ui-proof-20260610.json"
)
STEP37_TEXTONLY_SMOKE = Path(
    "build/current-all-local-model-smoke-step37-jang2l-crack-tools-nomedia-textonly-harness-20260606/other_Step-3.7-Flash-JANG_2L-CRACK/result.json"
)
STEP37_VLM_RUNTIME_AUDIT = Path(
    "build/current-step37-vlm-runtime-audit-after-source-live-media-proof-20260607.json"
)
LFM25_MXFP4_SMOKE = Path(
    "build/current-all-local-model-smoke-lfm25-mxfp4-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP4/result.json"
)
LFM25_MXFP8_SMOKE = Path(
    "build/current-all-local-model-smoke-lfm25-mxfp8-tools-nomedia-20260609/JANGQ_LFM2.5-8B-A1B-MXFP8/result.json"
)
NEMOTRON_OMNI_SMOKE = Path(
    "build/current-all-local-model-smoke-nemotron-omni-mxfp4-tools-nomedia-after-reasoning-budget-20260606/dealign.ai_Nemotron-Omni-Nano-MXFP4-CRACK/result.json"
)
NEMOTRON_OMNI_MEDIA_GATE = Path(
    "build/current-nemotron-omni-mxfp4-media-gate-20260607/SUMMARY.json"
)
DSV4_EXACTNESS_PREFLIGHT = Path(
    "build/current-dsv4-route-mode-code-exactness-preflight-after-mimo-classifier-refresh-20260608.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact": str(path), "exists": False, "status": "missing"}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive for local corrupt files
        return {
            "artifact": str(path),
            "exists": True,
            "status": "invalid_json",
            "error": str(exc),
        }
    if not isinstance(data, dict):
        return {"artifact": str(path), "exists": True, "status": "invalid_shape"}
    data = dict(data)
    data.setdefault("status", None)
    data["artifact"] = str(path)
    data["exists"] = True
    return data


def _dir_exists(path: Path) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "status": "present" if path.exists() else "missing",
    }


def _check(name: str, ok: bool, evidence: str, detail: Any = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "ok": bool(ok),
        "evidence": evidence,
    }
    if detail is not None:
        row["detail"] = detail
    return row


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _release_manifest_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    component_ok = _get(data, "current_proof_sweep", "component_ok")
    if not isinstance(component_ok, dict):
        component_ok = data.get("component_ok")
    component_ok = component_ok if isinstance(component_ok, dict) else {}
    return [
        _check(
            "release_manifest_exists",
            data.get("exists") is True,
            str(RELEASE_MANIFEST),
            data.get("status"),
        ),
        _check(
            "prepackage_ready",
            data.get("prepackage_ready") is True,
            str(RELEASE_MANIFEST),
            data.get("prepackage_ready"),
        ),
        _check(
            "release_ready",
            data.get("release_ready") is True,
            str(RELEASE_MANIFEST),
            data.get("release_ready"),
        ),
        _check(
            "packaged_integrity_component",
            component_ok.get("packaged_integrity_matrix") is True
            or component_ok.get("packaged_integrity_contracts") is True,
            str(RELEASE_MANIFEST),
            component_ok.get("packaged_integrity_matrix")
            if "packaged_integrity_matrix" in component_ok
            else component_ok.get("packaged_integrity_contracts"),
        ),
        _check(
            "real_ui_live_model_matrix",
            component_ok.get("real_ui_full_model_matrix") is True
            and component_ok.get("real_ui_live_model_proof") is True,
            str(RELEASE_MANIFEST),
            {
                "real_ui_full_model_matrix": component_ok.get(
                    "real_ui_full_model_matrix"
                ),
                "real_ui_live_model_proof": component_ok.get(
                    "real_ui_live_model_proof"
                ),
            },
        ),
    ]


def _mimo_classifier_checks(
    audit: dict[str, Any], classifier: dict[str, Any]
) -> list[dict[str, Any]]:
    component_ok = audit.get("component_ok")
    component_ok = component_ok if isinstance(component_ok, dict) else {}
    excluded = classifier.get("excluded_surfaces")
    excluded = excluded if isinstance(excluded, dict) else {}
    unresolved = classifier.get("unresolved_surfaces")
    unresolved = unresolved if isinstance(unresolved, dict) else {}
    classification = classifier.get("classification")
    exactness_green = component_ok.get("artifact_exactness") is True
    exactness_open = component_ok.get("artifact_exactness") is not True
    accepted_open_classifications = {
        "model_generated_literal_mutation_after_valid_parser_structure",
        "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes",
        "jangtq2_plain_literal_copy_fails_before_parser_or_json_repair",
        "jangtq2_literal_corruption_persists_without_cache_fastpath_router_and_tq_kernel_parity_passes",
        "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer",
    }
    expected_open_classification = (
        classification in accepted_open_classifications
        and unresolved.get("artifact_quantization_or_decode_logits_quality") is True
    )
    exactness_boundary_details = {
        "artifact_exactness": component_ok.get("artifact_exactness"),
        "classifier_status": classifier.get("status"),
        "classification": classification,
        "unresolved_surfaces": unresolved,
    }
    if (
        isinstance(classification, str)
        and classification.startswith("jangtq2_plain_literal_copy")
    ) or classification == (
        "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer"
    ):
        exactness_boundary_details["artifact_exactness_release_action"] = (
            "replace_all_routed_2bit_jangtq2_or_lift_gate_down_precision"
        )
        exactness_boundary_details["recommended_next_artifact_profiles"] = [
            "JANGTQ gate=3/up=2/down=3",
            "JANGTQ gate=3/up=3/down=3",
        ]
    return [
        _check(
            "mimo_no_source_exactness_classifier_exists",
            classifier.get("exists") is True,
            str(MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER),
            classifier.get("status"),
        ),
        _check(
            "mimo_no_source_classifier_no_source_load",
            classifier.get("source_vs_quant_load_performed") is False,
            str(MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER),
            classifier.get("source_vs_quant_load_skipped_reason"),
        ),
        _check(
            "mimo_no_source_classifier_excludes_parser_cache_sampling",
            excluded.get("parser_argument_rewrite") is True
            and excluded.get("prefix_paged_l2_or_kv_quant_primary_cause") is True
            and excluded.get("hidden_stochastic_sampling_primary_cause") is True,
            str(MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER),
            excluded,
        ),
        _check(
            "mimo_no_source_classifier_tracks_exactness_boundary",
            (exactness_open and expected_open_classification)
            or (exactness_green and classifier.get("status") == "pass"),
            str(MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER),
            exactness_boundary_details,
        ),
    ]


def _mimo_media_current_checks(
    media_runtime: dict[str, Any],
    video_audio: dict[str, Any],
    responses_tools_cache_ui: dict[str, Any],
) -> list[dict[str, Any]]:
    runtime_proven = media_runtime.get("proven")
    runtime_proven = runtime_proven if isinstance(runtime_proven, dict) else {}
    runtime_not_proven = media_runtime.get("not_proven")
    runtime_not_proven = runtime_not_proven if isinstance(runtime_not_proven, dict) else {}
    video_audio_proven = video_audio.get("proven")
    video_audio_proven = video_audio_proven if isinstance(video_audio_proven, dict) else {}
    video_audio_not_proven = video_audio.get("not_proven")
    video_audio_not_proven = (
        video_audio_not_proven if isinstance(video_audio_not_proven, dict) else {}
    )
    ui_cache = responses_tools_cache_ui.get("cache")
    ui_cache = ui_cache if isinstance(ui_cache, dict) else {}
    ui_runtime = responses_tools_cache_ui.get("runtime")
    ui_runtime = ui_runtime if isinstance(ui_runtime, dict) else {}
    source_media_runtime_green = (
        media_runtime.get("exists") is True
        and runtime_proven.get("api_routes_mllm") is True
        and runtime_proven.get("loader_overlay_auto_enabled") is True
        and runtime_proven.get("preserved_media_weights_bound") is True
        and runtime_proven.get("live_chat_image_200") is True
        and runtime_proven.get("prior_unsupported_media_400_cleared_for_source_image")
        is True
    )
    source_video_audio_route_green = (
        video_audio.get("exists") is True
        and video_audio_proven.get("source_server_loads_as_mllm") is True
        and video_audio_proven.get("media_weights_bound") is True
        and video_audio_proven.get("video_request_reaches_runtime") is True
        and video_audio_proven.get("video_http_200") is True
        and video_audio_proven.get("audio_request_reaches_runtime") is True
        and video_audio_proven.get("audio_http_200") is True
    )
    ui_responses_cache_green = (
        responses_tools_cache_ui.get("exists") is True
        and responses_tools_cache_ui.get("status") == "pass"
        and responses_tools_cache_ui.get("classification")
        == "dev_app_responses_tools_cache_green_exactness_still_bounded"
        and ui_cache.get("nativeCacheSubtype") == "mimo_v2_asymmetric_swa"
        and _positive_number(ui_cache.get("cacheHitTokens"))
        and _positive_number(ui_cache.get("l2TokensOnDisk"))
        and ui_runtime.get("quantizationProfile") == "JANGTQ_2"
    )
    return [
        _check(
            "mimo_jangtq2_current_source_media_runtime",
            source_media_runtime_green,
            str(MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE),
            {
                "status": media_runtime.get("status"),
                "proven": runtime_proven,
                "not_proven": runtime_not_proven,
            },
        ),
        _check(
            "mimo_jangtq2_current_source_video_audio_routes",
            source_video_audio_route_green,
            str(MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE),
            {
                "status": video_audio.get("status"),
                "proven": video_audio_proven,
                "not_proven": video_audio_not_proven,
            },
        ),
        _check(
            "mimo_jangtq2_dev_app_responses_tools_cache",
            ui_responses_cache_green,
            str(MIMO_JANGTQ2_RESPONSES_TOOLS_CACHE_UI),
            {
                "status": responses_tools_cache_ui.get("status"),
                "classification": responses_tools_cache_ui.get("classification"),
                "cache": ui_cache,
                "runtime": ui_runtime,
            },
        ),
        _check(
            "mimo_jangtq2_media_semantics_release_quality",
            video_audio_not_proven.get("video_semantic_correctness") is not True
            and video_audio_not_proven.get("solid_color_image_semantic_correctness")
            is not True
            and runtime_not_proven.get("mimo_exactness") is not True,
            str(MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE),
            {
                "source_not_proven": video_audio_not_proven,
                "runtime_not_proven": runtime_not_proven,
            },
        ),
    ]


def _mimo_checks(
    data: dict[str, Any],
    classifier: dict[str, Any],
    media_runtime: dict[str, Any],
    video_audio: dict[str, Any],
    responses_tools_cache_ui: dict[str, Any],
) -> list[dict[str, Any]]:
    component_ok = data.get("component_ok")
    component_ok = component_ok if isinstance(component_ok, dict) else {}
    blockers = data.get("blockers")
    blockers = blockers if isinstance(blockers, list) else []
    diagnostics = data.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    all_local_smoke = diagnostics.get("all_local_smoke")
    all_local_smoke = all_local_smoke if isinstance(all_local_smoke, dict) else {}
    jang2l_smoke = diagnostics.get("jang2l_all_local_smoke")
    jang2l_smoke = jang2l_smoke if isinstance(jang2l_smoke, dict) else {}
    classifier_unresolved = classifier.get("unresolved_surfaces")
    classifier_unresolved = (
        classifier_unresolved if isinstance(classifier_unresolved, dict) else {}
    )
    media_runtime_rows = _mimo_media_current_checks(
        media_runtime, video_audio, responses_tools_cache_ui
    )
    media_current_by_name = {row["name"]: row for row in media_runtime_rows}
    current_source_media_runtime = media_current_by_name[
        "mimo_jangtq2_current_source_media_runtime"
    ]["ok"]
    current_source_video_audio_routes = media_current_by_name[
        "mimo_jangtq2_current_source_video_audio_routes"
    ]["ok"]
    artifact_exactness_detail = {
        "blockers": blockers,
        "jangtq2_boundary": all_local_smoke.get("artifact_exactness_boundary"),
        "jang2l_boundary": jang2l_smoke.get("artifact_exactness_boundary"),
        "prompt_shape_first_token": diagnostics.get("prompt_shape_first_token"),
        "classifier": {
            "status": classifier.get("status"),
            "classification": classifier.get("classification"),
            "secondary_classification": classifier.get("secondary_classification"),
            "model_upload_action_required": classifier.get(
                "model_upload_action_required"
            ),
            "model_upload_action_reasons": classifier.get(
                "model_upload_action_reasons"
            ),
            "unresolved_surfaces": classifier_unresolved,
            "required_next_evidence": classifier.get("required_next_evidence"),
        },
    }
    required = [
        "manifest_integrity",
        "decode_speed_target",
        "api_cache_responses_contract",
        "prefix_paged_l2_cache_reproved",
        "long_prompt_coherence",
        "tool_protocol",
        "artifact_exactness",
        "cb_system_prompt_working_set_pressure",
        "source_vs_quant_requirement_satisfied",
        "media_weights_preserved",
        "media_runtime_capabilities_safe",
        "media_model_metadata_text_only_contract",
        "media_runtime_implementation",
        "mimo_media_wired",
    ]
    rows = [
        _check(
            "mimo_audit_exists",
            data.get("exists") is True,
            str(MIMO_AUDIT),
            data.get("status"),
        ),
        _check(
            "mimo_local_release_clearance",
            data.get("local_release_clearance") is True,
            str(MIMO_AUDIT),
            blockers,
        ),
    ]
    for name in required:
        detail = artifact_exactness_detail if name == "artifact_exactness" else None
        ok = component_ok.get(name) is True
        if name == "media_runtime_implementation":
            ok = ok or current_source_media_runtime
            detail = {
                "audit_component_ok": component_ok.get(name),
                "current_source_media_runtime": current_source_media_runtime,
                "source_evidence": str(MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE),
            }
        elif name == "mimo_media_wired":
            ok = ok or (
                current_source_media_runtime and current_source_video_audio_routes
            )
            detail = {
                "audit_component_ok": component_ok.get(name),
                "current_source_media_runtime": current_source_media_runtime,
                "current_source_video_audio_routes": current_source_video_audio_routes,
                "source_evidence": [
                    str(MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE),
                    str(MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE),
                ],
            }
        rows.append(
            _check(
                f"mimo_{name}",
                ok,
                str(MIMO_AUDIT),
                detail,
            )
        )
    rows.extend(media_runtime_rows)
    rows.extend(_mimo_classifier_checks(data, classifier))
    return rows


def _noheavy_api_cache_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    required = [
        "openai_chat_sampling_kwargs",
        "responses_sampling_kwargs",
        "streaming_cache_detail_usage",
        "responses_previous_response_history",
        "cache_stats_reuse_skip_telemetry",
        "cache_reuse_endpoints",
        "legacy_completions_output_caps_override_server_default",
        "request_output_caps_override_server_default",
        "prompt_context_caps_stay_separate_from_output_caps",
        "json_response_format_stream_validation",
        "responses_text_format_json_schema_preserved",
        "anthropic_bundle_defaults",
        "ollama_adapter_surface",
    ]
    rows = [
        _check(
            "noheavy_api_cache_contract_exists",
            data.get("exists") is True,
            str(NOHEAVY_API_CACHE),
            data.get("status"),
        ),
        _check(
            "noheavy_api_cache_contract_status_pass",
            data.get("status") == "pass",
            str(NOHEAVY_API_CACHE),
            data.get("status"),
        ),
    ]
    rows.extend(
        _check(name, checks.get(name) is True, str(NOHEAVY_API_CACHE))
        for name in required
    )
    return rows


def _api_surface_contract_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    required = [
        "openai_chat_completions_sampling_defaults",
        "openai_responses_sampling_defaults",
        "legacy_completions_output_caps_override_server_default",
        "chat_and_responses_output_caps_override_server_default",
        "prompt_context_caps_stay_separate_from_output_caps",
        "anthropic_adapter_bundle_defaults",
        "ollama_adapter_streaming_done_behavior",
        "chat_and_responses_streaming_cache_detail_usage",
        "responses_previous_response_history",
        "server_cache_and_tool_surfaces_named",
        "plain_attention_kv_status",
        "dsv4_native_cache_status",
        "zaya_typed_cca_status",
        "hybrid_ssm_partial_reuse",
        "turboquant_kv_runtime_contract",
        "turboquant_disk_roundtrip",
        "no_generic_tq_on_hybrid_ssm",
        "panel_request_builder_sampling_and_output_overrides",
        "panel_ollama_gateway_omits_disabled_sentinels",
        "panel_gateway_single_model_auto_switch_streaming",
        "panel_gateway_single_model_auto_switch_cache_endpoints",
        "panel_chat_override_policy_preserves_explicit_values",
        "panel_gateway_streaming_disconnect_epipe_guard",
        "panel_child_process_stdio_epipe_guard",
        "panel_backend_stderr_split_epipe_guard",
        "panel_gateway_ollama_proxy_response_error_guard",
        "panel_ipc_backend_request_epipe_guard",
        "panel_performance_health_epipe_guard",
        "panel_session_lifecycle_epipe_guard",
        "all_required_panel_api_markers_present",
    ]
    rows = [
        _check(
            "api_surface_contract_exists",
            data.get("exists") is True,
            str(API_SURFACE_CONTRACT),
            data.get("status"),
        ),
        _check(
            "api_surface_contract_status_pass",
            data.get("status") == "pass",
            str(API_SURFACE_CONTRACT),
            data.get("status"),
        ),
    ]
    rows.extend(
        _check(f"api_surface_{name}", checks.get(name) is True, str(API_SURFACE_CONTRACT))
        for name in required
    )
    return rows


def _panel_settings_contract_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    commands = data.get("commands")
    commands = commands if isinstance(commands, dict) else {}
    settings_counts = _get(
        commands,
        "panel_settings_contracts",
        "counts",
        default={},
    )
    required = [
        "panel_settings_contract_count",
        "dsv4_default_native_prefix_on",
        "dsv4_explicit_prefix_off_disables_native_flags",
        "dsv4_l2_explicit_off_preserves_prefix",
        "dsv4_generic_kv_flags_suppressed",
        "dsv4_pool_quant_controls_are_dsv4_only",
        "max_output_context_cli_split",
        "chat_max_output_is_per_chat_override",
        "non_dsv4_cache_toggles_preserved",
        "i18n_max_output_context_copy",
        "lazy_mmap_launch_memory_admission",
        "minimax_reasoning_parser_detection",
        "native_mtp_d3_default_policy",
        "model_family_parser_registry",
        "engine_cache_architecture_registry",
        "panel_emitted_flags_are_registered_engine_cli_flags",
        "panel_typecheck",
    ]
    rows = [
        _check(
            "panel_settings_contract_exists",
            data.get("exists") is True,
            str(PANEL_SETTINGS_CONTRACT),
            data.get("status"),
        ),
        _check(
            "panel_settings_contract_status_pass",
            data.get("status") == "pass",
            str(PANEL_SETTINGS_CONTRACT),
            data.get("status"),
        ),
        _check(
            "panel_settings_contract_coverage",
            _get(settings_counts, "test_files_passed") == 6
            and (_get(settings_counts, "tests_passed") or 0) >= 260,
            str(PANEL_SETTINGS_CONTRACT),
            settings_counts,
        ),
        _check(
            "panel_settings_no_missing_source_markers",
            data.get("missing_source_markers") == [],
            str(PANEL_SETTINGS_CONTRACT),
            data.get("missing_source_markers"),
        ),
    ]
    rows.extend(
        _check(f"panel_{name}", checks.get(name) is True, str(PANEL_SETTINGS_CONTRACT))
        for name in required
    )
    return rows


def _responses_raw_sse_parity_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    detail = {"missing_captures": data.get("missing_captures")}
    required = [
        "direct_capture_present",
        "gateway_capture_present",
        "tunnel_capture_present",
        "all_required_surfaces_present",
        "all_present_surfaces_parse_cleanly",
        "all_present_surfaces_match_expected_function_name",
        "all_present_surfaces_match_expected_arguments",
        "all_present_surfaces_have_authoritative_args",
        "authoritative_arguments_match_across_present_surfaces",
        "tunnel_expected_model_advertised",
        "local_responses_streaming_guards_pass",
        "local_empty_xml_arguments_fail_closed",
        "local_output_index_ordering_guard",
        "gateway_argument_stream_passthrough_guard",
        "responses_previous_response_history_guard",
        "all_present_surfaces_have_valid_output_item_indices",
        "all_present_surfaces_have_required_reasoning",
        "no_reasoning_disable_workaround",
    ]
    rows = _simple_artifact_checks("responses_raw_sse_parity", data)
    rows.extend(
        _check(
            f"responses_raw_sse_parity_{name}",
            data.get("status") == "pass"
            and checks.get(name) is True
            if name == "all_required_surfaces_present"
            else checks.get(name) is True,
            str(RESPONSES_RAW_SSE_PARITY),
            detail if name == "all_required_surfaces_present" else checks.get(name),
        )
        for name in required
    )
    return rows


def _tool_call_contract_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    required = [
        "tool_parser_residue_rejected_instead_of_executed",
        "schema_valid_dsml_tool_call_preserved",
        "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid",
        "dsv4_tool_preamble_suppressed_and_not_stored_without_call",
        "tool_choice_none_does_not_fallback_to_raw_dsml",
        "family_tool_parser_matrix_covers_no_leak_and_alias_edges",
        "panel_max_tool_iterations_caps_tool_loops",
        "all_required_tool_call_markers_present",
    ]
    rows = [
        _check(
            "tool_call_contract_exists",
            data.get("exists") is True,
            str(TOOL_CALL_CONTRACT),
            data.get("status"),
        )
    ]
    rows.extend(
        _check(name, checks.get(name) is True, str(TOOL_CALL_CONTRACT))
        for name in required
    )
    return rows


def _objective_requirement(
    data: dict[str, Any], needle: str
) -> dict[str, Any] | None:
    requirements = data.get("requirements")
    if not isinstance(requirements, list):
        return None
    for row in requirements:
        if not isinstance(row, dict):
            continue
        requirement = str(row.get("requirement") or row.get("name") or "")
        if needle in requirement:
            return row
    return None


def _n2_pro_397b_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    row = _objective_requirement(data, "N2 Pro 397B JANG1L/JANGTQ")
    details = row.get("details") if isinstance(row, dict) else {}
    details = details if isinstance(details, dict) else {}
    detail = {
        "status": row.get("status") if isinstance(row, dict) else "missing",
        "requirement": row.get("requirement") if isinstance(row, dict) else None,
        "evidence": row.get("evidence") if isinstance(row, dict) else None,
        "local_artifact_probe": details.get("local_artifact_probe"),
        "jang1l_live_gate": details.get("jang1l_live_gate"),
        "required_next_evidence": details.get("required_next_evidence"),
    }
    return [
        _check(
            "n2_pro_397b_objective_row_exists",
            row is not None,
            str(OBJECTIVE_DIGEST),
            data.get("status"),
        ),
        _check(
            "n2_pro_397b_release_clearance",
            isinstance(row, dict) and row.get("status") == "pass",
            str(OBJECTIVE_DIGEST),
            detail,
        ),
    ]


def _simple_artifact_checks(name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _check(f"{name}_artifact_exists", data.get("exists") is True, data["artifact"]),
        _check(
            f"{name}_status_pass",
            data.get("status") == "pass",
            data["artifact"],
            data.get("status"),
        ),
    ]


def _requests(data: dict[str, Any]) -> list[dict[str, Any]]:
    requests = data.get("requests")
    return [row for row in requests if isinstance(row, dict)] if isinstance(requests, list) else []


def _all_request_validations_clean(data: dict[str, Any]) -> bool:
    failures = data.get("failures")
    if failures != []:
        return False
    rows = _requests(data)
    return bool(rows) and all(row.get("validation_failures") == [] for row in rows)


def _has_record_fact_tool_call(data: dict[str, Any]) -> bool:
    for row in _requests(data):
        tool_calls = row.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            fn = _get(call, "function", default={})
            if (
                isinstance(fn, dict)
                and fn.get("name") == "record_fact"
                and '"blue-cat"' in str(fn.get("arguments"))
            ):
                return True
    return False


def _has_cache_reuse_request(data: dict[str, Any], expected_detail: str) -> bool:
    for row in _requests(data):
        details = _get(row, "usage", "prompt_tokens_details", default={})
        if (
            _positive_number(_get(details, "cached_tokens"))
            and details.get("cache_detail") == expected_detail
            and _get(row, "cache_summary", "has_cache_hit") is True
        ):
            return True
    return False


def _native_mixed_swa_cache_ok(
    native: dict[str, Any],
    *,
    family: str,
    subtype: str | None = None,
) -> bool:
    native = native if isinstance(native, dict) else {}
    subtype_ok = True if subtype is None else native.get("cache_subtype") == subtype
    return (
        native.get("family") == family
        and native.get("schema") == "mixed_swa_kv_v1"
        and native.get("cache_type") == "mixed_swa_kv"
        and subtype_ok
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
        and _get(native, "generic_turboquant_kv", "enabled") is False
        and _get(native, "storage_quantization", "enabled") is True
        and _get(native, "storage_quantization", "bits") == 4
    )


def _smoke_mixed_swa_checks(
    name: str,
    data: dict[str, Any],
    *,
    path: Path,
    family: str,
    tool_parser: str,
    reasoning_parser: str,
    expected_mllm: bool,
    cache_detail: str,
    subtype: str | None = None,
    cache_proof: dict[str, Any] | None = None,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    row = data.get("row") if isinstance(data.get("row"), dict) else {}
    capabilities = _get(data, "capabilities", "body", default={})
    cache_source = cache_proof if isinstance(cache_proof, dict) else data
    cache_evidence = cache_path or path
    native = (
        cache_source.get("native_cache")
        if isinstance(cache_source.get("native_cache"), dict)
        else _get(cache_source, "cache_after", "body", "native_cache", default={})
    )
    block_disk = _get(cache_source, "cache_after", "block_disk_cache", default={})
    if not isinstance(block_disk, dict) or not block_disk:
        block_disk = _get(cache_source, "cache_after", "body", "block_disk_cache", default={})
    totals = _get(cache_source, "cache_after", "cache_totals", default={})
    if not isinstance(totals, dict) or not totals:
        totals = _get(cache_source, "cache_after", "body", "cache_totals", default={})
    cache_reuse_ok = _has_cache_reuse_request(data, cache_detail) or (
        _get(cache_source, "checks", "cache_second_hit") is True
        and _get(cache_source, "second", "usage", "prompt_tokens_details", "cache_detail")
        == cache_detail
    )
    return _simple_artifact_checks(name, data) + [
        _check(f"{name}_request_validations_clean", _all_request_validations_clean(data), str(path), data.get("failures")),
        _check(f"{name}_mllm_classification", row.get("is_mllm") is expected_mllm, str(path), row.get("is_mllm")),
        _check(f"{name}_tool_parser", capabilities.get("tool_parser") == tool_parser, str(path), capabilities.get("tool_parser")),
        _check(f"{name}_reasoning_parser", capabilities.get("reasoning_parser") == reasoning_parser, str(path), capabilities.get("reasoning_parser")),
        _check(f"{name}_tool_call_record_fact", _has_record_fact_tool_call(data), str(path)),
        _check(f"{name}_cache_reuse_detail", cache_reuse_ok, str(cache_evidence), cache_detail),
        _check(f"{name}_native_mixed_swa_cache", _native_mixed_swa_cache_ok(native, family=family, subtype=subtype), str(cache_evidence), native),
        _check(f"{name}_block_l2_written", _positive_number(_get(block_disk, "disk_writes")) and _positive_number(_get(totals, "l2_block_tokens_on_disk")), str(cache_evidence), block_disk),
    ]


def _gemma4_12b_media_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    row = _get(data, "rows", "jang4m_image", default={})
    row = row if isinstance(row, dict) else {}
    checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
    return _simple_artifact_checks(
        "gemma4_12b_jang4m_media",
        data,
    ) + [
        _check(
            "gemma4_12b_media_rows_complete",
            data.get("status") == "pass"
            and _get(data, "checks", "all_rows_passed") is True
            and row.get("status") == "pass"
            and row.get("failures") == []
            and _get(row, "chat", "code") == 200
            and checks.get("vision_advertised") is True
            and checks.get("red_detected") is True
            and checks.get("no_channel_leak") is True,
            str(GEMMA4_12B_JANG4M_MEDIA_SMOKE),
            {
                "status": data.get("status"),
                "row_status": row.get("status"),
                "failures": row.get("failures"),
                "content": _get(row, "chat", "content"),
                "checks": checks,
            },
        )
    ]


def _gemma4_12b_issue191_startup_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    content = _get(data, "chat_response", "choices", default=[])
    first_choice = content[0] if isinstance(content, list) and content else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    visible = message.get("content") if isinstance(message, dict) else None
    return _simple_artifact_checks("gemma4_12b_issue191_startup", data) + [
        _check(
            "gemma4_12b_issue191_startup_visible_generation",
            data.get("status") == "pass"
            and checks.get("import_alias_ok") is True
            and checks.get("startup_health_ok") is True
            and checks.get("visible_generation_ok") is True
            and checks.get("post_chat_health_ok") is True
            and visible == "GEMMA4-OK"
            and first_choice.get("finish_reason") == "stop",
            str(GEMMA4_12B_ISSUE191_STARTUP_VISIBLE),
            {
                "status": data.get("status"),
                "model": data.get("model"),
                "checks": checks,
                "content": visible,
                "finish_reason": first_choice.get("finish_reason"),
            },
        )
    ]


def _gemma_qat_native_mxfp4_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    gemma12b_audio_ok = (
        checks.get("gemma4_12b_audio_weight_backed") is True
        or checks.get("gemma4_12b_audio_honestly_gated") is True
    )
    detail = {
        "missing_required_rows": data.get("missing_required_rows"),
        "open_required_rows": data.get("open_required_rows"),
        "source_live_smoke_open_rows": data.get("source_live_smoke_open_rows"),
    }
    required_rows = data.get("required_rows", {})
    required_rows = required_rows if isinstance(required_rows, dict) else {}
    qat_jang4m_keys = (
        "gemma4_e2b_qat_jang4m",
        "gemma4_e4b_qat_jang4m",
        "gemma4_12b_qat_jang4m",
        "gemma4_26b_qat_jang4m",
        "gemma4_31b_qat_jang4m",
    )
    qat_jang4m_checks: list[dict[str, Any]] = []
    for row_key in qat_jang4m_keys:
        row = required_rows.get(row_key)
        row = row if isinstance(row, dict) else {}
        check_prefix = f"gemma_qat_native_mxfp4_{row_key}"
        qat_jang4m_checks.extend(
            [
                _check(
                    f"{check_prefix}_present",
                    checks.get(f"{row_key}_present") is True or bool(row),
                    str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
                    row,
                ),
                _check(
                    f"{check_prefix}_open",
                    row.get("status") == "pass"
                    and row.get("live_proof_status") == "pass",
                    str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
                    row
                    or (
                        "Gemma4 QAT JANG_4M must remain open until autodetect, "
                        "generation_config defaults, Gemma4 parser, mixed-SWA/prefix "
                        "cache, TurboQuant KV boundary, L2, Responses streaming "
                        "args/content deltas, media honesty, UI/CLI, and installed-app "
                        "parity are live-proven."
                    ),
                ),
            ]
        )
    return _simple_artifact_checks("gemma_qat_native_mxfp4", data) + qat_jang4m_checks + [
        _check(
            "gemma_qat_native_mxfp4_gemma4_e2b_present",
            checks.get("gemma4_e2b_qat_native_mxfp4_present") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            data.get("required_rows", {}).get("gemma4_e2b_qat_native_mxfp4")
            if isinstance(data.get("required_rows"), dict)
            else None,
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_e2b_qat_jang4m_installed_app_ui_api_cache_proven",
            checks.get("gemma4_e2b_qat_jang4m_installed_app_ui_api_cache_proven") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            data.get("required_rows", {}).get("gemma4_e2b_qat_jang4m")
            if isinstance(data.get("required_rows"), dict)
            else None,
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_e4b_present",
            checks.get("gemma4_e4b_qat_native_mxfp4_present") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            data.get("required_rows", {}).get("gemma4_e4b_qat_native_mxfp4")
            if isinstance(data.get("required_rows"), dict)
            else None,
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_e4b_qat_jang4m_installed_app_ui_api_cache_proven",
            checks.get("gemma4_e4b_qat_jang4m_installed_app_ui_api_cache_proven") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            data.get("required_rows", {}).get("gemma4_e4b_qat_jang4m")
            if isinstance(data.get("required_rows"), dict)
            else None,
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_12b_present",
            checks.get("gemma4_12b_native_mxfp4_present") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_12b_qat_jang4m_installed_app_ui_api_cache_proven",
            checks.get("gemma4_12b_qat_jang4m_installed_app_ui_api_cache_proven") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            data.get("required_rows", {}).get("gemma4_12b_qat_jang4m")
            if isinstance(data.get("required_rows"), dict)
            else None,
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_26b_present",
            checks.get("gemma4_26b_present") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_31v_or_31b_present",
            checks.get("gemma4_31v_or_31b_present") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_12b_audio_weight_backed",
            gemma12b_audio_ok,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            {
                "gemma4_12b_audio_weight_backed": checks.get("gemma4_12b_audio_weight_backed"),
                "gemma4_12b_audio_honestly_gated": checks.get(
                    "gemma4_12b_audio_honestly_gated"
                ),
                "row": data.get("required_rows", {}).get("gemma4_12b_native_mxfp4")
                if isinstance(data.get("required_rows"), dict)
                else None,
            },
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_12b_video_runtime_proven",
            checks.get("gemma4_12b_video_runtime_proof_required") is not True
            or checks.get("gemma4_12b_video_runtime_source_proven") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            {
                "gemma4_12b_video_runtime_proof_required": checks.get(
                    "gemma4_12b_video_runtime_proof_required"
                ),
                "gemma4_12b_video_runtime_source_proven": checks.get(
                    "gemma4_12b_video_runtime_source_proven"
                ),
                "row": data.get("required_rows", {}).get("gemma4_12b_native_mxfp4")
                if isinstance(data.get("required_rows"), dict)
                else None,
            },
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_26b_video_runtime_proven",
            checks.get("gemma4_26b_video_runtime_proof_required") is not True
            or checks.get("gemma4_26b_video_runtime_source_proven") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            {
                "gemma4_26b_video_runtime_proof_required": checks.get(
                    "gemma4_26b_video_runtime_proof_required"
                ),
                "gemma4_26b_video_runtime_source_proven": checks.get(
                    "gemma4_26b_video_runtime_source_proven"
                ),
                "row": data.get("required_rows", {}).get("gemma4_26b_vl")
                if isinstance(data.get("required_rows"), dict)
                else None,
            },
        ),
        _check(
            "gemma_qat_native_mxfp4_gemma4_31v_or_31b_video_runtime_proven",
            checks.get("gemma4_31v_or_31b_video_runtime_proof_required") is not True
            or checks.get("gemma4_31v_or_31b_video_runtime_source_proven") is True,
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            {
                "gemma4_31v_or_31b_video_runtime_proof_required": checks.get(
                    "gemma4_31v_or_31b_video_runtime_proof_required"
                ),
                "gemma4_31v_or_31b_video_runtime_source_proven": checks.get(
                    "gemma4_31v_or_31b_video_runtime_source_proven"
                ),
                "row": data.get("required_rows", {}).get("gemma4_31v_or_31b_vl")
                if isinstance(data.get("required_rows"), dict)
                else None,
            },
        ),
        _check(
            "gemma_qat_native_mxfp4_all_source_live_smokes_present",
            checks.get("all_required_source_live_smokes_present") is True
            and data.get("missing_required_rows") == []
            and data.get("source_live_smoke_open_rows") == [],
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            detail,
        ),
        _check(
            "gemma_qat_native_mxfp4_all_live_proofs_present",
            data.get("status") == "pass"
            and checks.get("all_required_live_proofs_present") is True
            and data.get("missing_required_rows") == []
            and data.get("open_required_rows") == [],
            str(GEMMA_QAT_NATIVE_MXFP4_INVENTORY),
            detail,
        ),
    ]


def _step37_vlm_runtime_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = (
        data.get("mlx_vlm_step3p7_runtime")
        if isinstance(data.get("mlx_vlm_step3p7_runtime"), dict)
        else {}
    )
    progress = (
        data.get("source_owned_runtime_progress")
        if isinstance(data.get("source_owned_runtime_progress"), dict)
        else {}
    )
    contract = (
        data.get("mlx_vlm_implementation_contract")
        if isinstance(data.get("mlx_vlm_implementation_contract"), dict)
        else {}
    )
    live_media_proof = (
        data.get("live_media_proof")
        if isinstance(data.get("live_media_proof"), dict)
        else {}
    )
    required_progress = [
        "config_layer_available",
        "model_layer_available",
        "vision_layer_available",
        "projector_layer_available",
        "processor_layer_available",
        "image_embeds_merge_available",
        "pixel_values_processor_available",
        "vision_patch_embed_available",
        "vision_abs_posemb_resize_available",
        "vision_transformer_blocks_available",
        "vision_downsamplers_available",
        "pixel_values_to_projector_available",
        "placeholder_projected_token_count_available",
    ]
    runtime_symbols_ok = all(
        runtime.get(name) is True
        for name in (
            "has_model",
            "has_vision_model",
            "has_language_model",
            "has_model_config",
            "has_processor",
        )
    )
    progress_ok = all(progress.get(name) is True for name in required_progress)
    return [
        _check(
            "step37_vlm_runtime_audit_exists",
            data.get("exists") is True,
            str(STEP37_VLM_RUNTIME_AUDIT),
            data.get("status"),
        ),
        _check(
            "step37_vlm_noheavy_runtime_surface",
            data.get("status") == "pass"
            and data.get("mlx_vlm_step3p7_runtime_available") is True
            and runtime_symbols_ok
            and progress_ok
            and data.get("release_clearance") == "audit_does_not_block_release",
            str(STEP37_VLM_RUNTIME_AUDIT),
            {
                "runtime": runtime,
                "source_progress_release_clearance": progress.get("release_clearance"),
            },
        ),
        _check(
            "step37_vlm_rejects_fake_clearance_paths",
            {"text_only_bridge", "importable_stub"}.issubset(
                set(contract.get("explicitly_rejected_clearance_paths") or [])
            ),
            str(STEP37_VLM_RUNTIME_AUDIT),
            contract.get("explicitly_rejected_clearance_paths"),
        ),
        _check(
            "step37_real_vlm_runtime_complete",
            live_media_proof.get("pass") is True,
            str(STEP37_VLM_RUNTIME_AUDIT),
            live_media_proof
            or (
                "no-heavy Step3p7 VLM runtime surface exists, but live image/video "
                "media proof is still required before release clearance"
            ),
        ),
    ]


def _native_hybrid_ssm_cache_ok(native: dict[str, Any], *, family: str) -> bool:
    native = native if isinstance(native, dict) else {}
    return (
        native.get("family") == family
        and native.get("schema") == "hybrid_ssm_v1"
        and native.get("cache_type") == "hybrid_ssm_typed"
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
        and _get(native, "generic_turboquant_kv", "enabled") is False
        and _get(native, "attention_kv_storage_quantization", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "bits") == 4
        and _positive_number(native.get("ssm_entries"))
    )


def _request_by_name(data: dict[str, Any], name: str) -> dict[str, Any]:
    for row in _requests(data):
        if row.get("name") == name:
            return row
    return {}


def _check_row_by_name(data: dict[str, Any], name: str) -> dict[str, Any]:
    checks = data.get("checks")
    if not isinstance(checks, list):
        return {}
    for row in checks:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return {}


def _all_named_checks_pass(data: dict[str, Any]) -> bool:
    checks = data.get("checks")
    return (
        isinstance(checks, list)
        and bool(checks)
        and all(isinstance(row, dict) and row.get("ok") is True for row in checks)
    )


def _nemotron_omni_media_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    image = _request_by_name(data, "image_blue")
    video = _request_by_name(data, "video_blue")
    audio = _request_by_name(data, "audio_blue")
    recall = _request_by_name(data, "turn2_recall")
    cache_final = data.get("cache_final") if isinstance(data.get("cache_final"), dict) else {}
    native = cache_final.get("native_cache") if isinstance(cache_final.get("native_cache"), dict) else {}
    block_disk = cache_final.get("block_disk_cache") if isinstance(cache_final.get("block_disk_cache"), dict) else {}
    totals = cache_final.get("cache_totals") if isinstance(cache_final.get("cache_totals"), dict) else {}
    ssm_disk = _get(cache_final, "ssm_companion", "disk", default={})
    recall_details = _get(recall, "usage", "prompt_tokens_details", default={})
    log_tail = data.get("log_tail") if isinstance(data.get("log_tail"), str) else ""
    required_requests = {
        "image_blue": image,
        "video_blue": video,
        "audio_blue": audio,
        "turn2_recall": recall,
    }
    media_rows_ok = (
        str(data.get("status")).lower() == "pass"
        and data.get("passed") is True
        and _all_named_checks_pass(data)
        and all(row.get("code") == 200 for row in required_requests.values())
        and "blue" in str(image.get("content", "")).lower()
        and "blue" in str(video.get("content", "")).lower()
        and "blue" in str(audio.get("content", "")).lower()
        and "blue" in str(recall.get("content", "")).lower()
        and "cat" in str(recall.get("content", "")).lower()
    )
    cache_reuse_ok = (
        _positive_number(recall_details.get("cached_tokens"))
        and recall_details.get("cache_detail") == "paged+ssm"
        and _get(cache_final, "scheduler_stats", "cache_hit_tokens") == recall_details.get("cached_tokens")
        and _get(_check_row_by_name(data, "cache_stats_final_available"), "ok") is True
    )
    l2_ok = (
        _positive_number(block_disk.get("disk_writes"))
        and _positive_number(block_disk.get("disk_hits"))
        and _positive_number(totals.get("l2_block_tokens_on_disk"))
        and _positive_number(totals.get("l2_ssm_tokens_on_disk"))
        and _positive_number(ssm_disk.get("stores"))
        and _positive_number(ssm_disk.get("total_tokens_on_disk"))
    )
    return [
        _check(
            "nemotron_omni_media_gate_exists",
            data.get("exists") is True,
            str(NEMOTRON_OMNI_MEDIA_GATE),
            data.get("status"),
        ),
        _check(
            "nemotron_omni_media_rows_complete",
            media_rows_ok,
            str(NEMOTRON_OMNI_MEDIA_GATE),
            {
                "status": data.get("status"),
                "requests": {
                    name: {
                        "code": row.get("code"),
                        "content": row.get("content"),
                    }
                    for name, row in required_requests.items()
                },
            },
        ),
        _check(
            "nemotron_omni_media_cache_reuse_paged_ssm",
            cache_reuse_ok,
            str(NEMOTRON_OMNI_MEDIA_GATE),
            recall_details,
        ),
        _check(
            "nemotron_omni_media_l2_block_and_ssm",
            l2_ok,
            str(NEMOTRON_OMNI_MEDIA_GATE),
            {
                "block_disk": block_disk,
                "cache_totals": totals,
                "ssm_disk": ssm_disk,
            },
        ),
        _check(
            "nemotron_omni_media_native_hybrid_cache",
            _native_hybrid_ssm_cache_ok(native, family="nemotron_h"),
            str(NEMOTRON_OMNI_MEDIA_GATE),
            native,
        ),
        _check(
            "nemotron_omni_media_chat_completions_endpoint",
            "POST /v1/chat/completions" in log_tail,
            str(NEMOTRON_OMNI_MEDIA_GATE),
        ),
    ]


def _smoke_hybrid_ssm_checks(
    name: str,
    data: dict[str, Any],
    *,
    path: Path,
    family: str,
    tool_parser: str,
    reasoning_parser: str,
    expected_mllm: bool,
) -> list[dict[str, Any]]:
    row = data.get("row") if isinstance(data.get("row"), dict) else {}
    capabilities = _get(data, "capabilities", "body", default={})
    native = _get(data, "cache_after", "body", "native_cache", default={})
    block_disk = _get(data, "cache_after", "body", "block_disk_cache", default={})
    ssm_disk = _get(data, "cache_after", "body", "scheduler_cache", "ssm_companion_cache", "disk", default={})
    totals = _get(data, "cache_after", "body", "cache_totals", default={})
    return _simple_artifact_checks(name, data) + [
        _check(f"{name}_request_validations_clean", _all_request_validations_clean(data), str(path), data.get("failures")),
        _check(f"{name}_mllm_classification", row.get("is_mllm") is expected_mllm, str(path), row.get("is_mllm")),
        _check(f"{name}_tool_parser", capabilities.get("tool_parser") == tool_parser, str(path), capabilities.get("tool_parser")),
        _check(f"{name}_reasoning_parser", capabilities.get("reasoning_parser") == reasoning_parser, str(path), capabilities.get("reasoning_parser")),
        _check(f"{name}_tool_call_record_fact", _has_record_fact_tool_call(data), str(path)),
        _check(f"{name}_cache_reuse_detail", _has_cache_reuse_request(data, "paged+ssm"), str(path)),
        _check(f"{name}_native_hybrid_ssm_cache", _native_hybrid_ssm_cache_ok(native, family=family), str(path), native),
        _check(f"{name}_block_l2_hits_writes", _positive_number(_get(block_disk, "disk_writes")) and _positive_number(_get(block_disk, "disk_hits")), str(path), block_disk),
        _check(f"{name}_ssm_l2_written", _positive_number(_get(ssm_disk, "stores")) and _positive_number(_get(totals, "l2_ssm_tokens_on_disk")), str(path), ssm_disk),
    ]


def _native_hybrid_cache_ok(data: dict[str, Any]) -> bool:
    native = data if isinstance(data, dict) else {}
    return (
        native.get("schema") == "hybrid_ssm_v1"
        and native.get("cache_type") == "hybrid_ssm_typed"
        and native.get("prefix") is True
        and native.get("paged") is True
        and native.get("block_disk_l2") is True
        and _get(native, "generic_turboquant_kv", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "enabled") is True
        and _get(native, "attention_kv_storage_quantization", "bits") == 4
    )


def _mtp_runtime_active(data: dict[str, Any]) -> bool:
    mtp = data if isinstance(data, dict) else {}
    return (
        mtp.get("runtime_active") is True
        and mtp.get("runtime_available") is True
        and mtp.get("runtime_supported") is True
        and mtp.get("index_has_mtp_tensors") is True
        and mtp.get("status") == "native_runtime_active"
        and _positive_number(mtp.get("effective_depth"))
    )


def _qwen27_cancel_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    health = data.get("health_after") if isinstance(data.get("health_after"), dict) else {}
    if not health:
        health = data.get("health_before") if isinstance(data.get("health_before"), dict) else {}
    probe = data.get("probe") if isinstance(data.get("probe"), dict) else {}
    raw = data.get("raw") if isinstance(data.get("raw"), dict) else {}
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    return [
        _check("qwen27_cancel_artifact_exists", data.get("exists") is True, str(QWEN27_RESPONSES_CANCEL), data.get("status")),
        _check("qwen27_cancel_status_pass", data.get("status") == "pass", str(QWEN27_RESPONSES_CANCEL), data.get("status")),
        _check("qwen27_cancel_responses_streaming", request.get("stream") is True and bool(raw.get("response_id")), str(QWEN27_RESPONSES_CANCEL)),
        _check("qwen27_cancel_route_success", probe.get("cancel_route_present") is True and raw.get("cancel_status") == 200, str(QWEN27_RESPONSES_CANCEL)),
        _check("qwen27_cancel_no_bad_text", probe.get("bad_text_captured") is False and raw.get("stream_error") is None, str(QWEN27_RESPONSES_CANCEL)),
        _check("qwen27_cancel_mtp_runtime_active", _mtp_runtime_active(health.get("mtp") if isinstance(health, dict) else {}), str(QWEN27_RESPONSES_CANCEL)),
        _check("qwen27_cancel_hybrid_cache_policy", _native_hybrid_cache_ok(health.get("native_cache") if isinstance(health, dict) else {}), str(QWEN27_RESPONSES_CANCEL)),
    ]


def _qwen27_api_parity_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    health = data.get("health_after") if isinstance(data.get("health_after"), dict) else {}
    native = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    mtp = health.get("mtp") if isinstance(health.get("mtp"), dict) else {}
    block_disk = _get(health, "cache", "block_disk_cache", default={})
    ssm_disk = _get(health, "cache", "ssm_companion", "disk", default={})
    return [
        _check("qwen27_api_parity_artifact_exists", data.get("exists") is True, str(QWEN27_API_PARITY), data.get("status")),
        _check("qwen27_api_responses_text", _get(checks, "responses_text", "code") == 200 and _get(checks, "responses_text", "text_head") == "ACK", str(QWEN27_API_PARITY)),
        _check("qwen27_api_responses_required_tool", _get(checks, "responses_tool_required", "code") == 200 and _get(checks, "responses_tool_required", "has_record_fact") is True, str(QWEN27_API_PARITY)),
        _check("qwen27_api_anthropic", _get(checks, "anthropic_messages", "code") == 200 and _get(checks, "anthropic_messages", "text_head") == "ACK", str(QWEN27_API_PARITY)),
        _check("qwen27_api_ollama", _get(checks, "ollama_chat", "code") == 200 and _get(checks, "ollama_chat", "text_head") == "ACK", str(QWEN27_API_PARITY)),
        _check("qwen27_api_chat_stream_sse", _get(checks, "chat_stream_sse", "has_ack") is True, str(QWEN27_API_PARITY)),
        _check("qwen27_api_mtp_runtime_active", _mtp_runtime_active(mtp), str(QWEN27_API_PARITY)),
        _check("qwen27_api_hybrid_cache_policy", _native_hybrid_cache_ok(native), str(QWEN27_API_PARITY)),
        _check("qwen27_api_cache_hit_tokens", _positive_number(_get(health, "scheduler", "cache_hit_tokens")), str(QWEN27_API_PARITY)),
        _check("qwen27_api_block_l2_writes_hits", _positive_number(_get(block_disk, "disk_writes")) and _positive_number(_get(block_disk, "disk_hits")), str(QWEN27_API_PARITY)),
        _check("qwen27_api_ssm_l2_stores", _positive_number(_get(ssm_disk, "stores")) and _positive_number(_get(ssm_disk, "total_tokens_on_disk")), str(QWEN27_API_PARITY)),
    ]


def _qwen27_restart_l2_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    phase1 = _get(data, "phases", "phase1", default={})
    phase2 = _get(data, "phases", "phase2", default={})
    if not phase1 or not phase2:
        results = data.get("results") if isinstance(data.get("results"), list) else []
        if results and isinstance(results[0], dict):
            phase1 = results[0].get("first") if isinstance(results[0].get("first"), dict) else phase1
            phase2 = results[0].get("second") if isinstance(results[0].get("second"), dict) else phase2
    phase1_block = (
        _get(phase1, "block_disk_cache", default={})
        or _get(phase1, "cache_stats", "block_disk_cache", default={})
        or _get(phase1, "cache_after", "body", "block_disk_cache", default={})
        or _get(phase1, "health_after", "cache", "block_disk_cache", default={})
    )
    phase1_ssm = (
        _get(phase1, "ssm_companion_disk", default={})
        or _get(phase1, "cache_stats", "ssm_companion", "disk", default={})
        or _get(phase1, "cache_after", "body", "ssm_companion", "disk", default={})
        or _get(phase1, "health_after", "cache", "ssm_companion", "disk", default={})
    )
    phase2_block = (
        _get(phase2, "block_disk_cache", default={})
        or _get(phase2, "cache_stats", "block_disk_cache", default={})
        or _get(phase2, "cache_after", "body", "block_disk_cache", default={})
        or _get(phase2, "health_after", "cache", "block_disk_cache", default={})
    )
    phase2_usage_detail = (
        _get(phase2, "usage", "prompt_tokens_details", default={})
        or _get(phase2, "response", "usage", "prompt_tokens_details", default={})
    )
    phase2_cache_hit_tokens = (
        _get(phase2, "cache_hit_tokens")
        or _get(phase2, "cached_tokens")
        or _get(phase2_usage_detail, "cached_tokens")
        or _get(phase2, "health_after", "scheduler", "cache_hit_tokens")
    )
    phase2_native = _get(phase2, "native_cache", default={}) or _get(phase2, "health_after", "native_cache", default={})
    phase2_mtp = _get(phase2, "mtp", default={}) or _get(phase2, "health_after", "mtp", default={})
    return [
        _check("qwen27_restart_artifact_exists", data.get("exists") is True, str(QWEN27_RESTART_L2_RESTORE), data.get("status")),
        _check("qwen27_restart_status_pass", data.get("status") == "pass", str(QWEN27_RESTART_L2_RESTORE), data.get("status")),
        _check("qwen27_restart_phase1_store", _positive_number(_get(phase1_block, "disk_writes")) and _positive_number(_get(phase1_ssm, "stores")), str(QWEN27_RESTART_L2_RESTORE)),
        _check("qwen27_restart_phase2_disk_hit", _positive_number(phase2_cache_hit_tokens) and _get(phase2_usage_detail, "cache_detail") == "paged+ssm+disk", str(QWEN27_RESTART_L2_RESTORE)),
        _check("qwen27_restart_phase2_block_l2_hit", _positive_number(_get(phase2_block, "disk_hits")), str(QWEN27_RESTART_L2_RESTORE)),
        _check("qwen27_restart_phase2_native_cache", _native_hybrid_cache_ok(phase2_native), str(QWEN27_RESTART_L2_RESTORE)),
        _check("qwen27_restart_phase2_mtp_active", _mtp_runtime_active(phase2_mtp), str(QWEN27_RESTART_L2_RESTORE)),
    ]


def _qwen35_startup_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    health = data.get("health") if isinstance(data.get("health"), dict) else {}
    native = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    mtp = health.get("mtp") if isinstance(health.get("mtp"), dict) else {}
    routing = health.get("routing") if isinstance(health.get("routing"), dict) else {}
    return [
        _check("qwen35_startup_artifact_exists", data.get("exists") is True, str(QWEN35_STARTUP_MTP), data.get("status")),
        _check("qwen35_startup_model_loaded", health.get("model_loaded") is True and health.get("model_name") == "JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP", str(QWEN35_STARTUP_MTP)),
        _check("qwen35_startup_mtp_runtime_active", _mtp_runtime_active(mtp), str(QWEN35_STARTUP_MTP)),
        _check("qwen35_startup_mtp_depth_three", mtp.get("effective_depth") == 3, str(QWEN35_STARTUP_MTP), mtp.get("effective_depth")),
        _check("qwen35_startup_hybrid_cache_policy", _native_hybrid_cache_ok(native), str(QWEN35_STARTUP_MTP)),
        _check("qwen35_startup_trained_k_preserved", routing.get("trained_active_experts") == 8 and routing.get("effective_active_experts") == 8, str(QWEN35_STARTUP_MTP)),
    ]


def _qwen35_long_tool_cache_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    acceptance = data.get("acceptance") if isinstance(data.get("acceptance"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    later_rows = rows[1:]
    block_l2 = any(_positive_number(_get(row, "block_disk_cache", "disk_hits")) for row in rows if isinstance(row, dict))
    block_writes = any(_positive_number(_get(row, "block_disk_cache", "disk_writes")) for row in rows if isinstance(row, dict))
    ssm_l2 = any(_positive_number(_get(row, "ssm_companion", "disk", "total_tokens_on_disk")) for row in rows if isinstance(row, dict))
    http_errors = [
        {
            "turn": row.get("turn"),
            "round": round_row.get("round"),
            "http_status": round_row.get("http_status"),
            "error": round_row.get("error"),
        }
        for row in rows
        if isinstance(row, dict)
        for round_row in (row.get("rounds") or [])
        if isinstance(round_row, dict) and round_row.get("http_status")
    ]
    cache_details = [
        _get(row, "scheduler_stats", "last_cache_execution", "cache_detail")
        for row in later_rows
        if isinstance(row, dict)
    ]
    return [
        _check("qwen35_long_tool_cache_artifact_exists", data.get("exists") is True, str(QWEN35_LONG_TOOL_CACHE), data.get("status")),
        _check("qwen35_long_tool_cache_overall_pass", data.get("overall_pass") is True, str(QWEN35_LONG_TOOL_CACHE), data.get("overall_pass")),
        _check("qwen35_long_tool_cache_no_http_errors", not http_errors, str(QWEN35_LONG_TOOL_CACHE), http_errors[:5]),
        _check("qwen35_long_tool_cache_three_turns", data.get("turns") == 3 and len(rows) == 3, str(QWEN35_LONG_TOOL_CACHE)),
        _check("qwen35_long_tool_cache_previous_response_id", acceptance.get("previous_response_id_used") is True, str(QWEN35_LONG_TOOL_CACHE)),
        _check("qwen35_long_tool_cache_reuse_each_turn", acceptance.get("cache_reuse_each_turn_after_first") is True and all(_positive_number(row.get("cached_tokens")) for row in later_rows if isinstance(row, dict)), str(QWEN35_LONG_TOOL_CACHE), cache_details),
        _check("qwen35_long_tool_cache_paged_ssm_detail", all(detail == "paged+ssm" for detail in cache_details), str(QWEN35_LONG_TOOL_CACHE), cache_details),
        _check("qwen35_long_tool_cache_l2_block_and_ssm", block_l2 and block_writes and ssm_l2, str(QWEN35_LONG_TOOL_CACHE)),
        _check("qwen35_long_tool_cache_tool_calls", acceptance.get("tool_call_each_required_turn") is True, str(QWEN35_LONG_TOOL_CACHE)),
        _check("qwen35_long_tool_cache_tool_evidence_grounded", acceptance.get("tool_evidence_each_required_turn") is True, str(QWEN35_LONG_TOOL_CACHE), acceptance.get("tool_evidence_each_required_turn")),
        _check("qwen35_long_tool_cache_no_tool_markup_leak", acceptance.get("no_tool_markup_leak") is True, str(QWEN35_LONG_TOOL_CACHE)),
        _check("qwen35_long_tool_cache_no_loop_tail", acceptance.get("no_loop_like_tail") is True, str(QWEN35_LONG_TOOL_CACHE)),
        _check("qwen35_long_tool_cache_final_visible_no_tools", acceptance.get("final_turn_tools_disabled") is True and acceptance.get("final_turn_visible_output") is True, str(QWEN35_LONG_TOOL_CACHE)),
    ]


def _qwen35_restart_l2_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    phase1 = _get(data, "phases", "phase1", default={})
    phase2 = _get(data, "phases", "phase2", default={})
    phase1_block = (
        _get(phase1, "block_disk_cache", default={})
        or _get(phase1, "cache_stats", "block_disk_cache", default={})
        or _get(phase1, "health", "cache", "block_disk_cache", default={})
    )
    phase1_ssm = (
        _get(phase1, "ssm_companion_disk", default={})
        or _get(phase1, "cache_stats", "ssm_companion", "disk", default={})
        or _get(phase1, "health", "cache", "ssm_companion", "disk", default={})
    )
    phase2_block = (
        _get(phase2, "block_disk_cache", default={})
        or _get(phase2, "cache_stats", "block_disk_cache", default={})
        or _get(phase2, "health", "cache", "block_disk_cache", default={})
    )
    phase2_ssm = (
        _get(phase2, "ssm_companion_disk", default={})
        or _get(phase2, "cache_stats", "ssm_companion", "disk", default={})
        or _get(phase2, "health", "cache", "ssm_companion", "disk", default={})
    )
    phase2_usage_detail = _get(phase2, "usage", "prompt_tokens_details", default={})
    if not phase2_usage_detail:
        phase2_usage_detail = _get(phase2, "response", "usage", "prompt_tokens_details", default={})
    if not phase2_usage_detail:
        phase2_usage_detail = _get(phase2, "health", "scheduler", "last_cache_execution", default={})
    phase2_cache_hit_tokens = (
        _get(phase2, "cache_hit_tokens")
        or _get(phase2_usage_detail, "cached_tokens")
        or _get(phase2, "health", "scheduler", "cache_hit_tokens")
    )
    phase2_native = _get(phase2, "native_cache", default={}) or _get(phase2, "health", "native_cache", default={})
    phase2_mtp = _get(phase2, "mtp", default={}) or _get(phase2, "health", "mtp", default={})
    return [
        _check("qwen35_restart_artifact_exists", data.get("exists") is True, str(QWEN35_RESTART_L2_RESTORE), data.get("status")),
        _check("qwen35_restart_status_pass", data.get("status") == "pass", str(QWEN35_RESTART_L2_RESTORE), data.get("status")),
        _check("qwen35_restart_phase1_store", _positive_number(_get(phase1_block, "disk_writes")) and _positive_number(_get(phase1_ssm, "stores")), str(QWEN35_RESTART_L2_RESTORE)),
        _check("qwen35_restart_phase2_disk_hit", _positive_number(phase2_cache_hit_tokens) and "disk" in str(_get(phase2_usage_detail, "cache_detail") or ""), str(QWEN35_RESTART_L2_RESTORE), phase2_usage_detail),
        _check("qwen35_restart_phase2_block_l2_hit", _positive_number(_get(phase2_block, "disk_hits")), str(QWEN35_RESTART_L2_RESTORE)),
        _check("qwen35_restart_phase2_ssm_l2_hit", _positive_number(_get(phase2_ssm, "hits")), str(QWEN35_RESTART_L2_RESTORE)),
        _check("qwen35_restart_phase2_native_cache", _native_hybrid_cache_ok(phase2_native), str(QWEN35_RESTART_L2_RESTORE)),
        _check("qwen35_restart_phase2_mtp_active", _mtp_runtime_active(phase2_mtp), str(QWEN35_RESTART_L2_RESTORE)),
    ]


def _qwen35_raw_sse_parity_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    captures = data.get("captures") if isinstance(data.get("captures"), dict) else {}
    detail = {
        "missing_captures": data.get("missing_captures"),
        "conflicting_output_indices": {
            name: row.get("conflicting_output_indices")
            for name, row in captures.items()
            if isinstance(row, dict) and row.get("conflicting_output_indices")
        },
    }
    return [
        _check("qwen35_raw_sse_artifact_exists", data.get("exists") is True, str(QWEN35_RAW_SSE_PARITY), data.get("status")),
        _check("qwen35_raw_sse_status_pass", data.get("status") == "pass", str(QWEN35_RAW_SSE_PARITY), data.get("status")),
        _check("qwen35_raw_sse_same_model_surfaces", checks.get("all_present_surfaces_same_model") is True and checks.get("all_present_surfaces_match_expected_model") is True, str(QWEN35_RAW_SSE_PARITY), checks.get("all_present_surfaces_same_model")),
        _check("qwen35_raw_sse_authoritative_args", checks.get("all_present_surfaces_have_authoritative_args") is True and checks.get("all_present_surfaces_match_expected_arguments") is True, str(QWEN35_RAW_SSE_PARITY)),
        _check("qwen35_raw_sse_reasoning_events", checks.get("all_present_surfaces_have_required_reasoning") is True and checks.get("no_reasoning_disable_workaround") is True, str(QWEN35_RAW_SSE_PARITY)),
        _check("qwen35_raw_sse_valid_output_item_indices", checks.get("all_present_surfaces_have_valid_output_item_indices") is True, str(QWEN35_RAW_SSE_PARITY), detail),
        _check("qwen35_raw_sse_local_source_guards", checks.get("local_responses_streaming_guards_pass") is True and checks.get("local_output_index_ordering_guard") is True and checks.get("local_empty_xml_arguments_fail_closed") is True, str(QWEN35_RAW_SSE_PARITY)),
    ]


def _qwen_installed_video_checks(
    name: str,
    data: dict[str, Any],
    *,
    path: Path,
    expected_model: str,
    expected_depth: int,
) -> list[dict[str, Any]]:
    surfaces = data.get("provenSurfaces") if isinstance(data.get("provenSurfaces"), list) else []
    server = data.get("server") if isinstance(data.get("server"), dict) else {}
    health = server.get("health") if isinstance(server.get("health"), dict) else {}
    native = health.get("native_cache") if isinstance(health.get("native_cache"), dict) else {}
    mtp = health.get("mtp") if isinstance(health.get("mtp"), dict) else {}
    scheduler = health.get("scheduler") if isinstance(health.get("scheduler"), dict) else {}
    batch = scheduler.get("batch_generator") if isinstance(scheduler.get("batch_generator"), dict) else {}
    cache = health.get("cache") if isinstance(health.get("cache"), dict) else {}
    block_disk = cache.get("block_disk_cache") if isinstance(cache.get("block_disk_cache"), dict) else {}
    ssm_disk = _get(cache, "ssm_companion", "disk", default={})
    last_execution = scheduler.get("last_cache_execution") if isinstance(scheduler.get("last_cache_execution"), dict) else {}
    required_surfaces = {
        "installed_app_ui",
        "video_where_supported",
        "reasoning_display",
        "responses_api",
        "responses_delta_streaming",
        "responses_cache_detail_usage",
        "tool_l2_cache_integrated",
        "native_cache_status",
        "l2_disk_storage",
        "parser_leak_check",
        "language_leak_check",
    }
    return [
        _check(f"{name}_installed_video_artifact_exists", data.get("exists") is True, str(path), data.get("status")),
        _check(f"{name}_installed_video_status_pass", data.get("status") == "pass", str(path), data.get("status")),
        _check(f"{name}_installed_video_app_route", data.get("uiLaunchMode") == "installed-app" and data.get("installedAppPath") == "/Applications/vMLX.app", str(path), data.get("uiLaunchMode")),
        _check(f"{name}_installed_video_model_identity", data.get("modelPath") == expected_model and health.get("model_loaded") is True, str(path), data.get("modelPath")),
        _check(f"{name}_installed_video_requested", data.get("requestedVideo") is True, str(path), data.get("requestedVideo")),
        _check(f"{name}_installed_video_surfaces", required_surfaces.issubset(set(surfaces)), str(path), sorted(set(surfaces))),
        _check(f"{name}_installed_video_visible_output", data.get("visibleAssistantTurnsComplete") is True, str(path)),
        _check(f"{name}_installed_video_frames_processed", _get(batch, "num_images_processed") == 6, str(path), _get(batch, "num_images_processed")),
        _check(f"{name}_installed_video_hybrid_cache_policy", _native_hybrid_cache_ok(native), str(path), native),
        _check(f"{name}_installed_video_mtp_active", _mtp_runtime_active(mtp) and mtp.get("effective_depth") == expected_depth, str(path), mtp),
        _check(f"{name}_installed_video_cache_hit_detail", _positive_number(_get(last_execution, "cached_tokens")) and _get(last_execution, "cache_detail") == "paged+ssm", str(path), last_execution),
        _check(f"{name}_installed_video_l2_storage", _positive_number(_get(block_disk, "disk_writes")) and _positive_number(_get(ssm_disk, "stores")), str(path), {"block": block_disk, "ssm": ssm_disk}),
    ]


def _qwen27_long_context_cache_tail_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    cold = _get(data, "phases", "cold", default={})
    warm = _get(data, "phases", "warm", default={})
    cold_usage = _get(cold, "response", "usage", default={})
    warm_usage = _get(warm, "response", "usage", default={})
    warm_health = _get(warm, "health", default={})
    warm_stats = _get(warm, "cache_stats", default={})
    warm_details = (
        _get(warm, "response", "usage", "prompt_tokens_details", default={})
        or _get(warm_health, "scheduler", "last_cache_execution", default={})
    )
    block_disk = _get(warm_stats, "block_disk_cache", default={}) or _get(warm_health, "cache", "block_disk_cache", default={})
    ssm_disk = _get(warm_stats, "ssm_companion", "disk", default={}) or _get(warm_health, "cache", "ssm_companion", "disk", default={})
    native = _get(warm_health, "native_cache", default={})
    mtp = _get(warm_health, "mtp", default={})
    tq = _get(warm_health, "turboquant_kv_cache", default={})
    checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    return [
        _check("qwen27_long_context_artifact_exists", data.get("exists") is True, str(QWEN27_LONG_CONTEXT_CACHE_TAIL), data.get("status")),
        _check("qwen27_long_context_status_pass", data.get("status") == "pass", str(QWEN27_LONG_CONTEXT_CACHE_TAIL), data.get("status")),
        _check("qwen27_long_context_prompt_large", _positive_number(_get(cold_usage, "input_tokens")) and _get(cold_usage, "input_tokens") >= 30000 and _get(warm_usage, "input_tokens") == _get(cold_usage, "input_tokens"), str(QWEN27_LONG_CONTEXT_CACHE_TAIL), {"cold": cold_usage, "warm": warm_usage}),
        _check("qwen27_long_context_tail_markers", checks.get("cold_visible_tail_markers") is True and checks.get("warm_visible_tail_markers") is True, str(QWEN27_LONG_CONTEXT_CACHE_TAIL), checks),
        _check("qwen27_long_context_warm_cache_hit", _positive_number(_get(warm_details, "cached_tokens")) and _get(warm_details, "cached_tokens") >= 30000 and _get(warm_details, "cache_detail") in {"paged+ssm", "paged+ssm+disk"}, str(QWEN27_LONG_CONTEXT_CACHE_TAIL), warm_details),
        _check("qwen27_long_context_block_l2_written", _positive_number(_get(block_disk, "disk_writes")) and _positive_number(_get(block_disk, "total_tokens_on_disk")) and _get(block_disk, "total_tokens_on_disk") >= 30000, str(QWEN27_LONG_CONTEXT_CACHE_TAIL), block_disk),
        _check("qwen27_long_context_block_l2_hits", _positive_number(_get(block_disk, "disk_hits")), str(QWEN27_LONG_CONTEXT_CACHE_TAIL), block_disk),
        _check("qwen27_long_context_ssm_l2_written", _positive_number(_get(ssm_disk, "stores")) and _positive_number(_get(ssm_disk, "total_tokens_on_disk")) and _get(ssm_disk, "total_tokens_on_disk") >= 30000, str(QWEN27_LONG_CONTEXT_CACHE_TAIL), ssm_disk),
        _check("qwen27_long_context_native_hybrid_cache", _native_hybrid_cache_ok(native), str(QWEN27_LONG_CONTEXT_CACHE_TAIL), native),
        _check("qwen27_long_context_turboquant_attention_kv", _get(native, "generic_turboquant_kv", "enabled") is True and tq.get("enabled") is True, str(QWEN27_LONG_CONTEXT_CACHE_TAIL), {"native": native, "tq": tq}),
        _check("qwen27_long_context_mtp_active", _mtp_runtime_active(mtp), str(QWEN27_LONG_CONTEXT_CACHE_TAIL), mtp),
    ]


def _issue179_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    proven = data.get("proven") if isinstance(data.get("proven"), dict) else {}
    local_ui = data.get("local_real_ui") if isinstance(data.get("local_real_ui"), dict) else {}
    cancel_probe = (
        data.get("local_responses_cancel_probe")
        if isinstance(data.get("local_responses_cancel_probe"), dict)
        else {}
    )
    reporter_parity = (
        data.get("reporter_parity_comparison")
        if isinstance(data.get("reporter_parity_comparison"), dict)
        else {}
    )
    reporter_hash = (
        data.get("reporter_server_hash_parity")
        if isinstance(data.get("reporter_server_hash_parity"), dict)
        else {}
    )
    reporter_repro = (
        data.get("local_reporter_prompt_reproduction")
        if isinstance(data.get("local_reporter_prompt_reproduction"), dict)
        else {}
    )
    current_source_smoke = (
        data.get("current_source_minimax_small_smoke")
        if isinstance(data.get("current_source_minimax_small_smoke"), dict)
        else {}
    )
    current_source_smoke_checks = (
        current_source_smoke.get("checks")
        if isinstance(current_source_smoke.get("checks"), dict)
        else {}
    )
    not_proven = data.get("not_proven") if isinstance(data.get("not_proven"), list) else []
    return [
        _check(
            "issue179_root_cause_artifact_exists",
            data.get("exists") is True,
            str(ISSUE179_AUDIT),
            data.get("status"),
        ),
        _check(
            "issue179_root_cause_status_pass",
            data.get("status") == "pass",
            str(ISSUE179_AUDIT),
            data.get("status"),
        ),
        _check(
            "issue179_current_source_cancel_contract",
            proven.get("current_source_responses_cancel_contract_proven") is True
            and proven.get("current_source_responses_cancel_inactive_404_contract_proven") is True,
            str(ISSUE179_AUDIT),
        ),
        _check(
            "issue179_latest_public_dmg_cancel_route",
            proven.get("latest_public_dmg_has_responses_cancel_route") is True,
            str(ISSUE179_AUDIT),
        ),
        _check(
            "issue179_local_installed_cancel_route",
            proven.get("local_installed_bundle_has_responses_cancel_route") is True,
            str(ISSUE179_AUDIT),
        ),
        _check(
            "issue179_local_installed_cancel_live_probe",
            proven.get("local_installed_responses_cancel_live_probe") is True
            and cancel_probe.get("status") == "pass"
            and cancel_probe.get("response_id_seen") is True
            and cancel_probe.get("cancel_status") == 200
            and _get(cancel_probe, "probe", "cancel_route_present") is True
            and _get(cancel_probe, "probe", "bad_text_captured") is False,
            str(ISSUE179_AUDIT),
            cancel_probe,
        ),
        _check(
            "issue179_local_real_ui_clean",
            proven.get("local_real_ui_diagnostics_clean") is True
            and local_ui.get("all_required_clean") is True,
            str(ISSUE179_AUDIT),
            {
                "clean_count": local_ui.get("clean_count"),
                "required_count": local_ui.get("required_count"),
            },
        ),
        _check(
            "issue179_installed_session_settings_parity",
            proven.get("local_installed_issue179_session_settings_parity") is True,
            str(ISSUE179_AUDIT),
            local_ui.get("installed_session_settings_parity"),
        ),
        _check(
            "issue179_reporter_parity_artifact",
            reporter_parity.get("status") == "pass",
            str(ISSUE179_AUDIT),
            reporter_parity.get("failures") or not_proven,
        ),
        _check(
            "issue179_reporter_server_hash_parity",
            reporter_hash.get("status") == "pass",
            str(ISSUE179_AUDIT),
            {
                "failure": reporter_hash.get("failure"),
                "reporter_installed_server_sha256": reporter_hash.get("reporter_installed_server_sha256"),
                "latest_public_server_sha256": reporter_hash.get("latest_public_server_sha256"),
            },
        ),
        _check(
            "issue179_reporter_prompt_reproduction_or_log_proof",
            proven.get("local_reporter_prompt_reproduction_clean") is True
            and (
                proven.get("reporter_log_has_abort_before_visible_content") is True
                or reporter_repro.get("clean") is True
            ),
            str(ISSUE179_AUDIT),
            {
                "local_reporter_prompt_reproduction_clean": proven.get(
                    "local_reporter_prompt_reproduction_clean"
                ),
                "reporter_log_has_abort_before_visible_content": proven.get(
                    "reporter_log_has_abort_before_visible_content"
                ),
                "reporter_repro": reporter_repro,
            },
        ),
        _check(
            "issue179_current_source_minimax_small_smoke",
            current_source_smoke.get("all_checks_pass") is True
            and all(current_source_smoke_checks.values()),
            str(ISSUE179_AUDIT),
            {
                "path": current_source_smoke.get("path"),
                "status": current_source_smoke.get("status"),
                "checks": current_source_smoke_checks,
                "release_boundary": current_source_smoke.get("release_boundary"),
            },
        ),
    ]


def _dsv4_exactness_preflight_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    active_heavy = data.get("active_heavy_processes")
    active_heavy = active_heavy if isinstance(active_heavy, list) else []
    selected_cases = data.get("selected_cases")
    selected_cases = selected_cases if isinstance(selected_cases, list) else []
    launch_blockers = data.get("launch_blockers")
    launch_blockers = launch_blockers if isinstance(launch_blockers, list) else []
    memory_detail = {
        "status": data.get("status"),
        "reason": data.get("reason"),
        "available_for_gate_gb": data.get("available_for_gate_gb"),
        "required_available_gb": data.get("required_available_gb"),
        "memory_gap_gb": data.get("memory_gap_gb"),
        "strict_vm_stat_memory_gap_gb": data.get("strict_vm_stat_memory_gap_gb"),
        "preflight_memory_source": data.get("preflight_memory_source"),
    }
    safe_skip = (
        data.get("status") == "skipped"
        and data.get("reason") == "insufficient_vm_stat_memory"
        and data.get("did_not_launch") is True
        and data.get("launch_decision") == "do_not_launch"
        and data.get("launch_allowed") is False
        and "insufficient_memory" in launch_blockers
    )
    return [
        _check(
            "dsv4_exactness_preflight_artifact_exists",
            data.get("exists") is True,
            str(DSV4_EXACTNESS_PREFLIGHT),
            data.get("status"),
        ),
        _check(
            "dsv4_memory_preflight_floor_valid",
            data.get("floor_valid") is True
            and data.get("conservative_floor_valid") is True,
            str(DSV4_EXACTNESS_PREFLIGHT),
            {
                "floor_valid": data.get("floor_valid"),
                "conservative_floor_valid": data.get("conservative_floor_valid"),
            },
        ),
        _check(
            "dsv4_memory_preflight_process_context",
            data.get("active_heavy_process_count") == 0 and active_heavy == [],
            str(DSV4_EXACTNESS_PREFLIGHT),
            {
                "active_heavy_process_count": data.get("active_heavy_process_count"),
                "active_heavy_processes": active_heavy[:5],
            },
        ),
        _check(
            "dsv4_memory_preflight_launch_safely_blocked",
            safe_skip or data.get("status") == "pass",
            str(DSV4_EXACTNESS_PREFLIGHT),
            {
                "did_not_launch": data.get("did_not_launch"),
                "launch_decision": data.get("launch_decision"),
                "launch_allowed": data.get("launch_allowed"),
                "launch_blockers": launch_blockers,
            },
        ),
        _check(
            "dsv4_memory_preflight_resource_available",
            data.get("launch_allowed") is True
            and _positive_number(data.get("available_for_gate_gb"))
            and _positive_number(data.get("required_available_gb"))
            and data.get("available_for_gate_gb") >= data.get("required_available_gb"),
            str(DSV4_EXACTNESS_PREFLIGHT),
            memory_detail,
        ),
        _check(
            "dsv4_exactness_preflight_status_pass",
            data.get("status") == "pass",
            str(DSV4_EXACTNESS_PREFLIGHT),
            data.get("status"),
        ),
        _check(
            "dsv4_exactness_preflight_case_matrix_present",
            data.get("case_count") == len(selected_cases)
            and len(selected_cases) >= 10,
            str(DSV4_EXACTNESS_PREFLIGHT),
            {
                "case_count": data.get("case_count"),
                "selected_case_count": len(selected_cases),
            },
        ),
        _check(
            "dsv4_exact_code_file_long_output_complete",
            data.get("status") == "pass",
            str(DSV4_EXACTNESS_PREFLIGHT),
            data.get("reason") or data.get("status"),
        ),
    ]


def _build(root: Path) -> dict[str, Any]:
    release_manifest = _load_json(root / RELEASE_MANIFEST)
    mimo = _load_json(root / MIMO_AUDIT)
    mimo_classifier = _load_json(root / MIMO_NO_SOURCE_EXACTNESS_CLASSIFIER)
    mimo_media_runtime = _load_json(root / MIMO_JANGTQ2_MEDIA_RUNTIME_SOURCE)
    mimo_video_audio = _load_json(root / MIMO_JANGTQ2_VIDEO_AUDIO_SOURCE)
    mimo_responses_tools_cache_ui = _load_json(
        root / MIMO_JANGTQ2_RESPONSES_TOOLS_CACHE_UI
    )
    noheavy = _load_json(root / NOHEAVY_API_CACHE)
    responses_raw_sse = _load_json(root / RESPONSES_RAW_SSE_PARITY)
    api_surface = _load_json(root / API_SURFACE_CONTRACT)
    panel_settings = _load_json(root / PANEL_SETTINGS_CONTRACT)
    tool_call = _load_json(root / TOOL_CALL_CONTRACT)
    objective_digest = _load_json(root / OBJECTIVE_DIGEST)
    issue179 = _load_json(root / ISSUE179_AUDIT)
    qwen27 = _load_json(root / QWEN27_RESPONSES_CANCEL)
    qwen27_api = _load_json(root / QWEN27_API_PARITY)
    qwen27_restart = _load_json(root / QWEN27_RESTART_L2_RESTORE)
    qwen27_installed_video = _load_json(root / QWEN27_INSTALLED_VIDEO)
    qwen27_long_context = _load_json(root / QWEN27_LONG_CONTEXT_CACHE_TAIL)
    qwen35_startup = _load_json(root / QWEN35_STARTUP_MTP)
    qwen35_long_tool_cache = _load_json(root / QWEN35_LONG_TOOL_CACHE)
    qwen35_restart = _load_json(root / QWEN35_RESTART_L2_RESTORE)
    qwen35_raw_sse = _load_json(root / QWEN35_RAW_SSE_PARITY)
    qwen35_installed_video = _load_json(root / QWEN35_INSTALLED_VIDEO)
    gemma4 = _load_json(root / GEMMA4_12B_JANG4M_SMOKE)
    gemma4_autoq4_cache = _load_json(root / GEMMA4_12B_JANG4M_AUTOQ4_CACHE)
    gemma4_issue191_startup = _load_json(root / GEMMA4_12B_ISSUE191_STARTUP_VISIBLE)
    gemma4_media = _load_json(root / GEMMA4_12B_JANG4M_MEDIA_SMOKE)
    gemma_qat = _load_json(root / GEMMA_QAT_NATIVE_MXFP4_INVENTORY)
    step37 = _load_json(root / STEP37_TEXTONLY_SMOKE)
    step37_vlm = _load_json(root / STEP37_VLM_RUNTIME_AUDIT)
    lfm4 = _load_json(root / LFM25_MXFP4_SMOKE)
    lfm8 = _load_json(root / LFM25_MXFP8_SMOKE)
    nemotron = _load_json(root / NEMOTRON_OMNI_SMOKE)
    nemotron_media = _load_json(root / NEMOTRON_OMNI_MEDIA_GATE)
    dsv4 = _load_json(root / DSV4_EXACTNESS_PREFLIGHT)

    groups = {
        "release_packaging_ui": _release_manifest_checks(release_manifest),
        "api_cache_responses_contract": _noheavy_api_cache_checks(noheavy),
        "responses_raw_sse_parity": _responses_raw_sse_parity_checks(
            responses_raw_sse
        ),
        "api_surface_endpoints_contract": _api_surface_contract_checks(api_surface),
        "ui_settings_parser_cache_contract": _panel_settings_contract_checks(panel_settings),
        "tool_json_xml_code_contract": _tool_call_contract_checks(tool_call),
        "n2_pro_397b": _n2_pro_397b_checks(objective_digest),
        "mimo_v25_jangtq2": _mimo_checks(
            mimo,
            mimo_classifier,
            mimo_media_runtime,
            mimo_video_audio,
            mimo_responses_tools_cache_ui,
        ),
        "qwen36_mtp": _qwen27_cancel_checks(qwen27)
        + _qwen27_api_parity_checks(qwen27_api)
        + _qwen27_restart_l2_checks(qwen27_restart)
        + _qwen_installed_video_checks(
            "qwen27",
            qwen27_installed_video,
            path=QWEN27_INSTALLED_VIDEO,
            expected_model="/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
            expected_depth=3,
        )
        + _qwen27_long_context_cache_tail_checks(qwen27_long_context)
        + _qwen35_startup_checks(qwen35_startup)
        + _qwen35_long_tool_cache_checks(qwen35_long_tool_cache)
        + _qwen35_restart_l2_checks(qwen35_restart)
        + _qwen35_raw_sse_parity_checks(qwen35_raw_sse)
        + _qwen_installed_video_checks(
            "qwen35",
            qwen35_installed_video,
            path=QWEN35_INSTALLED_VIDEO,
            expected_model="/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
            expected_depth=3,
        ),
        "gemma4_12b": _smoke_mixed_swa_checks(
            "gemma4_12b_jang4m_nomedia",
            gemma4,
            path=GEMMA4_12B_JANG4M_SMOKE,
            family="gemma4",
            tool_parser="gemma4",
            reasoning_parser="gemma4",
            expected_mllm=True,
            cache_detail="paged+mixed_swa",
            cache_proof=gemma4_autoq4_cache,
            cache_path=GEMMA4_12B_JANG4M_AUTOQ4_CACHE,
        )
        + _gemma4_12b_issue191_startup_checks(gemma4_issue191_startup)
        + _gemma4_12b_media_checks(gemma4_media),
        "gemma_qat_native_mxfp4": _gemma_qat_native_mxfp4_checks(gemma_qat),
        "step37_flash": _smoke_mixed_swa_checks(
            "step37_textonly_nomedia",
            step37,
            path=STEP37_TEXTONLY_SMOKE,
            family="step3p7",
            subtype="step3p7_full_sliding_kv",
            tool_parser="step3p5",
            reasoning_parser="qwen3",
            expected_mllm=False,
            cache_detail="paged",
        )
        + _step37_vlm_runtime_checks(step37_vlm),
        "lfm25": _smoke_hybrid_ssm_checks(
            "lfm25_mxfp4_nomedia",
            lfm4,
            path=LFM25_MXFP4_SMOKE,
            family="lfm2",
            tool_parser="lfm2",
            reasoning_parser="qwen3",
            expected_mllm=False,
        )
        + _smoke_hybrid_ssm_checks(
            "lfm25_mxfp8_nomedia",
            lfm8,
            path=LFM25_MXFP8_SMOKE,
            family="lfm2",
            tool_parser="lfm2",
            reasoning_parser="qwen3",
            expected_mllm=False,
        ),
        "nemotron_omni": _smoke_hybrid_ssm_checks(
            "nemotron_omni_nomedia",
            nemotron,
            path=NEMOTRON_OMNI_SMOKE,
            family="nemotron_h",
            tool_parser="nemotron",
            reasoning_parser="deepseek_r1",
            expected_mllm=True,
        )
        + _nemotron_omni_media_checks(nemotron_media),
        "minimax": _issue179_checks(issue179),
        "dsv4_flash": _dsv4_exactness_preflight_checks(dsv4),
    }

    flat = [row for rows in groups.values() for row in rows]
    failed = [row for row in flat if not row["ok"]]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if not failed else "open",
        "release_ready": not failed,
        "groups": groups,
        "failed": failed,
        "failed_count": len(failed),
        "boundary": (
            "No-heavy checklist only. A pass here would mean current evidence "
            "claims every listed gate is green; it still must be backed by the "
            "live artifacts referenced by each row. Open rows are release blockers."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = _build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(f"status={result['status']}")
    print(f"failed_count={result['failed_count']}")
    for row in result["failed"][:40]:
        print(f"OPEN {row['name']}: {row['detail'] if 'detail' in row else row['evidence']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
