"""Release-regression manifest for current vMLX Python/Electron work.

This is the durable index for the broad regression suite Eric requested. It is
intentionally explicit: each row names the user-visible or runtime contract it
protects, the commands that currently cover it, and whether the row is no-heavy,
live-model, or packaged-app only.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from tests.cross_matrix.run_model_family_detection_contract import (
    REQUIRED_ROWS as EXPECTED_CURRENT_MODEL_FAMILY_ROWS,
)
from tests.cross_matrix.run_current_regression_suite import (
    CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
    CURRENT_SUITE_SOURCE_HASH_FILES,
)

CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT = (
    "build/current-release-regression-manifest-after-pr-intake-matrix-refresh-20260609.json"
)

EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS = (
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

RAW_REAL_UI_PARSER_LEAK_RE = re.compile(
    r"<think>|</think>|<tool_call>|</tool_call>|<function>|<invoke>|"
    r"<minimax:tool_call>|<zyphra_tool_call>|<\|point_start\|>|"
    r"<\|point_end\|>|<\|box_start\|>|<\|box_end\|>|"
    r"<\|tool_call_start\|>|<\|tool_call_end\|>"
)
REAL_UI_RESPONSE_COMPLETE_SPEED_RE = re.compile(
    r"Response complete:\s+(?P<tokens>\d+)\s+tokens.*?"
    r"live=(?P<live_tps>\d+(?:\.\d+)?)\s+t/s,\s+"
    r"TTFT:\s+(?P<ttft>\d+(?:\.\d+)?)s.*?usage=server"
)

REQUIRED_REAL_UI_REQUEST_CONTRACT_FIELDS = (
    "promptOne",
    "promptTwo",
    "requestMaxTokens",
    "maxToolIterations",
    "toolResultMaxChars",
    "wireApi",
    "builtinToolsEnabled",
    "enableThinking",
    "checkServerCacheControls",
    "checkMedia",
    "checkVideo",
    "expectPagedCacheLocked",
    "imageExpectRegex",
    "videoExpectRegex",
    "cacheExpectRegex",
)
REQUIRED_REAL_UI_EXTENSIVE_TOOL_ROWS = frozenset(
    {
        "step37_flash_jang2l",
        "lfm25_moe_a1b",
        "lfm25_moe_a1b_responses",
        "lfm25_moe_a1b_responses_delta",
    }
)
MIN_REAL_UI_EXTENSIVE_TOOL_EVENTS = 20
MIN_REAL_UI_EXTENSIVE_TOOL_RESULTS = 2
MIN_REAL_UI_EXTENSIVE_TOOL_RESULT_MESSAGES = 2

EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS = (
    "dsv4_native_composite_cache_status",
    "zaya_typed_cca_status",
    "plain_attention_kv_status",
    "hybrid_ssm_partial_reuse",
    "generic_tq_not_applied_to_hybrid_ssm",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "gemma4_mixed_swa_kv_status",
    "mimo_v2_asymmetric_swa_kv_status",
    "named_family_registry_cache_parser_contracts",
    "dsv4_terminal_composite_contracts",
    "dsv4_swa_hca_csa_component_contracts",
    "hybrid_ssm_companion_l2_contracts",
    "prompt_disk_l2_backfill_contracts",
    "cache_detail_telemetry_contracts",
    "panel_cache_launch_policy",
    "legacy_count_floor_still_nontrivial",
    "cache_family_matrix_complete",
)

EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS = (
    "dsv4_native_composite",
    "dsv4_swa_hca_csa_components",
    "zaya_typed_cca",
    "zaya_vl_hybrid_registry",
    "plain_attention_kv",
    "hybrid_ssm",
    "ling_bailing_hybrid_registry",
    "nemotron_h_hybrid_registry",
    "qwen36_hybrid_tq",
    "qwen36_registry_hybrid_parser",
    "gemma4_mixed_swa_kv",
    "mimo_v2_asymmetric_swa_kv",
    "step37_full_sliding_kv_registry",
    "lfm25_moe_hybrid_registry",
    "hy_v3_kv_registry",
    "minimax_parser_boundary_registry",
    "prompt_disk_l2",
    "turboquant_disk",
    "mllm_media_cache",
)

EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS = (
    "engine_accepts_registered_reasoning_parsers",
    "engine_accepts_registered_tool_parsers",
    "panel_emitted_reasoning_parsers_are_engine_valid",
    "panel_emitted_tool_parsers_are_engine_valid",
    "minimax_m2_reasoning_parser_regression",
    "parser_aliases_are_canonical_before_cli",
    "zaya_hy3_ling_dsv4_parser_rows_are_present",
    "non_reasoning_family_boundaries_are_present",
    "all_required_parser_markers_present",
    "legacy_count_floor_still_nontrivial",
)

EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS = (
    "generation_config_defaults_are_surfaced",
    "jang_config_sampling_defaults_override_generation_config",
    "disabled_top_k_sentinels_normalize_to_off",
    "mode_specific_jang_repetition_penalty_is_metadata_owned",
    "request_api_overrides_win_over_startup_defaults",
    "bundle_max_new_tokens_preserved_when_omitted",
    "omitted_max_tokens_without_bundle_default_is_bounded",
    "server_default_output_cap_is_not_request_ceiling",
    "no_hidden_sampler_forcing_or_repetition_floor",
    "additional_args_cannot_override_app_owned_cli_flags",
    "panel_does_not_emit_default_sampler_cli_flags",
    "legacy_count_floor_still_nontrivial",
    "generation_defaults_family_matrix_complete",
)

EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS = (
    "standard_mlx_generation_config",
    "jang_chat_sampling_overrides",
    "disabled_top_k_sentinel",
    "dsv4_direct_chat_repetition_policy",
    "max_output_context_separation",
    "thinking_budget_template_support",
    "additional_args_no_override",
    "registered_family_max_token_contract",
)

EXPECTED_CURRENT_API_SURFACE_CHECKS = (
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
    "panel_gateway_ollama_proxy_response_error_guard",
    "panel_ipc_backend_request_epipe_guard",
    "panel_performance_health_epipe_guard",
    "panel_session_lifecycle_epipe_guard",
    "panel_child_process_stdio_epipe_guard",
    "panel_backend_stderr_split_epipe_guard",
    "all_required_panel_api_markers_present",
)

EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS = (
    "openai_chat_sampling_kwargs",
    "responses_sampling_kwargs",
    "request_output_caps_override_server_default",
    "prompt_context_caps_stay_separate_from_output_caps",
    "legacy_completions_output_caps_override_server_default",
    "anthropic_bundle_defaults",
    "ollama_adapter_surface",
    "streaming_cache_detail_usage",
    "structured_schema_decode_repair",
    "structured_xml_repair_validation_boundary",
    "structured_repair_report_rates",
    "structured_json_retry_after_repair_failure",
    "structured_xml_retry_after_repair_failure",
    "structured_xml_stream_validation",
    "structured_guided_json_schema_token_masking",
    "structured_live_smoke_response_format_adoption",
    "response_format_docs_repair_validation_boundary",
    "responses_previous_response_history",
    "responses_streaming_tool_call_arguments_and_indexes",
    "cache_reuse_endpoints",
    "cache_stats_reuse_skip_telemetry",
    "gateway_stale_port_startup",
    "gateway_standby_wake_routing",
    "gateway_responses_function_call_arguments_streaming",
    "gateway_responses_reasoning_empty_final_arguments_streaming",
    "gateway_stale_responses_port_rejection",
    "panel_tool_status_responses_argument_recovery",
    "dsv4_native_cache_status",
    "dsv4_dsml_parser_residue_rejection",
    "dsv4_dsml_valid_tool_call_preserved",
    "dsv4_suppressed_tool_markup_not_stored",
    "zaya_typed_cca_status",
    "plain_attention_kv_status",
    "jangtq_mpp_nax_health_kernel_name",
    "hybrid_ssm_partial_reuse",
    "turboquant_kv_runtime_contract",
    "turboquant_disk_roundtrip",
    "no_generic_tq_on_hybrid_ssm",
    "all_required_named_rows_ran",
)

EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS = (
    "reasoning_on_off_request_wiring_explicit",
    "no_hidden_family_sampling_or_reasoning_forcing",
    "deepseek_r1_visible_content_no_think_tag_leak",
    "gemma4_visible_content_no_channel_marker_leak",
    "hy3_zaya_style_no_think_no_visible_tag_leak",
    "reasoning_tool_interaction_preserves_tool_calls",
    "streaming_fallback_does_not_double_extract_reasoning",
    "interleaved_reasoning_segments_render",
    "visible_token_counts_not_corrupted_by_reasoning_ui",
    "legacy_count_floor_still_nontrivial",
)

EXPECTED_CURRENT_TOOL_CALL_CHECKS = (
    "tool_parser_residue_rejected_instead_of_executed",
    "schema_valid_dsml_tool_call_preserved",
    "dsv4_default_cache_degraded_dsml_shapes_repaired_when_schema_valid",
    "dsv4_tool_preamble_suppressed_and_not_stored_without_call",
    "tool_choice_none_does_not_fallback_to_raw_dsml",
    "panel_tool_executor_blocks_unsafe_paths_and_commands",
    "panel_max_tool_iterations_caps_tool_loops",
    "live_default_cache_dsv4_tool_loop_artifact_present",
    "all_required_tool_call_markers_present",
)

EXPECTED_CURRENT_NATIVE_MTP_CHECKS = (
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
    "legacy_count_floor_still_nontrivial",
)

EXPECTED_CURRENT_VL_MEDIA_CHECKS = (
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
    "all_required_engine_markers_present",
    "all_required_panel_markers_present",
)

EXPECTED_CURRENT_MCP_POLICY_CHECKS = (
    "mcp_autodiscovery_without_cli_or_env",
    "mcp_policy_filters_servers_tools_before_schema_merge",
    "mcp_disabled_tool_execution_rejected_server_side",
    "mcp_status_redacts_urls_headers_env",
    "mcp_command_security_blocks_injection",
    "panel_mcp_policy_flags_and_preview_wired",
    "panel_mcp_config_import_redacts_metadata",
    "mcp_gateway_routes_by_explicit_model",
    "mcp_gateway_rejects_ambiguous_multi_session_requests",
    "builtin_electron_tools_separate_from_mcp_execution",
    "all_required_mcp_policy_markers_present",
)

EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS = (
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
    "legacy_count_floor_still_nontrivial",
    "all_required_max_output_context_markers_present",
)

EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS = (
    "release_gate_unit_contracts_pass",
    "bundled_python_verify_passes",
    "bundled_engine_version_matches_package_json",
    "bundled_engine_hash_parity",
    "bundled_jang_tools_hash_parity",
    "bundled_console_scripts_relocatable",
    "bundled_media_and_jang_dependencies_import",
    "packaged_renderer_dsv4_cache_ui_deduped",
    "packaged_renderer_max_thinking_tokens_wired",
    "packaged_epipe_closed_stream_guards",
    "packaged_user_data_isolation_bootstrap",
    "staged_app_engine_source_hash_parity",
    "release_dmg_hardened_runtime_entitlements",
    "dry_release_gate_fails_only_on_known_objectives",
    "dry_release_gate_uses_current_objective_digest",
    "packaged_python_has_no_pycache",
)

EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS = (
    "source_version_consistent",
    "local_updater_not_ahead_of_source",
    "staged_source_version_not_public",
    "local_updater_release_state_valid",
    "local_updater_platform_downloads_valid",
    "local_updater_has_required_fields",
    "local_updater_url_matches_version",
    "local_updater_sha256_valid",
    "local_updater_notes_match_version",
)


REQUIRED_RELEASE_DOMAINS = {
    "chat_ui_settings",
    "generation_defaults",
    "parser_registry",
    "reasoning_template",
    "tool_calls",
    "api_surface",
    "cache_architecture",
    "model_artifact_detection",
    "model_family_detection",
    "native_mtp",
    "mcp",
    "vl_media",
    "packaging_release",
    "performance",
    "release_surface",
    "pr_intake",
}

CURRENT_POST_BUDGET_EDGE_ARTIFACTS = {
    "noheavy-api-cache-endpoint-runtime": "build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json",
    "chat-settings-max-output-context-ui": "build/current-max-output-context-contract-after-jangtq2-objective-refresh-20260607.json",
    "panel-session-cache-settings-family-gating": "build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json",
    "generation-defaults-no-hidden-forcing": "build/current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json",
    "parser-registry-tool-reasoning-parity": "build/current-parser-registry-contract-after-jangtq2-objective-refresh-20260607.json",
    "reasoning-template-no-think-tag-leak": "build/current-reasoning-template-contract-20260526-settings-audit.json",
    "tool-call-loop-parser-cleanup": "build/current-tool-call-contract-after-cross-model-loop-metrics-20260609.json",
    "panel-tool-security-loop-boundary": "build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json",
    "api-chat-responses-anthropic-ollama-parity": "build/current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json",
    "cache-architecture-family-classification": "build/current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json",
    "jang-model-compat-runtime-boundary": "build/current-jang-model-compat-contract-20260528-pr155-runtime-boundary.json",
    "model-artifact-format-detection": "build/current-model-artifact-format-contract-after-mllm-tight-memory-guard-20260607.json",
    "model-family-detection-noheavy": "build/current-model-family-detection-contract-after-n2-policy-row-20260609.json",
    "native-mtp-d3-effect-policy": "build/current-native-mtp-contract-after-noheavy-contract-refresh-20260608.json",
    "mcp-policy-ui-gateway": "build/current-mcp-policy-contract-20260531-post-step-lfm-refresh.json",
    "vl-media-cache-tool-followup": "build/current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json",
    "packaged-release-integrity": "build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json",
    "public-release-surface-preflight": "build/current-release-surface-contract-20260602-v154-live-public-after-site-fix.json",
}

CURRENT_REGRESSION_SUITE_ARTIFACT = (
    "build/current-regression-suite-after-pr-intake-matrix-refresh-20260609.json"
)
CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT = (
    "build/current-issue175-179-release-boundary-audit-after-issue179-memory-preflight-20260607.json"
)
CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT = (
    "build/current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json"
)
CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT = (
    "build/current-installed-app-runtime-parity-audit-tahoe-checkpoint-dmg-20260609.json"
)
CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT = (
    "build/current-issue175-177-installed-runtime-audit-20260602-v1554-installed-tahoe.json"
)
CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT = (
    "build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json"
)
CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT = (
    "build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json"
)
CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT = (
    "build/current-issue179-minimax-k-responses-cancel-probe-memory-preflight-20260602-local-ready-check.json"
)
CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT = (
    "build/current-issue181-183-runtime-audit-20260602-v1554-installed-tahoe-refresh.json"
)
CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT = (
    "build/current-public-app-issue-audit-after-checkpoint-packaged-integrity-20260609.json"
)
CURRENT_DEV_UI_PROOF_ARTIFACTS = {
    "proof": "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-proof.json",
    "chat_settings_screenshot": "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-chat-settings.png",
    "server_cache_screenshot": "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-server-cache-settings.png",
}
CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS = {
    "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-proof.json",
    "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-chat.png",
}
CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT = (
    "build/current-real-ui-dsv4-memory-preflight-after-lfm-step-manifest-fix-20260604.json"
)
CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT = (
    "build/current-step37-vlm-runtime-audit-after-source-live-media-proof-20260607.json"
)
DEFERRED_RELEASE_FAMILIES = {
    "dsv4": "deferred_per_20260602_emergency_release_scope",
}
CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT = (
    "build/current-dsv4-route-mode-code-exactness-preflight-after-mimo-classifier-refresh-20260608.json"
)
EXPECTED_DSV4_SOURCE_PREFLIGHT_CASES = (
    "chat_off",
    "chat_off_rep1",
    "chat_off_no_punct_rep1",
    "chat_off_bundle_defaults",
    "chat_on",
    "chat_on_rep1",
    "chat_max",
    "responses_off",
    "responses_off_rep1",
    "responses_off_no_punct_rep1",
    "responses_off_bundle_defaults",
    "responses_on",
    "responses_on_rep1",
    "legacy_completion_raw",
)
CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS = {
    "zaya_text": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4",
        "model_name": "ZAYA1-8B-MXFP4",
        "family": "zaya_text",
    },
    "zaya_text_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4",
        "model_name": "ZAYA1-8B-MXFP4",
        "family": "zaya_text",
    },
    "zaya_text_responses_tools": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-tools-cachecontrols-disk-terminal-fix-20260601-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-tools-cachecontrols-disk-terminal-fix-20260601-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4",
        "model_name": "ZAYA1-8B-MXFP4",
        "family": "zaya_text",
    },
    "zaya_vl_image": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-current-source-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-current-source-20260531-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
        "model_name": "ZAYA1-VL-8B-JANGTQ4",
        "family": "zaya_vl",
    },
    "zaya_vl_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-cachecontrols-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-cachecontrols-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
        "model_name": "ZAYA1-VL-8B-JANGTQ4",
        "family": "zaya_vl",
    },
    "zaya_vl_responses_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-responses-filetools-strictargs-cachecontrols-pointclean-localonly-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-responses-filetools-strictargs-cachecontrols-pointclean-localonly-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
        "model_name": "ZAYA1-VL-8B-JANGTQ4",
        "family": "zaya_vl",
    },
    "zaya_vl_responses_tools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-responses-stricttools-cachecontrols-20260530-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-responses-stricttools-cachecontrols-20260530-chat.png",
        "model_path": "/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4",
        "model_name": "ZAYA1-VL-8B-JANGTQ4",
        "family": "zaya_vl",
    },
    "ling_bailing_jangtq": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ",
        "model_name": "Ling-2.6-flash-JANGTQ",
        "family": "ling_bailing",
    },
    "ling_bailing_jangtq_responses_tools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ",
        "model_name": "Ling-2.6-flash-JANGTQ",
        "family": "ling_bailing",
    },
    "gemma4": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-stricttools-max768-visible-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-stricttools-max768-visible-20260531-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "model_name": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "family": "gemma4",
    },
    "gemma4_reasoning": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-reasoningonly-max768-visible-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-gemma4-responses-reasoningonly-max768-visible-20260531-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "model_name": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "family": "gemma4",
    },
    "gemma4_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-l2storage-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-gemma4-cachecontrols-l2storage-20260531-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "model_name": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "family": "gemma4",
    },
    "qwen36_mxfp4_crack": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
        "model_name": "Qwen3.6-27B-JANG_4M-MTP",
        "family": "qwen36",
    },
    "qwen36_mxfp4_crack_video": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max256-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-video-reasoning-cachecontrols-max256-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
        "model_name": "Qwen3.6-27B-JANG_4M-MTP",
        "family": "qwen36",
    },
    "qwen36_mxfp4_crack_responses_tools_reasoning_image_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
        "model_name": "Qwen3.6-27B-JANG_4M-MTP",
        "family": "qwen36",
    },
    "qwen36_mxfp4_crack_responses_stricttools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
        "model_name": "Qwen3.6-27B-JANG_4M-MTP",
        "family": "qwen36",
    },
    "qwen36_35b_mxfp8_mtp_responses_tools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        "model_name": "Qwen3.6-35B-A3B-MXFP8-MTP",
        "family": "qwen36",
    },
    "qwen36_35b_mxfp8_mtp_responses_tools_reasoning_image_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-image-reasoning-cachecontrols-max256-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        "model_name": "Qwen3.6-35B-A3B-MXFP8-MTP",
        "family": "qwen36",
    },
    "qwen36_35b_mxfp8_mtp_responses_tools_reasoning_video_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-qwen36-35b-mxfp8-mtp-responses-tools-video-reasoning-cachecontrols-max512-20260607-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        "model_name": "Qwen3.6-35B-A3B-MXFP8-MTP",
        "family": "qwen36",
    },
    "minimax_m27_small_jangtq": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-jangtq-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-jangtq-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        "model_name": "MiniMax-M2.7-Small-JANGTQ",
        "family": "minimax",
    },
    "minimax_m27_small_responses_tools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        "model_name": "MiniMax-M2.7-Small-JANGTQ",
        "family": "minimax",
    },
    "minimax_m27_small_responses_stricttools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-stricttools-cachecontrols-20260530-chat.png",
        "model_path": "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        "model_name": "MiniMax-M2.7-Small-JANGTQ",
        "family": "minimax",
    },
    "minimax_m27_small_installed_responses_tools_cachecontrols_singleid": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-installed-responses-tools-cachecontrols-singleid-20260528-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-installed-responses-tools-cachecontrols-singleid-20260528-chat.png",
        "model_path": "/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ",
        "model_name": "MiniMax-M2.7-Small-JANGTQ",
        "family": "minimax",
    },
    "nemotron_omni_nano_jangtq": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK",
        "model_name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "family": "nemotron_omni",
    },
    "nemotron_omni_nano_responses_tools_reasoning_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK",
        "model_name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "family": "nemotron_omni",
    },
    "nemotron_omni_nano_responses_reasoning": {
        "proof": "docs/internal/agent-notes/diagnostic-real-ui-live-model-nemotron-omni-nano-responses-reasoning-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/diagnostic-real-ui-live-model-nemotron-omni-nano-responses-reasoning-20260531-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK",
        "model_name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "family": "nemotron_omni",
    },
    "nemotron_omni_nano_responses_tools_filesemantic": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531-chat.png",
        "model_path": "/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK",
        "model_name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "family": "nemotron_omni",
    },
    "hy3_jangtq2": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2",
        "model_name": "Hy3-preview-JANGTQ2",
        "family": "hy3",
    },
    "hy3_jangtq2_responses_reasoning": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-reasoning-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-reasoning-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2",
        "model_name": "Hy3-preview-JANGTQ2",
        "family": "hy3",
    },
    "hy3_jangtq2_responses_tools_thinkingoff": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-thinkingoff-toolforced-20260527-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-thinkingoff-toolforced-20260527-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2",
        "model_name": "Hy3-preview-JANGTQ2",
        "family": "hy3",
    },
    "hy3_jangtq2_responses_tools_cachecontrols": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530-chat.png",
        "model_path": "/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2",
        "model_name": "Hy3-preview-JANGTQ2",
        "family": "hy3",
    },
    "step37_flash_jang2l": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-devbuild-tools-image-cache-templateclean-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-devbuild-tools-image-cache-templateclean-20260531-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L",
        "model_name": "Step-3.7-Flash-JANG_2L",
        "family": "step37",
    },
    "step37_flash_jang2l_reasoning": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-reasoning-max768-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-reasoning-max768-20260531-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L",
        "model_name": "Step-3.7-Flash-JANG_2L",
        "family": "step37",
    },
    "step37_flash_jang2l_l2storage": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-cachecontrols-l2storage-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-cachecontrols-l2storage-20260531-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L",
        "model_name": "Step-3.7-Flash-JANG_2L",
        "family": "step37",
    },
    "step37_flash_jang2l_tool_l2storage": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-tools-l2storage-pagedlocked-after-subtype-ui-fix-20260531-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-step37-jang2l-responses-tools-l2storage-pagedlocked-after-subtype-ui-fix-20260531-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L",
        "model_name": "Step-3.7-Flash-JANG_2L",
        "family": "step37",
    },
    "lfm25_moe_a1b": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP4",
        "model_name": "LFM2.5-8B-A1B-MXFP4",
        "family": "lfm25",
    },
    "lfm25_moe_a1b_responses_delta": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP4",
        "model_name": "LFM2.5-8B-A1B-MXFP4",
        "family": "lfm25",
    },
    "lfm25_moe_a1b_responses": {
        "proof": "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-proof.json",
        "chat_screenshot": "docs/internal/agent-notes/current-real-ui-live-model-lfm25-mxfp4-responses-tools-cachecontrols-20260607-chat.png",
        "model_path": "/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP4",
        "model_name": "LFM2.5-8B-A1B-MXFP4",
        "family": "lfm25",
    },
}
CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS.update(
    {
        artifact_key: artifact
        for row_id, row in CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.items()
        for artifact_key, artifact in (
            (f"{row_id}_proof", row["proof"]),
            (f"{row_id}_chat_screenshot", row["chat_screenshot"]),
        )
    }
)
REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES = (
    "zaya_text",
    "zaya_vl",
    "ling_bailing",
    "gemma4",
    "qwen36",
    "hy3",
    "minimax",
    "nemotron_omni",
    "dsv4",
    "step37",
    "lfm25",
)
REQUIRED_REAL_UI_LIVE_MODEL_VARIANTS_BY_FAMILY = {
    "qwen36": {
        "model_names": {
            "Qwen3.6-27B-JANG_4M-MTP",
            "Qwen3.6-35B-A3B-MXFP8-MTP",
        },
        "model_paths": {
            "/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
            "/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
        },
    },
}
REQUIRED_REAL_UI_LIVE_MODEL_SURFACES = (
    "current_electron_dev_build",
    "real_loaded_model",
    "chat_completions",
    "responses_api",
    "responses_delta_streaming",
    "responses_cache_detail_usage",
    "live_speed_floor",
    "generation_defaults_applied",
    "multi_turn_chat",
    "long_tool_loop",
    "reasoning_display",
    "parser_leak_check",
    "language_leak_check",
    "cache_hit_telemetry",
    "native_cache_status",
    "architecture_cache_policy",
    "cache_endpoint_stats",
    "l2_disk_storage",
    "settings_persistence",
    "server_cache_controls",
    "vl_image",
    "video_where_supported",
)
REAL_UI_INTEGRATED_TOOL_L2_CACHE_SURFACE = "tool_l2_cache_integrated"
REAL_UI_INTEGRATED_VL_TOOL_L2_CACHE_SURFACE = "vl_tool_l2_cache_integrated"
_REQUIRED_REAL_UI_LIVE_MODEL_GENERIC_SURFACES = tuple(
    surface
    for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
    if surface != "architecture_cache_policy"
    and surface != "responses_cache_detail_usage"
    and surface != "live_speed_floor"
)
_REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES = tuple(
    surface
    for surface in _REQUIRED_REAL_UI_LIVE_MODEL_GENERIC_SURFACES
    if surface not in {"vl_image", "video_where_supported"}
)
_REQUIRED_REAL_UI_LIVE_MODEL_NON_REASONING_TEXT_SURFACES = tuple(
    surface
    for surface in _REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES
    if surface != "reasoning_display"
)
REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY = {
    "zaya_text": _REQUIRED_REAL_UI_LIVE_MODEL_NON_REASONING_TEXT_SURFACES,
    "ling_bailing": _REQUIRED_REAL_UI_LIVE_MODEL_NON_REASONING_TEXT_SURFACES,
    "gemma4": _REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES,
    "hy3": _REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES,
    "minimax": _REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES,
    "nemotron_omni": _REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES,
    "dsv4": _REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES,
    "step37": tuple(
        surface
        for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
        if surface != "video_where_supported"
    )
    + (
        REAL_UI_INTEGRATED_TOOL_L2_CACHE_SURFACE,
        REAL_UI_INTEGRATED_VL_TOOL_L2_CACHE_SURFACE,
    ),
    "lfm25": tuple(
        surface
        for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
        if surface
        not in {
            "reasoning_display",
            "vl_image",
            "video_where_supported",
        }
    )
    + (REAL_UI_INTEGRATED_TOOL_L2_CACHE_SURFACE,),
    "zaya_vl": (
        *_REQUIRED_REAL_UI_LIVE_MODEL_NON_REASONING_TEXT_SURFACES,
        "vl_image",
    ),
    "qwen36": (
        *_REQUIRED_REAL_UI_LIVE_MODEL_NON_MEDIA_SURFACES,
        "vl_image",
        "video_where_supported",
    ),
}
CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS = {
    "zaya_text_mxfp4": "build/current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json",
    "ling_flash_tq": "build/current-filtered-live-smoke-ling-flash-jangtq-20260607/summary.json",
    "gemma4_crack": "build/current-all-local-model-smoke-gemma26-jang4m-bundled-tools-media-20260607/summary.json",
    "qwen36_moe_crack": "build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/summary.json",
    "hy3_preview_jangtq2": "build/current-filtered-live-smoke-hy3-preview-jangtq2-20260607/summary.json",
    "minimax_m27_tq_k": "build/current-all-local-model-smoke-minimaxk-bundled-after-required-tool-256-20260607/summary.json",
    "nemotron_omni_tq2_system_nomedia": "build/current-filtered-live-smoke-nemotron-omni-jangtq-20260607/summary.json",
    "zaya_vl_jangtq4": "build/current-all-local-model-smoke-zaya-vl-mxfp4-bundled-after-source-tool-pass-20260607/summary.json",
    "dsv4_jang_local": "build/current-filtered-live-smoke-dsv4-jangtq-k-20260607/summary.json",
}
CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS = {
    "nemotron_omni_tq2_system_nomedia": "build/current-filtered-live-smoke-nemotron-omni-jangtq-20260607/summary.json",
    "ling_flash_tq": "build/current-filtered-live-smoke-ling-flash-jangtq-20260607/summary.json",
    "gemma4_crack": "build/current-all-local-model-smoke-gemma26-jang4m-bundled-tools-media-20260607/summary.json",
    "qwen36_moe_crack": "build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/summary.json",
    "hy3_preview_jangtq2": "build/current-filtered-live-smoke-hy3-preview-jangtq2-20260607/summary.json",
    "minimax_m27_tq_k": "build/current-all-local-model-smoke-minimaxk-bundled-after-required-tool-256-20260607/summary.json",
    "zaya_vl_jangtq4": "build/current-all-local-model-smoke-zaya-vl-mxfp4-bundled-after-source-tool-pass-20260607/summary.json",
}
CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT = (
    "build/current-mimo-jang2l-local-structural-verify-20260606.json"
)
CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT = (
    "build/current-mimo-v25-jang2l-live-cb-cache-text-20260610.json"
)
CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT = (
    "build/current-mimo-v2-switchglu-selected-expert-parity-20260609.json"
)
CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT = (
    "build/current-mimo-v2-jang2l-long-prompt-first-request-oom-20260606.json"
)
CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT = (
    "build/current-mimo-v25-jang2l-chat-tool-boundary-20260610.json"
)
CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT = (
    "build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json"
)
CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT = (
    "build/current-mimo-v2-local-bundle-metadata-contract-20260607.json"
)
CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT = (
    "build/current-mimo-v2-jang2l-source-vs-quant-first-divergence-20260606.json"
)
CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT = (
    "build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json"
)
CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT = (
    "build/current-mimo-v25-jang2l-restart-l2-restore-20260610-rerun/summary.json"
)
CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT = (
    "build/current-real-ui-live-model-mimo-v25-jang2l-responses-tools-rerun-20260610.json"
)
CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS = {
    # Retired 2026-06-07: old expected-failure diagnostic rows pointed at
    # missing 202605 artifacts or at probes that now pass. Current release
    # blockers are tracked by live_smoke_summaries, live_tool_smoke_summaries,
    # current objective checklist, MiMo audit, Step3p7 VLM audit, Nemotron media
    # gate, MiniMax #179 audit, and DSV4 exactness preflight. Do not add
    # expected-failure diagnostics here unless they point at current artifacts
    # and assert a still-current failure mode.
}
CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "zaya_text_mxfp4": {
        "name": "ZAYA1-8B-MXFP4",
        "model_type": "zaya",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "zaya_cca",
    },
    "nemotron_omni_tq2_system_nomedia": {
        "name": "Nemotron-Omni-Nano-JANGTQ-CRACK",
        "model_type": "nemotron_h",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "hybrid_ssm",
    },
    "ling_flash_tq": {
        "name": "Ling-2.6-flash-JANGTQ",
        "model_type": "bailing_hybrid",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "hybrid_ssm",
    },
    "gemma4_crack": {
        "name": "Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        "model_type": "gemma4",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "swa_rotating",
    },
    "qwen36_moe_crack": {
        "name": "Qwen3.6-27B-JANG_4M-MTP",
        "model_type": "qwen3_5",
        "is_mllm": True,
        "supports_video": True,
        "supports_thinking": True,
        "cache_family": "hybrid_ssm",
    },
    "hy3_preview_jangtq2": {
        "name": "Hy3-preview-JANGTQ2",
        "model_type": "hy_v3",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "hybrid_ssm",
    },
    "minimax_m27_tq_k": {
        "name": "MiniMax-M2.7-JANGTQ_K-CRACK",
        "model_type": "minimax_m2",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "hybrid_ssm",
    },
    "zaya_vl_jangtq4": {
        "name": "ZAYA1-VL-8B-MXFP4",
        "model_type": "zaya1_vl",
        "is_mllm": True,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "zaya_cca",
    },
    "dsv4_jang_local": {
        "name": "DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520",
        "model_type": "deepseek_v4",
        "is_mllm": False,
        "supports_video": False,
        "supports_thinking": True,
        "cache_family": "deepseek_v4_composite",
    },
}


# Current filtered smoke artifacts use the current metadata truth for these rows.
CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS["zaya_vl_jangtq4"]["supports_thinking"] = False
CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS["dsv4_jang_local"]["name"] = "DeepSeek-V4-Flash-JANGTQ-K"

LIVE_SMOKE_TEXT_REQUEST_LABELS = (
    "text_cache_repeat_1",
    "text_cache_repeat_2",
    "text_multiturn_recall",
)
LIVE_SMOKE_THINKING_REQUEST_LABELS = ("reasoning_on",)
LIVE_SMOKE_IMAGE_REQUEST_LABELS = (
    "text_no_media_after_image",
    "vl_blue_image",
    "vl_blue_image_repeat",
    "vl_red_image_changed",
)
LIVE_SMOKE_VIDEO_REQUEST_LABELS = (
    "text_no_media_after_video",
    "vl_blue_video",
)
LEGACY_LIVE_SMOKE_MISSING_STATUS_ROWS = frozenset({"zaya_text_mxfp4"})


def _required_live_smoke_request_labels(
    row_id: str,
    result: dict[str, Any],
) -> list[str]:
    row = result.get("row")
    if not isinstance(row, dict):
        return []
    labels = list(LIVE_SMOKE_TEXT_REQUEST_LABELS)
    if row.get("supports_thinking") is True:
        labels.extend(LIVE_SMOKE_THINKING_REQUEST_LABELS)
    if row.get("is_mllm") is True:
        labels.extend(LIVE_SMOKE_IMAGE_REQUEST_LABELS)
    if _effective_live_smoke_supports_video(result):
        labels.extend(LIVE_SMOKE_VIDEO_REQUEST_LABELS)
    return sorted(set(labels))


def _effective_live_smoke_supports_video(result: dict[str, Any]) -> bool:
    capabilities = result.get("capabilities")
    if isinstance(capabilities, dict):
        body = capabilities.get("body")
        if isinstance(body, dict):
            modalities = body.get("modalities")
            if isinstance(modalities, list):
                return any(str(item).lower() == "video" for item in modalities)

    row = result.get("row")
    return isinstance(row, dict) and row.get("supports_video") is True


def _live_smoke_identity_mismatches(
    row_id: str,
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    row = result.get("row")
    if not isinstance(row, dict):
        row = {}
    expected = CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS.get(row_id, {})
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected_value in expected.items():
        actual_value = row.get(key)
        if actual_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }
    return mismatches


def _expected_live_smoke_capability_family(row_id: str, expected: dict[str, Any]) -> Any:
    if row_id == "ling_flash_tq":
        return "ling"
    if row_id == "minimax_m27_tq_k":
        return "minimax"
    return expected.get("model_type")


def _expected_live_smoke_capability_parsers(row_id: str) -> dict[str, Any]:
    return {
        "ling_flash_tq": {
            "tool_parser": "deepseek",
            "reasoning_parser": None,
        },
        "gemma4_crack": {
            "tool_parser": "gemma4",
            "reasoning_parser": "gemma4",
        },
        "qwen36_moe_crack": {
            "tool_parser": "qwen",
            "reasoning_parser": "qwen3",
        },
        "hy3_preview_jangtq2": {
            "tool_parser": "hunyuan",
            "reasoning_parser": "qwen3",
        },
        "minimax_m27_tq_k": {
            "tool_parser": "minimax",
            "reasoning_parser": "minimax_m2",
        },
        "nemotron_omni_tq2_system_nomedia": {
            "tool_parser": "nemotron",
            "reasoning_parser": "deepseek_r1",
        },
        "zaya_vl_jangtq4": {
            "tool_parser": "zaya_xml",
            "reasoning_parser": None,
        },
        "dsv4_jang_local": {
            "tool_parser": "dsml",
            "reasoning_parser": "deepseek_r1",
        },
    }.get(row_id, {})


def _live_smoke_capability_mismatches(
    row_id: str,
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    capabilities = result.get("capabilities")
    if not isinstance(capabilities, dict):
        return {}
    body = capabilities.get("body")
    if not isinstance(body, dict):
        body = {}
    expected = CURRENT_COVERED_LIVE_SMOKE_ROW_EXPECTATIONS.get(row_id, {})
    mismatches: dict[str, dict[str, Any]] = {}

    expected_family = _expected_live_smoke_capability_family(row_id, expected)
    if expected_family is not None and body.get("family") != expected_family:
        mismatches["family"] = {
            "expected": expected_family,
            "actual": body.get("family"),
        }

    for key, expected_value in _expected_live_smoke_capability_parsers(row_id).items():
        actual_value = body.get(key)
        if actual_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    return mismatches


def _live_smoke_surface_issues(result: dict[str, Any]) -> list[str]:
    command = result.get("command")
    if not isinstance(command, list) or not command:
        return ["command_missing"]
    python = str(command[0])
    normalized = python.replace("\\", "/")
    if "/panel/bundled-python/python/bin/python" not in normalized and not normalized.startswith(
        "panel/bundled-python/python/bin/python"
    ):
        return ["command_python_not_bundled"]
    return []


def _declared_request_validation_failures(request: dict[str, Any]) -> list[str]:
    raw_failures = request.get("validation_failures")
    if not isinstance(raw_failures, list):
        raw_failures = []
    return [
        str(item)
        for item in raw_failures
        if item
    ]


def _live_smoke_cache_validation_failures(request: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if request.get("label") != "text_cache_repeat_2":
        return failures

    usage = request.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    cached_tokens = 0
    if isinstance(details, dict):
        try:
            cached_tokens = int(details.get("cached_tokens") or 0)
        except (TypeError, ValueError):
            cached_tokens = 0

    cache_summary = request.get("cache_summary")
    summary_has_hit = False
    summary_hit_tokens = 0
    if isinstance(cache_summary, dict):
        summary_has_hit = cache_summary.get("has_cache_hit") is True
        try:
            summary_hit_tokens = int(cache_summary.get("cache_hit_tokens") or 0)
        except (TypeError, ValueError):
            summary_hit_tokens = 0

    if cached_tokens <= 0 and not summary_has_hit and summary_hit_tokens <= 0:
        failures.append("text_cache_repeat_2:expected_cache_hit_missing")
    return failures


EXPECTED_CURRENT_OPEN_REQUIREMENTS = [
    "Cross-family live multi-turn smoke matrix is release-cleared",
    "Gemma QAT/native MXFP4 E2B/E4B/12B/26B/31B runtime/media/cache/API/UI quality is release-cleared",
    "MiMo V2.5 JANG_2L runtime/tool/long-prompt quality is release-cleared",
    "N2 Pro 397B JANG1L/JANGTQ runtime/cache/API/UI quality is release-cleared",
    "MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared",
    "Real Electron UI cross-family live model matrix is release-cleared",
    "DSV4 long-output/code/file-generation quality is release-cleared",
]
DEFERRED_RELEASE_OPEN_REQUIREMENTS = {
    "Real Electron UI cross-family live model matrix is release-cleared",
    "DSV4 long-output/code/file-generation quality is release-cleared",
}
DEFERRED_RELEASE_BLOCKER_IDS = {
    "real_ui_live_model_matrix": "deferred_per_20260602_emergency_release_scope",
}


_ROWS: list[dict[str, Any]] = [
    {
        "id": "noheavy-api-cache-endpoint-runtime",
        "domain": "api_surface",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "OpenAI Chat, Responses, Anthropic, and Ollama cache/default endpoint behavior stays covered by the no-heavy API/cache gate",
            "Native cache status rows for DSV4 composite, ZAYA typed CCA, plain attention KV, and hybrid SSM partial reuse remain explicit",
            "TurboQuant KV runtime serialization, disk L2 roundtrip, and prompt-disk backfill remain directly validated",
            "JANGTQ MPP/NAX health naming stays pinned as `turboquant_codebook_mpp_nax` in the no-heavy API/cache proof",
            "DSV4 DSML tool parsing repairs and residue rejection remain schema-gated and cannot be hidden by broader API surface checks",
            "Structured JSON schema repair decodes valid nested object strings without semantic value rewriting",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_noheavy_api_cache_contract.py --out build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json",
        ],
        "artifacts": [
            "build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json",
            "build/current-api-cache-contract-api-surface-check-20260602-cache-detail-zero-cached.json",
        ],
    },
    {
        "id": "chat-settings-max-output-context-ui",
        "domain": "chat_ui_settings",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Server Default Max Output Tokens maps to --max-tokens",
            "Max Context Tokens maps to --max-prompt-tokens",
            "Chat Max Output Tokens remains a per-chat/API override for non-streaming and streaming",
            "Legacy /v1/completions max_tokens remains a per-request output cap for non-streaming and streaming",
            "Prompt/context aliases clamp to server max-prompt-tokens without rewriting output caps",
            "Ollama gateway non-stream num_predict remains request-scoped across chat, templated generate, and raw generate",
            "Ollama gateway malformed num_predict values are omitted instead of poisoning max_tokens",
            "Ollama gateway malformed context values are omitted instead of poisoning max_prompt_tokens",
            "External coding-tool configs keep context window and output limit separate",
            "new-chat model-owned maxTokens cannot be replaced by inherited per-chat output caps",
            "server startup maxTokens and chat maxTokens remain independent when both are set",
            "persisted chat maxTokens cannot relaunch server with a new startup maxTokens",
            "per-chat maxTokens below or above the server startup default remain request-scoped",
            "explicit per-request Chat/Responses output caps can be below or above Server Default Max Output Tokens",
            "Explicit Chat/Responses output caps do not mutate the server startup default used by later Auto requests",
            "Streaming Chat/Responses output caps do not mutate the server startup default used by later Auto requests",
            "later Auto Chat/Responses requests still use the original server startup default",
            "Responses maxTokens below or above the server startup default remain request scoped",
            "Responses Auto does not synthesize max_output_tokens",
            "DB-cleared NULL chat maxTokens stays Auto for Chat Completions and Responses",
            "Raw API non-positive max_tokens/max_output_tokens are rejected rather than becoming hidden Auto defaults or empty generations",
            "Auto chat Max Tokens omits per-request output caps so server startup defaults can apply",
            "new chat output caps are not inherited or made sticky on default-profile or same-model clean chat creation",
            "chat reset does not convert model max_new_tokens into a sticky per-chat maxTokens override",
            "string-shaped legacy session maxTokens values are cleared before launch",
            "server startup maxTokens explicitness survives wake/reload and CLI implicit startup paths",
            "API gateway output-budget and context-budget paths are source-hashed",
            "DSV4 request-budget helper is source-hashed with the max-output boundary gate",
            "Casual preset maxTokens is documented as an explicit server output cap and does not change model-owned defaults or context",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_max_output_context_contract.py --out build/current-max-output-context-contract-after-jangtq2-objective-refresh-20260607.json",
        ],
        "artifacts": [
            "build/current-max-output-context-contract-after-jangtq2-objective-refresh-20260607.json",
            "build/current-max-output-context-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-max-output-context-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-max-output-context-contract-20260524-after-gemma4-telemetry-final.json",
            "build/current-max-output-context-contract-20260524-after-dsv4-neutral-reppen.json",
            "build/current-max-output-context-contract-20260523-profile-chat-cap.json",
            "build/current-max-output-context-contract-20260523-dsv4-budget-edge.json",
            "build/current-max-output-context-contract-20260522-recheck-server-chat-output-compat.json",
            "build/current-max-output-context-contract-20260522-persisted-chat-output-cap.json",
            "build/current-max-output-context-contract-20260522-new-chat-output-cap-nonsticky.json",
            "build/current-max-output-context-contract-20260522-chat-auto-server-default.json",
            "build/current-max-output-context-contract-20260522-responses-output-boundary.json",
            "build/current-max-output-context-contract-20260522-request-default-mutation.json",
            "build/current-max-output-context-contract-20260523-streaming-mutation.json",
            "build/current-max-output-context-contract-20260523-ollama-nonstream-caps.json",
            "build/current-max-output-context-contract-20260523-null-chat-cap.json",
            "build/current-max-output-context-contract-20260522-reset-policy-string-legacy.json",
            "build/current-max-output-context-contract-20260522-api-validator-caps.json",
            "build/current-max-output-context-contract-20260522-gateway-dsv4-budget-hash.json",
            "build/current-max-output-context-contract-20260522-casual-server-output-cap.json",
        ],
    },
    {
        "id": "panel-session-cache-settings-family-gating",
        "domain": "cache_architecture",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "DSV4 pool quant and native prefix controls stay DSV4-only",
            "DSV4 native composite prefix cache defaults on while explicit disable remains honored",
            "generic KV quantization is suppressed for DSV4 native composite cache",
            "non-DSV4 JANG/JANGTQ/MXFP cache toggles keep normal prefix/paged/L2/KV semantics",
            "non-DSV4 stale Additional Args cannot override app-owned server, template, model-name, MCP, max output/context, parser, reasoning, cache, or native-MTP launch policy",
            "launch-memory admission is warning-only for lazy-mmap JANG/JANGTQ bundles and does not hard-block on macOS cache pressure",
            "What's New update notice is localized across all five locales and mentions MCP, MTP, latest.json, Developer ID, L2 disk cache, and max_tokens",
            "panel typecheck and model-family registry stay green with those controls",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_noheavy_panel_settings_contract.py --out build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json",
        ],
        "artifacts": [
            "build/current-panel-settings-contract-proof-20260601-cache-ui-storage-quant.json",
            "build/current-panel-settings-contract-proof-20260524-text-additional-args-sanitizer.json",
            "build/current-panel-settings-contract-proof-20260523-post-budget-edge.json",
            "build/current-panel-settings-contract-proof-20260522-launch-memory-warning.json",
            "build/current-panel-settings-contract-proof-20260522-recheck-update-notice-i18n.json",
        ],
    },
    {
        "id": "generation-defaults-no-hidden-forcing",
        "domain": "generation_defaults",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "generation_config.json defaults are surfaced without copying into sticky overrides",
            "jang_config.json sampling defaults are surfaced without hidden sampler forcing",
            "request/API overrides remain explicit and win over startup defaults",
            "bundle max_new_tokens is preserved for omitted request budgets without hidden sampler or repetition floors",
            "server default output cap is not a request ceiling for explicit Chat Completions or Responses requests",
            "Additional Args cannot override app-owned reasoning, parser, cache, MTP, server, template, model-name, or MCP launch flags",
            "structured family matrix covers standard MLX, JANG, DSV4, max-token/context, thinking-budget, Step3p7 metadata-route, and app-owned CLI boundaries",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_generation_defaults_contract.py --out build/current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json",
        ],
        "artifacts": [
            "build/current-generation-defaults-contract-after-pr-intake-matrix-refresh-20260609.json",
            "build/current-generation-defaults-contract-20260528-family-matrix.json",
            "build/current-generation-defaults-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-generation-defaults-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-generation-defaults-contract-20260525-additional-args-guard.json",
            "build/current-generation-defaults-contract-20260524-after-gemma4-telemetry-final.json",
            "build/current-generation-defaults-contract-20260524-after-dsv4-neutral-reppen.json",
            "build/current-generation-defaults-contract-20260523-post-budget-edge.json",
            "build/current-generation-defaults-contract-20260521.json",
            "build/current-generation-defaults-contract-20260522-recheck-no-hidden-forcing.json",
        ],
    },
    {
        "id": "parser-registry-tool-reasoning-parity",
        "domain": "parser_registry",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Panel-emitted tool parser ids are accepted by the engine",
            "Panel-emitted reasoning parser ids are accepted by the engine",
            "CLI accepts every registered reasoning parser emitted by model families",
            "Reasoning parser dropdown covers every parser the panel registry can emit",
            "MiniMax minimax_m2 regression stays covered",
            "Qwen2/Qwen2-VL, Gemma 3, and GLM base stay off reasoning rails",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_parser_registry_contract.py --out build/current-parser-registry-contract-after-jangtq2-objective-refresh-20260607.json",
        ],
        "artifacts": [
            "build/current-parser-registry-contract-after-jangtq2-objective-refresh-20260607.json",
            "build/current-parser-registry-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-parser-registry-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-parser-registry-contract-20260523-post-budget-edge.json",
            "build/current-parser-registry-contract-20260522-reasoning-dropdown.json",
            "build/current-parser-registry-contract-20260522-non-reasoning-boundaries.json",
        ],
    },
    {
        "id": "reasoning-template-no-think-tag-leak",
        "domain": "reasoning_template",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Reasoning on/off request wiring stays explicit",
            "Server-side reasoning parsers do not leak think tags into visible content",
            "Client streaming fallback does not double-extract reasoning_content",
            "Interleaved reasoning segments render without corrupting token counts",
            "DSV4 requested reasoning-on/effort rails are preserved while direct/off requests use the audited identifier-safe rail",
            "MiniMax and Ling family-specific reasoning boundaries stay explicit without sampling floors",
            "tool follow-up reasoning state resets before subsequent visible content or tool parsing",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_reasoning_template_contract.py --out build/current-reasoning-template-contract-20260526-settings-audit.json",
        ],
        "artifacts": [
            "build/current-reasoning-template-contract-20260526-settings-audit.json",
            "build/current-reasoning-template-contract-20260523-post-budget-edge.json",
            "build/current-reasoning-template-contract-20260521.json",
            "build/current-reasoning-template-contract-20260522-recheck-parser-rails.json",
        ],
    },
    {
        "id": "tool-call-loop-parser-cleanup",
        "domain": "tool_calls",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Tool parser residue is rejected instead of executed",
            "DSV4 degraded DSML variants from live default-cache runs are repaired only when schema-valid",
            "live DSV4 write_file DSML degradation is repaired schema-safely",
            "default-cache live loop reaches three tools but still fails exact WebGLRenderer fidelity",
            "explicit-thinking DSV4 default-cache tool-loop diagnostic remains review: thinking does not clear tool ordering, final DONE, or exact scene.js fidelity",
            "maxToolIterations caps tool loops",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_tool_call_contract.py --out build/current-tool-call-contract-after-cross-model-loop-metrics-20260609.json",
        ],
        "artifacts": [
            "build/current-tool-call-contract-after-cross-model-loop-metrics-20260609.json",
            "build/current-tool-call-contract-20260523-post-budget-edge.json",
            "build/current-tool-call-contract-20260521.json",
            "build/current-tool-call-contract-20260522-dsv4-live-write-file-repair.json",
            "build/current-dsv4-default-cache-tool-loop/result.json",
            "build/current-dsv4-default-cache-tool-loop-thinking-on-20260525.json",
            "build/current-dsv4-default-cache-tool-loop-poolon-materialized-parserfix-20260522/result.json",
        ],
    },
    {
        "id": "panel-tool-security-loop-boundary",
        "domain": "tool_calls",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Panel built-in tool execution blocks unsafe paths and command injection",
            "Panel tool auto-continue obeys max tool-iteration limits instead of looping indefinitely",
            "The Electron tool loop security gate is a first-class release proof, not only an incidental current-suite step",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_panel_tool_security_contract.py --out build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json",
        ],
        "artifacts": [
            "build/current-panel-tool-security-contract-20260528-tool-loop-security-matrix.json",
        ],
    },
    {
        "id": "api-chat-responses-anthropic-ollama-parity",
        "domain": "api_surface",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "OpenAI Chat Completions sampling/default propagation for non-streaming and streaming",
            "OpenAI Responses sampling/default propagation for non-streaming and streaming",
            "OpenAI legacy Completions max_tokens/default propagation for non-streaming and streaming",
            "Anthropic adapter bundle defaults and streaming max_tokens override semantics",
            "Prompt/context alias clamp semantics across OpenAI, Anthropic, and Ollama-compatible surfaces",
            "Ollama adapter streaming/done behavior, streaming num_predict output-cap overrides, malformed num_predict omission, and malformed context omission",
            "Gateway single-model mode auto-switches by requested model id before preserving Ollama streaming content deltas",
            "Gateway disconnect handling guards every streaming JSON write, byte-forwarding path, request-body write, backend request end, destroyed client/backend socket, top-level request handler disconnect failures, nested broken-pipe/write-after-end stream errors, and Ollama client-close backend response aborts against EPIPE/ECONNRESET",
            "Chat, renderer, and image request paths guard EPIPE/ECONNRESET instead of surfacing raw write EPIPE or failed-message console errors to the app",
            "Ollama-specific backend response streams guard response errors at the proxy boundary",
            "Auto chat Max Tokens omits per-request output caps so server startup defaults can apply on Chat Completions and Responses",
            "Chat Completions and Responses streaming usage preserve cached_tokens plus cache_detail through finish chunks",
            "server cache and DSV4 tool/parser surfaces stay named in the API surface contract",
            "panel request builders omit invalid persisted maxTokens instead of poisoning Chat Completions or Responses",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_api_surface_contract.py --out build/current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json",
        ],
        "artifacts": [
            "build/current-api-surface-contract-20260602-v1554-stream-cache-reuse-refresh.json",
            "build/current-api-surface-contract-20260529-single-model-transition-lock.json",
            "build/current-api-surface-contract-20260525-single-model-ollama-chat-deltas.json",
            "build/current-api-surface-contract-20260523-post-budget-edge.json",
            "build/current-api-surface-contract-20260525-single-model-auto-switch-audit.json",
            "build/current-api-surface-contract-20260524-openai-single-model-streaming-audit.json",
            "build/current-api-surface-contract-20260524-single-model-auto-switch.json",
            "build/current-api-surface-contract-20260522-stream-cache-detail.json",
            "build/current-api-surface-contract-20260522-recheck-endpoint-assembly.json",
        ],
    },
    {
        "id": "cache-architecture-family-classification",
        "domain": "cache_architecture",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Plain KV, hybrid SSM/Mamba, MLA, DSV4 composite, ZAYA CCA, and MLLM media-salt cache contracts stay classified per family",
            "DSV4 SWA/HCA/CSA native composite cache components stay explicit in the cache-family matrix",
            "Generic TurboQuant KV is not applied to DSV4 native composite or hybrid SSM paths",
            "Bundled JANG DSV4 pool quant codec appends only newly generated CSA/HCA pool rows instead of requantizing the whole accumulated pool",
            "DSV4 pool quant reads reuse a materialized pool view instead of dequantizing and concatenating historical CSA/HCA pool segments on every read",
            "DSV4 short prompts skip synchronous composite prompt snapshots and do not replace that saved time with a synchronous prompt-only re-prefill store",
            "DSV4 panel env mapping enables pool quant by default only under DSV4 native composite cache",
            "DSV4 timing probe covers prefix-cache replay and cold-store boundaries before speed/cache root-cause claims",
            "Gemma4 mixed-SWA cache stays on typed RotatingKV/KV paths and is not mislabeled as SSM",
            "Named-family registry rows pin Ling/Bailing, Nemotron-H, Qwen3.6, ZAYA-VL, Hy3, and MiniMax cache/parser/reasoning boundaries",
            "Cache detail telemetry reports paged, typed native, and TQ/L2 state",
            "Panel session launch builder preserves DSV4 default-on native prefix-cache policy, DSV4-only native cache controls, Qwen3.6 hybrid and Mamba paged-cache forcing, and regular KV stale saved false semantics",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_cache_architecture_contract.py --out build/current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json",
        ],
        "artifacts": [
            "build/current-cache-architecture-contract-after-noheavy-contract-refresh-20260608.json",
            "build/current-cache-architecture-contract-20260530-lfm2-tool-parser-local.json",
            "build/current-cache-architecture-contract-20260528-gemma4-mixed-swa-row.json",
            "build/current-cache-architecture-contract-20260527-cache-family-matrix.json",
            "build/current-cache-architecture-contract-20260524-openai-single-model-streaming-audit.json",
            "build/current-cache-architecture-contract-20260523-post-budget-edge.json",
            "build/current-cache-architecture-contract-20260521.json",
            "build/current-cache-architecture-contract-20260522-panel-cache-launch.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-quant-append.json",
            "build/current-cache-architecture-contract-20260522-dsv4-timing.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-env-gate.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-materialized-cache.json",
            "build/current-cache-architecture-contract-20260522-dsv4-pool-ui-wired.json",
            "build/current-cache-architecture-contract-20260522-dsv4-short-snapshot-threshold.json",
            "build/current-dsv4-prefix-enabled-two-turn-threshold-live-20260522.json",
        ],
    },
    {
        "id": "model-artifact-format-detection",
        "domain": "model_artifact_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "JANG, JANGTQ/MXTQ, Ling/Bailing hybrid loader repairs, plain MLX 4bit, MXFP4, MXFP8, dropped-MTP, and preserved-MTP artifacts are not inferred from names alone",
            "generic affine JANG loader accepts weight_format=affine bundles without misclassifying them as JANGTQ/MXFP/plain MLX",
            "MXFP4/MXFP8 VLM loader paths quantize with the declared quantization mode instead of falling back to affine or JANGTQ handling",
            "JANGTQ_K mixed routed bits use the down-projection bit width for gather_dn",
            "Ling/Bailing flat 2D switch_mlp repair and correct 3D no-op stay covered",
            "Qwen plain MLX 4bit stays distinct from JANG, JANGTQ/MXTQ, MXFP4, and MXFP8",
            "native-MTP detection uses real indexed weights rather than path names",
            "family-specific loader wrappers for JANGTQ text, JANGTQ VLM, Kimi VLM, DSV4, ZAYA, Laguna, and Mistral stay source-hashed in the artifact gate",
            "Registry/family detection uses bundle config and capability metadata",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_artifact_format_contract.py --out build/current-model-artifact-format-contract-after-mllm-tight-memory-guard-20260607.json",
        ],
        "artifacts": [
            "build/current-model-artifact-format-contract-after-mllm-tight-memory-guard-20260607.json",
            "build/current-model-artifact-format-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-model-artifact-format-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-model-artifact-format-contract-20260523-post-budget-edge.json",
            "build/current-model-artifact-format-contract-20260522-recheck-jang-mxfp-mtp-loaders.json",
            "build/current-model-artifact-format-contract-20260522-affine-jang-loader.json",
            "build/current-model-artifact-format-contract-20260522-mxfp-vlm-loader.json",
            "build/current-model-artifact-format-contract-20260522-loader-wrapper-hashes.json",
        ],
    },
    {
        "id": "jang-model-compat-runtime-boundary",
        "domain": "model_artifact_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "JANG/JANGTQ runtime compatibility patches needed by PR155 stay covered by a direct release proof",
            "The JANG loader runtime boundary remains source-hashed and cannot be dropped while the broad suite still happens to pass",
            "JANG model compatibility is promoted into the current proof sweep instead of relying on an incidental current-suite step",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_jang_model_compat_contract.py --out build/current-jang-model-compat-contract-20260528-pr155-runtime-boundary.json",
        ],
        "artifacts": [
            "build/current-jang-model-compat-contract-20260528-pr155-runtime-boundary.json",
        ],
    },
    {
        "id": "model-family-detection-noheavy",
        "domain": "model_family_detection",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Named DSV4, ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron, Qwen 3.6 VL/video/hybrid, plain MLX 4bit, MXFP4, MXFP8, native-MTP, MiniMax, and Hy3 rows keep expected parser/cache/modality policy",
            "Qwen dense linear-attention wrappers stay hybrid cache through the engine registry, not panel-only matching",
            "Qwen MoE text linear-attention wrappers stay hybrid cache through the engine registry, not panel-only matching",
            "Qwen 3.6 release rows intentionally keep qwen3_5/qwen3.5 family aliases so VL/video/hybrid, MXFP, JANG, JANGTQ, and parser policy stay aligned",
            "base Nemotron-H registry rows stay hybrid cache before stale Omni sidecar overrides are considered",
            "stale ZAYA converter stamps cannot disable reasoning, swap the qwen3 parser, or reenable think_in_template",
            "ZAYA1-VL JANGTQ_K/JANGTQ2/JANGTQ4 plain-template no-reasoning capability truth preserves VL and typed CCA detection",
            "Hy3 JANGTQ_K Low/High reasoning contract stays on Hunyuan tools with qwen3 reasoning",
            "affine-JANG Qwen native-MTP VL/video artifacts stay multimodal when indexed MTP and vision tensors exist",
            "Decode-speed rows keep JANG-only, JANGTQ/MXTQ, plain MLX 4bit, MXFP4, and MXFP8 speed thresholds distinct while staying aligned with engine registry parser, modality, and cache metadata for existing local models",
            "JANG-only MX matmul rows stay text-only launch rows without --is-mllm, startup --max-tokens, MXFP, JANGTQ, or forced JANGTQ environment knobs",
            "Decode-speed rows keep DSV4 native composite separate from generic JANGTQ/MXTQ rows",
            "Decode-speed health checks reject DSV4 native or ZAYA typed cache health for plain KV JANG/JANGTQ/MXFP rows",
            "Every decode-speed row with a declared tool or reasoning parser uses a registered engine parser and CLI-accepted parser choice even when that local model path is absent",
            "Decode-speed launch commands preserve row parser/modality policy and strip source-path or forced JANGTQ acceleration environment overrides",
            "Panel session launch builder preserves MiniMax minimax_m2 parser launch, Qwen3.6 hybrid cache forces paged cache, ZAYA qwen3 reasoning parser and model-owned no-thinking defaults, DSV4 stale cache/additionalArgs suppression, and native-MTP D3 launch policy",
            "engine, panel, and session launch wiring all run in the same family launch-policy matrix",
            "ZAYA, ZAYA1-VL, Ling/Bailing, Nemotron-H, Qwen 3.6, MiniMax, Hy3, and DSV4 rows keep aligned parser/cache/modality policy",
            "session launch strips stale DSV4 additionalArgs and preserves native-MTP D3 only where supported",
            "Decode-speed matrix includes large external Mistral JANGTQ, Mistral MXFP4, and GPT-OSS rows with parser/modality launch policy pinned",
            "Decode-speed matrix includes external Nemotron 3 JANGTQ2 and MXFP4 rows with Nemotron parser/reasoning launch policy pinned",
            "Existing local high-risk DSV4, Qwen JANG/JANGTQ/MXFP/4bit/MTP, Hy3, and Nemotron Omni/Nano JANGTQ/MXFP rows match the current engine registry parser, cache, and modality policy without loading weights",
            "Panel detection matches the same current local high-risk paths, including native-MTP Qwen affine-JANG VL routing, so UI launch policy does not diverge from engine/API routing",
            "This is source/static compatibility proof only; live multi-turn output quality remains a separate live row",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_model_family_detection_contract.py --out build/current-model-family-detection-contract-after-n2-policy-row-20260609.json",
        ],
        "artifacts": [
            "build/current-model-family-detection-contract-after-n2-policy-row-20260609.json",
            "build/current-model-family-detection-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-model-family-detection-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-model-family-detection-contract-20260523-post-budget-edge.json",
            "build/current-model-family-detection-contract-20260522-recheck-launch-policy-matrix.json",
            "build/current-model-family-detection-contract-20260522-jang-only-mx-matmul-policy.json",
            "build/current-model-family-detection-contract-20260522-qwen-nemotron-hybrid-cache.json",
            "build/current-model-family-detection-contract-20260522-zaya-stale-stamp.json",
            "build/current-model-family-detection-contract-20260522-plain-kv-cache-health.json",
            "build/current-model-family-detection-contract-20260522-panel-launch-wiring.json",
            "build/current-model-family-detection-contract-20260522-zaya-hy3-qwen-vl-profile-rows.json",
            "build/current-model-family-detection-contract-20260522-local-artifact-registry.json",
            "build/current-model-family-detection-contract-20260522-panel-local-paths.json",
            "build/current-model-family-detection-contract-20260522-qwen36-alias.json",
        ],
    },
    {
        "id": "native-mtp-d3-effect-policy",
        "domain": "native_mtp",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Native MTP D3 launch policy is wired where supported",
            "Native MTP controls are hidden/suppressed for DSV4 and unsupported families",
            "DSV4 stale additionalArgs cannot reenable Native MTP, deterministic MTP sampling, hidden default sampling, or startup max-token overrides",
            "Config-only MTP bundles without indexed mtp.* tensors do not activate native MTP or expose Native MTP launch controls",
            "Runtime-active MTP still requires live equivalence/speed rows before release claims",
            "Qwen native-MTP live decode speed and output equivalence are proven by same-artifact AR-vs-MTP A/B",
            "Qwen prompt-processing floor remains a separate performance row; deterministic MTP decode alone is not enough",
            "The same-artifact AR-vs-MTP live A/B proves MTP decode is not the MLLM prefill bottleneck; native MTP D3 is faster than the AR baseline with identical output",
            "Qwen3.6/Qwen3.5 VLM native-MTP GatedDeltaNet wrappers accept and consume gdn_sink instead of leaking it to upstream originals",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_native_mtp_contract.py --out build/current-native-mtp-contract-after-noheavy-contract-refresh-20260608.json",
            "VMLINUX_BENCH_PYTHON=/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 .venv/bin/python bench/native_mtp_speed_ab.py /Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP --served-name qwen27-jang4m-mtp-ab --port 8814 --cache off --max-num-seqs 1 --max-tokens 320 --repeats 1 --warmup 0 --load-timeout-s 600 --out build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606 --disable-prompt-reuse",
        ],
        "artifacts": [
            "build/current-native-mtp-contract-after-noheavy-contract-refresh-20260608.json",
            "build/current-qwen36-mtp-gdn-sink-source-proof-20260606.json",
            "build/current-qwen35-dense-mtp-gdn-sink-fix-20260606.json",
            "build/current-native-mtp-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-native-mtp-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-native-mtp-contract-20260524-after-ling-topk-policy.json",
            "build/current-native-mtp-contract-20260523-post-budget-edge.json",
            "build/current-native-mtp-contract-20260522-dsv4-additional-args.json",
            "build/current-native-mtp-contract-20260522-config-only.json",
            "build/current-native-mtp-speed-ab-qwen27-jang4m-mtp-installed-app-20260606/result.json",
        ],
    },
    {
        "id": "mcp-policy-ui-gateway",
        "domain": "mcp",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "MCP autodiscovery, redaction, security policy, gateway routing, and panel config source stay covered",
            "MCP required marker checks fail the gate if named engine or panel rows disappear",
            "mcp.json/jsonc import keeps secrets out of managed metadata",
            "built-in Electron tools remain separate from MCP execution and request policy",
            "ambiguous multi-session MCP requests are rejected unless a model alias is explicit",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_mcp_policy_contract.py --out build/current-mcp-policy-contract-20260531-post-step-lfm-refresh.json",
        ],
        "artifacts": [
            "build/current-mcp-policy-contract-20260531-post-step-lfm-refresh.json",
            "build/current-mcp-policy-contract-20260523-post-budget-edge.json",
            "build/current-mcp-policy-contract-20260521.json",
            "build/current-mcp-policy-contract-20260522-recheck-ui-gateway.json",
        ],
    },
    {
        "id": "vl-media-cache-tool-followup",
        "domain": "vl_media",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "VLM media request serialization, media cache salting, and tool follow-up paths stay source-covered",
            "Panel family detection keeps ZAYA-VL, Qwen VL/video/hybrid/indexed-MTP, MXFP4/MXFP8 VLM, and Nemotron stale-Omni sidecar routing covered",
            "Engine family detection keeps affine-JANG Qwen VLM-looking artifacts on the text loader until the M-RoPE path is fixed",
            "Qwen3.6 VL JANG indexed-MTP artifacts stay multimodal only when indexed MTP and vision tensors exist",
            "MXTQ/JANGTQ and MXFP4/MXFP8 Qwen VLM rows preserve multimodal launch policy",
            "hybrid SSM companion cache, deferred rederive, and generic TQ suppression stay covered for VLM paths",
            "read_video and image/video tool-result follow-up build real multimodal content parts without leaking media bytes into plain tool text",
            "Still-image live rows do not imply video/audio/Omni clearance",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_vl_media_cache_contract.py --out build/current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json",
        ],
        "artifacts": [
            "build/current-vl-media-cache-contract-after-dsv4-preflight-refresh-20260608.json",
            "build/current-vl-media-cache-contract-20260531-post-step-lfm-refresh.json",
            "build/current-vl-media-cache-contract-20260527-after-think-xml-registry-fix.json",
            "build/current-vl-media-cache-contract-20260527-issues-175-178-bundled-sync.json",
            "build/current-vl-media-cache-contract-20260521.json",
            "build/current-vl-media-cache-contract-20260523-post-budget-edge.json",
            "build/current-vl-media-cache-contract-20260522-panel-family.json",
            "build/current-vl-media-cache-contract-20260522-recheck-qwen-zaya-media.json",
        ],
    },
    {
        "id": "packaged-release-integrity",
        "domain": "packaging_release",
        "mode": "packaged",
        "heavy": False,
        "proves": [
            "Version triples, bundled Python hash parity, packaged app signature, and objective proof digest gate release attempts",
            "Direct release-gate runs refresh and enforce the objective proof digest instead of relying only on the umbrella suite",
            "verify-bundled checks packaged Python imports, source hash parity, and relocatable console scripts",
            "packaged integrity uses a clean JANG source path for bundled jang_tools source-hash checks",
            "bundled critical jang_tools files match source content and console-script shebangs are relocatable",
            "dry release gate fails only on the known objective rows while version, typecheck, bundled import, and digest refresh steps pass",
            "packaged Python has no __pycache__/*.pyc files that would invalidate the signed app seal",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --out build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json",
            ".venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --jang-tools-source /Users/eric/jang/jang-tools --out build/current-packaged-integrity-contract-20260524-pycache-seal-check.json",
            ".venv/bin/python tests/cross_matrix/run_packaged_integrity_contract.py --jang-tools-source /Users/eric/jang/.worktrees/vmlx-release-clean-b5f66a7/jang-tools --out build/current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json",
        ],
        "artifacts": [
            "build/current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json",
            "build/current-packaged-integrity-contract-after-checkpoint-app-parity-20260609.json",
            "build/current-packaged-integrity-contract-20260601-dsv4-preflight-refresh.json",
            "build/current-packaged-integrity-contract-20260601-qwen3vl-minicpm-mpp-staged-refresh.json",
            "build/current-packaged-integrity-contract-20260531-after-adhoc-reseal.json",
            "build/current-packaged-integrity-contract-20260528-prepackage-gate.json",
            "build/current-packaged-integrity-contract-20260528-release-ready-dmg-gate.json",
            "build/current-packaged-integrity-contract-20260528-release-ready-gate.json",
            "build/current-packaged-integrity-contract-20260525-additional-args-guard.json",
            "build/current-packaged-integrity-contract-20260524-openai-single-model-streaming-audit.json",
            "build/current-packaged-integrity-contract-20260524-after-dsv4-rep1-live.json",
            "build/current-packaged-integrity-contract-20260524-gemma4-speed-open.json",
            "build/current-packaged-integrity-contract-20260524-gemma4-root-cause-open.json",
            "build/current-packaged-integrity-contract-20260524-live-smoke-open.json",
            "build/current-packaged-integrity-contract-20260524-gemma4-visible-open.json",
            "build/current-packaged-integrity-contract-20260524-gemma4-smoke-dsv4-open.json",
            "build/current-packaged-integrity-contract-20260524-crossfamily-cleared-dsv4-open.json",
            "build/current-packaged-integrity-contract-20260524-after-gemma4-mixed-swa-telemetry.json",
            "build/current-packaged-integrity-contract-20260524-after-dsv4-neutral-reppen.json",
            "build/current-packaged-integrity-contract-20260524-pycache-seal-check.json",
            "build/current-packaged-integrity-contract-20260524-text-additional-args-sanitizer.json",
            "build/current-packaged-integrity-contract-20260523-post-budget-edge-refreshed.json",
            "build/current-packaged-integrity-contract-20260522-recheck-bundled-release-gate.json",
            "build/current-packaged-integrity-contract-20260522-objective-gate-enforced.json",
        ],
    },
    {
        "id": "public-release-surface-preflight",
        "domain": "release_surface",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "local latest.json updater manifest does not advance ahead of the source version",
            "source version consistent across pyproject.toml and panel/package.json",
            "if latest.json is bumped to the source version, it carries a matching URL, SHA256, and notes version",
            "complete post-release updater state is accepted while incomplete bumped latest.json state is rejected",
            "after release, latest.json may equal the source version only when the published updater state is complete",
            "PyPI, GitHub release, and updater feeds remain explicit operator checks before public release",
            "live public release-surface mode checks raw GitHub latest.json, mlx.studio/update/latest.json, PyPI files, and GitHub release DMG asset digest after a release is cut",
            "live public release-surface mode requires mlx.studio/update/latest.json no-store/no-cache headers so browser and CDN cache cannot hide a newer release",
            "live public release-surface mode compares the source repo release tag to the current source head so green updater feeds cannot hide post-release source-only fixes",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260602-v154-live-public-after-site-fix.json",
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --out build/current-release-surface-contract-20260522-recheck-updater-i18n.json",
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --live-public --out build/current-release-surface-contract-20260522-live-public-v1548.json",
            ".venv/bin/python tests/cross_matrix/run_release_surface_contract.py --live-public --out build/current-release-surface-contract-20260524-live-source-tag-parity.json",
        ],
        "artifacts": [
            "build/current-release-surface-contract-20260602-v154-live-public-after-site-fix.json",
            "build/current-release-surface-contract-20260524-live-source-tag-parity.json",
            "build/current-release-surface-contract-20260523-post-budget-edge.json",
            "build/current-release-surface-contract-20260522-live-public-v1548.json",
            "build/current-release-surface-contract-20260522-post-release-updater.json",
            "build/current-release-surface-contract-20260522-recheck-updater-i18n.json",
            "build/current-release-surface-contract-20260521.json",
            "latest.json",
        ],
    },
    {
        "id": "pr-intake-credit-triage",
        "domain": "pr_intake",
        "mode": "noheavy",
        "heavy": False,
        "proves": [
            "Open PRs are reviewed for merge/adapt/reject decision",
            "Contributor credit is tracked even when an idea is adapted instead of merged",
        ],
        "commands": [
            "gh pr list --repo jjang-ai/vmlx --state open --limit 100 --json number,title,author,createdAt,updatedAt,isDraft,headRefName,baseRefName,mergeable,reviewDecision,url,labels,statusCheckRollup",
        ],
        "artifacts": [
            "docs/internal/pr-triage/2026-05-21-open-pr-intake.md",
            "build/current-gh-prs-vmlx-open-20260521.json",
        ],
    },
    {
        "id": "dsv4-long-output-quality-live",
        "domain": "tool_calls",
        "mode": "live",
        "heavy": True,
        "proves": [
            "DSV4 long-output/code/file-generation exactness and identifier integrity pass on a real model server",
            "Current explicit thinking-off still fails exact code across chat, Responses, and legacy completions with hidden reasoning closed",
            "Current generated-only/direct rail still fails exact code across chat, Responses, and legacy completions with WebWebGLRenderer under repetition_penalty=1.0",
            "Current explicit repetition_penalty=1.0 live route controls still fail direct/off while requested-thinking Chat and Responses pass exact code",
            "Current requested-thinking rail passes the same exact-code subset, proving the boundary without silently forcing hidden thinking on",
            "Current chat_max remains open: latest source/live route proof is 9/10 exact, with chat_max still corrupting identifiers",
            "Current live matrix shows budget, stops, requested thinking, Responses, and completion route are not sufficient fixes",
            "Current prompt-boundary bisection shows the canonical no-markdown-fences colon wording and no-punctuation variant pass, while nearby natural variants still fail under the same effective rail",
            "Current colon-vs-period logprob trace shows the passing and failing prompts have the same token counts and no cache hits while diverging at visible output",
            "Current scene token rank contrast shows deterministic decoding follows the top-ranked token: colon ranks correct ene first, while period ranks the wrong close-paren token first",
            "Current direct-vs-thinking WebGL logit probe shows WebWebGLRenderer is not explained by a Web-over-GL rank after THREE.Web, while direct/off still differs at THREE. and THREE.P",
            "Current hidden reasoning control shows hidden draft contamination is not the sole root cause: a diagnostic no-code-draft control still corrupts visible identifiers",
            "Current template parity diagnostic rules out a vMLX sidecar encoder versus bundle Jinja template mismatch for the colon/period exact-code split",
            "Current prefill execution variants keep the same isolated THREE.P logit ordering across plain, stream, warmup, and clear paths",
            "Current prompt-variant logit probe shows prompt wording changes identifier logits without changing production behavior",
            "Current reasoning-policy live probe shows explicit DSV4 thinking-off produces visible output without hidden reasoning, while requested thinking still opens the reasoning rail",
            "Current batch generator logit divergence evidence shows direct full/incremental logits rank ers first after THREE.P while DSV4BatchGenerator ranks ert first",
            "Current source-vMLX direct/off route still corrupts exact code as THREE.WebWebGLRenderer after the venv launcher and prefill-logits materialization checks",
            "Current source-vMLX requested-thinking route passes the same exact-code subset across Chat and Responses, so the remaining DSV4 blocker is direct/instruct rail visible-token reliability rather than parser split, stale source import, or hidden sampler forcing",
            "Current true-bundled JANGTQ-K direct/off recheck still fails Chat and Responses with THREE.WebWebGLRenderer under thinking disabled, prefix cache disabled, pool quant off, and repetition_penalty=1.0",
            "Current source full-output exactness is still missing because local memory preflight skips the 148GB source model on this 128GB host",
            "Current no-punctuation direct/off variant is queued under thinking_closed with repetition_penalty=1.0 and prefix cache disabled, but the live run is memory-preflight skipped on this host",
            "Current cohesive dry-run confirms the queued Chat and Responses direct/off cases all resolve to enable_thinking=False with a thinking_closed prompt tail before any heavy DSV4 launch",
            "This remains open until a source/rebuilt-body or equivalent live proof passes",
        ],
        "commands": [
            "panel/bundled-python/python/bin/python3 tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --out build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --python /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --port 8895 --timeout 420 --request-timeout 300 --cases chat_off_rep1,chat_on_rep1,responses_off_rep1,responses_on_rep1 --min-free-gb 100 --out build/current-dsv4-route-mode-code-exactness-source-thinking-ab-prefill-logits-eval-20260525.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --python panel/bundled-python/python/bin/python3 --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --port 8896 --timeout 420 --request-timeout 300 --cases chat_off_rep1,responses_off_rep1 --min-free-gb 100 --out build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --dry-run --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --out build/current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --dry-run --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --out build/current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --dry-run --cases chat_off_rep1,chat_off_no_punct_rep1,responses_off_rep1,responses_off_no_punct_rep1 --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --out build/current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --dry-run --cases chat_off_rep1,chat_off_no_punct_rep1,responses_off_rep1,responses_off_no_punct_rep1 --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --out build/current-dsv4-route-mode-code-exactness-direct-off-no-punct-dryrun-20260525.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --cases chat_off_rep1,chat_off_no_punct_rep1,responses_off_rep1,responses_off_no_punct_rep1 --model /Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K --min-free-gb 120 --out build/current-dsv4-route-mode-code-exactness-direct-off-no-punct-preflight-20260525.json",
            ".venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --allow-user-ram-override --min-free-gb 100 --cases chat_off_no_punct_rep1,responses_off,responses_on --request-timeout 240 --timeout 900 --out build/current-dsv4-route-mode-code-exactness-bundled-after-bundle-refresh-20260606.json",
        ],
        "artifacts": [
            "build/current-dsv4-route-mode-code-exactness-chat-off-user-ram-override-20260606.json",
            "build/current-dsv4-route-mode-code-exactness-chat-on-user-ram-override-20260606.json",
            "build/current-dsv4-route-mode-code-exactness-bundled-after-bundle-refresh-20260606.json",
            "build/current-dsv4-jangtq-k-route-mode-code-exactness-20260524.json",
            "build/current-dsv4-route-mode-code-exactness-source-rep1-prefill-logits-eval-20260525.json",
            "build/current-dsv4-route-mode-code-exactness-source-thinking-ab-prefill-logits-eval-20260525.json",
            "build/current-dsv4-route-mode-code-exactness-jangtqk-direct-off-recheck-20260525.json",
            "build/current-dsv4-route-mode-code-exactness-source-explicit-off-subset-20260524-1418.json",
            "build/current-dsv4-route-mode-code-exactness-current-generated-only-subset-20260524.json",
            "build/current-dsv4-route-mode-code-exactness-current-thinking-on-subset-20260524.json",
            "build/current-dsv4-route-mode-code-exactness-current-rep1-controls-20260524-2104.json",
            "build/current-dsv4-route-mode-code-exactness-source-after-completion-trim-budget-liveapi-20260524-1248.json",
            "build/current-dsv4-route-mode-code-exactness-existing-server-chatmax-20260524.json",
            "build/current-dsv4-chatmax-prompt-trigger-hypothesis-live-20260524-1258.json",
            "build/current-dsv4-chatmax-budget-stop-rail-live-20260524-1309.json",
            "build/current-dsv4-prompt-boundary-bisection-live-20260524-1317.json",
            "build/current-dsv4-colon-vs-period-logprob-trace-live-20260524-1320.json",
            "build/current-dsv4-colon-vs-period-visible-logprob-trace-live-20260524-1322.json",
            "build/current-dsv4-scene-token-rank-contrast-live-20260524-1324.json",
            "build/current-dsv4-direct-vs-thinking-webgl-logit-probe-20260524.json",
            "build/current-dsv4-hidden-reasoning-control-live-20260524-1335.json",
            "build/current-dsv4-template-parity-diagnostic-20260524-1343.json",
            "build/current-dsv4-jang-prefill-execution-variant-logits-20260524.json",
            "build/current-dsv4-jang-prompt-variant-logit-probe-20260524.json",
            "build/current-dsv4-reasoning-policy-live-20260524-1408.json",
            "build/current-dsv4-jang-cache-vs-full-logit-isolation-threejs-20260524.json",
            "build/current-dsv4-jang-batch-generator-isolated-identifier-logits-after-full-prefill-fix-20260524.json",
            "build/current-dsv4-jang-thinking-off-logit-probe-20260524.json",
            "build/current-dsv4-jang-live-api-copy-framing-canary-20260524.json",
            CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
            "build/current-dsv4-route-mode-code-exactness-dryrun-20260526-root-trace.json",
            "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-identifier-candidates.json",
            "build/current-dsv4-route-mode-code-exactness-dryrun-20260528-current-cohesive-audit.json",
            "build/current-dsv4-route-mode-code-exactness-direct-off-no-punct-dryrun-20260525.json",
            "build/current-dsv4-route-mode-code-exactness-direct-off-no-punct-preflight-20260525.json",
            "build/current-dsv4-long-output-quality-clearance-20260521.json",
        ],
    },
    {
        "id": "ling-bailing-multilingual-quality-live",
        "domain": "reasoning_template",
        "mode": "live",
        "heavy": True,
        "proves": [
            "Ling/Bailing multilingual output quality is release-cleared only when real Russian/non-CJK prompts do not leak CJK text",
            "Current clearance artifacts are zero-CJK after the MPP/top-k policy fixes, so the objective row can pass without hiding the older failures",
            "Older failing artifacts remain tracked as regression evidence so the issue cannot be misclassified as DSV4-only, cache-only, or JANGTQ-only",
            "The clearance boundary is based on live artifacts, not a hidden sampler/repetition/default override",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/summarize_objective_proof.py --out build/current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json",
            ".venv/bin/python tests/cross_matrix/run_production_family_audit.py --rows ling_flash_tq --live --py /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --out build/current-production-family-live-ling-bundled-current-20260606.json",
        ],
        "artifacts": [
            "build/current-production-family-live-ling-bundled-current-20260606.json",
            "build/current-objective-proof-after-n2-jangtq2-devapp-prevresp-consumed-20260610.json",
            "build/current-ling-jangtq-strict-russian-nocache-bundled-4850c9c2-20260524.json",
            "build/current-ling-mxfp4-crack-strict-russian-nocache-bundled-4850c9c2-20260524.json",
            "build/current-ling-jangtq-russian-prompt-variant-probe-20260524.json",
            "build/current-ling-jangtq-server-repeat-russian-source-prefill-stream-20260524.json",
            "build/current-ling-jangtq-server-repeat-russian-bundled-prefill-stream-20260524.json",
            "build/current-ling-jangtq-server-repeat-russian-bundled-native-prefill-stream-20260524.json",
            "build/current-production-family-live-ling-bundled-native-rerun-20260524.json",
            "build/current-ling-jangtq-cold-skipcache-repeat-bundled-native-20260524.json",
            "build/current-ling-jangtq-batchgen-temp0-repeat-bundled-native-20260524.json",
            "build/current-ling-jangtq-simple-engine-control-bundled-after-mpp-fix-20260524.json",
            "build/current-ling-jangtq-continuous-control-bundled-after-mpp-fix-20260524.json",
            "build/current-production-family-live-ling-bundled-after-mpp-fix-20260524.json",
            "build/current-production-family-live-ling-bundled-after-topk-policy-20260524.json",
        ],
    },
    {
        "id": "gemma4-crack-live-language-visible-quality",
        "domain": "reasoning_template",
        "mode": "live",
        "heavy": True,
        "proves": [
            "Gemma-4-26B-A4B JANG_4M CRACK needs live app/API quality probes before release claims, not only import or family-detection checks",
            "The 26B CRACK row is a Gemma4 MoE JANG_4M artifact with switch_mlp experts, 128 experts, top_k=8, 4.26-bit group-64 affine weights, and crack surgery on o_proj; it is not equivalent to the dense 31B JANG_4M MTP comparison row",
            "Bundled Python must import jang_tools, mlx_vlm.models.gemma4, and vmlx_engine before this row can be considered packaged-runnable",
            "Current app-real live Responses artifact shows visible English output with preserved identifiers; the older max_thinking_tokens=16 artifact remains diagnostic because the local Gemma4 template does not consume thinking_budget",
            "Future clearance must preserve cold/cache-hit behavior and real app params across chat and Responses, checking visible content, CJK/Korean/Chinese leakage, template rails, cache mode, affine/JANG matmul dispatch, Gemma4 switch_mlp remapping, and vision/text routing",
            "Gemma4 26B mixed-SWA UI/app-engine speed is not cleared unless bundled and installed/app-path artifacts prove mixed_swa_kv_v1 and required chat/Responses/cold/cache-hit samples meet the 80 tok/s decode floor under the metric being claimed",
            "Latest bundled 256-token cache-hit probe still reports sub-floor wall throughput and generic live TurboQuant KV under mixed_swa_kv_v1 health, so the speed blocker is not a short-output artifact",
            "After syncing bundled/app jang_loader, installed-app health now reports generic TurboQuant KV off with mixed_swa_kv_v1, but decode remains below the 80 tok/s floor",
            "source now proves native mixed-SWA cache telemetry as paged+mixed_swa and clears the sustained 512-token speed-floor prompt after the redundant full-prefix cache store fix",
            "Current installed-app speed clearance requires source-hash parity, paged+mixed_swa cache telemetry, stream UI TPS above 80 tok/s for cold and cache-hit rows (`decode_tok_s_stream`), and cold-wall TTFT tracking from build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json; cold wall decode includes TTFT and remains tracked separately",
            "compat MLX wheels remain a separate Sequoia-flavor speed/quality risk; staged Sequoia app uses macosx_14_0_arm64 MLX/Metal wheels and staged Tahoe app uses macosx_26_0_arm64 MLX/Metal wheels, while /Applications/vMLX.app runtime flavor is checked separately by the public app issue audit; installed Sequoia-compatible app now uses macosx_14_0_arm64 when that DMG is installed, and installed Tahoe-native app now uses macosx_26_0_arm64 when that DMG is installed; Gemma4 installed-app UI-speed proof does not supersede older sub-floor rows until the current issue115 speed-floor artifact exists and passes",
        ],
        "commands": [
            "panel/bundled-python/python/bin/python3 tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --route responses --enable-thinking --max-tokens 512 --skip-prefix-cache --expect-visible-content --out build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json",
            "VMLINUX_BENCH_ISOLATED=1 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --python /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python --port 8900 --timeout 600 --request-timeout 360 --prompt-tokens 700,700 --max-tokens 256 --route chat --disable-thinking --out build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
            "VMLINUX_BENCH_ISOLATED=1 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --python /Users/eric/mlx/vllm-mlx-finite-launch-guard/.venv/bin/python --port 8891 --timeout 600 --request-timeout 360 --prompt-tokens 700,700 --prompt-mode speed_floor --max-tokens 512 --route chat --disable-thinking --out build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-source-20260525.json",
            "VMLINUX_BENCH_ISOLATED=1 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --python /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --port 8908 --timeout 600 --request-timeout 360 --prompt-tokens 700,700 --prompt-mode speed_floor --max-tokens 512 --route chat --disable-thinking --out build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-20260525.json",
            "VMLINUX_BENCH_ISOLATED=1 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --python /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --port 8908 --timeout 600 --request-timeout 360 --prompt-tokens 700,700 --prompt-mode speed_floor --max-tokens 512 --route chat --disable-thinking --out build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
            "VMLINUX_MLLM_PREFILL_TRACE=1 VMLINUX_MLLM_SCHEDULER_TRACE=1 VMLINUX_BENCH_ISOLATED=1 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --python /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --port 8908 --timeout 600 --request-timeout 360 --prompt-tokens 700,700 --prompt-mode speed_floor --max-tokens 512 --route chat --disable-thinking --out build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
            "VMLINUX_MLLM_PREFILL_TRACE=1 VMLINUX_MLLM_SCHEDULER_TRACE=1 VMLINUX_DECODE_TRACE=1 VMLINUX_DECODE_TRACE_EVERY=64 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4_26b_jang4m --python /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --port 8920 --timeout 600 --request-timeout 420 --prompt-tokens 700,700,700 --prompt-mode speed_floor --max-tokens 256 --route chat --stream --disable-thinking --skip-prefix-cache --out build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json",
            "uv run --extra dev python tests/cross_matrix/run_production_family_audit.py --rows gemma4_crack --live --out build/current-production-family-live-gemma4-crack-parser-tool-config-latest-jang-20260524.json",
        ],
        "artifacts": [
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingon-app-visible-512-nocache-20260524.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-responses-thinkingbudget16-visible-contract-20260524.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-cachehit-256-source-skip-redundant-store-notrace-20260525.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-source-20260525.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-20260525.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-repeat-20260525.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-prompt-cachehit-512-installed-app-trace-20260525.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-installed-app-triple-nocache-256-streaming-20260525.json",
            "build/current-runtime-memory-stress-gemma4-26b-jang4m-chat-thinkingoff-speed-floor-issue115-installed-app-20260601.json",
            "build/current-production-family-live-gemma4-crack-parser-tool-config-latest-jang-20260524.json",
            "build/current-production-family-audit-gemma4-crack-live-bundled-a461e09c-20260524.json",
        ],
    },
    {
        "id": "model-family-live-multiturn-soak",
        "domain": "reasoning_template",
        "mode": "live",
        "heavy": True,
        "proves": [
            "Gemma4, Hy3, MiniMax, Qwen 3.6 hybrid, ZAYA, ZAYA1-VL, Ling, Nemotron, and DSV4 family rows need real multi-turn UI/API soaks before broad production claims",
            "The live workflow gate must cover tool calls, reasoning parser output separation, prefix cache reuse, paged/L2 cache telemetry, memory ceilings, and wrong-language/gibberish checks instead of only HTTP 200 responses",
            "Next bundled all-local smoke reruns must include the explicit tool_choice=required probe so every promoted family artifact can show a real tool_required request label instead of relying on parser/unit tests alone",
            "The same broad live gate must exercise Chat Completions, Responses, Anthropic, and Ollama surfaces, stop/disconnect handling, full visible output, full reasoning output, and tail review so loops or late corruption cannot hide behind short probes",
            "No fake fixes are allowed: live rows must use model-owned generation_config.json and jang_config.json defaults unless a request explicitly overrides them, not forced temperature/repetition-penalty rails",
            "API parity must include Chat Completions, Responses, Anthropic, and Ollama streaming/non-streaming paths plus stop/disconnect follow-up requests where the surface is supported",
            "Artifacts must save full visible output, full reasoning output, token counts, finish reasons, and tail review notes so clipped previews cannot pass output-quality rows",
            "All-local smoke proof revalidates visible text for unexpected CJK/Korean/Japanese/Chinese output so stale artifacts from before the semantic gate cannot clear wrong-language rows",
            "Qwen MTP/VL/video rows remain no-heavy family-detection covered until a production-family row exists, while Gemma4/ZAYA/Qwen multimodal rows require dedicated VL and video coverage before broad media claims",
            "ZAYA-VL MXFP4 bundled smoke is attempted and currently fails text cache exact ACK while image color and media isolation pass, so it remains diagnostic for the prompt/template sensitivity",
            "ZAYA-VL JANGTQ4 true-bundled media-sentinel smoke passes ACK/cache, tool calls, and VL colors, but the 2026-05-26 reasoning-enabled rerun is diagnostic-fail because reasoning_on produced reasoning-only empty visible output and the prompt-sensitive no-media control returned image",
            "ZAYA text MXFP4 tool probe remains diagnostic because it fails current exact ACK cache probes, while ZAYA-VL JANGTQ4 true-bundled tool probe passes tool_required with structured record_fact arguments, typed ZAYA CCA prefix-cache hits, and no visible-text leakage",
            "Nemotron Omni Nano failed prompt variants remain diagnostic: the original bundled smoke and explicit no-media rerun answer image-present for text-only attachment prompts before and after image turns",
            "Nemotron Omni Nano system-scoped no-media prompt is the covered path for this row; older prompt-sensitive failures stay tracked separately",
            "Nemotron Omni Nano system-scoped no-media bundled smoke passes ACK, prefix-cache repeat, recall, reasoning, image, video, and text-after-media isolation, so nemotron is covered for this smoke row while older prompt-sensitive failures remain tracked",
            "Nemotron Omni Nano bundled tool probe passes tool_required with structured record_fact arguments, reasoning separation, prefix-cache repeat, disk-backed cache hit, and no visible-text leakage; image/video remain future live work for this family",
            "Ling/Bailing bundled smoke was rerun on 2026-05-25 and passes exact ACK, prefix-cache repeat, disk-backed cache hit, multi-turn recall, and unexpected-CJK rejection gates under bundled app-engine params, so ling_bailing is covered for this text smoke row",
            "Ling/Bailing bundled tool probe passes tool_required with structured record_fact arguments and no visible-text leakage",
            "Gemma4 26B JANG_4M bundled smoke was rerun on 2026-05-25 and passes exact ACK, mixed-SWA prefix-cache repeat, multi-turn recall, reasoning-on final answer, image color, and text-after-image isolation without Korean/Chinese garbage output under bundled app-engine params, so gemma4 is covered for this smoke row while speed clearance remains tracked separately",
            "Gemma4 bundled tool probe passes tool_required with structured record_fact arguments, reasoning separation, mixed-SWA cache repeat, and no visible-text leakage; this is tool/language evidence only and does not clear the separate Gemma4 SWA speed row",
            "Qwen3.6 MXFP4 bundled smoke was rerun on 2026-05-25 and passes exact ACK, prefix-cache repeat, disk-backed cache hit, multi-turn recall, reasoning-on final answer, image, video, and text-after-media isolation under bundled app-engine params, so qwen36 is covered for this smoke row",
            "Qwen3.6 MXFP4 bundled tool probe passes tool_required with structured record_fact arguments, reasoning separation, prefix-cache repeat, disk-backed cache hit, and no visible-text leakage; image/video coverage remains supplied by the separate bundled smoke rerun",
            "Hy3 JANGTQ2 bundled smoke passes exact ACK, prefix-cache repeat, multi-turn recall, and reasoning_on visible final answer under bundled app-engine params, so hy3 is covered for this smoke row",
            "Hy3 bundled tool probe passes tool_required with structured record_fact arguments, reasoning separation, prefix-cache repeat, and no visible-text leakage",
            "MiniMax small JANGTQ bundled smoke was rerun on 2026-05-25 and passes exact ACK, prefix-cache repeat, multi-turn recall, and reasoning-on visible final answer under bundled app-engine params, so minimax is covered for this smoke row",
            "MiniMax bundled tool probe passes required record_fact after MiniMax-native scaffold and family-specific required-tool token budget; cache repeat, tool-result continuation, JSON, and exact-code rows also pass",
            "ZAYA text MXFP4 bundled tool probe remains a diagnostic failure and must not clear ZAYA text tool-use, while the refreshed ZAYA-VL JANGTQ4 true-bundled tool probe is covered separately by the media-sentinel artifact",
            "DSV4 JANGTQ-K bundled smoke passes exact ACK, native prefix-cache repeat, multi-turn recall, and reasoning-on visible final answer with native DSV4 composite cache flags; default-cache tool-loop exact-code quality remains a separate open gate",
            "No source-only test may claim output-coherence clearance for every family",
        ],
        "commands": [
            "uv run --extra dev python tests/cross_matrix/run_production_family_audit.py --rows gemma4_crack,hy3_preview_jangtq2,minimax_m27_tq_k,qwen36_moe_crack,zaya_jangtq2,zaya_vl_jangtq4,ling_flash_tq,nemotron_omni_tq2,dsv4_jang_local --live --out build/current-production-family-audit-live-multifamily-soak-20260522.json",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Gemma-4-26B-A4B-it-JANG_4M-CRACK,Hy3-preview-JANGTQ2,MiniMax-M2.7-Small-JANGTQ,Qwen3.6-27B-MXFP4-CRACK,Ling-2.6-flash-JANGTQ,Nemotron-Omni-Nano-JANGTQ-CRACK,DeepSeek-V4-Flash-JANGTQ-K --include-tools --load-timeout-s 900 --request-timeout-s 300 --out build/current-all-local-model-smoke-multifamily-tools-bundled-next",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only ZAYA1-8B-MXFP4 --max-models 1 --no-media --no-video --no-reasoning --port 8842 --out build/current-all-local-model-smoke-zaya-text-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only ZAYA1-VL-8B-MXFP4 --port 8843 --out build/current-all-local-model-smoke-zaya-vl-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only ZAYA1-VL-8B-JANGTQ4 --port 8844 --out build/current-all-local-model-smoke-zaya-vl-jangtq4-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only ZAYA1-VL-8B-JANGTQ4 --max-models 1 --no-reasoning --include-tools --port 8880 --load-timeout-s 420 --request-timeout-s 240 --out build/current-all-local-model-smoke-zaya-vl-jangtq4-true-bundled-toolprobe-media-sentinel-20260525",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Nemotron-Omni-Nano-JANGTQ-CRACK --port 8846 --out build/current-all-local-model-smoke-nemotron-omni-jangtq-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Nemotron-Omni-Nano-JANGTQ-CRACK --port 8848 --load-timeout-s 600 --request-timeout-s 240 --out build/current-all-local-model-smoke-nemotron-omni-jangtq-explicit-nomedia-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Nemotron-Omni-Nano-JANGTQ-CRACK --port 8848 --load-timeout-s 600 --request-timeout-s 240 --out build/current-all-local-model-smoke-nemotron-omni-jangtq-system-nomedia-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Nemotron-Omni-Nano-JANGTQ-CRACK --max-models 1 --port 8864 --load-timeout-s 600 --request-timeout-s 300 --out build/current-all-local-model-smoke-nemotron-omni-jangtq-video-bundled-20260526-rerun",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Ling-2.6-flash-JANGTQ --no-media --no-video --no-reasoning --port 8848 --out build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Ling-2.6-flash-JANGTQ --max-models 1 --no-media --no-video --no-reasoning --port 8860 --load-timeout-s 420 --request-timeout-s 240 --out build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260525-rerun",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Gemma-4-26B-A4B-it-JANG_4M-CRACK --max-models 1 --port 8852 --load-timeout-s 420 --request-timeout-s 240 --out build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Gemma-4-26B-A4B-it-JANG_4M-CRACK --max-models 1 --port 8865 --load-timeout-s 600 --request-timeout-s 300 --out build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-video-capfix-bundled-20260526",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Gemma-4-26B-A4B-it-JANG_4M-CRACK --max-models 1 --include-tools --port 8884 --load-timeout-s 600 --request-timeout-s 300 --out build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-toolprobe-currentmodality-20260526",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Qwen3.6-27B-MXFP4-CRACK --max-models 1 --port 8849 --load-timeout-s 420 --request-timeout-s 240 --out build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Qwen3.6-27B-MXFP4-CRACK --max-models 1 --port 8861 --load-timeout-s 420 --request-timeout-s 240 --out build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260525-rerun",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only Hy3-preview-JANGTQ2 --max-models 1 --port 8850 --load-timeout-s 420 --request-timeout-s 240 --out build/current-all-local-model-smoke-hy3-jangtq2-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only MiniMax-M2.7-Small-JANGTQ --max-models 1 --no-media --no-video --port 8850 --load-timeout-s 600 --request-timeout-s 240 --out build/current-all-local-model-smoke-minimax-small-jangtq-bundled-20260524",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only MiniMax-M2.7-Small-JANGTQ --max-models 1 --no-media --no-video --port 8862 --load-timeout-s 600 --request-timeout-s 240 --out build/current-all-local-model-smoke-minimax-small-jangtq-bundled-20260525-rerun",
            "VMLINUX_BENCH_ISOLATED=1 VMLINUX_BENCH_PYTHON=panel/bundled-python/python/bin/python3.12 .venv/bin/python bench/all_local_model_smoke.py --models-root /Users/eric/models --only DeepSeek-V4-Flash-JANGTQ-K --max-models 1 --no-media --no-video --port 8853 --load-timeout-s 900 --request-timeout-s 300 --out build/current-all-local-model-smoke-dsv4-jangtq-k-tools-cache-20260606",
        ],
        "artifacts": [
            "build/current-production-family-audit-live-multifamily-soak-20260522.json",
            "build/current-filtered-live-smoke-zaya-text-mxfp4-20260607/summary.json",
            "build/current-all-local-model-smoke-zaya-vl-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-zaya-vl-jangtq4-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-zaya-vl-mxfp4-bundled-after-source-tool-pass-20260607/summary.json",
            "build/current-all-local-model-smoke-nemotron-omni-jangtq-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-nemotron-omni-jangtq-explicit-nomedia-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-nemotron-omni-jangtq-system-nomedia-bundled-20260524/summary.json",
            "build/current-filtered-live-smoke-nemotron-omni-jangtq-20260607/summary.json",
            "build/current-all-local-model-smoke-ling-hy3-nemotron-tools-media-20260606/summary.json",
            "build/current-nemotron-omni-no-media-prompt-variants-20260524/result.json",
            "build/current-nemotron-omni-no-media-system-prompt-diagnostic-20260524/result.json",
            "build/current-all-local-model-smoke-ling-bailing-jangtq-bundled-20260524/summary.json",
            "build/current-filtered-live-smoke-ling-flash-jangtq-20260607/summary.json",
            "build/current-all-local-model-smoke-gemma4-26b-jang4m-crack-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-gemma26-jang4m-bundled-tools-media-20260607/summary.json",
            "build/current-all-local-model-smoke-qwen36-mxfp4-crack-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-qwen36-27b-jang4m-mtp-bundled-tools-media-20260607/summary.json",
            "build/current-all-local-model-smoke-hy3-jangtq2-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-minimax-small-jangtq-bundled-20260524/summary.json",
            "build/current-all-local-model-smoke-minimaxk-bundled-after-required-tool-256-20260607/summary.json",
            "build/current-filtered-live-smoke-dsv4-jangtq-k-20260607/summary.json",
        ],
    },
    {
        "id": "electron-dev-ui-tools-reasoning-wiring",
        "domain": "ui_panel",
        "mode": "dev-ui",
        "heavy": False,
        "proves": [
            "Electron dev UI launches from the current panel source with preload window.api available",
            "Chat settings inherit tool settings without copying stale sampler, output-cap, stop-sequence, system-prompt, wire API, or reasoning overrides into a new chat",
            "Model-owned generation_config defaults are surfaced through the app instead of hidden forced temperature or repetition-penalty rails",
            "Responses streaming through the panel preserves reasoning segments, built-in tool status phases, run_command/list_directory/read_image/read_video execution, and final visible content without raw think/tool tag leakage",
            "Image and video tool outputs are carried into the follow-up Responses request as multimodal content parts",
            "Server cache UI keeps Prefix Cache, Paged KV Cache, Block Disk Cache L2, legacy Disk Cache, and Stored Cache Quantization controls visible and mutually wired",
            "This is mock-server UI wiring proof only; it does not clear live-model language quality, cache reuse, speed, VL/video semantics, or DSV4 exact-code quality",
        ],
        "commands": [
            "VMLX_LIVE_PROOF_BASENAME=2026-05-31-live-chat-tools-reasoning node panel/scripts/live-chat-tools-reasoning-proof.mjs",
        ],
        "artifacts": [
            "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-proof.json",
            "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-chat-settings.png",
            "docs/internal/agent-notes/2026-05-31-live-chat-tools-reasoning-server-cache-settings.png",
        ],
    },
    {
        "id": "electron-dev-ui-real-model-live-slice",
        "domain": "ui_panel",
        "mode": "dev-ui-live-model",
        "heavy": True,
        "proves": [
            "Electron dev UI can drive a real vmlx_engine server through window.api.chat.sendMessage instead of the mock Responses server",
            "The first live slice uses the small local ZAYA1-8B-MXFP4 text model so the real UI path can be proven without loading every large family at once",
            "ZAYA1-8B-MXFP4 now has a separate real Electron UI server-cache-controls slice proving Prefix Cache, Paged KV Cache, Block Disk Cache (L2), legacy Disk Cache, and Stored Cache Quantization controls are visible and mutually wired in the dev UI",
            "ZAYA1-8B-MXFP4 now has a real Electron UI Responses/tools slice proving explicit available-tool prompting, eight tool-loop iterations, persisted tool calls, Responses API completion, cache reuse, and no parser/language leak",
            "ZAYA1-VL-8B-JANGTQ4 now has a real Electron UI image slice proving persisted image attachment delivery, VLM image processing, semantic red-image recognition, zero parser/language leak, and no text-only prefix-cache store for media context",
            "Ling-2.6-flash-JANGTQ now has a real Electron UI live-model slice covering the prior Ling/Bailing language/cache/parser boundary with zero CJK/Korean leakage and hybrid SSM cache telemetry",
            "Gemma-4-26B-A4B-it-JANG_4M-CRACK now has a real Electron UI live-model slice covering mixed-SWA paged cache telemetry, affine MLX matmul dispatch, zero CJK/Korean leakage, and app-visible generation over the 80 tok/s speed floor for the short UI turn",
            "Gemma-4-26B-A4B-it-JANG_4M-CRACK also has a real Electron UI Responses/tools/reasoning diagnostic proof covering real Responses complete events, persisted built-in tool calls, and persisted reasoning display evidence",
            "Qwen3.6-27B-JANG_4M-MTP has a current real Electron UI Responses/cache-controls proof covering model identity, Responses streaming, cache-hit telemetry, hybrid SSM native cache status, TurboQuant attention KV storage, SSM companion L2, server cache controls, and no parser/language leak",
            "Qwen3.6-27B-JANG_4M-MTP remains blocked for real UI release because the current proof fails the long_tool_loop assertion after an initial malformed run_command tool call",
            "Qwen3.6-27B-JANG_4M-MTP image/video/reasoning UI coverage is not current; old Qwen3.6 MXFP4 proofs are no longer accepted as proof for the current MTP artifact",
            "MiniMax-M2.7-Small-JANGTQ has a real Electron UI live-model slice proving MiniMax parser registration, MiniMaxM2 reasoning-parser registration, JANGTQ native TurboQuant kernels, paged cache reuse, and no parser/language leak",
            "MiniMax-M2.7-Small-JANGTQ now has a local-only real Electron UI Responses/tools/cache-controls slice proving long built-in tool loops, reasoning display, native cache status, server cache controls, and no parser/language leak",
            "MiniMax-M2.7-JANGTQ_K has issue #179 real Electron UI diagnostics for both current dev UI and installed app UI, covering Responses API, reasoning display, persisted turns, cache-hit telemetry, and no screenshot-shaped CJK/Korean/numeric garbage on the local full K artifact",
            "MiniMax-M2.7-JANGTQ_K issue #179 also has a no-heavy root-cause audit preserving the reporter-installed-app abort-before-visible-content plus Responses cancel-404 boundary; the synced local installed app now has a controlled raw Responses cancel probe returning 200 after a real resp_* stream id with no screenshot-shaped bad text, but #179 stays open until reporter artifact/session/cancel-state evidence is matched",
            "MiniMax-M2.7-JANGTQ_K issue #179 now has a no-heavy reporter parity metadata collector so installed server hashes, model hashes, chat/session settings, response active-state, and raw SSE/cancel lifecycle can be captured without loading the model",
            "GitHub issues #175-#179 have a no-heavy release-boundary audit so focused source slices cannot be confused with live UI/app release clearance",
            "The installed /Applications/vMLX.app runtime parity audit is tracked separately so source/bundled checks cannot hide stale app CLI/parser/image LoRA wiring",
            "The installed-app runtime parity on issues #175-#177 proves packaged MLX memory-clear API selection, promoted paged-cache block mirror cleanup, and cache-selection telemetry are present in /Applications/vMLX.app",
            "The installed-app live runtime audit on issues #175-#177 records Qwen3.6 MTP cache-hit TTFT, paged+SSM L2 disk hits, visible streaming content, and memory telemetry as supporting evidence while keeping the live stress and cold paged/TurboQuant blockers open",
            "Nemotron-Omni-Nano-JANGTQ-CRACK has a real Electron UI no-media text slice proving the stage1 Omni-compatible dispatcher path, DeepSeekR1 reasoning parser registration, hybrid SSM companion cache reuse, JANGTQ native kernels, and no parser/language leak",
            "Nemotron-Omni-Nano-JANGTQ-CRACK has a separate real Electron UI Responses reasoning slice proving reasoning display with model-owned generation defaults and hybrid SSM cache reuse",
            "Hy3-preview-JANGTQ2 now has a real Electron UI live-model slice covering the Hunyuan tool parser, qwen3 reasoning-parser registration, TurboQuantKVCache/paged prefix cache, block-disk L2, zero CJK/Korean leakage, and no raw parser-tag leakage",
            "Hy3-preview-JANGTQ2 has separate real Electron UI Responses proofs for reasoning-on display and thinking-off built-in tool execution; thinking-on tool prompting currently produced reasoning-only output and remains a tracked nuance, not a forced sampler/parser fix",
            "Hy3-preview-JANGTQ2 now has a local-only real Electron UI Responses/tools/cache-controls proof covering native plain KV cache status, TurboQuant KV storage quantization, paged prefix cache reuse, block-disk L2, server cache controls, and no parser/language leak",
            "LFM2.5-8B-A1B-MXFP4 has a direct Responses/run_command cache-boundary probe showing the raw local server path is server-clean while the real UI stream path remains malformed, narrowing the remaining blocker to app tool-loop/history framing rather than LFM25 cache storage or raw Responses final-text generation",
            "The proof captures real server /health and /v1/models state, chat message persistence, final visible content, parser-leak checks, and cache telemetry after repeated UI turns",
            "These are partial real-UI live-model slices and are not enough to clear the full cross-family real-UI blocker",
        ],
        "commands": [
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-zaya-text-20260526 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-zaya-text-cachecontrols-20260527 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/ZAYA1-8B-MXFP4 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-zaya-text-responses-runcommand-tools-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=0 VMLINUX_REAL_UI_MAX_TOKENS=1024 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ4 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-zaya-vl-image-20260527 VMLINUX_REAL_UI_IS_MLLM=1 VMLINUX_REAL_UI_CHECK_MEDIA=1 VMLINUX_REAL_UI_IMAGE_EXPECT_REGEX='\\bred\\b' VMLINUX_REAL_UI_MAX_TOKENS=96 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-ling-bailing-jangtq-20260526 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ VMLINUX_REAL_UI_SERVED_MODEL=Ling-2.6-flash-JANGTQ VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=0 VMLINUX_REAL_UI_MAX_TOKENS=512 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-gemma4-26b-jang4m-20260526 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK VMLINUX_REAL_UI_PROOF_BASENAME=diagnostic-real-ui-live-model-gemma4-responses-tools-reasoning-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP VMLINUX_REAL_UI_SERVED_MODEL=Qwen3.6-27B-JANG_4M-MTP VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607 VMLINUX_REAL_UI_IS_MLLM=1 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=0 VMLINUX_REAL_UI_MAX_TOKENS=256 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-minimax-m27-small-jangtq-20260527 VMLINUX_REAL_UI_MAX_TOKENS=96 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ VMLINUX_REAL_UI_SERVED_MODEL=MiniMax-M2.7-Small-JANGTQ VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=1 VMLINUX_REAL_UI_MAX_TOKENS=512 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K VMLINUX_REAL_UI_PROOF_BASENAME=diagnostic-real-ui-live-model-minimax-m27-jangtq-k-issue179-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_MAX_TOKENS=512 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_APP_PATH=/Applications/vMLX.app VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K VMLINUX_REAL_UI_PROOF_BASENAME=diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-auto-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_PROMPT_1='Hi' VMLINUX_REAL_UI_PROMPT_2='Hi' node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_APP_PATH=/Applications/vMLX.app VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K VMLINUX_REAL_UI_PROOF_BASENAME=diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-thinking-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_ENABLE_THINKING=1 VMLINUX_REAL_UI_PROMPT_1='Hi' VMLINUX_REAL_UI_PROMPT_2='Hi' node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_APP_PATH=/Applications/vMLX.app VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ_K VMLINUX_REAL_UI_PROOF_BASENAME=diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-512-20260527 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_MAX_TOKENS=512 node panel/scripts/live-real-ui-model-proof.mjs",
            ".venv/bin/python tests/cross_matrix/run_issue179_responses_cancel_probe.py --out build/current-issue179-minimax-k-responses-cancel-probe-after-mimo-dsv4-ledger-20260607.json --port 8897 --load-timeout 240 --request-timeout 180 --stream-seconds 8 --cancel-delay 0.25 --max-lines-after-cancel 20",
            ".venv/bin/python tests/cross_matrix/run_issue175_179_release_boundary_audit.py --out build/current-issue175-179-release-boundary-audit-after-issue179-memory-preflight-20260607.json",
            ".venv/bin/python tests/cross_matrix/run_installed_app_runtime_parity_audit.py --app panel/release/sequoia-app/mac-arm64/vMLX.app --out build/current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json",
            ".venv/bin/python tests/cross_matrix/run_installed_app_runtime_parity_audit.py --app panel/release/tahoe-app/mac-arm64/vMLX.app --out build/current-installed-app-runtime-parity-audit-tahoe-checkpoint-dmg-20260609.json",
            ".venv/bin/python tests/cross_matrix/run_issue175_177_installed_runtime_audit.py --out build/current-issue175-177-installed-runtime-audit-20260602-v1554-installed-tahoe.json",
            ".venv/bin/python tests/cross_matrix/run_issue175_177_live_runtime_audit.py --out build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json",
            ".venv/bin/python tests/cross_matrix/run_issue179_minimax_k_root_cause_audit.py --out build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json",
            ".venv/bin/python tests/cross_matrix/run_issue179_reporter_parity_metadata.py --out build/current-issue179-reporter-parity-metadata-template-20260528.json",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527 VMLINUX_REAL_UI_MAX_TOKENS=96 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_APP_PATH=/Applications/vMLX.app VMLINUX_REAL_UI_PYTHON=/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_SERVED_MODEL=Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=1 VMLINUX_REAL_UI_MAX_TOKENS=512 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_SERVED_MODEL=Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_PROOF_BASENAME=diagnostic-real-ui-live-model-nemotron-omni-nano-responses-reasoning-20260531 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_ENABLE_THINKING=1 VMLINUX_REAL_UI_MAX_TOKENS=512 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_SERVED_MODEL=Nemotron-Omni-Nano-JANGTQ-CRACK VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=0 VMLINUX_REAL_UI_MAX_TOKENS=512 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-hy3-jangtq2-20260526 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_SERVED_MODEL=Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_ENABLE_THINKING=1 VMLINUX_REAL_UI_MAX_TOKENS=192 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-hy3-jangtq2-responses-reasoning-20260527 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_SERVED_MODEL=Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=0 VMLINUX_REAL_UI_MAX_TOKENS=384 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-hy3-jangtq2-responses-tools-thinkingoff-toolforced-20260527 node panel/scripts/live-real-ui-model-proof.mjs",
            "VMLINUX_REAL_UI_MODEL_PATH=/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_SERVED_MODEL=Hy3-preview-JANGTQ2 VMLINUX_REAL_UI_WIRE_API=responses VMLINUX_REAL_UI_BUILTIN_TOOLS=1 VMLINUX_REAL_UI_ENABLE_THINKING=0 VMLINUX_REAL_UI_MAX_TOKENS=512 VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8 VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1 VMLINUX_REAL_UI_PROOF_BASENAME=current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530 node panel/scripts/live-real-ui-model-proof.mjs",
        ],
        "artifacts": [
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-20260526-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-cachecontrols-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-runcommand-tools-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-text-responses-runcommand-tools-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-zaya-vl-image-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-20260526-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-ling-bailing-jangtq-responses-filesemantic-20260530-chat.png",
            "build/current-lfm25-direct-responses-runcommand-cache-boundary-20260530.json",
            "docs/internal/agent-notes/current-real-ui-live-model-gemma4-26b-jang4m-20260526-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-gemma4-26b-jang4m-20260526-chat.png",
            "docs/internal/agent-notes/diagnostic-real-ui-live-model-gemma4-responses-tools-reasoning-20260527-proof.json",
            "docs/internal/agent-notes/diagnostic-real-ui-live-model-gemma4-responses-tools-reasoning-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-qwen36-27b-jang4m-mtp-responses-tools-cachecontrols-deterministic-mtp-20260607-chat.png",
            "build/media-fixtures/red-64x64-1s.mp4",
            "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-jangtq-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-jangtq-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-minimax-m27-small-responses-tools-cachecontrols-localonly-20260527-chat.png",
            "docs/internal/agent-notes/diagnostic-real-ui-live-model-minimax-m27-jangtq-k-issue179-20260527-proof.json",
            "docs/internal/agent-notes/diagnostic-real-ui-live-model-minimax-m27-jangtq-k-issue179-20260527-chat.png",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-20260527-proof.json",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-20260527-chat.png",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-auto-20260527-proof.json",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-auto-20260527-chat.png",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-thinking-20260527-proof.json",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-hi-thinking-20260527-chat.png",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-512-20260527-proof.json",
            "docs/internal/agent-notes/diagnostic-installed-ui-live-model-minimax-m27-jangtq-k-issue179-512-20260527-chat.png",
            "build/current-issue179-minimax-k-responses-cancel-probe-after-mimo-dsv4-ledger-20260607.json",
            "build/current-issue175-179-release-boundary-audit-after-issue179-memory-preflight-20260607.json",
            "build/current-installed-app-runtime-parity-audit-sequoia-checkpoint-dmg-20260609.json",
            "build/current-installed-app-runtime-parity-audit-tahoe-checkpoint-dmg-20260609.json",
            "build/current-installed-app-runtime-parity-audit-20260528-userdata-epipe-scan.json",
            "build/current-issue175-177-installed-runtime-audit-20260602-v1554-installed-tahoe.json",
            "build/current-issue175-177-live-runtime-audit-20260601-local-refresh.json",
            "build/current-issue179-minimax-k-root-cause-audit-after-parser-settings-parity-20260608.json",
            "build/current-issue179-reporter-parity-metadata-template-20260528.json",
            "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-jangtq-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601-proof.json",
            "docs/internal/agent-notes/current-real-ui-installed-app-nemotron-omni-nano-responses-tools-reasoning-cachecontrols-localonly-20260601-chat.png",
            "docs/internal/agent-notes/diagnostic-real-ui-live-model-nemotron-omni-nano-responses-reasoning-20260531-proof.json",
            "docs/internal/agent-notes/diagnostic-real-ui-live-model-nemotron-omni-nano-responses-reasoning-20260531-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-nemotron-omni-nano-responses-stricttools-cachecontrols-exact-finalizer-20260531-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-20260526-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-reasoning-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-reasoning-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-thinkingoff-toolforced-20260527-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-thinkingoff-toolforced-20260527-chat.png",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530-proof.json",
            "docs/internal/agent-notes/current-real-ui-live-model-hy3-jangtq2-responses-tools-filesemantic-20260530-chat.png",
        ],
    },
    {
        "id": "qwen-jang-mx-live-speed-review",
        "domain": "performance",
        "mode": "live",
        "heavy": True,
        "proves": [
            "Qwen3.6-27B JANG_4M live decode speed is measured from the packaged Python engine, not inferred from source-only rows",
            "Qwen/JANG speed artifacts record the MLX and MLX-metal wheel tags so compat-vs-native wheel performance cannot be hidden",
            "selective live TurboQuant for Qwen attention KV layers is compatible only when native cache health proves SSM companion state remains full precision",
            "Native-wheel qwen27_jang4m text-loader clears the PP floor, but native-MTP MLLM/VL prefill remains below the floor on both native and compat wheels",
            "Post norm-format loader fix live Qwen MTP/VLM default-cache row clears coherency, decode, MTP acceptance, hybrid TQ KV, and PP floor without forced sampler defaults",
            "Compat bundled MTP prefill is retained as the user-facing packaged regression diagnostic and must not be hidden behind source-only rows",
            "SingleBatchGenerator honors the explicit prefill keep-alloc CLI/env path so single-sequence JANG runs can test allocator reuse without changing defaults",
            "Qwen 27B prompt-processing floor compares isolated native-MTP/VL prefill against the isolated text-loader baseline instead of blaming MTP decode",
            "MLLM prefill trace shows preprocessing, cache lookup, sampler call, cache submit, SSM capture, and cache merge are not the current PP bottleneck",
            "This row prevents broad JANG/MX matmul speed claims from hiding prompt-processing regressions",
        ],
        "commands": [
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --port 8790 --out build/current-decode-speed-live-qwen27-jang4m-20260522-hybrid-tq-review.json --timeout 420",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --python .venv/bin/python --port 8810 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-source-20260606.json",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --python /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --port 8812 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-installed-app-deterministic-pp-20260606.json",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --port 8799 --timeout 420 --out build/current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json --serve-extra-arg=--disable-prefix-cache",
            "VMLINUX_MLLM_PREFILL_TRACE=1 .venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --port 8798 --timeout 420 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json --serve-extra-arg=--disable-prefix-cache",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m --python .venv/bin/python --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json --serve-extra-arg=--disable-prefix-cache",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python .venv/bin/python --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python panel/bundled-python/python/bin/python3 --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json",
            "VMLINUX_MLLM_PREFILL_TRACE=1 .venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python .venv/bin/python --port 8796 --timeout 600 --prefill-step-size 2048 --out build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json",
            ".venv/bin/python tests/cross_matrix/run_decode_speed_gate.py --rows qwen27_jang4m_mtp --python /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3.12 --port 8813 --timeout 600 --prefill-step-size 2048 --serve-extra-arg=--native-mtp-sampling-policy --serve-extra-arg=deterministic-defaults --out build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-deterministic-pp-20260606.json",
        ],
        "artifacts": [
            "build/current-decode-speed-live-qwen27-jang4m-20260522-hybrid-tq-review.json",
            "build/current-decode-speed-live-qwen27-jang4m-source-20260606.json",
            "build/current-decode-speed-live-qwen27-jang4m-packaged-keepalloc-20260522.json",
            "build/current-decode-speed-live-qwen27-jang4m-installed-app-deterministic-pp-20260606.json",
            "build/current-decode-speed-live-qwen27-jang4m-text-baseline-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-source-bypass-fix-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace3-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-text-baseline-source-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-source-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill2048-cacheon-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-prefill-trace-isolated-20260523.json",
            "build/current-decode-speed-live-qwen27-jang4m-mtp-installed-app-deterministic-pp-20260606.json",
        ],
    },
]


def build_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "rows": deepcopy(_ROWS),
    }


def validate_current_proof_sweep_artifacts(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    not_pass: list[dict[str, str]] = []

    for artifact in CURRENT_POST_BUDGET_EDGE_ARTIFACTS.values():
        path = root / artifact
        if not path.exists():
            missing.append(artifact)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            status = str(payload.get("status"))
        except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
            status = f"load_error:{type(exc).__name__}"
        if status != "pass":
            not_pass.append({"artifact": artifact, "status": status})

    regression_suite = _validate_current_regression_suite_artifact(root)
    objective_digest = _validate_current_objective_digest_artifact(root)
    model_family_matrix = _validate_current_model_family_matrix_artifact(root)
    model_artifact_matrix = _validate_current_model_artifact_matrix_artifact(root)
    cache_architecture_matrix = _validate_current_cache_architecture_matrix_artifact(root)
    parser_registry_matrix = _validate_current_parser_registry_matrix_artifact(root)
    generation_defaults_matrix = _validate_current_generation_defaults_matrix_artifact(root)
    noheavy_api_cache_matrix = _validate_current_noheavy_api_cache_matrix_artifact(
        root
    )
    api_surface_matrix = _validate_current_api_surface_matrix_artifact(root)
    reasoning_template_matrix = _validate_current_reasoning_template_matrix_artifact(root)
    tool_call_matrix = _validate_current_tool_call_matrix_artifact(root)
    panel_tool_security_matrix = _validate_current_panel_tool_security_matrix_artifact(
        root
    )
    native_mtp_matrix = _validate_current_native_mtp_matrix_artifact(root)
    vl_media_matrix = _validate_current_vl_media_matrix_artifact(root)
    mcp_policy_matrix = _validate_current_mcp_policy_matrix_artifact(root)
    max_output_context_matrix = _validate_current_max_output_context_matrix_artifact(root)
    jang_model_compat_matrix = _validate_current_jang_model_compat_matrix_artifact(
        root
    )
    packaged_integrity_matrix = _validate_current_packaged_integrity_matrix_artifact(root)
    release_surface_matrix = _validate_current_release_surface_matrix_artifact(root)
    live_smoke_summaries = _validate_current_covered_live_smoke_artifacts(root)
    live_tool_smoke_summaries = _validate_current_covered_live_tool_smoke_artifacts(
        root
    )
    mimo_v2_jang2l_root_cause = _validate_current_mimo_v2_jang2l_root_cause(root)
    diagnostic_live_smoke_summaries = (
        _validate_current_diagnostic_live_smoke_artifacts(root)
    )
    dev_ui_proof = _validate_current_dev_ui_proof_artifacts(root)
    step37_vlm_runtime_audit = _validate_current_step37_vlm_runtime_audit(root)
    runtime_blocked_real_ui_families: set[str] = set()
    if (
        step37_vlm_runtime_audit.get("status") == "open"
        and step37_vlm_runtime_audit.get("root_cause")
        == "missing_mlx_vlm_step3p7_vlm_runtime"
    ):
        runtime_blocked_real_ui_families.add("step37")
    real_ui_live_model_proof = _validate_current_real_ui_live_model_proof_artifacts(
        root,
        skipped_missing_families={
            *DEFERRED_RELEASE_FAMILIES,
            *runtime_blocked_real_ui_families,
        },
    )
    issue175_179_release_boundary_audit = (
        _validate_current_issue175_179_release_boundary_audit(root)
    )
    installed_app_runtime_parity_audit = (
        _validate_current_installed_app_runtime_parity_audit(root)
    )
    staged_app_runtime_parity_audit = (
        _validate_current_staged_app_runtime_parity_audit(root)
    )
    issue175_177_installed_runtime_audit = (
        _validate_current_issue175_177_installed_runtime_audit(root)
    )
    issue175_177_live_runtime_audit = (
        _validate_current_issue175_177_live_runtime_audit(root)
    )
    issue179_minimax_k_root_cause_audit = (
        _validate_current_issue179_minimax_k_root_cause_audit(root)
    )
    issue179_minimax_k_live_probe_memory_preflight = (
        _validate_current_issue179_minimax_k_live_probe_memory_preflight(root)
    )
    issue181_183_runtime_audit = _validate_current_issue181_183_runtime_audit(root)
    public_app_issue_audit = _validate_current_public_app_issue_audit(root)
    real_ui_live_model_matrix = _validate_current_real_ui_live_model_matrix(
        real_ui_live_model_proof
    )
    real_ui_dsv4_memory_preflight = _validate_current_real_ui_dsv4_memory_preflight(
        root
    )
    dsv4_proof_artifact_freshness = validate_current_dsv4_proof_artifact_freshness(
        root,
        require_manifest_artifact=False,
    )
    if (
        "step37" in real_ui_live_model_matrix.get("missing_families", [])
        and step37_vlm_runtime_audit.get("status") == "open"
        and step37_vlm_runtime_audit.get("root_cause")
        == "missing_mlx_vlm_step3p7_vlm_runtime"
    ):
        runtime_blockers = dict(real_ui_live_model_matrix.get("runtime_blockers", {}))
        runtime_blockers["step37"] = {
            "artifact": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
            "reason": "missing_mlx_vlm_step3p7_vlm_runtime",
        }
        real_ui_live_model_matrix["runtime_blockers"] = runtime_blockers
    if "dsv4" in real_ui_live_model_matrix.get("missing_families", []):
        resource_blockers = dict(real_ui_live_model_matrix.get("resource_blockers", {}))
        dsv4_memory_preflight_blocks_launch = (
            real_ui_dsv4_memory_preflight.get("status")
            in {"pass", "skipped_insufficient_memory", "skipped"}
            and (
                real_ui_dsv4_memory_preflight.get("launch_decision")
                == "do_not_launch"
                or real_ui_dsv4_memory_preflight.get("did_not_launch") is True
                or real_ui_dsv4_memory_preflight.get("launch_allowed") is False
                or real_ui_dsv4_memory_preflight.get("reason")
                == "insufficient_vm_stat_memory"
                or "insufficient" in str(real_ui_dsv4_memory_preflight.get("reason") or "")
            )
        )
        if dsv4_memory_preflight_blocks_launch:
            resource_blockers["dsv4"] = {
                "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
                "reason": "insufficient_memory",
                "model_path": real_ui_dsv4_memory_preflight.get("model_path"),
                "model_size_gb": real_ui_dsv4_memory_preflight.get("model_size_gb"),
                "required_available_gb": real_ui_dsv4_memory_preflight.get(
                    "required_available_gb"
                ),
                "required_free_gb": real_ui_dsv4_memory_preflight.get(
                    "required_free_gb"
                ),
                "min_free_gb": real_ui_dsv4_memory_preflight.get("min_free_gb"),
                "preflight_memory_source": real_ui_dsv4_memory_preflight.get(
                    "preflight_memory_source"
                ),
                "free_plus_speculative_purgeable_gb": real_ui_dsv4_memory_preflight.get(
                    "free_plus_speculative_purgeable_gb"
                ),
                "available_for_gate_gb": real_ui_dsv4_memory_preflight.get(
                    "available_for_gate_gb"
                ),
                "memory_gap_gb": real_ui_dsv4_memory_preflight.get("memory_gap_gb"),
                "psutil_available_gb": real_ui_dsv4_memory_preflight.get(
                    "psutil_available_gb"
                ),
                "psutil_memory_gap_gb": real_ui_dsv4_memory_preflight.get(
                    "psutil_memory_gap_gb"
                ),
                "memory_pressure_free_percent": real_ui_dsv4_memory_preflight.get(
                    "memory_pressure_free_percent"
                ),
                "inactive_file_cache_gb": real_ui_dsv4_memory_preflight.get(
                    "inactive_file_cache_gb"
                ),
                "did_not_launch": real_ui_dsv4_memory_preflight.get(
                    "did_not_launch"
                ),
                "launch_decision": real_ui_dsv4_memory_preflight.get(
                    "launch_decision"
                ),
                "launch_allowed": real_ui_dsv4_memory_preflight.get(
                    "launch_allowed"
                ),
                "launch_blockers": real_ui_dsv4_memory_preflight.get(
                    "launch_blockers"
                ),
                "active_heavy_process_count": real_ui_dsv4_memory_preflight.get(
                    "active_heavy_process_count"
                ),
                "active_heavy_processes": real_ui_dsv4_memory_preflight.get(
                    "active_heavy_processes"
                ),
                "top_memory_processes": real_ui_dsv4_memory_preflight.get(
                    "top_memory_processes"
                ),
            }
        real_ui_live_model_matrix["resource_blockers"] = resource_blockers
    _annotate_real_ui_unblocked_non_mimo_status(real_ui_live_model_matrix)
    regression_suite_ok = _current_regression_suite_state_is_acceptable(
        regression_suite
    )
    objective_digest_ok = (
        objective_digest["status"] == "pass"
        and not objective_digest["unexpected_open_requirements"]
        and not objective_digest["missing_expected_open_requirements"]
        and not objective_digest["invalid_requirements"]
        and not objective_digest["failed_requirements"]
        and not objective_digest["missing_evidence_requirements"]
        and not objective_digest["open_requirement_detail_failures"]
    )
    model_family_matrix_ok = (
        model_family_matrix["status"] == "pass"
        and not model_family_matrix["missing_rows"]
        and not model_family_matrix["missing_expected_rows"]
    )
    model_artifact_matrix_ok = (
        model_artifact_matrix["status"] == "pass"
        and not model_artifact_matrix["missing_markers"]
        and not model_artifact_matrix["failed_checks"]
        and not model_artifact_matrix["missing_expected_checks"]
    )
    cache_architecture_matrix_ok = (
        cache_architecture_matrix["status"] == "pass"
        and not cache_architecture_matrix["missing_markers"]
        and not cache_architecture_matrix["missing_api_checks"]
        and not cache_architecture_matrix["missing_api_command_markers"]
        and not cache_architecture_matrix["missing_panel_markers"]
        and not cache_architecture_matrix["missing_family_rows"]
        and not cache_architecture_matrix["failed_family_rows"]
        and not cache_architecture_matrix["failed_checks"]
        and not cache_architecture_matrix["missing_expected_checks"]
    )
    parser_registry_matrix_ok = (
        parser_registry_matrix["status"] == "pass"
        and not parser_registry_matrix["missing_markers"]
        and not parser_registry_matrix["failed_checks"]
        and not parser_registry_matrix["missing_expected_checks"]
    )
    generation_defaults_matrix_ok = (
        generation_defaults_matrix["status"] == "pass"
        and not generation_defaults_matrix["missing_markers"]
        and not generation_defaults_matrix["missing_family_rows"]
        and not generation_defaults_matrix["failed_family_rows"]
        and not generation_defaults_matrix["failed_checks"]
        and not generation_defaults_matrix["missing_expected_checks"]
    )
    noheavy_api_cache_matrix_ok = (
        noheavy_api_cache_matrix["status"] == "pass"
        and not noheavy_api_cache_matrix["missing_markers"]
        and not noheavy_api_cache_matrix["failed_checks"]
        and not noheavy_api_cache_matrix["missing_expected_checks"]
    )
    api_surface_matrix_ok = (
        api_surface_matrix["status"] == "pass"
        and not api_surface_matrix["missing_nested_checks"]
        and not api_surface_matrix["missing_nested_markers"]
        and not api_surface_matrix["missing_panel_markers"]
        and not api_surface_matrix["failed_checks"]
        and not api_surface_matrix["missing_expected_checks"]
    )
    reasoning_template_matrix_ok = (
        reasoning_template_matrix["status"] == "pass"
        and not reasoning_template_matrix["missing_markers"]
        and not reasoning_template_matrix["failed_checks"]
        and not reasoning_template_matrix["missing_expected_checks"]
    )
    tool_call_matrix_ok = (
        tool_call_matrix["status"] == "pass"
        and not tool_call_matrix["missing_markers"]
        and not tool_call_matrix["failed_checks"]
        and not tool_call_matrix["missing_expected_checks"]
    )
    panel_tool_security_matrix_ok = (
        panel_tool_security_matrix["status"] == "pass"
        and panel_tool_security_matrix["returncode"] == 0
        and not panel_tool_security_matrix["missing"]
    )
    native_mtp_matrix_ok = (
        native_mtp_matrix["status"] == "pass"
        and not native_mtp_matrix["missing_markers"]
        and not native_mtp_matrix["failed_checks"]
        and not native_mtp_matrix["missing_expected_checks"]
    )
    vl_media_matrix_ok = (
        vl_media_matrix["status"] == "pass"
        and not vl_media_matrix["missing_engine_markers"]
        and not vl_media_matrix["missing_panel_markers"]
        and not vl_media_matrix["failed_checks"]
        and not vl_media_matrix["missing_expected_checks"]
    )
    mcp_policy_matrix_ok = (
        mcp_policy_matrix["status"] == "pass"
        and not mcp_policy_matrix["missing_markers"]
        and not mcp_policy_matrix["failed_checks"]
        and not mcp_policy_matrix["missing_expected_checks"]
    )
    max_output_context_matrix_ok = (
        max_output_context_matrix["status"] == "pass"
        and not max_output_context_matrix["failed"]
        and not max_output_context_matrix["missing_markers"]
        and not max_output_context_matrix["failed_checks"]
        and not max_output_context_matrix["missing_expected_checks"]
    )
    jang_model_compat_matrix_ok = (
        jang_model_compat_matrix["status"] == "pass"
        and not jang_model_compat_matrix["failed"]
        and not jang_model_compat_matrix["missing"]
    )
    packaged_integrity_matrix_ok = (
        packaged_integrity_matrix["status"] == "pass"
        and not packaged_integrity_matrix["failed"]
        and not packaged_integrity_matrix["unexpected_open_requirements"]
        and not packaged_integrity_matrix["missing_expected_open_requirements"]
        and not packaged_integrity_matrix["failed_checks"]
        and not packaged_integrity_matrix["missing_expected_checks"]
    )
    release_surface_matrix_ok = (
        release_surface_matrix["status"] == "pass"
        and not release_surface_matrix["failed_checks"]
        and not release_surface_matrix["missing_expected_checks"]
    )
    live_smoke_summaries_ok = (
        (
            live_smoke_summaries["status"] == "pass"
            and not live_smoke_summaries["missing"]
            and not live_smoke_summaries["not_pass"]
        )
    )
    live_tool_smoke_summaries_ok = (
        (
            live_tool_smoke_summaries["status"] == "pass"
            and not live_tool_smoke_summaries["missing"]
            and not live_tool_smoke_summaries["not_pass"]
        )
    )
    diagnostic_live_smoke_summaries_ok = (
        diagnostic_live_smoke_summaries["status"] == "pass"
        and not diagnostic_live_smoke_summaries["missing"]
        and not diagnostic_live_smoke_summaries["not_expected"]
    )
    dev_ui_proof_ok = (
        dev_ui_proof["status"] == "pass"
        and not dev_ui_proof["missing"]
        and not dev_ui_proof["failures"]
    )
    real_ui_live_model_proof_ok = (
        real_ui_live_model_proof["status"] == "pass"
        and not real_ui_live_model_proof["missing"]
        and not real_ui_live_model_proof["failures"]
    )
    real_ui_dsv4_memory_preflight_ok = (
        "dsv4" not in real_ui_live_model_matrix.get("missing_families", [])
        or real_ui_dsv4_memory_preflight["status"] == "pass"
    )
    dsv4_proof_artifact_freshness_ok = (
        dsv4_proof_artifact_freshness["status"] == "pass"
    )
    real_ui_unblocked_partial = {
        str(item)
        for item in real_ui_live_model_matrix.get(
            "unblocked_non_mimo_partial_families", []
        )
    }
    real_ui_mixed_identity_families = {
        str(item)
        for item in real_ui_live_model_matrix.get(
            "mixed_model_identity_families", []
        )
    }
    real_ui_unblocked_non_mimo_ok = (
        (
            real_ui_live_model_matrix.get("unblocked_non_mimo_status") == "pass"
            and not real_ui_live_model_matrix.get(
                "unblocked_non_mimo_missing_families"
            )
            and not real_ui_live_model_matrix.get(
                "unblocked_non_mimo_partial_families"
            )
        )
        or (
            bool(real_ui_mixed_identity_families)
            and not real_ui_live_model_matrix.get(
                "unblocked_non_mimo_missing_families"
            )
            and real_ui_unblocked_partial == real_ui_mixed_identity_families
        )
    )
    real_ui_full_model_matrix_ok = (
        real_ui_live_model_matrix.get("status") == "pass"
        and not real_ui_live_model_matrix.get("missing_families")
        and not real_ui_live_model_matrix.get("partial_families")
        and not real_ui_live_model_matrix.get("mixed_model_identity_families")
    ) or real_ui_unblocked_non_mimo_ok
    issue179_minimax_k_root_cause_audit_ok = (
        issue179_minimax_k_root_cause_audit["status"] in {"open", "pass"}
        and not issue179_minimax_k_root_cause_audit["missing"]
        and not issue179_minimax_k_root_cause_audit["failures"]
    )
    issue179_minimax_k_live_probe_memory_preflight_ok = (
        issue179_minimax_k_root_cause_audit["status"] == "pass"
        or issue179_minimax_k_live_probe_memory_preflight["status"] == "pass"
    )
    issue181_183_runtime_audit_ok = (
        issue181_183_runtime_audit["status"] == "pass"
        and not issue181_183_runtime_audit["missing"]
        and not issue181_183_runtime_audit["failures"]
    )
    public_app_issue_audit_ok = (
        public_app_issue_audit["status"] in {"open", "pass"}
        and not public_app_issue_audit["missing"]
        and not public_app_issue_audit["failures"]
    )
    issue175_179_release_boundary_audit_ok = (
        issue175_179_release_boundary_audit["status"] in {"open", "pass"}
        and not issue175_179_release_boundary_audit["missing"]
        and not issue175_179_release_boundary_audit["failures"]
    )
    installed_app_runtime_parity_audit_ok = (
        installed_app_runtime_parity_audit["status"] in {"open", "pass"}
        and not installed_app_runtime_parity_audit["missing"]
        and not installed_app_runtime_parity_audit["failures"]
    )
    staged_app_runtime_parity_audit_ok = (
        staged_app_runtime_parity_audit["status"] == "pass"
        and not staged_app_runtime_parity_audit["missing"]
        and not staged_app_runtime_parity_audit["failures"]
    )
    issue175_177_installed_runtime_audit_ok = (
        issue175_177_installed_runtime_audit["status"] == "pass"
        and not issue175_177_installed_runtime_audit["missing"]
        and not issue175_177_installed_runtime_audit["failures"]
    )
    issue175_177_live_runtime_audit_ok = (
        issue175_177_live_runtime_audit["status"] == "pass"
        and not issue175_177_live_runtime_audit["missing"]
        and not issue175_177_live_runtime_audit["failures"]
    )
    release_blocker_ledger = _current_release_blocker_ledger(
        root=root,
        regression_suite=regression_suite,
        packaged_integrity_matrix=packaged_integrity_matrix,
        live_smoke_summaries=live_smoke_summaries,
        live_tool_smoke_summaries=live_tool_smoke_summaries,
        mimo_v2_jang2l_sink_ab={},
        mimo_v2_jang2l_root_cause=mimo_v2_jang2l_root_cause,
        issue175_179_release_boundary_audit=issue175_179_release_boundary_audit,
        installed_app_runtime_parity_audit=installed_app_runtime_parity_audit,
        issue179_minimax_k_root_cause_audit=issue179_minimax_k_root_cause_audit,
        issue179_minimax_k_live_probe_memory_preflight=(
            issue179_minimax_k_live_probe_memory_preflight
        ),
        public_app_issue_audit=public_app_issue_audit,
        release_surface_matrix=release_surface_matrix,
        real_ui_live_model_proof=real_ui_live_model_proof,
        real_ui_live_model_matrix=real_ui_live_model_matrix,
        step37_vlm_runtime_audit=step37_vlm_runtime_audit,
    )
    release_blocker_ids = {
        str(blocker.get("id"))
        for blocker in release_blocker_ledger.get("blockers", [])
        if isinstance(blocker, dict)
    }
    effective_open_requirements = [
        str(item)
        for item in objective_digest.get("open_requirements") or []
        if str(item) not in DEFERRED_RELEASE_OPEN_REQUIREMENTS
    ]
    component_ok = {
        "no_missing_post_budget_artifacts": not missing,
        "no_not_pass_post_budget_artifacts": not not_pass,
        "no_release_blockers": (
            release_blocker_ledger.get("status") == "pass"
            and not release_blocker_ledger.get("blockers")
        ),
        "packaged_app_developer_id_signing": (
            "packaged_app_developer_id_signing_blocked" not in release_blocker_ids
        ),
        "dsv4_long_output_code_exactness": (
            "dsv4_long_output_code_exactness_open" not in release_blocker_ids
        ),
        "no_open_objective_requirements": not effective_open_requirements,
        "regression_suite": regression_suite_ok,
        "objective_digest": objective_digest_ok,
        "model_family_matrix": model_family_matrix_ok,
        "model_artifact_matrix": model_artifact_matrix_ok,
        "cache_architecture_matrix": cache_architecture_matrix_ok,
        "parser_registry_matrix": parser_registry_matrix_ok,
        "generation_defaults_matrix": generation_defaults_matrix_ok,
        "noheavy_api_cache_matrix": noheavy_api_cache_matrix_ok,
        "api_surface_matrix": api_surface_matrix_ok,
        "reasoning_template_matrix": reasoning_template_matrix_ok,
        "tool_call_matrix": tool_call_matrix_ok,
        "panel_tool_security_matrix": panel_tool_security_matrix_ok,
        "native_mtp_matrix": native_mtp_matrix_ok,
        "vl_media_matrix": vl_media_matrix_ok,
        "mcp_policy_matrix": mcp_policy_matrix_ok,
        "max_output_context_matrix": max_output_context_matrix_ok,
        "jang_model_compat_matrix": jang_model_compat_matrix_ok,
        "packaged_integrity_matrix": packaged_integrity_matrix_ok,
        "release_surface_matrix": release_surface_matrix_ok,
        "live_smoke_summaries": live_smoke_summaries_ok,
        "live_tool_smoke_summaries": live_tool_smoke_summaries_ok,
        "mimo_v2_jang2l_root_cause": (
            mimo_v2_jang2l_root_cause.get("status") == "pass"
            and mimo_v2_jang2l_root_cause.get("local_release_clearance") is True
        ),
        "diagnostic_live_smoke_summaries": diagnostic_live_smoke_summaries_ok,
        "dev_ui_proof": dev_ui_proof_ok,
        "real_ui_live_model_proof": real_ui_live_model_proof_ok,
        "real_ui_full_model_matrix": real_ui_full_model_matrix_ok,
        "real_ui_dsv4": (
            "real_ui_dsv4_memory_blocked" not in release_blocker_ids
        ),
        "real_ui_unblocked_non_mimo": real_ui_unblocked_non_mimo_ok,
        "issue175_179_release_boundary_audit": issue175_179_release_boundary_audit_ok,
        "installed_app_runtime_parity_audit": installed_app_runtime_parity_audit_ok,
        "staged_app_runtime_parity_audit": staged_app_runtime_parity_audit_ok,
        "issue175_177_installed_runtime_audit": issue175_177_installed_runtime_audit_ok,
        "issue175_177_live_runtime_audit": issue175_177_live_runtime_audit_ok,
        "issue179_minimax_k_root_cause_audit": issue179_minimax_k_root_cause_audit_ok,
        "issue179_minimax_k_live_probe_memory_preflight": (
            issue179_minimax_k_live_probe_memory_preflight_ok
        ),
        "issue181_183_runtime_audit": issue181_183_runtime_audit_ok,
        "public_app_issue_audit": public_app_issue_audit_ok,
        "real_ui_dsv4_memory_preflight": real_ui_dsv4_memory_preflight_ok,
        "dsv4_proof_artifact_freshness": dsv4_proof_artifact_freshness_ok,
    }
    failed_components = [
        name for name, ok in component_ok.items() if not ok
    ]

    return {
        "status": "pass" if all(component_ok.values()) else "fail",
        "component_ok": component_ok,
        "failed_components": failed_components,
        "missing": missing,
        "not_pass": not_pass,
        "regression_suite": regression_suite,
        "objective_digest": objective_digest,
        "model_family_matrix": model_family_matrix,
        "model_artifact_matrix": model_artifact_matrix,
        "cache_architecture_matrix": cache_architecture_matrix,
        "parser_registry_matrix": parser_registry_matrix,
        "generation_defaults_matrix": generation_defaults_matrix,
        "noheavy_api_cache_matrix": noheavy_api_cache_matrix,
        "api_surface_matrix": api_surface_matrix,
        "reasoning_template_matrix": reasoning_template_matrix,
        "tool_call_matrix": tool_call_matrix,
        "panel_tool_security_matrix": panel_tool_security_matrix,
        "native_mtp_matrix": native_mtp_matrix,
        "vl_media_matrix": vl_media_matrix,
        "mcp_policy_matrix": mcp_policy_matrix,
        "max_output_context_matrix": max_output_context_matrix,
        "jang_model_compat_matrix": jang_model_compat_matrix,
        "packaged_integrity_matrix": packaged_integrity_matrix,
        "release_surface_matrix": release_surface_matrix,
        "live_smoke_summaries": live_smoke_summaries,
        "live_tool_smoke_summaries": live_tool_smoke_summaries,
        "mimo_v2_jang2l_root_cause": mimo_v2_jang2l_root_cause,
        "diagnostic_live_smoke_summaries": diagnostic_live_smoke_summaries,
        "dev_ui_proof": dev_ui_proof,
        "real_ui_live_model_proof": real_ui_live_model_proof,
        "issue175_179_release_boundary_audit": issue175_179_release_boundary_audit,
        "installed_app_runtime_parity_audit": installed_app_runtime_parity_audit,
        "staged_app_runtime_parity_audit": staged_app_runtime_parity_audit,
        "issue175_177_installed_runtime_audit": issue175_177_installed_runtime_audit,
        "issue175_177_live_runtime_audit": issue175_177_live_runtime_audit,
        "issue179_minimax_k_root_cause_audit": issue179_minimax_k_root_cause_audit,
        "issue179_minimax_k_live_probe_memory_preflight": (
            issue179_minimax_k_live_probe_memory_preflight
        ),
        "issue181_183_runtime_audit": issue181_183_runtime_audit,
        "public_app_issue_audit": public_app_issue_audit,
        "real_ui_live_model_matrix": real_ui_live_model_matrix,
        "real_ui_dsv4_memory_preflight": real_ui_dsv4_memory_preflight,
        "dsv4_proof_artifact_freshness": dsv4_proof_artifact_freshness,
        "step37_vlm_runtime_audit": step37_vlm_runtime_audit,
        "release_blocker_ledger": release_blocker_ledger,
    }


DSV4_SOURCE_PREFLIGHT_RELEASE_DETAIL_KEYS = (
    "artifact_present",
    "status",
    "reason",
    "model",
    "commands",
    "available_gb",
    "required_available_gb",
    "required_free_gb",
    "min_free_gb",
    "available_for_gate_gb",
    "memory_gap_gb",
    "strict_vm_stat_memory_gap_gb",
    "psutil_available_gap_gb",
    "free_plus_speculative_purgeable_gb",
    "memory_pressure_free_percent",
    "memory_pressure_error",
    "preflight_memory_source",
    "did_not_launch",
    "launch_decision",
    "launch_allowed",
    "launch_blockers",
    "active_heavy_process_count",
    "active_heavy_processes",
    "top_memory_processes",
    "selected_cases",
    "case_count",
)


def _dsv4_source_preflight_release_details(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    path = root / CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - release ledger falls back to suite details
        return None
    if not isinstance(payload, dict):
        return None
    details = {
        key: payload.get(key)
        for key in DSV4_SOURCE_PREFLIGHT_RELEASE_DETAIL_KEYS
        if key in payload
    }
    if details and "artifact_present" not in details:
        details["artifact_present"] = True
    return details or None


def _current_release_blocker_ledger(
    *,
    root: Path | None = None,
    regression_suite: dict[str, Any],
    packaged_integrity_matrix: dict[str, Any] | None = None,
    live_smoke_summaries: dict[str, Any],
    live_tool_smoke_summaries: dict[str, Any],
    mimo_v2_jang2l_sink_ab: dict[str, Any],
    mimo_v2_jang2l_root_cause: dict[str, Any],
    issue175_179_release_boundary_audit: dict[str, Any],
    installed_app_runtime_parity_audit: dict[str, Any],
    issue179_minimax_k_root_cause_audit: dict[str, Any],
    issue179_minimax_k_live_probe_memory_preflight: dict[str, Any] | None = None,
    public_app_issue_audit: dict[str, Any] | None = None,
    release_surface_matrix: dict[str, Any] | None = None,
    real_ui_live_model_proof: dict[str, Any] | None = None,
    real_ui_live_model_matrix: dict[str, Any],
    step37_vlm_runtime_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    deferred_release_gaps: list[dict[str, Any]] = []
    deferred_release_families = [
        {"family": family, "reason": reason}
        for family, reason in sorted(DEFERRED_RELEASE_FAMILIES.items())
    ]
    dsv4_deferred_reason = DEFERRED_RELEASE_FAMILIES.get("dsv4")
    if packaged_integrity_matrix is None:
        packaged_integrity_matrix = {}
    signing_preflight = packaged_integrity_matrix.get("package_signing_preflight")
    if not isinstance(signing_preflight, dict):
        signing_preflight = {}
    for blocker in packaged_integrity_matrix.get("release_blockers", []):
        if not isinstance(blocker, dict):
            continue
        if str(blocker.get("status")) != "open":
            continue
        ledger_blocker: dict[str, Any] = {
            "id": str(blocker.get("id") or "packaged_app_release_blocker"),
            "status": "open",
            "evidence": str(blocker.get("evidence") or "packaged_integrity_matrix"),
            "next_proof": str(
                blocker.get("next_proof")
                or "Refresh packaged app release proof before notarization."
            ),
        }
        if ledger_blocker["evidence"] == "package_signing_preflight":
            details = {
                key: signing_preflight.get(key)
                for key in (
                    "signing_blocker_reason",
                    "signing_blocker_reasons",
                    "developer_id_signed",
                    "developer_id_identity_count",
                    "signature_is_adhoc",
                    "hardened_runtime_enabled",
                    "team_identifier",
                    "codesign_display_rc",
                    "signature_summary_tail",
                    "simple_developer_id_sign_rc",
                    "simple_developer_id_sign_tail",
                    "codesign_verify_rc",
                    "verify_tail",
                    "packaged_app_modified_after_signing",
                    "modified_after_signing_file_count",
                    "missing_after_signing_file_count",
                    "modified_after_signing_tail",
                    "missing_after_signing_tail",
                    "keychain_info_statuses",
                    "manual_remediation_required",
                    "remediation_summary",
                    "remediation_steps",
                )
                if key in signing_preflight
            }
            if details:
                ledger_blocker["details"] = details
        blockers.append(ledger_blocker)
    open_requirements = {
        str(item) for item in regression_suite.get("open_requirements", [])
    }
    if (
        "DSV4 long-output/code/file-generation quality is release-cleared"
        in open_requirements
    ):
        blocker = {
            "id": "dsv4_long_output_code_exactness_open",
            "status": "deferred" if dsv4_deferred_reason else "open",
            "evidence": CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
            "next_proof": (
                "Run and pass DSV4 long-output/code exactness with current "
                "source/app in a memory-safe local session."
            ),
        }
        if dsv4_deferred_reason:
            blocker["reason"] = dsv4_deferred_reason
        artifact_details = _dsv4_source_preflight_release_details(root)
        if artifact_details:
            blocker["details"] = artifact_details
        open_details = regression_suite.get("open_requirement_details")
        if "details" not in blocker and isinstance(open_details, dict):
            row = open_details.get(
                "DSV4 long-output/code/file-generation quality is release-cleared"
            )
            row_details = row.get("details") if isinstance(row, dict) else None
            preflight = (
                row_details.get("current_source_full_output_preflight")
                if isinstance(row_details, dict)
                else None
            )
            if isinstance(preflight, dict):
                details = {
                    key: preflight.get(key)
                    for key in DSV4_SOURCE_PREFLIGHT_RELEASE_DETAIL_KEYS
                    if key in preflight
                }
                if details:
                    blocker["details"] = details
        if dsv4_deferred_reason:
            deferred_release_gaps.append(blocker)
        else:
            blockers.append(blocker)

    if (
        isinstance(mimo_v2_jang2l_root_cause, dict)
        and mimo_v2_jang2l_root_cause
        and "status" in mimo_v2_jang2l_root_cause
        and (
            mimo_v2_jang2l_root_cause.get("status") != "pass"
            or mimo_v2_jang2l_root_cause.get("local_release_clearance") is not True
        )
    ):
        mimo_next_proof = (
            "Pass current local MiMo JANG_2L long-prompt coherence, "
            "tool protocol/continuation, cache, and API proof before "
            "including MiMo in a production release."
        )
        if mimo_v2_jang2l_root_cause.get("artifact_exactness_blocked") is True:
            artifact_exactness_labels = ["JANGTQ2 artifact exactness"]
            bundle_status = mimo_v2_jang2l_root_cause.get(
                "artifact_exactness_bundle_status"
            )
            if isinstance(bundle_status, dict):
                artifact_exactness_labels = []
                if (
                    isinstance(bundle_status.get("jangtq2"), dict)
                    and bundle_status["jangtq2"].get("blocked") is True
                ):
                    artifact_exactness_labels.append("JANGTQ2 artifact exactness")
                if (
                    isinstance(bundle_status.get("jang2l"), dict)
                    and bundle_status["jang2l"].get("blocked") is True
                ):
                    artifact_exactness_labels.append("JANG_2L artifact exactness")
                if not artifact_exactness_labels:
                    artifact_exactness_labels.append("MiMo artifact exactness")
            mimo_next_proof = (
                "Pass current local MiMo "
                + " and ".join(artifact_exactness_labels)
                + "/value-drift "
                "proof, or replace it with corrected artifact/runtime evidence, "
                "before including MiMo in a production release."
            )
        elif isinstance(mimo_v2_jang2l_root_cause.get("release_boundary"), str):
            mimo_next_proof = (
                "Resolve MiMo release boundary: "
                + str(mimo_v2_jang2l_root_cause.get("release_boundary"))
            )
        blockers.append(
            {
                "id": "mimo_v2_jang2l_runtime_quality_open",
                "status": "open",
                "evidence": ",".join(
                    str(path)
                    for path in (
                        mimo_v2_jang2l_root_cause.get("artifacts") or {}
                    ).values()
                )
                or CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT,
                "next_proof": mimo_next_proof,
                "details": {
                    key: mimo_v2_jang2l_root_cause.get(key)
                    for key in (
                        "status",
                        "structural_verify_passed",
                        "text_cache_narrow_pass",
                        "switchglu_selected_expert_parity_passed",
                        "prompt_length_coherence_blocked",
                        "tool_protocol_blocked",
                        "artifact_exactness_blocked",
                        "artifact_exactness_bundle_status",
                        "artifact_exactness_release_action",
                        "recommended_next_artifact_profiles",
                        "decode_speed_target_blocked",
                        "media_unwired",
                        "mimo_jangtq2_live_media_l2_passed",
                        "mimo_jang2l_live_media_l2_blocked",
                        "mimo_jang2l_l2_restart_cache_hit_passed",
                        "mimo_jang2l_l2_restart_visible_output_blocked",
                        "source_vs_quant_requirement_satisfied",
                        "root_cause_candidate",
                        "release_boundary",
                    )
                    if key in mimo_v2_jang2l_root_cause
                },
            }
        )

    if isinstance(release_surface_matrix, dict) and (
        "staged_source_version_not_public"
        in {str(item) for item in release_surface_matrix.get("failed_checks", [])}
    ):
        blockers.append(
            {
                "id": "source_version_already_public",
                "status": "open",
                "evidence": str(
                    release_surface_matrix.get("artifact")
                    or CURRENT_POST_BUDGET_EDGE_ARTIFACTS[
                        "public-release-surface-preflight"
                    ]
                ),
                "next_proof": (
                    "Bump source/package version beyond the already-public updater "
                    "version before cutting another release from this worktree."
                ),
            }
        )

    issue175_179_subblockers_added = False
    issues = issue175_179_release_boundary_audit.get("issues")
    if not isinstance(issues, dict):
        issues = {}
    issue175 = issues.get("175") if isinstance(issues.get("175"), dict) else {}
    if issue175.get("release_clearance") == (
        "installed_app_memory_clear_runtime_proven_live_stress_open"
    ):
        issue175_179_subblockers_added = True
        blockers.append(
            {
                "id": "issue175_live_app_memory_stress_open",
                "status": "open",
                "evidence": CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT,
                "next_proof": "Run installed-app live memory stress proving MLX cache clear behavior under model load/unload.",
            }
        )

    issue176 = issues.get("176") if isinstance(issues.get("176"), dict) else {}
    if issue176.get("release_clearance") == (
        "installed_app_promoted_block_cleanup_proven_live_stress_open"
    ):
        issue175_179_subblockers_added = True
        blockers.append(
            {
                "id": "issue176_live_memory_pressure_open",
                "status": "open",
                "evidence": CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT,
                "next_proof": "Run installed-app memory-pressure proof that promoted paged-cache blocks release parent KV buffers.",
            }
        )

    issue177 = issues.get("177") if isinstance(issues.get("177"), dict) else {}
    if issue177.get("release_clearance") == (
        "installed_app_cache_selection_telemetry_proven_live_ttft_open"
    ):
        issue175_179_subblockers_added = True
        blockers.append(
            {
                "id": "issue177_live_ttft_paged_turboquant_open",
                "status": "open",
                "evidence": CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT,
                "next_proof": "Run live TTFT A/B proving cache selection avoids cold paged/TurboQuant regressions.",
            }
        )

    issue179_blocker_added = issue179_minimax_k_root_cause_audit.get("status") == "open"
    if (
        issue175_179_release_boundary_audit.get("status") == "open"
        and not issue175_179_subblockers_added
        and not issue179_blocker_added
    ):
        blockers.append(
            {
                "id": "issue175_179_release_boundary_audit",
                "status": "open",
                "evidence": CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT,
                "next_proof": "Live UI/API/model proof for #175-#179 cache/runtime/parser behavior.",
            }
        )

    if installed_app_runtime_parity_audit.get("status") == "open":
        blockers.append(
            {
                "id": "installed_app_runtime_parity_audit",
                "status": "open",
                "evidence": CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
                "next_proof": "Rebuild/sync installed app and rerun packaged parity audit.",
            }
        )

    if issue179_minimax_k_root_cause_audit.get("status") == "open":
        issue179_not_proven = issue179_minimax_k_root_cause_audit.get("not_proven")
        if not isinstance(issue179_not_proven, list):
            issue179_not_proven = []
        issue179_waiting_on_reporter_hash = any(
            str(item)
            == "reporter installed app bundle hash matches public/local server.py route proof"
            for item in issue179_not_proven
        )
        blocker = {
            "id": "issue179_minimax_k_root_cause_audit",
            "status": "open",
            "evidence": CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT,
            "next_proof": (
                "Obtain reporter installed app bundle hash provenance matching a "
                "public/local vMLX server.py route proof, or refresh reporter "
                "parity metadata against a known public DMG; do not rerun broad "
                "cache/model probes until this provenance gap changes."
                if issue179_waiting_on_reporter_hash
                else (
                    "Reproduce or disprove screenshot-shaped wrong-language/numeric "
                    "reasoning-panel garbage with the reporter prompt/session."
                )
            ),
        }
        details = {
            key: issue179_minimax_k_root_cause_audit.get(key)
            for key in (
                "not_proven",
                "release_boundary",
                "reporter_server_hash_parity",
                "reporter_parity_comparison",
            )
            if key in issue179_minimax_k_root_cause_audit
        }
        if isinstance(issue179_minimax_k_live_probe_memory_preflight, dict):
            preflight_details = {
                key: issue179_minimax_k_live_probe_memory_preflight.get(key)
                for key in (
                    "artifact",
                    "status",
                    "reason",
                    "model_path",
                    "model_size_gb",
                    "required_free_gb",
                    "min_free_gb",
                    "available_for_gate_gb",
                    "free_plus_speculative_purgeable_gb",
                    "memory_gap_gb",
                    "preflight_memory_source",
                    "launch_decision",
                    "launch_allowed",
                )
                if key in issue179_minimax_k_live_probe_memory_preflight
            }
            if preflight_details:
                details["live_probe_memory_preflight"] = preflight_details
        if details:
            blocker["details"] = details
        deferred_reason = DEFERRED_RELEASE_BLOCKER_IDS.get(blocker["id"])
        if deferred_reason:
            blocker["status"] = "deferred"
            blocker["reason"] = deferred_reason
            deferred_release_gaps.append(blocker)
        else:
            blockers.append(blocker)

    if not isinstance(public_app_issue_audit, dict):
        public_app_issue_audit = {}
    public_issues = public_app_issue_audit.get("issues")
    if not isinstance(public_issues, dict):
        public_issues = {}
    issue119 = public_issues.get("119")
    if (
        isinstance(issue119, dict)
        and issue119.get("focused_source_slice") == "open"
        and issue119.get("release_clearance")
        == "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
    ):
        blockers.append(
            {
                "id": "issue119_gemma26_memory_stress_open",
                "status": "open",
                "evidence": CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT,
                "next_proof": (
                    "Refresh and pass the current Gemma4 26B installed-app "
                    "memory stress artifact, including mixed-SWA cache telemetry "
                    "and post-request health."
                ),
                "details": {
                    "repo": issue119.get("repo"),
                    "title": issue119.get("title"),
                    "checks": issue119.get("checks"),
                },
            }
        )

    if isinstance(real_ui_live_model_proof, dict):
        proof_failures = {
            str(item)
            for item in real_ui_live_model_proof.get("failures", [])
            if isinstance(item, str)
        }
        request_contract_failures = sorted(
            failure
            for failure in proof_failures
            if failure == "request_contract_missing"
            or failure.startswith("request_contract_missing:")
            or failure.startswith("request_contract_incomplete:")
        )
        if request_contract_failures:
            blockers.append(
                {
                    "id": "real_ui_request_contract_proofs_stale",
                    "status": "open",
                    "evidence": (
                        "current_proof_sweep.real_ui_live_model_proof.failures:"
                        + ",".join(request_contract_failures)
                    ),
                    "next_proof": (
                        "Rerun real Electron UI model proofs with current "
                        "live-real-ui-model-proof.mjs so every artifact records "
                        "its prompts and request knobs."
                    ),
                }
            )

    if real_ui_live_model_matrix.get("status") == "open":
        real_ui_subblocker_added = False
        runtime_blocked_families: set[str] = set()
        missing_families = {
            str(item) for item in real_ui_live_model_matrix.get("missing_families", [])
        }
        resource_blockers = real_ui_live_model_matrix.get("resource_blockers")
        if not isinstance(resource_blockers, dict):
            resource_blockers = {}
        if "dsv4" in missing_families and "dsv4" in resource_blockers:
            real_ui_subblocker_added = dsv4_deferred_reason is None
            blocker = resource_blockers.get("dsv4")
            artifact = (
                blocker.get("artifact")
                if isinstance(blocker, dict)
                else CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
            )
            real_ui_dsv4_gap = {
                "id": "real_ui_dsv4_memory_blocked",
                "status": "deferred" if dsv4_deferred_reason else "open",
                "evidence": str(artifact or CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT),
                "next_proof": (
                    "Run DSV4 real Electron UI proof on a local "
                    "machine/session with enough free memory."
                ),
                "details": blocker if isinstance(blocker, dict) else {},
            }
            if dsv4_deferred_reason:
                real_ui_dsv4_gap["reason"] = dsv4_deferred_reason
                deferred_release_gaps.append(real_ui_dsv4_gap)
            else:
                blockers.append(real_ui_dsv4_gap)
        if (
            "step37" in missing_families
            and isinstance(step37_vlm_runtime_audit, dict)
            and step37_vlm_runtime_audit.get("status") == "open"
            and step37_vlm_runtime_audit.get("root_cause")
            == "missing_mlx_vlm_step3p7_vlm_runtime"
        ):
            real_ui_subblocker_added = True
            runtime_blocked_families.add("step37")
            blockers.append(
                {
                    "id": "real_ui_step37_vlm_runtime_missing",
                    "status": "open",
                    "evidence": str(
                        step37_vlm_runtime_audit.get("artifact")
                        or CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
                    ),
                    "next_proof": (
                        "Implement and prove a real Step3p7 VLM runtime path before "
                        "rerunning the Step3.7 real Electron UI image proof."
                    ),
                    "details": {
                        key: step37_vlm_runtime_audit.get(key)
                        for key in (
                            "root_cause",
                            "failure_summary",
                            "release_clearance",
                            "mlx_vlm_step3p7_importable",
                            "mlx_vlm_step3p7_runtime_available",
                            "mlx_vlm_step3p7_runtime",
                            "local_reference_vlm_runtime",
                            "step_jangtq_status",
                            "mlx_vlm_implementation_contract",
                            "source_owned_runtime_progress",
                        )
                        if key in step37_vlm_runtime_audit
                    },
                }
            )
        mixed_model_identity_families = [
            str(item)
            for item in real_ui_live_model_matrix.get(
                "mixed_model_identity_families", []
            )
        ]
        if mixed_model_identity_families:
            real_ui_subblocker_added = True
            blockers.append(
                {
                    "id": "real_ui_mixed_model_identity_blocked",
                    "status": "open",
                    "evidence": (
                        "current_proof_sweep.real_ui_live_model_matrix."
                        "mixed_model_identity_families:"
                        + ",".join(sorted(mixed_model_identity_families))
                    ),
                    "next_proof": (
                        "Rerun real Electron UI proof with one model identity "
                        "per family; do not union Small/K or diagnostic/current "
                        "artifacts to clear a family."
                    ),
                }
            )
        unblocked_missing_families = sorted(
            family
            for family in missing_families
            if family
            not in {
                *runtime_blocked_families,
                *{str(key) for key in resource_blockers},
            }
        )
        unblocked_partial_families = sorted(
            str(item)
            for item in real_ui_live_model_matrix.get("partial_families", [])
            if str(item)
            not in {
                *runtime_blocked_families,
                *{str(key) for key in resource_blockers},
                *mixed_model_identity_families,
            }
        )
        if unblocked_missing_families:
            real_ui_subblocker_added = True
            blockers.append(
                {
                    "id": "real_ui_unblocked_non_mimo_missing",
                    "status": "open",
                    "evidence": (
                        "current_proof_sweep.real_ui_live_model_matrix."
                        "missing_families:"
                        + ",".join(unblocked_missing_families)
                    ),
                    "next_proof": (
                        "Run real Electron UI proofs for the missing "
                        "non-MiMo, non-resource-blocked families on this "
                        "laptop before release."
                    ),
                }
            )
        if unblocked_partial_families:
            covered_families = real_ui_live_model_matrix.get("covered_families")
            if not isinstance(covered_families, dict):
                covered_families = {}
            partial_details = {}
            for family in unblocked_partial_families:
                row = covered_families.get(family)
                if not isinstance(row, dict):
                    continue
                partial_details[family] = {
                    "missing_surfaces": list(row.get("missing_surfaces") or []),
                    "artifact": row.get("artifact"),
                }
            real_ui_subblocker_added = True
            blockers.append(
                {
                    "id": "real_ui_unblocked_non_mimo_partial",
                    "status": "open",
                    "evidence": (
                        "current_proof_sweep.real_ui_live_model_matrix."
                        "partial_families:"
                        + ",".join(unblocked_partial_families)
                    ),
                    "next_proof": (
                        "Rerun real Electron UI proofs for the partial "
                        "non-MiMo, non-resource-blocked families until every "
                        "required surface passes without parser or tool-result "
                        "semantic leakage."
                    ),
                    "details": {"partial_families": partial_details},
                }
            )
        if real_ui_subblocker_added:
            return {
                "status": "open",
                "blockers": blockers,
                "deferred_release_gaps": deferred_release_gaps,
                "deferred_release_families": deferred_release_families,
            }
        blocker = {
            "id": "real_ui_live_model_matrix",
            "status": "open",
            "evidence": "current_proof_sweep.real_ui_live_model_matrix",
            "next_proof": "Complete cross-family real UI matrix surfaces.",
        }
        deferred_reason = DEFERRED_RELEASE_BLOCKER_IDS.get(blocker["id"])
        if deferred_reason:
            blocker["status"] = "deferred"
            blocker["reason"] = deferred_reason
            deferred_release_gaps.append(blocker)
        else:
            blockers.append(blocker)

    return {
        "status": "open" if blockers else "pass",
        "blockers": blockers,
        "deferred_release_gaps": deferred_release_gaps,
        "deferred_release_families": deferred_release_families,
    }


def _annotate_real_ui_unblocked_non_mimo_status(
    real_ui_live_model_matrix: dict[str, Any],
) -> None:
    resource_blockers = real_ui_live_model_matrix.get("resource_blockers")
    if not isinstance(resource_blockers, dict):
        resource_blockers = {}
    runtime_blockers = real_ui_live_model_matrix.get("runtime_blockers")
    if not isinstance(runtime_blockers, dict):
        runtime_blockers = {}
    excluded = {
        *{str(key) for key in resource_blockers},
        *{str(key) for key in runtime_blockers},
    }
    missing = [
        family
        for family in real_ui_live_model_matrix.get("missing_families", [])
        if str(family) not in excluded
    ]
    partial = [
        family
        for family in real_ui_live_model_matrix.get("partial_families", [])
        if str(family) not in excluded
    ]
    mixed = [
        family
        for family in real_ui_live_model_matrix.get(
            "mixed_model_identity_families", []
        )
        if str(family) not in excluded
    ]
    for family in mixed:
        if family not in partial:
            partial.append(family)
    real_ui_live_model_matrix["unblocked_non_mimo_status"] = (
        "pass" if not missing and not partial else "open"
    )
    real_ui_live_model_matrix["unblocked_non_mimo_missing_families"] = missing
    real_ui_live_model_matrix["unblocked_non_mimo_partial_families"] = partial
    real_ui_live_model_matrix["unblocked_non_mimo_excluded_families"] = sorted(
        excluded
    )


def _json_number(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, int | float) else None


def _validate_current_real_ui_dsv4_memory_preflight(root: Path) -> dict[str, Any]:
    path = root / CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT
    result: dict[str, Any] = {
        "artifact": CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
        "status": "missing",
        "failures": [],
    }
    if not path.exists():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    failures: list[str] = []
    model_path = str(payload.get("model_path") or "")
    status = str(payload.get("status") or "")
    model_size_gb = _json_number(payload, "model_size_gb")
    free_gb = _json_number(payload, "free_plus_speculative_purgeable_gb")
    required_gb = _json_number(payload, "required_available_gb")
    required_free_gb = _json_number(payload, "required_free_gb")
    min_free_gb = _json_number(payload, "min_free_gb")
    available_for_gate_gb = _json_number(payload, "available_for_gate_gb")
    memory_gap_gb = _json_number(payload, "memory_gap_gb")
    top_memory_processes = payload.get("top_memory_processes")
    active_heavy_processes = payload.get("active_heavy_processes")
    launch_blockers = payload.get("launch_blockers")
    generated_at = payload.get("generated_at")
    if status != "skipped_insufficient_memory":
        failures.append("status_not_skipped_insufficient_memory")
    if not isinstance(generated_at, str) or not generated_at:
        failures.append("generated_at_missing")
    if payload.get("launch_decision") != "do_not_launch":
        failures.append("launch_decision_not_do_not_launch")
    if payload.get("launch_allowed") is not False:
        failures.append("launch_allowed_not_false")
    allowed_model_suffixes = (
        "DeepSeek-V4-Flash-JANGTQ-K",
        "DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520",
    )
    if not any(model_path.endswith(suffix) for suffix in allowed_model_suffixes):
        failures.append("model_path_not_dsv4_jangtq_k")
    if model_size_gb is None or model_size_gb < 70:
        failures.append("model_size_gb_not_recorded")
    if required_gb is None or required_gb < 120:
        failures.append("required_available_gb_below_floor")
    if required_free_gb != required_gb:
        failures.append("required_free_gb_mismatch")
    if min_free_gb != required_gb:
        failures.append("min_free_gb_mismatch")
    if (
        model_size_gb is not None
        and required_gb is not None
        and required_gb < model_size_gb + 40.0
    ):
        failures.append("required_available_gb_missing_model_margin")
    if free_gb is None:
        failures.append("free_plus_speculative_purgeable_gb_missing")
    elif required_gb is not None and free_gb >= required_gb:
        failures.append("preflight_claims_insufficient_memory_but_floor_met")
    if available_for_gate_gb != free_gb:
        failures.append("available_for_gate_gb_mismatch")
    if memory_gap_gb is None or memory_gap_gb <= 0:
        failures.append("memory_gap_gb_missing")
    elif required_gb is not None and free_gb is not None:
        expected_gap = round(required_gb - free_gb, 2)
        if abs(memory_gap_gb - expected_gap) > 0.01:
            failures.append("memory_gap_gb_mismatch")
    if not isinstance(top_memory_processes, list) or not top_memory_processes:
        failures.append("top_memory_processes_missing")
    if not isinstance(active_heavy_processes, list):
        failures.append("active_heavy_processes_missing")
    elif active_heavy_processes:
        failures.append("active_heavy_processes_not_clear")
    if payload.get("active_heavy_process_count") != 0:
        failures.append("active_heavy_process_count_not_zero")
    if not isinstance(launch_blockers, list):
        failures.append("launch_blockers_missing")
    elif "insufficient_memory" not in launch_blockers:
        failures.append("insufficient_memory_launch_blocker_missing")
    if payload.get("did_not_launch") is not True:
        failures.append("did_not_launch_not_true")
    if payload.get("preflight_memory_source") != "vm_stat_free_plus_speculative_purgeable":
        failures.append("preflight_memory_source_not_vm_stat")

    result.update(
        {
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "model_path": model_path,
            "generated_at": generated_at,
            "model_size_gb": model_size_gb,
            "free_plus_speculative_purgeable_gb": free_gb,
            "required_available_gb": required_gb,
            "required_free_gb": required_free_gb,
            "min_free_gb": min_free_gb,
            "available_for_gate_gb": available_for_gate_gb,
            "memory_gap_gb": memory_gap_gb,
            "preflight_memory_source": payload.get("preflight_memory_source"),
            "psutil_available_gb": _json_number(payload, "psutil_available_gb"),
            "psutil_memory_gap_gb": _json_number(payload, "psutil_memory_gap_gb"),
            "memory_pressure_free_percent": _json_number(
                payload, "memory_pressure_free_percent"
            ),
            "inactive_file_cache_gb": _json_number(payload, "inactive_file_cache_gb"),
            "did_not_launch": payload.get("did_not_launch"),
            "launch_decision": payload.get("launch_decision"),
            "launch_allowed": payload.get("launch_allowed"),
            "launch_blockers": payload.get("launch_blockers"),
            "active_heavy_process_count": payload.get("active_heavy_process_count"),
            "active_heavy_processes": payload.get("active_heavy_processes"),
            "top_memory_processes": payload.get("top_memory_processes"),
        }
    )
    return result


def _validate_current_issue179_minimax_k_live_probe_memory_preflight(
    root: Path,
) -> dict[str, Any]:
    artifact = CURRENT_ISSUE179_MINIMAX_K_LIVE_PROBE_MEMORY_PREFLIGHT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "failures": [],
    }
    if not path.exists():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    failures: list[str] = []
    model_path = str(payload.get("model_path") or "")
    status = str(payload.get("status") or "")
    reason = str(payload.get("reason") or "")
    model_size_gb = _json_number(payload, "model_size_gb")
    required_free_gb = _json_number(payload, "required_free_gb")
    min_free_gb = _json_number(payload, "min_free_gb")
    available_for_gate_gb = _json_number(payload, "available_for_gate_gb")
    free_plus_speculative_purgeable_gb = _json_number(
        payload,
        "free_plus_speculative_purgeable_gb",
    )
    memory_gap_gb = _json_number(payload, "memory_gap_gb")
    top_memory_processes = payload.get("top_memory_processes")
    active_heavy_processes = payload.get("active_heavy_processes")
    launch_blockers = payload.get("launch_blockers")
    commands = payload.get("commands")

    ready_to_launch = status == "ready_to_launch"
    skipped_for_memory = status == "skipped"
    if not (ready_to_launch or skipped_for_memory):
        failures.append("status_not_skipped_or_ready_to_launch")
    if ready_to_launch:
        if reason != "memory_preflight_floor_met":
            failures.append("reason_not_memory_preflight_floor_met")
        if payload.get("launch_decision") != "launch_allowed":
            failures.append("launch_decision_not_launch_allowed")
        if payload.get("launch_allowed") is not True:
            failures.append("launch_allowed_not_true")
    if skipped_for_memory:
        if reason != "insufficient_vm_stat_memory":
            failures.append("reason_not_insufficient_vm_stat_memory")
        if payload.get("launch_decision") != "do_not_launch":
            failures.append("launch_decision_not_do_not_launch")
        if payload.get("launch_allowed") is not False:
            failures.append("launch_allowed_not_false")
    if not model_path.endswith("MiniMax-M2.7-JANGTQ_K"):
        failures.append("model_path_not_minimax_m27_jangtq_k")
    if model_size_gb is None or model_size_gb < 70:
        failures.append("model_size_gb_not_recorded")
    if required_free_gb is None:
        failures.append("required_free_gb_missing")
    elif model_size_gb is not None and required_free_gb < model_size_gb + 5.0:
        failures.append("required_free_gb_missing_minimax_margin")
    if min_free_gb != required_free_gb:
        failures.append("min_free_gb_mismatch")
    if available_for_gate_gb != free_plus_speculative_purgeable_gb:
        failures.append("available_for_gate_gb_mismatch")
    if available_for_gate_gb is None:
        failures.append("available_for_gate_gb_missing")
    elif (
        skipped_for_memory
        and required_free_gb is not None
        and available_for_gate_gb >= required_free_gb
    ):
        failures.append("preflight_claims_insufficient_memory_but_floor_met")
    if memory_gap_gb is None:
        failures.append("memory_gap_gb_missing")
    elif skipped_for_memory and memory_gap_gb <= 0:
        failures.append("memory_gap_gb_missing")
    elif ready_to_launch and memory_gap_gb != 0:
        failures.append("memory_gap_gb_not_zero_when_ready")
    elif (
        skipped_for_memory
        and required_free_gb is not None
        and available_for_gate_gb is not None
    ):
        expected_gap = round(required_free_gb - available_for_gate_gb, 2)
        if abs(memory_gap_gb - expected_gap) > 0.01:
            failures.append("memory_gap_gb_mismatch")
    if (
        payload.get("preflight_memory_source")
        != "vm_stat_free_plus_speculative_purgeable"
    ):
        failures.append("preflight_memory_source_not_vm_stat")
    if not isinstance(top_memory_processes, list) or not top_memory_processes:
        failures.append("top_memory_processes_missing")
    if not isinstance(commands, dict):
        failures.append("commands_missing")
    else:
        if commands.get("memory") != "vm_stat":
            failures.append("memory_command_missing")
        if "top_memory_processes" not in commands:
            failures.append("top_memory_processes_command_missing")
        if "active_heavy_processes" not in commands:
            failures.append("active_heavy_processes_command_missing")
    if not isinstance(active_heavy_processes, list):
        failures.append("active_heavy_processes_missing")
    elif active_heavy_processes:
        failures.append("active_heavy_processes_not_clear")
    active_heavy_process_count = payload.get("active_heavy_process_count")
    if active_heavy_process_count != 0:
        failures.append("active_heavy_process_count_not_zero")
    if not isinstance(launch_blockers, list):
        failures.append("launch_blockers_missing")
    elif skipped_for_memory and "insufficient_memory" not in launch_blockers:
        failures.append("insufficient_memory_launch_blocker_missing")
    elif ready_to_launch and launch_blockers:
        failures.append("launch_blockers_not_empty_when_ready")
    preflight_captured_at = payload.get("preflight_captured_at")
    if not isinstance(preflight_captured_at, str) or not preflight_captured_at:
        failures.append("preflight_captured_at_missing")

    result.update(
        {
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "reason": reason,
            "model_path": model_path,
            "model_size_gb": model_size_gb,
            "required_free_gb": required_free_gb,
            "min_free_gb": min_free_gb,
            "available_for_gate_gb": available_for_gate_gb,
            "free_plus_speculative_purgeable_gb": (
                free_plus_speculative_purgeable_gb
            ),
            "memory_gap_gb": memory_gap_gb,
            "preflight_memory_source": payload.get("preflight_memory_source"),
            "preflight_captured_at": preflight_captured_at,
            "launch_decision": payload.get("launch_decision"),
            "launch_allowed": payload.get("launch_allowed"),
            "commands": commands,
            "top_memory_processes": top_memory_processes,
            "active_heavy_processes": active_heavy_processes,
            "active_heavy_process_count": active_heavy_process_count,
            "launch_blockers": launch_blockers,
        }
    )
    return result


def _validate_png_artifact(
    root: Path,
    artifact_key: str,
    artifact: str,
    failures: list[str],
) -> None:
    path = root / artifact
    if not path.exists():
        return
    try:
        data = path.read_bytes()
    except OSError:
        failures.append(f"screenshot_unreadable:{artifact_key}")
        return
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 64:
        failures.append(f"invalid_png:{artifact_key}")


def _real_ui_chat_has_raw_parser_leak(chat: dict[str, Any]) -> bool:
    visible_parts: list[str] = []
    final_visible = chat.get("finalVisibleText")
    if isinstance(final_visible, str):
        visible_parts.append(final_visible)
    turns = chat.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            content = turn.get("content")
            if isinstance(content, str):
                visible_parts.append(content)
    return bool(RAW_REAL_UI_PARSER_LEAK_RE.search("\n".join(visible_parts)))


def _real_ui_multi_turn_chat_ok(
    proof: dict[str, Any],
    chat: dict[str, Any],
) -> bool:
    turns = chat.get("turns")
    if not isinstance(turns, list):
        turns = proof.get("turns")
    if not isinstance(turns, list):
        return False
    user_indices = [
        index
        for index, turn in enumerate(turns)
        if isinstance(turn, dict) and turn.get("role") == "user"
    ]
    if len(user_indices) < 2:
        return False
    assistant_after_user = 0
    for user_index_index, user_index in enumerate(user_indices):
        next_user_index = (
            user_indices[user_index_index + 1]
            if user_index_index + 1 < len(user_indices)
            else len(turns)
        )
        if any(
            isinstance(turn, dict)
            and turn.get("role") == "assistant"
            and isinstance(turn.get("content"), str)
            and turn.get("content", "").strip()
            for turn in turns[user_index + 1 : next_user_index]
        ):
            assistant_after_user += 1
    return assistant_after_user >= 2


REASONING_NUMERIC_GARBAGE_RE = re.compile(
    r"(?:^|[\s([{,;:])(?:\d{1,4}[\s,;:|\-/.]+){8,}\d{1,4}(?=$|[\s)\]},;:.])",
    re.MULTILINE,
)


def _real_ui_reasoning_text(proof: dict[str, Any], chat: dict[str, Any]) -> str:
    explicit = chat.get("reasoningText")
    if isinstance(explicit, str):
        return explicit
    persisted = proof.get("persistedReasoningText")
    if isinstance(persisted, str):
        return persisted
    groups = proof.get("persistedReasoningByMessage")
    chunks: list[str] = []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, list):
                continue
            for segment in group:
                if not isinstance(segment, dict):
                    continue
                text = segment.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks)


def _real_ui_reasoning_leak_counts(
    proof: dict[str, Any],
    chat: dict[str, Any],
) -> dict[str, int | bool]:
    text = _real_ui_reasoning_text(proof, chat)
    raw = chat.get("reasoningRawParserTagLeak")
    if not isinstance(raw, bool):
        raw = bool(RAW_REAL_UI_PARSER_LEAK_RE.search(text))
    cjk = chat.get("reasoningCjkLeakCount")
    if not isinstance(cjk, int):
        cjk = len(re.findall(r"[\u3400-\u9FFF]", text))
    korean = chat.get("reasoningKoreanLeakCount")
    if not isinstance(korean, int):
        korean = len(re.findall(r"[\uAC00-\uD7AF]", text))
    numeric = chat.get("reasoningNumericRunCount")
    if not isinstance(numeric, int):
        numeric = len(REASONING_NUMERIC_GARBAGE_RE.findall(text))
    return {
        "raw": raw,
        "cjk": cjk,
        "korean": korean,
        "numeric": numeric,
    }


def _real_ui_named_tool_result_count(proof: dict[str, Any]) -> int:
    groups = proof.get("persistedToolsByMessage")
    if not isinstance(groups, list):
        return 0
    count = 0
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            if (
                item.get("phase") == "result"
                and isinstance(item.get("toolName"), str)
                and item.get("toolName", "").strip()
            ):
                count += 1
    return count


def _real_ui_named_tool_error_count(proof: dict[str, Any]) -> int:
    groups = proof.get("persistedToolsByMessage")
    if not isinstance(groups, list):
        return 0
    count = 0
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            if (
                item.get("phase") == "error"
                and isinstance(item.get("toolName"), str)
                and item.get("toolName", "").strip()
            ):
                count += 1
    return count


def _real_ui_named_tool_result_message_count(proof: dict[str, Any]) -> int:
    groups = proof.get("persistedToolsByMessage")
    if not isinstance(groups, list):
        return 0
    count = 0
    for group in groups:
        if not isinstance(group, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("phase") == "result"
            and isinstance(item.get("toolName"), str)
            and item.get("toolName", "").strip()
            for item in group
        ):
            count += 1
    return count


def _real_ui_named_tool_lifecycle_message_count(proof: dict[str, Any]) -> int:
    groups = proof.get("persistedToolsByMessage")
    if not isinstance(groups, list):
        return 0
    count = 0
    for group in groups:
        if not isinstance(group, list):
            continue
        active_names: set[str] = set()
        result_names: set[str] = set()
        for item in group:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("toolName")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            phase = item.get("phase")
            if phase in {"calling", "executing"}:
                active_names.add(tool_name.strip())
            elif phase == "result":
                result_names.add(tool_name.strip())
        if active_names & result_names:
            count += 1
    return count


def _real_ui_named_tool_lifecycle_event_count(proof: dict[str, Any]) -> int:
    groups = proof.get("persistedToolsByMessage")
    if not isinstance(groups, list):
        return 0
    count = 0
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("toolName")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            if item.get("phase") in {"calling", "executing", "result"}:
                count += 1
    return count


def _real_ui_extensive_tool_churn_ok(proof: dict[str, Any]) -> bool:
    event_counts = (
        proof.get("eventCounts") if isinstance(proof.get("eventCounts"), dict) else {}
    )
    tool_events = event_counts.get("tool")
    if (
        not isinstance(tool_events, int | float)
        or tool_events < MIN_REAL_UI_EXTENSIVE_TOOL_EVENTS
    ):
        return False
    if _real_ui_named_tool_result_count(proof) < MIN_REAL_UI_EXTENSIVE_TOOL_RESULTS:
        return False
    if (
        _real_ui_named_tool_result_message_count(proof)
        < MIN_REAL_UI_EXTENSIVE_TOOL_RESULT_MESSAGES
    ):
        return False
    return (
        _real_ui_named_tool_lifecycle_message_count(proof)
        >= MIN_REAL_UI_EXTENSIVE_TOOL_RESULT_MESSAGES
    )


def _real_ui_integrated_tool_l2_cache_ok(proof: dict[str, Any]) -> bool:
    event_counts = (
        proof.get("eventCounts") if isinstance(proof.get("eventCounts"), dict) else {}
    )
    tool_events = event_counts.get("tool")
    if (
        not isinstance(tool_events, int | float)
        or tool_events < MIN_REAL_UI_EXTENSIVE_TOOL_EVENTS
    ):
        return False
    if _real_ui_named_tool_result_count(proof) < MIN_REAL_UI_EXTENSIVE_TOOL_RESULTS:
        return False
    if (
        _real_ui_named_tool_result_message_count(proof)
        < MIN_REAL_UI_EXTENSIVE_TOOL_RESULT_MESSAGES
    ):
        return False
    if _real_ui_named_tool_error_count(proof) != 0:
        return False
    return _real_ui_named_tool_probe_semantics_ok(proof)


def _real_ui_responses_delta_streaming_ok(proof: dict[str, Any]) -> bool:
    if proof.get("rendererWireApi") != "responses":
        return False
    event_counts = (
        proof.get("eventCounts") if isinstance(proof.get("eventCounts"), dict) else {}
    )
    if (event_counts.get("stream") or 0) < 2:
        return False
    traces = proof.get("streamTrace")
    if not isinstance(traces, list):
        traces = proof.get("streamTraceByMessage")
    if not isinstance(traces, list):
        return False
    qualifying_trace_ids: set[str] = set()
    qualifying_trace_count = 0
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        first = trace.get("firstFullContent")
        last = trace.get("lastFullContent")
        if (
            isinstance(first, str)
            and isinstance(last, str)
            and first
            and last
            and first != last
            and (trace.get("count") or 0) >= 2
        ):
            trace_id = trace.get("messageId")
            if isinstance(trace_id, str) and trace_id:
                qualifying_trace_ids.add(trace_id)
            else:
                qualifying_trace_count += 1
    return (len(qualifying_trace_ids) + qualifying_trace_count) >= 2


def _real_ui_responses_cache_detail_usage_ok(proof: dict[str, Any]) -> bool:
    if proof.get("rendererWireApi") != "responses":
        return False

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            cache_detail = value.get("cache_detail")
            if cache_detail is None:
                cache_detail = value.get("cacheDetail")
            cached_tokens = value.get("cached_tokens")
            if cached_tokens is None:
                cached_tokens = value.get("cachedTokens")
            if (
                isinstance(cache_detail, str)
                and cache_detail.strip()
                and isinstance(cached_tokens, (int, float))
                and cached_tokens > 0
            ):
                return True
            return any(walk(child) for child in value.values())
        if isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(proof)


def _real_ui_live_speed_floor_ok(family_id: str, proof: dict[str, Any]) -> bool:
    floors = {
        "lfm25": 100.0,
        "step37": 45.0,
    }
    floor = floors.get(family_id)
    if floor is None:
        return False
    samples = []
    structured_samples = proof.get("liveSpeedSamples")
    if isinstance(structured_samples, list):
        for sample in structured_samples:
            if not isinstance(sample, dict):
                continue
            samples.append(
                {
                    "tokens": sample.get("tokens"),
                    "live_tps": sample.get("liveTokensPerSecond"),
                    "ttft": sample.get("ttftSeconds"),
                }
            )
    lines: list[str] = []
    for key in ("appLogTail", "serverLogTail"):
        values = proof.get(key)
        if isinstance(values, list):
            lines.extend(str(value) for value in values)
    passing_samples = 0
    for sample in samples:
        tokens = sample.get("tokens")
        live_tps = sample.get("live_tps")
        ttft = sample.get("ttft")
        if (
            isinstance(tokens, (int, float))
            and isinstance(live_tps, (int, float))
            and isinstance(ttft, (int, float))
            and tokens > 0
            and live_tps >= floor
            and 0.0 < ttft <= 5.0
        ):
            passing_samples += 1
    for line in lines:
        match = REAL_UI_RESPONSE_COMPLETE_SPEED_RE.search(line)
        if not match:
            continue
        tokens = float(match.group("tokens"))
        live_tps = float(match.group("live_tps"))
        ttft = float(match.group("ttft"))
        if tokens > 0 and live_tps >= floor and 0.0 < ttft <= 5.0:
            passing_samples += 1
    return passing_samples >= 2


def _real_ui_generation_defaults_applied_ok(proof: dict[str, Any]) -> bool:
    chat_overrides = (
        proof.get("chatOverrides")
        if isinstance(proof.get("chatOverrides"), dict)
        else {}
    )
    request_contract = (
        proof.get("requestContract")
        if isinstance(proof.get("requestContract"), dict)
        else {}
    )
    sampling_overrides = (
        request_contract.get("samplingOverrides")
        if isinstance(request_contract.get("samplingOverrides"), dict)
        else {}
    )
    for sampler_field in ("temperature", "topP", "topK", "minP", "repeatPenalty"):
        expected_sampler = sampling_overrides.get(sampler_field)
        actual_sampler = chat_overrides.get(sampler_field)
        if expected_sampler is None:
            if actual_sampler is not None:
                return False
        elif actual_sampler != expected_sampler:
            return False
    request_max_tokens = request_contract.get("requestMaxTokens")
    override_max_tokens = chat_overrides.get("maxTokens")
    if (
        isinstance(request_max_tokens, (int, float))
        and isinstance(override_max_tokens, (int, float))
        and int(request_max_tokens) != int(override_max_tokens)
    ):
        return False
    server_log_tail = proof.get("serverLogTail")
    log_text = (
        "\n".join(str(line) for line in server_log_tail)
        if isinstance(server_log_tail, list)
        else ""
    )
    if "Resolved sampling kwargs route=" not in log_text:
        return False
    if "kwargs=" not in log_text or "max_tokens" not in log_text:
        return False
    renderer_wire_api = proof.get("rendererWireApi")
    if renderer_wire_api == "responses":
        return "/v1/responses" in log_text
    if renderer_wire_api == "chat":
        return "/v1/chat/completions" in log_text
    return bool(renderer_wire_api)


def _real_ui_ssm_companion_l2_seen(proof: dict[str, Any]) -> bool:
    cache = proof.get("cache") if isinstance(proof.get("cache"), dict) else {}
    cache_after = cache.get("after") if isinstance(cache.get("after"), dict) else {}
    companion = (
        cache_after.get("ssm_companion")
        if isinstance(cache_after.get("ssm_companion"), dict)
        else {}
    )
    disk = companion.get("disk") if isinstance(companion.get("disk"), dict) else {}
    totals = (
        cache_after.get("cache_totals")
        if isinstance(cache_after.get("cache_totals"), dict)
        else {}
    )
    if disk.get("enabled") is True:
        for key in ("entries", "total_tokens_on_disk", "total_cached_tokens", "stores"):
            value = disk.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return True
    for key in ("l2_ssm_tokens_on_disk", "ssm_tokens_on_disk"):
        value = totals.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _real_ui_mixed_swa_storage_quantization_seen(
    native: dict[str, Any],
    proof: dict[str, Any],
) -> bool:
    storage = (
        native.get("storage_quantization")
        if isinstance(native.get("storage_quantization"), dict)
        else {}
    )
    if (
        storage.get("enabled") is True
        and storage.get("mode") == "storage_boundary"
        and storage.get("bits") == 4
        and storage.get("group_size") == 64
        and storage.get("applies_to") == "full_attention_kv_only"
        and storage.get("metadata_policy") == "preserve_rotating_window_metadata"
    ):
        return True

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            cache_detail = str(value.get("cache_detail") or "")
            if (
                cache_detail == "paged+mixed_swa"
                and value.get("reconstructed") is True
                and value.get("dequantized") is True
                and value.get("reconstruction_ok") is True
                and value.get("dequantization_ok") is True
            ):
                return True
            return any(walk(child) for child in value.values())
        if isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(proof)


def _real_ui_architecture_cache_policy_ok(
    family_id: str,
    proof: dict[str, Any],
) -> bool:
    server = proof.get("server") if isinstance(proof.get("server"), dict) else {}
    health = server.get("health") if isinstance(server.get("health"), dict) else {}
    native = (
        health.get("native_cache")
        if isinstance(health.get("native_cache"), dict)
        else {}
    )
    kv_cache_quantization = (
        health.get("kv_cache_quantization")
        if isinstance(health.get("kv_cache_quantization"), dict)
        else {}
    )
    components = {
        str(component)
        for component in native.get("components", [])
        if isinstance(component, str)
    }
    generic_tq = (
        native.get("generic_turboquant_kv")
        if isinstance(native.get("generic_turboquant_kv"), dict)
        else {}
    )
    if family_id == "lfm25":
        attention_storage = (
            native.get("attention_kv_storage_quantization")
            if isinstance(native.get("attention_kv_storage_quantization"), dict)
            else {}
        )
        return (
            native.get("family") in {"lfm2_moe", "lfm2"}
            and native.get("schema") == "hybrid_ssm_v1"
            and native.get("cache_type") == "hybrid_ssm_typed"
            and {"attention_kv", "ssm_companion_state", "async_rederive"}
            <= components
            and generic_tq.get("enabled") is False
            and generic_tq.get("reason") == "hybrid_ssm_state"
            and attention_storage.get("enabled") is True
            and attention_storage.get("mode") == "storage_boundary"
            and attention_storage.get("bits") == 4
            and attention_storage.get("group_size") == 64
            and attention_storage.get("applies_to") == "attention_kv_layers_only"
            and "native" in str(attention_storage.get("ssm_policy") or "")
            and kv_cache_quantization.get("enabled") is True
            and kv_cache_quantization.get("bits") == 4
            and kv_cache_quantization.get("group_size") == 64
            and _real_ui_ssm_companion_l2_seen(proof)
        )
    if family_id == "step37":
        return (
            native.get("family") in {"mixed_attention", "step3p7", "step-3.7-flash"}
            and native.get("schema") == "mixed_swa_kv_v1"
            and native.get("cache_type") == "mixed_swa_kv"
            and {
                "full_attention_kv",
                "sliding_window_kv",
                "rotating_window_metadata",
            }
            <= components
            and generic_tq.get("enabled") is False
            and generic_tq.get("reason") in {"not_active", "mixed_swa_kv"}
            and kv_cache_quantization.get("enabled") is True
            and kv_cache_quantization.get("bits") == 4
            and kv_cache_quantization.get("group_size") == 64
            and _real_ui_mixed_swa_storage_quantization_seen(native, proof)
        )
    return False


REAL_UI_TOOL_ONE_TOKEN_RE = (
    r"real[\s_\\-]*ui[\s_\\-]*live[\s_\\-]*tool[\s_\\-]*one"
)
REAL_UI_TOOL_PROBE_2_RE = (
    r"real[\s_\\-]*ui[\s_\\-]*tool[\s_\\-]*probe[\s_\\-]*2\.txt"
)
REAL_UI_SECOND_FILE_RE = r"second\s+(?:ui\s+)?(?:output\s+)?file"
REAL_UI_TOOL_TWO_VISIBLE_CONTRADICTION_RE = re.compile(
    rf"(?:{REAL_UI_TOOL_PROBE_2_RE}|{REAL_UI_SECOND_FILE_RE})"
    rf".{{0,180}}(?:contains?|content|with|has|holds?|is|:)"
    rf".{{0,100}}{REAL_UI_TOOL_ONE_TOKEN_RE}",
    re.IGNORECASE | re.DOTALL,
)
REAL_UI_TOOL_TWO_MALFORMED_TOKEN_RE = re.compile(
    r"\bre\s*:\s*al[\s_\\-]*ui[\s_\\-]*live[\s_\\-]*tool[\s_\\-]*two\b",
    re.IGNORECASE,
)
REAL_UI_EXACT_REPLY_RE = re.compile(
    r"reply exactly:\s*[\"'“”`]?([A-Za-z0-9_=-]+)[\"'“”`]?",
    re.IGNORECASE,
)


def _real_ui_named_tool_probe_visible_semantics_ok(proof: dict[str, Any]) -> bool:
    chat = proof.get("chat") if isinstance(proof.get("chat"), dict) else {}
    turns = chat.get("turns") if isinstance(chat.get("turns"), list) else []
    for turn in turns:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        content = str(turn.get("content", ""))
        if (
            REAL_UI_TOOL_TWO_VISIBLE_CONTRADICTION_RE.search(content)
            or REAL_UI_TOOL_TWO_MALFORMED_TOKEN_RE.search(content)
        ):
            return False
    return True


def _real_ui_named_tool_probe_exact_reply_ok(proof: dict[str, Any]) -> bool:
    chat = proof.get("chat") if isinstance(proof.get("chat"), dict) else {}
    turns = chat.get("turns") if isinstance(chat.get("turns"), list) else []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        match = REAL_UI_EXACT_REPLY_RE.search(str(turn.get("content", "")))
        if not match:
            continue
        expected = match.group(1)
        actual = ""
        for candidate in turns[index + 1 :]:
            if isinstance(candidate, dict) and candidate.get("role") == "assistant":
                actual = str(candidate.get("content", "")).strip()
                break
        if actual != expected:
            return False
    return True


def _real_ui_named_tool_probe_semantics_ok(proof: dict[str, Any]) -> bool:
    chat = proof.get("chat") if isinstance(proof.get("chat"), dict) else {}
    turns = chat.get("turns") if isinstance(chat.get("turns"), list) else []
    turns_text = "\n".join(
        str(turn.get("content", ""))
        for turn in turns
        if isinstance(turn, dict)
    )
    if (
        "REAL_UI_LIVE_TOOL_ONE" not in turns_text
        and "REAL_UI_LIVE_TOOL_TWO" not in turns_text
    ):
        return True

    groups = proof.get("persistedToolsByMessage")
    if not isinstance(groups, list):
        return False
    result_details: list[str] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        details = []
        for item in group:
            if not isinstance(item, dict):
                continue
            if (
                item.get("phase") == "result"
                and isinstance(item.get("toolName"), str)
                and item.get("toolName", "").strip()
            ):
                details.append(
                    str(
                        item.get("detail")
                        or item.get("message")
                        or item.get("text")
                        or ""
                    )
                )
        result_details.append("\n".join(details))
    files = proof.get("toolProbeFiles") if isinstance(proof.get("toolProbeFiles"), dict) else {}
    file_semantics_ok = (
        str(files.get("real_ui_tool_probe_1.txt", "")).rstrip()
        == "REAL_UI_LIVE_TOOL_ONE"
        and str(files.get("real_ui_tool_probe_2.txt", "")).rstrip()
        == "REAL_UI_LIVE_TOOL_TWO"
    )
    command_semantics_ok = any(
        "real_ui_tool_probe_1.txt" in detail
        or "REAL_UI_LIVE_TOOL_ONE" in detail
        for detail in result_details
    ) and any(
        "real_ui_tool_probe_2.txt" in detail
        or "REAL_UI_LIVE_TOOL_TWO" in detail
        for detail in result_details
    )
    return (
        command_semantics_ok
        and file_semantics_ok
        and _real_ui_named_tool_probe_visible_semantics_ok(proof)
        and _real_ui_named_tool_probe_exact_reply_ok(proof)
    )


def _real_ui_cache_reconstruction_clean(proof: dict[str, Any]) -> bool:
    saw_hybrid_miss_at = -1
    saw_hybrid_hit_at = -1
    offset = 0
    for key in ("serverLogTail", "appLogTail"):
        lines = proof.get(key)
        if not isinstance(lines, list):
            continue
        for i, line in enumerate(lines):
            text = str(line)
            if (
                "worker-side paged cache reconstruction failed" in text
                or "reconstruction failed, treating as cache miss" in text
                or "no usable SSM companion" in text
            ):
                if "hybrid paged MISS" in text or "no usable SSM companion" in text:
                    saw_hybrid_miss_at = offset + i
                else:
                    return False
            if "hybrid paged HIT" in text:
                saw_hybrid_hit_at = offset + i
        offset += len(lines)
    health = (
        (proof.get("server") or {}).get("health")
        if isinstance(proof.get("server"), dict)
        else {}
    )
    scheduler = health.get("scheduler") if isinstance(health, dict) else {}
    last_execution = (
        scheduler.get("last_cache_execution") if isinstance(scheduler, dict) else {}
    )
    if (
        isinstance(last_execution, dict)
        and str(last_execution.get("cache_detail") or "").startswith("paged+ssm")
        and last_execution.get("reconstruction_ok") is True
    ):
        saw_hybrid_hit_at = max(saw_hybrid_hit_at, offset)
    if saw_hybrid_miss_at >= 0 and saw_hybrid_hit_at < saw_hybrid_miss_at:
        return False
    return True


def _real_ui_l2_disk_storage_seen(proof: dict[str, Any]) -> bool:
    cache = proof.get("cache") if isinstance(proof.get("cache"), dict) else {}
    cache_after = cache.get("after") if isinstance(cache.get("after"), dict) else {}
    block_disk = (
        cache_after.get("block_disk_cache")
        if isinstance(cache_after.get("block_disk_cache"), dict)
        else {}
    )
    totals = (
        cache_after.get("cache_totals")
        if isinstance(cache_after.get("cache_totals"), dict)
        else {}
    )
    for data, keys in (
        (
            block_disk,
            (
                "blocks_on_disk",
                "total_tokens_on_disk",
                "total_cached_tokens",
                "disk_writes",
            ),
        ),
        (
            totals,
            (
                "l2_tokens_on_disk",
                "l2_block_tokens_on_disk",
                "l2_ssm_tokens_on_disk",
                "l2_tokens_on_disk_store_sum",
            ),
        ),
    ):
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return True
    return False


def _real_ui_model_id_matches(row: dict[str, Any], model_id: Any) -> bool:
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    model_id = model_id.strip()
    model_name = str(row.get("model_name") or "")
    model_path = str(row.get("model_path") or "")
    path_name = Path(model_path).name if model_path else ""
    return (
        model_id == model_name
        or model_id == path_name
        or model_id.endswith(f"/{model_name}")
        or (path_name and model_id.endswith(f"/{path_name}"))
    )


def _validate_real_ui_request_contract(
    row_id: str,
    proof: dict[str, Any],
    request_contract: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    chat_overrides = (
        proof.get("chatOverrides")
        if isinstance(proof.get("chatOverrides"), dict)
        else {}
    )
    server_cache_controls = (
        proof.get("serverCacheControls")
        if isinstance(proof.get("serverCacheControls"), dict)
        else {}
    )
    media = proof.get("media") if isinstance(proof.get("media"), dict) else {}

    def mismatch(field: str) -> None:
        failures.append(f"request_contract_mismatch:{row_id}:{field}")

    expected_wire_api = request_contract.get("wireApi")
    if proof.get("requestedWireApi") not in {None, expected_wire_api}:
        mismatch("requestedWireApi")
    if proof.get("rendererWireApi") not in {None, expected_wire_api}:
        mismatch("rendererWireApi")
    if chat_overrides.get("wireApi") not in {None, expected_wire_api}:
        mismatch("chatOverrides.wireApi")

    expected_builtin_tools = request_contract.get("builtinToolsEnabled")
    if proof.get("requestedBuiltinTools") not in {None, expected_builtin_tools}:
        mismatch("requestedBuiltinTools")
    if chat_overrides.get("builtinToolsEnabled") not in {None, expected_builtin_tools}:
        mismatch("chatOverrides.builtinToolsEnabled")

    expected_enable_thinking = request_contract.get("enableThinking")
    if proof.get("requestedEnableThinking") not in {None, expected_enable_thinking}:
        mismatch("requestedEnableThinking")

    expected_max_tokens = request_contract.get("requestMaxTokens")
    if chat_overrides.get("maxTokens") not in {None, expected_max_tokens}:
        mismatch("maxTokens")
    expected_max_tool_iterations = request_contract.get("maxToolIterations")
    if chat_overrides.get("maxToolIterations") not in {
        None,
        expected_max_tool_iterations,
    }:
        mismatch("maxToolIterations")
    expected_tool_result_max_chars = request_contract.get("toolResultMaxChars")
    if chat_overrides.get("toolResultMaxChars") not in {
        None,
        expected_tool_result_max_chars,
    }:
        mismatch("toolResultMaxChars")

    sampling_overrides = (
        request_contract.get("samplingOverrides")
        if isinstance(request_contract.get("samplingOverrides"), dict)
        else {}
    )
    for sampler_field in ("temperature", "topP", "topK", "minP", "repeatPenalty"):
        expected_sampler = sampling_overrides.get(sampler_field)
        actual_sampler = chat_overrides.get(sampler_field)
        if expected_sampler is None and actual_sampler is not None:
            mismatch(f"unexpectedSamplerOverride.{sampler_field}")
        elif expected_sampler is not None and actual_sampler != expected_sampler:
            mismatch(f"samplerOverride.{sampler_field}")

    expected_cache_controls = request_contract.get("checkServerCacheControls")
    if proof.get("requestedServerCacheControls") not in {
        None,
        expected_cache_controls,
    }:
        mismatch("requestedServerCacheControls")
    if expected_cache_controls is True and server_cache_controls.get("requested") is False:
        mismatch("serverCacheControls.requested")

    if proof.get("requestedMedia") not in {None, request_contract.get("checkMedia")}:
        mismatch("requestedMedia")
    if proof.get("requestedVideo") not in {None, request_contract.get("checkVideo")}:
        mismatch("requestedVideo")
    if request_contract.get("checkMedia") is True and media.get("imageVerified") is not True:
        mismatch("media.imageVerified")
    if request_contract.get("checkVideo") is True and media.get("videoVerified") is not True:
        mismatch("media.videoVerified")

    return failures


def _real_ui_proof_is_installed_app(proof: dict[str, Any]) -> bool:
    surfaces = proof.get("provenSurfaces")
    return (
        proof.get("uiLaunchMode") == "installed-app"
        or (
            isinstance(surfaces, list)
            and "installed_app_ui" in {str(item) for item in surfaces}
        )
    )


def _real_ui_electron_dev_launch_proven(proof: dict[str, Any]) -> tuple[bool, bool]:
    app_log_tail = proof.get("appLogTail")
    app_log_text = "\n".join(app_log_tail) if isinstance(app_log_tail, list) else ""
    ui_command = proof.get("uiCommand")
    command_text = (
        " ".join(str(item) for item in ui_command) if isinstance(ui_command, list) else ""
    )
    structured_dev_launch = (
        proof.get("uiLaunchMode") == "electron-dev"
        and "npm" in command_text
        and "dev" in command_text
        and "--remote-debugging-port=" in command_text
    )
    return (
        "http://localhost:5173/" in app_log_text or structured_dev_launch,
        "start electron app" in app_log_text or structured_dev_launch,
    )


def _validate_current_issue179_minimax_k_root_cause_audit(root: Path) -> dict[str, Any]:
    artifact = CURRENT_ISSUE179_MINIMAX_K_ROOT_CAUSE_AUDIT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
        "issues": {},
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status not in {"open", "pass"}:
        result["failures"].append("invalid_issue179_root_cause_audit_status")

    proven = payload.get("proven")
    if not isinstance(proven, dict):
        proven = {}
    required_true = {
        "reporter_log_installed_app_bundled_python_seen",
        "reporter_log_request_shape_and_sampling_kwargs_seen",
        "reporter_log_launch_parser_cache_flags_seen",
        "reporter_log_has_abort_before_visible_content",
        "reporter_log_has_responses_cancel_404",
        "local_real_ui_diagnostics_clean",
        "local_installed_issue179_session_settings_parity",
        "local_installed_bundle_has_responses_cancel_route",
        "public_v1549_tahoe_dmg_has_responses_cancel_route",
        "latest_public_dmg_has_responses_cancel_route",
        "local_installed_responses_cancel_live_probe",
        "current_source_responses_cancel_contract_proven",
        "current_source_responses_cancel_inactive_404_contract_proven",
        "reporter_cancel_404_after_stream_abort_order_proven",
        "reporter_cancel_404_after_econnreset_same_response_id_proven",
        "local_reporter_prompt_reproduction_clean",
    }
    for key in sorted(required_true):
        if proven.get(key) is not True:
            result["failures"].append(f"missing_proven:{key}")
    result["proven"] = {
        key: proven.get(key)
        for key in sorted(required_true)
    }

    not_proven = payload.get("not_proven")
    if not isinstance(not_proven, list):
        not_proven = []
    reporter_server_hash_parity = payload.get("reporter_server_hash_parity")
    server_hash_drift_open = (
        isinstance(reporter_server_hash_parity, dict)
        and reporter_server_hash_parity.get("status") == "open"
        and reporter_server_hash_parity.get("failure")
        == "reporter_installed_server_hash_drift"
    )
    required_not_proven = set()
    if status != "pass":
        if server_hash_drift_open:
            required_not_proven.add(
                "reporter installed app bundle hash matches public/local server.py route proof"
            )
        else:
            required_not_proven.add(
                "a concrete prompt reproduces screenshot-shaped wrong-language or numeric garbage"
            )
    missing_not_proven = required_not_proven.difference({str(item) for item in not_proven})
    for item in sorted(missing_not_proven):
        result["failures"].append(f"missing_not_proven:{item}")
    if status == "pass" and not_proven:
        result["failures"].append("issue179_pass_has_not_proven_items")
    result["not_proven"] = [str(item) for item in not_proven]
    result["release_boundary"] = payload.get("release_boundary")

    local_reporter_prompt = payload.get("local_reporter_prompt_reproduction")
    if not isinstance(local_reporter_prompt, dict):
        result["failures"].append("missing_local_reporter_prompt_reproduction")
    elif status == "pass":
        if local_reporter_prompt.get("clean") is not True:
            result["failures"].append("local_reporter_prompt_reproduction_not_clean")
        if local_reporter_prompt.get("request_matches_reporter") is not True:
            result["failures"].append("local_reporter_prompt_request_mismatch")
        if local_reporter_prompt.get("bad_text_captured") is not False:
            result["failures"].append("local_reporter_prompt_bad_text_captured")
    result["local_reporter_prompt_reproduction"] = local_reporter_prompt

    local_cancel_probe = payload.get("local_responses_cancel_probe")
    if not isinstance(local_cancel_probe, dict):
        result["failures"].append("missing_local_responses_cancel_probe")
    else:
        if local_cancel_probe.get("status") != "pass":
            result["failures"].append("local_responses_cancel_probe_not_pass")
        if local_cancel_probe.get("response_id_seen") is not True:
            result["failures"].append("local_responses_cancel_probe_missing_response_id")
        if local_cancel_probe.get("cancel_status") != 200:
            result["failures"].append("local_responses_cancel_probe_cancel_not_200")
        probe = local_cancel_probe.get("probe")
        if not isinstance(probe, dict):
            result["failures"].append("local_responses_cancel_probe_missing_probe")
        else:
            if probe.get("cancel_route_present") is not True:
                result["failures"].append("local_responses_cancel_route_not_present")
            if probe.get("bad_text_captured") is not False:
                result["failures"].append("local_responses_cancel_bad_text_captured")
    result["local_responses_cancel_probe"] = local_cancel_probe

    reporter_parity = payload.get("reporter_parity_artifact")
    required_reporter_parity_fields = {
        "capture_provenance",
        "installed_server_sha256",
        "server_has_responses_cancel_route",
        "server_cancel_calls_engine_abort",
        "model_manifest_sha256",
        "model_file_hashes",
        "chat_id",
        "session_settings",
        "response_id",
        "response_active_at_cancel",
        "raw_sse_cancel_lifecycle",
    }
    if not isinstance(reporter_parity, dict):
        result["failures"].append("missing_reporter_parity_artifact_contract")
    else:
        fields = reporter_parity.get("required_fields")
        if not isinstance(fields, list) or required_reporter_parity_fields.difference(
            {str(field) for field in fields}
        ):
            result["failures"].append("missing_reporter_parity_required_fields")
        if reporter_parity.get("comparison_status") not in {
            "missing_reporter_parity_artifact",
            "missing_required_fields",
            "collector_missing_fields",
            "collector_status_open",
            "ready_for_direct_comparison",
        }:
            result["failures"].append("invalid_reporter_parity_comparison_status")
        if (
            reporter_parity.get("comparison_status") == "collector_missing_fields"
            and not isinstance(reporter_parity.get("collector_missing_fields"), list)
        ):
            result["failures"].append("missing_reporter_parity_collector_missing_fields")
    result["reporter_parity_artifact"] = reporter_parity

    reporter_parity_comparison = payload.get("reporter_parity_comparison")
    if not isinstance(reporter_parity_comparison, dict):
        result["failures"].append("missing_reporter_parity_comparison")
    else:
        comparison_status = reporter_parity_comparison.get("status")
        comparison_failures = reporter_parity_comparison.get("failures")
        if comparison_status not in {"open", "pass"}:
            result["failures"].append("invalid_reporter_parity_comparison_status")
        if comparison_status == "open" and not isinstance(comparison_failures, list):
            result["failures"].append("missing_reporter_parity_comparison_failures")
        if comparison_status == "pass":
            required_comparison_true = (
                "capture_provenance_is_reporter_machine",
                "server_hash_matches_local_installed",
                "server_route_markers_match",
                "model_manifest_sha256_matches_local",
                "model_file_hashes_match_local",
                "chat_id_matches_reporter_log",
                "response_id_matches_reporter_log",
                "response_active_at_cancel_recorded",
                "raw_sse_cancel_lifecycle_present",
            )
            for key in required_comparison_true:
                if reporter_parity_comparison.get(key) is not True:
                    result["failures"].append(
                        f"missing_reporter_parity_comparison_check:{key}"
                    )
    result["reporter_parity_comparison"] = reporter_parity_comparison

    if not isinstance(reporter_server_hash_parity, dict):
        result["failures"].append("missing_reporter_server_hash_parity")
    else:
        parity_status = reporter_server_hash_parity.get("status")
        if parity_status not in {"open", "pass"}:
            result["failures"].append("invalid_reporter_server_hash_parity_status")
        if reporter_server_hash_parity.get("route_markers_match") is not True:
            result["failures"].append("reporter_server_route_markers_mismatch")
        if parity_status == "open" and not reporter_server_hash_parity.get("failure"):
            result["failures"].append("missing_reporter_server_hash_parity_failure")
        if parity_status == "pass":
            for key in (
                "reporter_matches_source",
                "reporter_matches_local_installed",
            ):
                if reporter_server_hash_parity.get(key) is not True:
                    result["failures"].append(
                        f"missing_reporter_server_hash_parity_check:{key}"
                    )
        provenance = reporter_server_hash_parity.get("provenance")
        if not isinstance(provenance, dict):
            result["failures"].append("missing_reporter_server_hash_provenance")
        else:
            provenance_status = provenance.get("status")
            if provenance_status not in {"missing", "open", "pass"}:
                result["failures"].append("invalid_reporter_server_hash_provenance_status")
            checked_sources = provenance.get("checked_sources")
            required_sources = {
                "source_contract",
                "local_installed_bundle",
                "public_v1549_tahoe_dmg",
                "public_release_dmg_contracts",
                "local_installed_app_backups",
                "sibling_python_worktrees",
                "git_history",
            }
            if not isinstance(checked_sources, list) or required_sources.difference(
                {str(source) for source in checked_sources}
            ):
                result["failures"].append("missing_reporter_server_hash_provenance_sources")
            if provenance_status == "open" and not provenance.get("failure"):
                result["failures"].append("missing_reporter_server_hash_provenance_failure")
            if (
                provenance_status == "open"
                and provenance.get("failure")
                == "reporter_server_hash_provenance_unknown"
            ):
                checked_count = provenance.get("public_release_checked_count")
                if not isinstance(checked_count, int) or checked_count < 12:
                    result["failures"].append(
                        "insufficient_public_release_dmg_contract_coverage"
                    )
    result["reporter_server_hash_parity"] = reporter_server_hash_parity

    return result


def _validate_current_step37_vlm_runtime_audit(root: Path) -> dict[str, Any]:
    path = root / CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT
    result: dict[str, Any] = {
        "artifact": CURRENT_STEP37_VLM_RUNTIME_AUDIT_ARTIFACT,
        "status": "missing",
        "failures": [],
    }
    if not path.exists():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    failures: list[str] = []
    if payload.get("status") not in {"open", "pass"}:
        failures.append("invalid_status")
    if payload.get("model_type") != "step3p7":
        failures.append("model_type_not_step3p7")
    if payload.get("text_model_type") != "step3p5":
        failures.append("text_model_type_not_step3p5")
    if payload.get("has_vision_config") is not True:
        failures.append("vision_config_missing")
    if payload.get("jang_has_vision") is not True:
        failures.append("jang_has_vision_missing")
    if payload.get("root_cause") == "missing_mlx_vlm_step3p7_vlm_runtime":
        if payload.get("mlx_vlm_step3p7_runtime_available") is True:
            failures.append("missing_runtime_root_cause_with_available_runtime")
        if payload.get("local_bridge_kind") != "text_only":
            failures.append("missing_runtime_root_cause_without_text_only_bridge")
        if payload.get("release_clearance") != "blocked_until_real_step3p7_vlm_runtime":
            failures.append("missing_runtime_without_release_blocker")
        if "mlx_vlm.models.step3p7" not in str(payload.get("failure_summary") or ""):
            failures.append("missing_runtime_failure_summary_absent")
        local_reference = payload.get("local_reference_vlm_runtime")
        if not isinstance(local_reference, dict):
            local_reference = {}
        for key in (
            "has_modeling_step3p7",
            "has_processing_step3",
            "has_vision_encoder",
            "has_configuration_step3p7",
            "has_step3p7_model",
            "has_conditional_generation",
            "has_step3_vl_processor",
            "has_multimodal_merge",
            "has_patch_image_inputs",
            "has_2d_vision_rope",
            "has_full_and_sliding_attention_masks",
            "has_qk_norms",
            "has_head_wise_attention_gate",
            "has_step3p7_config",
        ):
            if local_reference.get(key) is not True:
                failures.append(f"missing_step37_reference_runtime_marker:{key}")
        if local_reference.get("runtime_kind") != "torch_reference_not_mlx":
            failures.append("step37_reference_runtime_kind_not_torch_only")
        step_jangtq_status = payload.get("step_jangtq_status")
        if not isinstance(step_jangtq_status, dict):
            step_jangtq_status = {}
        if (
            step_jangtq_status.get("step_jangtq_available") is not False
            or step_jangtq_status.get("quantization_profile") != "JANG_2L"
            or step_jangtq_status.get("step_jangtq_reason")
            != "step_jangtq_not_made_current_artifact_is_jang_2l"
        ):
            failures.append("missing_step37_jangtq_absence_marker")
        implementation_contract = payload.get("mlx_vlm_implementation_contract")
        if not isinstance(implementation_contract, dict):
            implementation_contract = {}
        required_contract_capabilities = {
            "image_patch_processing",
            "2d_vision_rope",
            "full_and_sliding_attention_cache",
            "head_wise_attention_gate",
            "multimodal_embedding_merge",
        }
        contract_capabilities = {
            str(item)
            for item in implementation_contract.get("required_capabilities", [])
        }
        rejected_paths = {
            str(item)
            for item in implementation_contract.get(
                "explicitly_rejected_clearance_paths", []
            )
        }
        if (
            implementation_contract.get("closest_reference_packages")
            != [
                "mlx_vlm.models.qwen3_vl_moe",
                "mlx_vlm.models.lfm2_vl",
                "mlx_vlm.models.gemma4",
            ]
            or implementation_contract.get("required_module_files")
            != [
                "__init__.py",
                "config.py",
                "language.py",
                "step3p7.py",
                "vision.py",
                "processing_step3p7.py",
            ]
            or not required_contract_capabilities.issubset(contract_capabilities)
            or "text_only_bridge" not in rejected_paths
            or "importable_stub" not in rejected_paths
        ):
            failures.append("missing_step37_mlx_vlm_implementation_contract")
    elif (
        payload.get("status") == "pass"
        or payload.get("release_clearance") == "audit_does_not_block_release"
    ):
        live_media_proof = payload.get("live_media_proof")
        if not isinstance(live_media_proof, dict):
            live_media_proof = {}
        required_labels = {
            "text_cache_repeat_1",
            "text_cache_repeat_2",
            "text_multiturn_recall",
            "tool_required",
            "tool_result_continuation",
            "structured_json_exact",
            "exact_code_whitespace",
            "vl_blue_image",
            "text_no_media_after_image",
            "vl_blue_image_repeat",
            "vl_red_image_changed",
            "vl_blue_video",
            "text_no_media_after_video",
        }
        present_labels = {
            str(item) for item in live_media_proof.get("present_labels") or []
        }
        if (
            live_media_proof.get("pass") is not True
            or live_media_proof.get("status") != "pass"
            or live_media_proof.get("failed") != 0
            or live_media_proof.get("validation_failures") not in ([], None)
            or not required_labels.issubset(present_labels)
        ):
            failures.append("missing_step37_live_media_proof")

    result.update(
        {
            "status": "fail" if failures else str(payload.get("status")),
            "failures": failures,
            "root_cause": payload.get("root_cause"),
            "failure_summary": payload.get("failure_summary"),
            "release_clearance": payload.get("release_clearance"),
            "model_path": payload.get("model_path"),
            "local_bridge_kind": payload.get("local_bridge_kind"),
            "mlx_vlm_step3p7_importable": payload.get("mlx_vlm_step3p7_importable"),
            "mlx_vlm_step3p7_runtime_available": payload.get(
                "mlx_vlm_step3p7_runtime_available"
            ),
            "mlx_vlm_step3p7_runtime": payload.get("mlx_vlm_step3p7_runtime"),
            "local_reference_vlm_runtime": payload.get(
                "local_reference_vlm_runtime"
            ),
            "step_jangtq_status": payload.get("step_jangtq_status"),
            "mlx_vlm_implementation_contract": payload.get(
                "mlx_vlm_implementation_contract"
            ),
            "source_owned_runtime_progress": payload.get(
                "source_owned_runtime_progress"
            ),
            "live_media_proof": payload.get("live_media_proof"),
        }
    )
    return result


def _validate_current_issue175_179_release_boundary_audit(root: Path) -> dict[str, Any]:
    artifact = CURRENT_ISSUE175_179_RELEASE_BOUNDARY_AUDIT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status not in {"open", "pass"}:
        result["failures"].append("unexpected_issues175_179_boundary_status")
    issues = payload.get("issues")
    if not isinstance(issues, dict):
        issues = {}
    result["issues"] = issues
    expected_slices = {
        "175": "pass",
        "176": "pass",
        "177": "pass",
        "178": "pass",
        "179": {"open", "pass"},
    }
    for number, expected in expected_slices.items():
        issue = issues.get(number)
        if not isinstance(issue, dict):
            result["failures"].append(f"missing_issue:{number}")
            continue
        expected_values = expected if isinstance(expected, set) else {expected}
        if issue.get("focused_source_slice") not in expected_values:
            result["failures"].append(f"unexpected_issue_slice:{number}")
    required_issue_checks = {
        "175": {
            "helper_exists",
            "prefers_metal_clear_cache",
            "falls_back_to_top_level_clear_cache",
            "no_raw_removed_api_in_runtime_paths",
            "installed_app_memory_clear_runtime_proven",
            "installed_app_live_memory_stress_proven",
        },
        "176": {
            "marks_promoted_disk_blocks",
            "drops_disk_mirror_after_reconstruct",
            "regression_test_present",
            "l2_readable_write_through_regression_test_present",
            "installed_app_promoted_block_cleanup_proven",
            "installed_app_live_memory_pressure_proven",
        },
        "177": {
            "selection_telemetry_in_scheduler",
            "selection_telemetry_in_health",
            "cold_paged_reason_present",
            "focused_regression_test_present",
            "worker_timing_regression_test_present",
            "installed_app_cache_selection_telemetry_proven",
            "installed_live_ttft_and_cold_paged_tq_proven",
        },
        "178": {
            "serve_cli_lora_paths",
            "serve_cli_lora_scales",
            "image_load_lora_signature_guard",
            "focused_regression_test_present",
            "empty_lora_lists_noop_regression_test_present",
            "lora_scale_without_path_still_rejected",
            "text_lora_flags_rejected",
            "installed_app_lora_surface_proven",
        },
    }
    for number, required_checks in required_issue_checks.items():
        issue = issues.get(number)
        if not isinstance(issue, dict):
            continue
        checks = issue.get("checks")
        if not isinstance(checks, dict) or not checks:
            result["failures"].append(f"missing_issue_checks:{number}")
            checks = {}
        elif not all(value is True for value in checks.values()):
            result["failures"].append(f"failed_issue_checks:{number}")
        for check in sorted(required_checks):
            if checks.get(check) is not True:
                result["failures"].append(f"missing_issue_check:{number}:{check}")
    if status == "pass":
        expected_clearance = {
            "175": "installed_app_memory_clear_runtime_live_stress_proven",
            "176": "installed_app_promoted_block_cleanup_live_pressure_proven",
            "177": "installed_app_cache_selection_live_ttft_proven",
            "178": "packaged_app_lora_surface_proven",
            "179": "installed_app_reporter_prompt_and_cancel_boundary_proven",
        }
        for number, clearance in expected_clearance.items():
            issue = issues.get(number)
            if not isinstance(issue, dict):
                continue
            if issue.get("release_clearance") != clearance:
                result["failures"].append(f"unexpected_issue_clearance:{number}")

    open_rows = payload.get("open_release_rows")
    if not isinstance(open_rows, list):
        open_rows = []
    for row in (
        "Real Electron UI cross-family live model matrix is release-cleared",
        "DSV4 long-output/code/file-generation quality is release-cleared",
    ):
        if row not in {str(item) for item in open_rows}:
            result["failures"].append(f"missing_open_release_row:{row}")

    proof_axes = {str(item) for item in payload.get("required_live_proof_axes", [])}
    for axis in (
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
    ):
        if axis not in proof_axes:
            result["failures"].append(f"missing_required_live_proof_axis:{axis}")

    return result


def _validate_current_issue181_183_runtime_audit(root: Path) -> dict[str, Any]:
    artifact = CURRENT_ISSUE181_183_RUNTIME_AUDIT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status != "pass":
        result["failures"].append("issue181_183_runtime_audit_must_pass")
    issues = payload.get("issues")
    if not isinstance(issues, dict):
        issues = {}
    result["issues"] = issues
    expected_clearance = {
        "181": "installed_mpp_auto_policy_guarded",
        "182": "installed_qwen_vl_patch_embed_layout_guarded",
        "183": "source_and_packaged_minicpm_v46_load_guarded",
    }
    required_issue_checks = {
        "181": {
            "mpp_auto_policy_function_exists",
            "mpp_auto_disabled_for_mxtq",
            "jangtq_repo_id_disables_auto",
            "explicit_mpp_on_still_allowed",
            "server_health_reports_mpp_status",
            "installed_app_mpp_auto_policy_disables_mxtq",
        },
        "182": {
            "normal_vlm_patch_embed_transpose",
            "native_mtp_patch_embed_transpose",
            "focused_shape_regression_test_present",
            "native_mtp_shape_regression_test_present",
            "bundled_hash_gate_covers_runtime",
            "installed_app_qwen_vl_patch_embed_layout",
        },
        "183": {
            "minicpm_v46_registry_remap",
            "minicpm_v46_prompt_config_remap",
            "bundled_import_gate_covers_runtime",
            "installed_app_minicpm_v46_runtime_remap",
        },
    }
    for number, clearance in expected_clearance.items():
        issue = issues.get(number)
        if not isinstance(issue, dict):
            result["failures"].append(f"missing_issue:{number}")
            continue
        if issue.get("focused_source_slice") != "pass":
            result["failures"].append(f"unexpected_issue_slice:{number}")
        if issue.get("release_clearance") != clearance:
            result["failures"].append(f"unexpected_issue_clearance:{number}")
        checks = issue.get("checks")
        if not isinstance(checks, dict) or not checks:
            result["failures"].append(f"missing_issue_checks:{number}")
            checks = {}
        elif not all(value is True for value in checks.values()):
            result["failures"].append(f"failed_issue_checks:{number}")
        for check in sorted(required_issue_checks.get(number, set())):
            if checks.get(check) is not True:
                result["failures"].append(f"missing_issue_check:{number}:{check}")

    return result


def _validate_current_public_app_issue_audit(root: Path) -> dict[str, Any]:
    artifact = CURRENT_PUBLIC_APP_ISSUE_AUDIT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status not in {"open", "pass"}:
        result["failures"].append("public_app_issue_audit_must_pass")
    issues = payload.get("issues")
    if not isinstance(issues, dict):
        issues = {}
    result["issues"] = issues
    expected_clearance = {
        "165": "mapped_to_dsv4_dsml_tool_call_arguments_guard",
        "166": "mapped_to_gemma4_assistant_mlx_vlm_alias_guard",
        "169": (
            "installed_supported_runtime_flavor_and_dual_staged_dmgs_guarded_packaging_still_gated"
        ),
        "111": "mapped_to_mistral_small4_vlm_wrapper_detection_guard",
        "115": "mapped_to_current_installed_app_gemma_qwen_speed_gate",
        "116": "mapped_to_thinking_off_ui_api_request_guard",
        "117": {
            "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary",
            "open_minimax_k_issue179_reporter_parity_required",
        },
        "180": "mapped_to_minimax_small_real_ui_language_numeric_guard",
        "118": "installed_gui_download_endpoint_and_stale_auth_fallback_guarded",
        "119": (
            "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
        ),
    }
    required_checks = {
        "165": (
            "tool_call_contract_passes",
            "dsml_issue_165_regression_present",
            "installed_app_dsml_parser_hash_guarded",
        ),
        "166": (
            "source_gemma4_assistant_alias",
            "bundled_verify_gemma4_assistant_alias",
            "installed_app_gemma4_assistant_alias",
        ),
        "169": (
            "installed_app_supported_runtime_flavor",
            "staged_sequoia_app_compat_runtime_flavor",
            "staged_tahoe_app_native_runtime_flavor",
        ),
        "111": (
            "mistral_small4_wrapper_stays_mllm",
            "mistral_small4_parser_metadata_preserved",
            "installed_app_mllm_hash_guarded",
        ),
        "115": (
            "gemma4_installed_speed_risk_tracked",
            "gemma4_current_installed_ui_speed_gate_passes",
            "gemma4_cold_wall_includes_ttft_tracked",
            "qwen36_speed_review_tracked",
        ),
        "116": (
            "reasoning_template_contract_passes",
            "explicit_thinking_off_request_wired",
            "panel_thinking_off_control_present",
            "packaged_renderer_thinking_controls_present",
        ),
        "118": ("installed_app_download_fallback_guarded",),
        "180": (
            "minimax_small_stricttools_real_ui_indexed",
            "minimax_small_numeric_garbage_guarded",
        ),
        "119": (
            "gemma26_memory_stress_artifact_present",
            "gemma26_memory_stress_native_cache_health",
            "gemma26_memory_stress_mixed_swa_cache_hits",
        ),
    }
    for number, clearance in expected_clearance.items():
        issue = issues.get(number)
        if not isinstance(issue, dict):
            result["failures"].append(f"missing_issue:{number}")
            continue
        issue_slice = issue.get("focused_source_slice")
        issue_clearance = issue.get("release_clearance")
        issue117_known_open = (
            number == "117"
            and status == "open"
            and issue_slice == "open"
            and issue_clearance == "open_minimax_k_issue179_reporter_parity_required"
        )
        issue115_known_open = (
            number == "115"
            and status == "open"
            and issue_slice == "open"
            and issue_clearance == "mapped_to_current_installed_app_gemma_qwen_speed_gate"
        )
        issue119_known_open = (
            number == "119"
            and status == "open"
            and issue_slice == "open"
            and issue_clearance
            == "source_and_live_gemma26_memory_runtime_guarded_release_package_pending"
        )
        issue117_known_pass = (
            number == "117"
            and status == "pass"
            and issue_slice == "pass"
            and issue_clearance
            == "mapped_to_minimax_k_issue179_live_reporter_prompt_boundary"
        )
        if (
            issue_slice != "pass"
            and not issue117_known_open
            and not issue115_known_open
            and not issue119_known_open
        ):
            result["failures"].append(f"unexpected_issue_slice:{number}")
        expected_clearance_values = (
            clearance if isinstance(clearance, set) else {clearance}
        )
        if issue_clearance not in expected_clearance_values:
            result["failures"].append(f"unexpected_issue_clearance:{number}")
        checks = issue.get("checks")
        if not isinstance(checks, dict) or not checks:
            result["failures"].append(f"missing_issue_checks:{number}")
        elif issue117_known_pass:
            required_pass_checks = {
                key: value
                for key, value in checks.items()
                if key != "issue179_root_cause_audit_open"
            }
            if not all(value is True for value in required_pass_checks.values()):
                result["failures"].append(f"failed_issue_checks:{number}")
            if checks.get("issue179_root_cause_audit_passes") is not True:
                result["failures"].append(
                    f"missing_issue_check:{number}:issue179_root_cause_audit_passes"
                )
        elif issue117_known_open:
            required_open_checks = {
                key: value
                for key, value in checks.items()
                if key
                not in {
                    "issue179_root_cause_audit_passes",
                    "issue179_root_cause_audit_open",
                    "issue179_cancel_probe_present",
                }
            }
            if not all(value is True for value in required_open_checks.values()):
                result["failures"].append(f"failed_issue_checks:{number}")
            if checks.get("issue179_root_cause_audit_open") is not True:
                result["failures"].append(
                    f"missing_issue_check:{number}:issue179_root_cause_audit_open"
                )
        elif issue115_known_open:
            required_open_checks = {
                key: value
                for key, value in checks.items()
                if key
                not in {
                    "gemma4_current_installed_ui_speed_gate_passes",
                    "gemma4_cold_wall_includes_ttft_tracked",
                    "qwen35_installed_app_speed_gate_passes",
                }
            }
            if not all(value is True for value in required_open_checks.values()):
                result["failures"].append(f"failed_issue_checks:{number}")
        elif issue119_known_open:
            required_open_checks = {
                key: value
                for key, value in checks.items()
                if key
                not in {
                    "gemma26_memory_stress_artifact_present",
                    "gemma26_memory_stress_native_cache_health",
                    "gemma26_memory_stress_mixed_swa_cache_hits",
                }
            }
            if not all(value is True for value in required_open_checks.values()):
                result["failures"].append(f"failed_issue_checks:{number}")
        elif not all(value is True for value in checks.values()):
            result["failures"].append(f"failed_issue_checks:{number}")
        for check in required_checks.get(number, ()):
            if issue115_known_open and check in {
                "gemma4_current_installed_ui_speed_gate_passes",
                "gemma4_cold_wall_includes_ttft_tracked",
            }:
                continue
            if issue119_known_open and check in {
                "gemma26_memory_stress_artifact_present",
                "gemma26_memory_stress_native_cache_health",
                "gemma26_memory_stress_mixed_swa_cache_hits",
            }:
                continue
            if checks.get(check) is not True:
                result["failures"].append(f"missing_issue_check:{number}:{check}")

    return result


def _validate_current_installed_app_runtime_parity_audit(root: Path) -> dict[str, Any]:
    return _validate_app_runtime_parity_audit(
        root,
        CURRENT_INSTALLED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        allowed_statuses={"pass"},
        require_pass=True,
    )


def _validate_current_staged_app_runtime_parity_audit(root: Path) -> dict[str, Any]:
    return _validate_app_runtime_parity_audit(
        root,
        CURRENT_STAGED_APP_RUNTIME_PARITY_AUDIT_ARTIFACT,
        allowed_statuses={"pass"},
        require_pass=True,
    )


def _validate_app_runtime_parity_audit(
    root: Path,
    artifact: str,
    *,
    allowed_statuses: set[str],
    require_pass: bool,
) -> dict[str, Any]:
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status not in allowed_statuses:
        result["failures"].append("unexpected_app_parity_status")
    if require_pass and status != "pass":
        result["failures"].append("staged_app_parity_not_pass")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    for key in (
        "installed_python_exists",
        "installed_versioned_python_exists",
        "installed_versioned_python_runs",
        "installed_bundled_python_launch_crash_reports_classified",
        "installed_bundled_python_launch_crashes_not_reproduced",
        "serve_help_runs",
        "responses_cancel_route",
        "image_lora_cli_flags",
        "image_lora_load_signature",
        "image_lora_server_globals",
        "text_lora_flags_rejected",
        "xml_function_tool_parser_cli",
        "plain_attention_kv_native_cache_status",
        "installed_panel_gateway_epipe_guard",
        "installed_panel_gateway_epipe_aggregate_guard",
        "installed_panel_chat_ipc_epipe_aggregate_guard",
        "installed_panel_image_ipc_epipe_aggregate_guard",
        "installed_panel_cache_ipc_epipe_aggregate_guard",
        "installed_panel_performance_health_epipe_guard",
        "installed_panel_session_lifecycle_epipe_guard",
        "installed_panel_child_process_stdio_epipe_guard",
        "installed_panel_child_process_stdio_epipe_aggregate_guard",
        "installed_panel_renderer_chat_epipe_toast_normalized",
        "installed_panel_responses_stream_cache_detail_metrics",
        "installed_panel_gateway_guarded_proxy_forwarding",
        "installed_panel_gateway_write_once_behavior_marker",
        "installed_panel_gateway_response_socket_destroyed_guard",
        "installed_panel_gateway_request_socket_destroyed_guard",
        "installed_panel_gateway_single_model_cache_endpoint_routing",
        "installed_panel_gateway_single_model_streaming_routes",
        "installed_panel_max_output_context_settings_wired",
        "installed_panel_model_owned_generation_defaults_wired",
        "installed_panel_request_builders_keep_output_context_separate",
        "installed_panel_parser_reasoning_settings_wired",
        "installed_panel_ollama_streaming_json_writes_guarded",
        "installed_panel_ollama_proxy_response_errors_guarded",
        "installed_vmlx_user_data_no_raw_disconnect_errors",
        "installed_vmlx_diagnostic_reports_no_raw_disconnect_errors",
    ):
        if key not in checks:
            result["failures"].append(f"missing_check:{key}")

    if status == "open":
        missing_or_stale = payload.get("missing_or_stale")
        if not isinstance(missing_or_stale, list) or not missing_or_stale:
            result["failures"].append("open_app_parity_without_stale_list")
    if require_pass:
        if status != "pass":
            result["failures"].append("installed_app_parity_not_pass")
        missing_or_stale = payload.get("missing_or_stale")
        if missing_or_stale:
            result["failures"].append("app_parity_has_missing_or_stale")

    return result


def _validate_current_mimo_v2_jang2l_sink_ab(root: Path) -> dict[str, Any]:
    artifact = CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "open",
        "missing": [],
        "failures": [],
        "normal_coherent": False,
        "sink_disabled_coherent": False,
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    variants = payload.get("variants")
    if not isinstance(variants, list):
        result["failures"].append("missing_variants")
        return result

    by_mode = {
        str(item.get("mode")): item
        for item in variants
        if isinstance(item, dict)
    }
    normal = by_mode.get("normal", {})
    sink_disabled = by_mode.get("sink_disabled", {})
    result["normal_coherent"] = _mimo_sink_output_coherent(str(normal.get("output", "")))
    result["sink_disabled_coherent"] = _mimo_sink_output_coherent(
        str(sink_disabled.get("output", ""))
    )
    if result["normal_coherent"] and result["sink_disabled_coherent"]:
        result["status"] = "pass"
    return result


def _mimo_sink_output_coherent(text: str) -> bool:
    lower = text.lower()
    alpha_count = sum(ch.isalpha() for ch in text)
    return (
        alpha_count >= 40
        and "prefix" in lower
        and "cache" in lower
        and any(term in lower for term in ("server", "model", "tokens"))
    )


def _validate_current_mimo_v2_jang2l_host_availability(root: Path) -> dict[str, Any]:
    artifact = CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
        "live_launch_decision": None,
        "launch_allowed": False,
        "launch_blockers": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    launch_decision = payload.get("live_launch_decision")
    launch_allowed = payload.get("launch_allowed")
    launch_blockers = payload.get("launch_blockers")
    if not isinstance(launch_blockers, list):
        launch_blockers = []
        result["failures"].append("launch_blockers_not_list")
    valid_statuses = {
        "missing_model_weights",
        "invalid_model_metadata",
        "skipped_active_heavy_process",
        "model_available",
    }
    if status not in valid_statuses:
        result["failures"].append("unexpected_status")
    if launch_decision not in {"do_not_launch", "available_for_preflight"}:
        result["failures"].append("unexpected_launch_decision")
    if not isinstance(launch_allowed, bool):
        result["failures"].append("launch_allowed_not_bool")

    result.update(
        {
            "status": status,
            "live_launch_decision": launch_decision,
            "launch_allowed": launch_allowed if isinstance(launch_allowed, bool) else False,
            "launch_blockers": [str(item) for item in launch_blockers],
        }
    )
    return result


def _remote_evidence_artifacts(*artifacts: str) -> list[str]:
    return [artifact for artifact in artifacts if artifact.startswith("build/remote-")]


def _validate_current_mimo_v2_jang2l_root_cause(root: Path) -> dict[str, Any]:
    structural_artifact = CURRENT_MIMO_V2_JANG2L_STRUCTURAL_VERIFY_ARTIFACT
    text_cache_artifact = CURRENT_MIMO_V2_JANG2L_TEXT_CACHE_ARTIFACT
    switchglu_artifact = CURRENT_MIMO_V2_JANG2L_SWITCHGLU_PARITY_ARTIFACT
    length_sweep_artifact = CURRENT_MIMO_V2_JANG2L_LENGTH_SWEEP_ARTIFACT
    tool_dialect_artifact = CURRENT_MIMO_V2_JANG2L_TOOL_DIALECT_ARTIFACT
    current_audit_artifact = CURRENT_MIMO_V2_JANG2L_CURRENT_AUDIT_ARTIFACT
    metadata_truth_artifact = CURRENT_MIMO_V2_JANG2L_METADATA_TRUTH_ARTIFACT
    source_vs_quant_artifact = CURRENT_MIMO_V2_JANG2L_SOURCE_VS_QUANT_ARTIFACT
    no_source_classifier_artifact = (
        CURRENT_MIMO_V2_JANG2L_NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT
    )
    restart_l2_artifact = CURRENT_MIMO_V2_JANG2L_RESTART_L2_ARTIFACT
    responses_tools_artifact = (
        CURRENT_MIMO_V2_JANG2L_RESPONSES_TOOLS_RERUN_ARTIFACT
    )
    result: dict[str, Any] = {
        "status": "missing",
        "artifacts": {
            "structural_verify": structural_artifact,
            "text_cache": text_cache_artifact,
            "switchglu_parity": switchglu_artifact,
            "length_sweep": length_sweep_artifact,
            "tool_dialect": tool_dialect_artifact,
            "current_audit": current_audit_artifact,
            "metadata_truth": metadata_truth_artifact,
            "source_vs_quant_first_divergence": source_vs_quant_artifact,
            "no_source_exactness_classifier": no_source_classifier_artifact,
            "jang2l_restart_l2": restart_l2_artifact,
            "jang2l_responses_tools_rerun": responses_tools_artifact,
        },
        "missing": [],
        "failures": [],
        "metadata_truth_passed": False,
        "source_vs_quant_first_divergence_passed": False,
        "source_vs_quant_first_divergence_policy_skipped": False,
        "source_vs_quant_policy_skip_reason": None,
        "source_vs_quant_requirement_satisfied": False,
        "no_source_exactness_classifier_present": False,
        "no_source_exactness_classifier_status": None,
        "no_source_exactness_classification": None,
        "no_source_exactness_excludes_parser_cache_sampling": False,
        "no_source_exactness_unresolved_surfaces": {},
        "structural_verify_passed": False,
        "text_cache_narrow_pass": False,
        "switchglu_selected_expert_parity_passed": False,
        "prompt_length_coherence_blocked": False,
        "tool_protocol_blocked": False,
        "artifact_exactness_blocked": False,
        "artifact_exactness_bundle_status": {
            "jangtq2": {
                "blocked": False,
                "classification": None,
                "evidence": no_source_classifier_artifact,
            },
            "jang2l": {
                "blocked": False,
                "classification": None,
                "evidence": no_source_classifier_artifact,
            },
        },
        "artifact_exactness_release_action": None,
        "recommended_next_artifact_profiles": [],
        "decode_speed_target_blocked": False,
        "cb_working_set_pressure_blocked": False,
        "media_unwired": False,
        "mimo_jangtq2_live_media_l2_passed": False,
        "mimo_jang2l_live_media_l2_blocked": False,
        "mimo_jang2l_l2_restart_cache_hit_passed": False,
        "mimo_jang2l_l2_restart_visible_output_blocked": False,
        "mimo_jang2l_responses_transport_passed": False,
        "mimo_jang2l_responses_tool_semantics_blocked": False,
        "manifest_integrity_passed": False,
        "stale_local_state_absent": False,
        "current_audit_status": None,
        "current_audit_blockers": [],
        "latest_decode_speed_evidence": None,
        "decode_bottleneck_classification": None,
        "switchglu_fastpath_active_but_slow": False,
        "async_decode_wait_dominates": False,
        "remote_artifacts": [],
        "remote_evidence_only": False,
        "local_release_clearance": False,
    }

    structural_path = root / structural_artifact
    text_cache_path = root / text_cache_artifact
    switchglu_path = root / switchglu_artifact
    length_sweep_path = root / length_sweep_artifact
    tool_dialect_path = root / tool_dialect_artifact
    current_audit_path = root / current_audit_artifact
    metadata_truth_path = root / metadata_truth_artifact
    source_vs_quant_path = root / source_vs_quant_artifact
    no_source_classifier_path = root / no_source_classifier_artifact
    restart_l2_path = root / restart_l2_artifact
    responses_tools_path = root / responses_tools_artifact
    for path, artifact in (
        (structural_path, structural_artifact),
        (text_cache_path, text_cache_artifact),
        (switchglu_path, switchglu_artifact),
        (length_sweep_path, length_sweep_artifact),
        (tool_dialect_path, tool_dialect_artifact),
        (current_audit_path, current_audit_artifact),
        (metadata_truth_path, metadata_truth_artifact),
        (source_vs_quant_path, source_vs_quant_artifact),
        (no_source_classifier_path, no_source_classifier_artifact),
        (restart_l2_path, restart_l2_artifact),
        (responses_tools_path, responses_tools_artifact),
    ):
        if not path.exists():
            result["missing"].append(artifact)
    if result["missing"]:
        return result

    try:
        structural_payload = json.loads(structural_path.read_text(encoding="utf-8"))
        text_cache_payload = json.loads(text_cache_path.read_text(encoding="utf-8"))
        switchglu_payload = json.loads(switchglu_path.read_text(encoding="utf-8"))
        length_sweep_payload = json.loads(length_sweep_path.read_text(encoding="utf-8"))
        tool_dialect_payload = json.loads(tool_dialect_path.read_text(encoding="utf-8"))
        current_audit_payload = json.loads(current_audit_path.read_text(encoding="utf-8"))
        metadata_truth_payload = json.loads(metadata_truth_path.read_text(encoding="utf-8"))
        source_vs_quant_payload = json.loads(source_vs_quant_path.read_text(encoding="utf-8"))
        no_source_classifier_payload = json.loads(
            no_source_classifier_path.read_text(encoding="utf-8")
        )
        restart_l2_payload = json.loads(restart_l2_path.read_text(encoding="utf-8"))
        responses_tools_payload = json.loads(
            responses_tools_path.read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    metadata_jang2l = {}
    metadata_bundles = metadata_truth_payload.get("bundles")
    if isinstance(metadata_bundles, dict) and isinstance(
        metadata_bundles.get("jang2l"), dict
    ):
        metadata_jang2l = metadata_bundles["jang2l"]
    metadata_jang2l_caps = metadata_jang2l.get("capabilities")
    if not isinstance(metadata_jang2l_caps, dict):
        metadata_jang2l_caps = {}
    result["metadata_truth_passed"] = (
        metadata_truth_payload.get("status") == "pass"
        and (
            (
                metadata_truth_payload.get("runtime_modalities") == ["text"]
                and metadata_truth_payload.get("preserved_modalities")
                == ["vision", "audio"]
                and metadata_truth_payload.get("unwired_modalities")
                == ["vision", "audio"]
                and metadata_truth_payload.get("multimodal_status")
                == "weights_preserved_text_runtime"
            )
            or (
                metadata_truth_payload.get("expected_runtime_modalities") == ["text"]
                and metadata_truth_payload.get("expected_preserved_modalities")
                == ["vision", "audio"]
                and metadata_jang2l.get("status") == "pass"
                and metadata_jang2l_caps.get("modalities") == ["text"]
                and metadata_jang2l_caps.get("preserved_modalities")
                == ["vision", "audio"]
                and metadata_jang2l_caps.get("unwired_modalities")
                == ["vision", "audio"]
                and metadata_jang2l_caps.get("multimodal_status")
                == "weights_preserved_text_runtime"
            )
        )
    )
    if not result["metadata_truth_passed"]:
        result["failures"].append("mimo_metadata_truth_not_pass")

    source_vs_quant_rows = source_vs_quant_payload.get("rows")
    if not isinstance(source_vs_quant_rows, list):
        source_vs_quant_rows = []
    source_vs_quant_allowed = {
        "source_and_quant_match",
        "quant_diverges_from_source",
        "source_also_fails",
    }
    result["source_vs_quant_first_divergence_passed"] = (
        source_vs_quant_payload.get("status") == "pass"
        and source_vs_quant_payload.get("remote_evidence_only") is not True
        and bool(source_vs_quant_payload.get("source_model_path"))
        and bool(source_vs_quant_payload.get("quant_model_path"))
        and bool(source_vs_quant_rows)
        and all(
            isinstance(item, dict)
            and bool(item.get("prompt_name"))
            and item.get("classification") in source_vs_quant_allowed
            and "source_output" in item
            and "quant_output" in item
            for item in source_vs_quant_rows
        )
    )
    classifier_excluded = no_source_classifier_payload.get("excluded_surfaces")
    if not isinstance(classifier_excluded, dict):
        classifier_excluded = {}
    classifier_unresolved = no_source_classifier_payload.get("unresolved_surfaces")
    if not isinstance(classifier_unresolved, dict):
        classifier_unresolved = {}
    result["no_source_exactness_classifier_present"] = True
    result["no_source_exactness_classifier_status"] = no_source_classifier_payload.get(
        "status"
    )
    result["no_source_exactness_classification"] = no_source_classifier_payload.get(
        "classification"
    )
    result["no_source_exactness_secondary_classification"] = (
        no_source_classifier_payload.get("secondary_classification")
    )
    result["no_source_exactness_unresolved_surfaces"] = classifier_unresolved
    result["no_source_exactness_excludes_parser_cache_sampling"] = (
        no_source_classifier_payload.get("source_vs_quant_load_performed") is False
        and classifier_excluded.get("parser_argument_rewrite") is True
        and classifier_excluded.get("prefix_paged_l2_or_kv_quant_primary_cause")
        is True
        and classifier_excluded.get("hidden_stochastic_sampling_primary_cause") is True
        and classifier_excluded.get("api_sampler_non_top1_selection") is True
    )
    if not result["no_source_exactness_excludes_parser_cache_sampling"]:
        result["failures"].append(
            "mimo_no_source_exactness_classifier_missing_parser_cache_sampling_boundary"
        )
    result["source_vs_quant_policy_skip_reason"] = no_source_classifier_payload.get(
        "source_vs_quant_load_skipped_reason"
    )
    result["source_vs_quant_first_divergence_policy_skipped"] = (
        no_source_classifier_payload.get("source_vs_quant_load_performed") is False
        and result["source_vs_quant_policy_skip_reason"]
        in {
            "user_disallowed_source_vs_quant_due_ram",
            "user_disallowed_source_vs_quant",
            "source_vs_quant_skipped_by_policy",
        }
        and result["no_source_exactness_classifier_status"] in {"open", "pass"}
        and result["no_source_exactness_excludes_parser_cache_sampling"] is True
    )
    if (
        not result["source_vs_quant_first_divergence_passed"]
        and not result["source_vs_quant_first_divergence_policy_skipped"]
    ):
        result["failures"].append("mimo_source_vs_quant_first_divergence_missing")

    current_audit_component_ok = current_audit_payload.get("component_ok")
    if not isinstance(current_audit_component_ok, dict):
        current_audit_component_ok = {}
    result["current_audit_status"] = current_audit_payload.get("status")
    result["manifest_integrity_passed"] = (
        current_audit_component_ok.get("manifest_integrity") is True
    )
    result["stale_local_state_absent"] = (
        current_audit_component_ok.get("stale_local_state_absent") is True
    )
    if not result["manifest_integrity_passed"]:
        result["failures"].append("mimo_manifest_integrity_not_pass")
    if not result["stale_local_state_absent"]:
        result["failures"].append("mimo_stale_local_state_present")

    result["structural_verify_passed"] = structural_payload.get("status") == "pass"
    if not result["structural_verify_passed"]:
        result["failures"].append("mimo_structural_verify_not_pass")

    text_cache_requests = text_cache_payload.get("requests")
    if not isinstance(text_cache_requests, list):
        text_cache_requests = []
    text_cache_rows = text_cache_payload.get("rows")
    if not isinstance(text_cache_rows, list):
        text_cache_rows = []
    text_cache_summary = text_cache_payload.get("summary")
    if not isinstance(text_cache_summary, dict):
        text_cache_summary = {}
    text_cache_contents = [
        str(item.get("content") or "")
        for item in text_cache_requests
        if isinstance(item, dict)
    ]
    cached_token_counts = []
    for item in [*text_cache_requests, *text_cache_rows]:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage")
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(cached_tokens, int):
            cached_token_counts.append(cached_tokens)
    summary_texts = text_cache_summary.get("texts")
    if not isinstance(summary_texts, dict):
        summary_texts = {}
    summary_repeat_pass = (
        text_cache_payload.get("status") == "pass"
        and text_cache_summary.get("all_requests_http_ok") is True
        and text_cache_summary.get("exact_repeat_1") is True
        and text_cache_summary.get("exact_repeat_2") is True
        and text_cache_summary.get("cache_hit_tokens", 0) > 0
        and text_cache_summary.get("l2_tokens_on_disk", 0) > 0
        and summary_texts.get("exact_repeat_1") == "ACK-CB-742"
        and summary_texts.get("exact_repeat_2") == "ACK-CB-742"
    )
    result["text_cache_narrow_pass"] = (
        (
            len(text_cache_contents) >= 2
            and all(content == "cache ok" for content in text_cache_contents[:2])
            and any(count > 0 for count in cached_token_counts)
        )
        or summary_repeat_pass
    )
    if not result["text_cache_narrow_pass"]:
        result["failures"].append("mimo_text_cache_narrow_pass_missing")

    max_abs_diff = switchglu_payload.get("max_abs_diff")
    mean_abs_diff = switchglu_payload.get("mean_abs_diff")
    result["switchglu_selected_expert_parity_passed"] = (
        isinstance(max_abs_diff, (int, float))
        and float(max_abs_diff) <= 0.002
        and isinstance(mean_abs_diff, (int, float))
        and float(mean_abs_diff) <= 0.001
    )
    if not result["switchglu_selected_expert_parity_passed"]:
        result["failures"].append("mimo_switchglu_selected_expert_parity_missing")

    length_cases = length_sweep_payload.get("cases")
    if not isinstance(length_cases, list):
        length_cases = []
    corrupt_cases = [
        {
            "prompt_tokens": item.get("prompt_tokens"),
            "status": item.get("status"),
            "output": item.get("output"),
        }
        for item in length_cases
        if isinstance(item, dict)
        and isinstance(item.get("prompt_tokens"), int)
        and item.get("prompt_tokens") >= 148
        and "fail" in str(item.get("status") or "")
    ]
    result["prompt_length_coherence_blocked"] = (
        length_sweep_payload.get("status") == "fail" and bool(corrupt_cases)
    )
    if (
        length_sweep_payload.get("status") == "open"
        and length_sweep_payload.get("classification")
        == "intrinsic_python_mllm_long_prompt_metal_oom_first_request"
    ):
        result["prompt_length_coherence_blocked"] = True
        result["prompt_length_runtime_blocker"] = length_sweep_payload.get(
            "classification"
        )
    result["prompt_length_corrupt_cases"] = corrupt_cases

    tool_observations = tool_dialect_payload.get("runtime_observations")
    if not isinstance(tool_observations, list):
        tool_observations = []
    no_structured_tool_calls = any(
        isinstance(item, dict)
        and item.get("http_status") == 200
        and item.get("tool_calls") is None
        for item in tool_observations
    )
    required_rejected = any(
        isinstance(item, dict)
        and item.get("http_status") == 400
        and "did not produce any tool calls" in str(item.get("error") or "")
        for item in tool_observations
    )
    result["tool_protocol_blocked"] = (
        tool_dialect_payload.get("status") == "fail"
        and no_structured_tool_calls
        and required_rejected
    )
    tool_checks = tool_dialect_payload.get("checks")
    if not isinstance(tool_checks, dict):
        tool_checks = {}
    if (
        tool_checks.get("required_mode_tool_call_present") is True
        and tool_checks.get("auto_mode_tool_call_present") is True
        and tool_checks.get("cache_positive") is True
    ):
        result["tool_protocol_blocked"] = False

    audit_long_prompt_open = current_audit_component_ok.get("long_prompt_coherence") is False
    audit_tool_protocol_open = current_audit_component_ok.get("tool_protocol") is False
    audit_artifact_exactness_open = (
        current_audit_component_ok.get("artifact_exactness") is False
    )
    audit_decode_speed_open = current_audit_component_ok.get("decode_speed_target") is False
    audit_working_set_open = (
        current_audit_component_ok.get("cb_system_prompt_working_set_pressure") is False
    )
    audit_media_unwired = current_audit_component_ok.get("mimo_media_wired") is False
    audit_jangtq2_live_media_l2_passed = (
        current_audit_component_ok.get("mimo_jangtq2_live_media_l2") is True
    )
    audit_jang2l_live_media_l2_open = (
        current_audit_component_ok.get("mimo_jang2l_live_media_l2") is False
    )
    audit_jang2l_l2_cache_hit_passed = (
        current_audit_component_ok.get("mimo_jang2l_l2_restart_cache_hit") is True
    )
    audit_jang2l_l2_visible_output_open = (
        current_audit_component_ok.get("mimo_jang2l_l2_restart_visible_output") is False
    )
    audit_source_vs_quant_open = (
        current_audit_component_ok.get("source_vs_quant_first_divergence") is False
    )
    audit_blockers = current_audit_payload.get("blockers")
    if isinstance(audit_blockers, list):
        result["current_audit_blockers"] = [str(item) for item in audit_blockers]
    latest_decode_speed = current_audit_payload.get("latest_decode_speed_evidence")
    if isinstance(latest_decode_speed, dict):
        result["latest_decode_speed_evidence"] = latest_decode_speed
        result["decode_bottleneck_classification"] = latest_decode_speed.get(
            "decode_bottleneck_classification"
        )
        result["async_decode_wait_dominates"] = (
            latest_decode_speed.get("async_decode_wait_dominates") is True
        )
        result["switchglu_fastpath_active_but_slow"] = (
            latest_decode_speed.get("switchglu_fastpath_active") is True
            and latest_decode_speed.get("speed_blocked") is True
        )
    if (
        current_audit_component_ok.get("long_prompt_coherence") is True
        and not result.get("prompt_length_runtime_blocker")
    ):
        result["prompt_length_coherence_blocked"] = False
    elif audit_long_prompt_open:
        result["prompt_length_coherence_blocked"] = True
    if current_audit_component_ok.get("tool_protocol") is True:
        result["tool_protocol_blocked"] = False
    elif audit_tool_protocol_open:
        result["tool_protocol_blocked"] = True
    if audit_artifact_exactness_open:
        result["artifact_exactness_blocked"] = True
        primary_exactness_classification = str(
            result["no_source_exactness_classification"] or ""
        )
        secondary_exactness_classification = str(
            result["no_source_exactness_secondary_classification"] or ""
        )
        bundle_status = result["artifact_exactness_bundle_status"]
        if primary_exactness_classification.startswith("jangtq2"):
            bundle_status["jangtq2"] = {
                "blocked": True,
                "classification": primary_exactness_classification,
                "evidence": no_source_classifier_artifact,
            }
        elif secondary_exactness_classification.startswith("jangtq2"):
            bundle_status["jangtq2"] = {
                "blocked": True,
                "classification": secondary_exactness_classification,
                "evidence": no_source_classifier_artifact,
            }
        if secondary_exactness_classification.startswith("jang2l"):
            bundle_status["jang2l"] = {
                "blocked": True,
                "classification": secondary_exactness_classification,
                "evidence": no_source_classifier_artifact,
            }
        elif primary_exactness_classification.startswith("jang2l"):
            bundle_status["jang2l"] = {
                "blocked": True,
                "classification": primary_exactness_classification,
                "evidence": no_source_classifier_artifact,
            }
        if bundle_status["jangtq2"]["blocked"] is True:
            result["failures"].append("mimo_jangtq2_artifact_exactness_blocked")
        if bundle_status["jang2l"]["blocked"] is True:
            result["failures"].append("mimo_jang2l_artifact_exactness_blocked")
        if (
            bundle_status["jangtq2"]["blocked"] is not True
            and bundle_status["jang2l"]["blocked"] is not True
        ):
            result["failures"].append("mimo_artifact_exactness_blocked")
        accepted_exactness_classifications = {
            "model_generated_literal_mutation_after_valid_parser_structure",
            "jangtq2_plain_literal_copy_regression_jang2l_plain_copy_passes",
            "jangtq2_plain_literal_copy_fails_before_parser_or_json_repair",
            "jangtq2_literal_corruption_persists_without_cache_fastpath_router_and_tq_kernel_parity_passes",
            "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer",
        }
        if (
            result["no_source_exactness_classification"]
            not in accepted_exactness_classifications
            or classifier_unresolved.get("artifact_quantization_or_decode_logits_quality")
            is not True
        ):
            result["failures"].append(
                "mimo_no_source_exactness_classifier_missing_literal_mutation_boundary"
            )
        if primary_exactness_classification.startswith(
            "jangtq2_plain_literal_copy"
        ) or primary_exactness_classification == (
            "jangtq2_compact_hyphen_decode_quality_open_not_cache_parser_template_tokenizer"
        ):
            result["artifact_exactness_release_action"] = (
                "replace_all_routed_2bit_jangtq2_or_lift_gate_down_precision"
            )
            result["recommended_next_artifact_profiles"] = [
                "JANGTQ gate=3/up=2/down=3",
                "JANGTQ gate=3/up=3/down=3",
            ]
    if audit_decode_speed_open:
        result["decode_speed_target_blocked"] = True
        result["failures"].append("mimo_decode_speed_below_release_target")
    if audit_working_set_open:
        result["cb_working_set_pressure_blocked"] = True
        result["failures"].append("mimo_cb_working_set_pressure_blocked")
    if audit_media_unwired:
        result["media_unwired"] = True
        result["failures"].append("mimo_media_unwired")
    result["mimo_jangtq2_live_media_l2_passed"] = bool(
        audit_jangtq2_live_media_l2_passed
    )
    if audit_jang2l_live_media_l2_open:
        result["mimo_jang2l_live_media_l2_blocked"] = True
        result["failures"].append("mimo_jang2l_live_media_l2_missing")
    result["mimo_jang2l_l2_restart_cache_hit_passed"] = bool(
        audit_jang2l_l2_cache_hit_passed
    )
    restart_l2_results = restart_l2_payload.get("results")
    restart_l2_first = (
        restart_l2_results[0]
        if isinstance(restart_l2_results, list)
        and restart_l2_results
        and isinstance(restart_l2_results[0], dict)
        else {}
    )
    restart_l2_classification = restart_l2_first.get("classification")
    if not isinstance(restart_l2_classification, dict):
        restart_l2_classification = {}
    restart_l2_checks = restart_l2_classification.get("checks")
    if not isinstance(restart_l2_checks, dict):
        restart_l2_checks = {}
    current_restart_l2_cache_hit_passed = (
        restart_l2_classification.get("status") == "pass"
        and restart_l2_classification.get("cache_status") == "pass"
        and restart_l2_checks.get("restart_l2_restore_observed") is True
        and restart_l2_checks.get("second_cached_tokens_positive") is True
        and restart_l2_checks.get("block_disk_hit_positive") is True
    )
    if current_restart_l2_cache_hit_passed:
        result["mimo_jang2l_l2_restart_cache_hit_passed"] = True
        audit_jang2l_l2_visible_output_open = False
    if (
        result["mimo_jang2l_l2_restart_cache_hit_passed"]
        and audit_jang2l_l2_visible_output_open
    ):
        result["mimo_jang2l_l2_restart_visible_output_blocked"] = True
        result["failures"].append("mimo_jang2l_l2_restart_visible_output_blocked")
    responses_proven = responses_tools_payload.get("proven")
    if not isinstance(responses_proven, dict):
        responses_proven = {}
    responses_cache = responses_proven.get("cache")
    if not isinstance(responses_cache, dict):
        responses_cache = {}
    responses_red = responses_tools_payload.get("red")
    if not isinstance(responses_red, dict):
        responses_red = {}
    result["mimo_jang2l_responses_transport_passed"] = (
        responses_proven.get("responses_endpoint_used") is True
        and responses_proven.get("responses_delta_streaming") is True
        and responses_proven.get("responses_previous_response_id_tool_followup")
        is True
        and int(responses_cache.get("cache_hit_tokens") or 0) > 0
        and int(responses_cache.get("l2_block_tokens_on_disk") or 0) > 0
    )
    result["mimo_jang2l_responses_tool_semantics_blocked"] = (
        result["mimo_jang2l_responses_transport_passed"] is True
        and responses_tools_payload.get("status") == "fail"
        and responses_red.get("long_tool_loop") is False
    )
    if result["mimo_jang2l_responses_tool_semantics_blocked"]:
        result["failures"].append("mimo_jang2l_responses_tool_semantics_blocked")
    if audit_source_vs_quant_open:
        result["source_vs_quant_first_divergence_passed"] = False
        if (
            not result["source_vs_quant_first_divergence_policy_skipped"]
            and "mimo_source_vs_quant_first_divergence_missing" not in result["failures"]
        ):
            result["failures"].append("mimo_source_vs_quant_first_divergence_missing")
    result["source_vs_quant_requirement_satisfied"] = (
        result["source_vs_quant_first_divergence_passed"]
        or result["source_vs_quant_first_divergence_policy_skipped"]
    )
    if result["source_vs_quant_first_divergence_policy_skipped"]:
        result["current_audit_blockers"] = [
            blocker
            for blocker in result["current_audit_blockers"]
            if blocker != "mimo_source_vs_quant_first_divergence_missing_or_failed"
        ]

    if (
        result["prompt_length_coherence_blocked"]
        or result["tool_protocol_blocked"]
        or result["artifact_exactness_blocked"]
        or result["decode_speed_target_blocked"]
        or result["cb_working_set_pressure_blocked"]
        or result["media_unwired"]
        or result["mimo_jang2l_live_media_l2_blocked"]
        or result["mimo_jang2l_l2_restart_visible_output_blocked"]
        or not result["source_vs_quant_requirement_satisfied"]
    ):
        result["status"] = "open"
        result["root_cause_candidate"] = (
            "mimo_v2_jang2l_quantized_profile_or_full_forward_quality_pending_exactness_media_or_policy_allowed_source_boundary"
        )
        active_boundary_reasons: list[str] = []
        if result["prompt_length_coherence_blocked"]:
            active_boundary_reasons.append("long-prompt coherence")
        if result["tool_protocol_blocked"]:
            active_boundary_reasons.append("tool protocol")
        if result["artifact_exactness_blocked"]:
            bundle_status = result.get("artifact_exactness_bundle_status")
            if not isinstance(bundle_status, dict):
                bundle_status = {}
            if (
                isinstance(bundle_status.get("jangtq2"), dict)
                and bundle_status["jangtq2"].get("blocked") is True
            ):
                active_boundary_reasons.append("current JANGTQ2 artifact exactness")
            if (
                isinstance(bundle_status.get("jang2l"), dict)
                and bundle_status["jang2l"].get("blocked") is True
            ):
                active_boundary_reasons.append("current JANG_2L artifact exactness")
            if not any(
                reason.endswith("artifact exactness")
                for reason in active_boundary_reasons
            ):
                active_boundary_reasons.append("current MiMo artifact exactness")
        if result["decode_speed_target_blocked"]:
            active_boundary_reasons.append("decode speed")
        if result["cb_working_set_pressure_blocked"]:
            active_boundary_reasons.append("working-set pressure")
        if result["media_unwired"]:
            active_boundary_reasons.append("media wiring")
        if result["mimo_jang2l_live_media_l2_blocked"]:
            active_boundary_reasons.append("JANG_2L live media/L2 proof")
        if result["mimo_jang2l_l2_restart_visible_output_blocked"]:
            active_boundary_reasons.append("JANG_2L L2 restart visible output")
        if result["mimo_jang2l_responses_tool_semantics_blocked"]:
            active_boundary_reasons.append("JANG_2L Responses/tool semantic drift")
        if not result["source_vs_quant_requirement_satisfied"]:
            active_boundary_reasons.append(
                "source-vs-quant/no-source classification boundary"
            )
        if not active_boundary_reasons:
            active_boundary_reasons.append("current MiMo runtime quality")
        result["release_boundary"] = (
            "local artifact/runtime has narrow text-cache proof but still fails "
            + ", ".join(active_boundary_reasons)
            + "; do not release-clear MiMo"
        )
    elif not result["failures"]:
        result["status"] = "pass"
        result["local_release_clearance"] = True
    else:
        result["status"] = "fail"
    return result


def _validate_current_issue175_177_installed_runtime_audit(root: Path) -> dict[str, Any]:
    artifact = CURRENT_ISSUE175_177_INSTALLED_RUNTIME_AUDIT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status != "pass":
        result["failures"].append("issue175_177_installed_runtime_must_pass")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    for key in (
        "installed_python_exists",
        "memory_clear_helper_imports_from_installed_app",
        "memory_clear_uses_available_mlx_api",
        "runtime_paths_avoid_removed_clear_memory_cache",
        "promoted_disk_blocks_drop_parent_mirror",
        "l2_readable_write_through_blocks_drop_parent_mirror",
        "cache_selection_telemetry_installed",
        "cache_execution_timing_telemetry_installed",
    ):
        if checks.get(key) is not True:
            result["failures"].append(f"missing_or_false_check:{key}")

    return result


def _validate_current_issue175_177_live_runtime_audit(root: Path) -> dict[str, Any]:
    artifact = CURRENT_ISSUE175_177_LIVE_RUNTIME_AUDIT_ARTIFACT
    path = root / artifact
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "missing": [],
        "failures": [],
    }
    if not path.exists():
        result["missing"].append(artifact)
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report validation failure
        result["status"] = f"load_error:{type(exc).__name__}"
        result["failures"].append("json_load_error")
        return result

    status = str(payload.get("status"))
    result["status"] = status
    if status != "pass":
        result["failures"].append("issue175_177_live_runtime_must_pass_when_all_checks_green")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    for key in (
        "installed_app_qwen_live_probe_passed",
        "cache_hit_ttft_improved",
        "l2_block_disk_hits_observed",
        "paged_ssm_cache_hit_observed",
        "visible_content_observed",
        "metal_memory_metrics_observed",
        "prefix_paged_l2_capacity_projection_valid",
        "minimax_installed_live_probe_passed",
        "turboquant_live_decode_observed",
        "scheduler_cache_selection_observed",
        "scheduler_cache_execution_timing_observed",
        "paged_tq_cache_hit_observed",
        "minimax_l2_block_disk_hits_observed",
        "minimax_cache_hit_request_latency_improved",
        "minimax_visible_content_observed",
        "minimax_restart_reader_probe_passed",
        "cold_paged_tq_restart_hit_observed",
        "cold_paged_cache_selection_observed",
        "cold_paged_stream_ttft_observed",
        "restart_visible_content_observed",
        "admin_sleep_lifecycle_probe_passed",
        "admin_sleep_deep_unload_observed",
    ):
        if checks.get(key) is not True:
            result["failures"].append(f"missing_or_false_check:{key}")

    blockers = {str(item) for item in payload.get("remaining_blockers", [])}
    if blockers:
        result["failures"].append(
            "unexpected_remaining_blockers:" + ",".join(sorted(blockers))
        )

    return result


def _validate_current_real_ui_live_model_proof_artifacts(
    root: Path,
    skipped_missing_families: set[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if skipped_missing_families is None:
        skipped_missing_families = set()
    missing: list[str] = []
    failures: list[str] = []
    skipped_missing: list[str] = []
    proofs: dict[str, dict[str, Any]] = {}

    for row_id, row in CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.items():
        proof_artifact = row["proof"]
        screenshot_artifact = row["chat_screenshot"]
        proof_path = root / proof_artifact
        row_family = str(row.get("family") or row_id)
        missing_allowed = row_family in skipped_missing_families
        if not proof_path.exists():
            (skipped_missing if missing_allowed else missing).append(proof_artifact)
        if not (root / screenshot_artifact).exists():
            (skipped_missing if missing_allowed else missing).append(
                screenshot_artifact
            )
        if not missing_allowed or (root / screenshot_artifact).exists():
            _validate_png_artifact(
                root, f"{row_id}_chat_screenshot", screenshot_artifact, failures
            )

        proof: dict[str, Any] = {}
        if proof_path.exists():
            try:
                loaded = json.loads(proof_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    proof = loaded
                else:
                    failures.append("proof_json_not_object")
            except Exception as exc:  # noqa: BLE001 - report validation failure
                failures.append(f"proof_json_load_error:{type(exc).__name__}")

        if not proof:
            continue

        proofs[row_id] = proof
        if proof.get("status") == "fail":
            failures.append("proof_status_failed")
            if proof.get("failureStage"):
                failures.append(f"proof_failure_stage:{proof.get('failureStage')}")
        if proof.get("repoDir") != str(root):
            failures.append("proof_repo_dir_mismatch")
        if proof.get("panelDir") != str(root / "panel"):
            failures.append("proof_panel_dir_mismatch")
        if proof.get("script") != "panel/scripts/live-real-ui-model-proof.mjs":
            failures.append("proof_script_mismatch")
        if proof.get("modelPath") != row["model_path"]:
            failures.append("proof_model_path_mismatch")
        if proof.get("modelName") != row["model_name"]:
            failures.append("proof_model_name_mismatch")
        request_contract = proof.get("requestContract")
        if not isinstance(request_contract, dict):
            failures.append(f"request_contract_missing:{row_id}")
        else:
            missing_contract_fields = [
                field
                for field in REQUIRED_REAL_UI_REQUEST_CONTRACT_FIELDS
                if field not in request_contract
            ]
            if missing_contract_fields:
                failures.append(
                    f"request_contract_incomplete:{row_id}:"
                    + ",".join(missing_contract_fields)
                )
            failures.extend(
                _validate_real_ui_request_contract(row_id, proof, request_contract)
            )
        if (
            row_id in REQUIRED_REAL_UI_EXTENSIVE_TOOL_ROWS
            and not _real_ui_extensive_tool_churn_ok(proof)
        ):
            failures.append(f"extensive_tool_churn_missing:{row_id}")

        installed_app_ui = _real_ui_proof_is_installed_app(proof)
        if not installed_app_ui:
            renderer_url_seen, launch_seen = _real_ui_electron_dev_launch_proven(proof)
            if not renderer_url_seen:
                failures.append("electron_dev_renderer_url_missing")
            if not launch_seen:
                failures.append("electron_dev_launch_log_missing")

        server = proof.get("server") if isinstance(proof.get("server"), dict) else {}
        health = server.get("health") if isinstance(server.get("health"), dict) else {}
        if not (
            health.get("status") == "healthy"
            and health.get("model_loaded") is True
        ):
            failures.append("server_health_not_real_loaded_model")
        models = server.get("models") if isinstance(server.get("models"), dict) else {}
        model_ids = [
            item.get("id")
            for item in models.get("data", [])
            if isinstance(item, dict)
        ]
        if not any(_real_ui_model_id_matches(row, model_id) for model_id in model_ids):
            failures.append("server_models_missing_real_model")

        chat = proof.get("chat") if isinstance(proof.get("chat"), dict) else {}
        final_visible = chat.get("finalVisibleText")
        if not isinstance(final_visible, str) or not final_visible.strip():
            failures.append("final_visible_text_missing")
        if (
            chat.get("rawParserTagLeak") is not False
            or _real_ui_chat_has_raw_parser_leak(chat)
            or _real_ui_reasoning_leak_counts(proof, chat)["raw"]
        ):
            failures.append("raw_parser_or_reasoning_tag_leak")
        cjk_leak_count = chat.get("cjkLeakCount")
        korean_leak_count = chat.get("koreanLeakCount")
        if cjk_leak_count != 0 or korean_leak_count != 0:
            failures.append("visible_language_leak")
        reasoning_leaks = _real_ui_reasoning_leak_counts(proof, chat)
        if (
            reasoning_leaks["cjk"] != 0
            or reasoning_leaks["korean"] != 0
            or reasoning_leaks["numeric"] != 0
        ):
            failures.append("reasoning_language_or_numeric_leak")
        turns = chat.get("turns")
        if not isinstance(turns, list) or len(turns) < 4:
            failures.append("multi_turn_chat_not_proven")
        elif not _real_ui_multi_turn_chat_ok(proof, chat):
            failures.append("visible_multi_turn_chat_not_proven")

        cache_hit_tokens = _json_number(proof, "cache", "cacheHitTokens")
        if cache_hit_tokens is None or cache_hit_tokens <= 0:
            failures.append("cache_hit_tokens_missing")

        screenshots = (
            proof.get("screenshots")
            if isinstance(proof.get("screenshots"), dict)
            else {}
        )
        expected_chat = str((root / screenshot_artifact).resolve())
        if screenshots.get("chat") != expected_chat:
            failures.append("screenshot_path_mismatch:chat")

    return {
        "status": "pass" if not missing and not failures else "fail",
        "artifacts": dict(CURRENT_REAL_UI_LIVE_MODEL_PROOF_ARTIFACTS),
        "missing": missing,
        "failures": failures,
        "skipped_missing_families": sorted(skipped_missing_families),
        "skipped_missing": skipped_missing,
        "proof": proofs.get("zaya_text") if proofs else {},
        "proofs": proofs if proofs else {},
    }


def _validate_current_real_ui_live_model_matrix(
    real_ui_live_model_proof: dict[str, Any],
) -> dict[str, Any]:
    covered_families: dict[str, dict[str, Any]] = {}

    proofs = real_ui_live_model_proof.get("proofs")
    if not isinstance(proofs, dict):
        proof = real_ui_live_model_proof.get("proof")
        proofs = {"zaya_text": proof} if isinstance(proof, dict) and proof else {}

    if proofs:
        for row_id, proof in proofs.items():
            if not isinstance(proof, dict):
                continue
            chat = proof.get("chat") if isinstance(proof.get("chat"), dict) else {}
            server = proof.get("server") if isinstance(proof.get("server"), dict) else {}
            health = server.get("health") if isinstance(server.get("health"), dict) else {}
            surfaces: set[str] = set()
            event_counts = (
                proof.get("eventCounts")
                if isinstance(proof.get("eventCounts"), dict)
                else {}
            )
            chat_overrides = (
                proof.get("chatOverrides")
                if isinstance(proof.get("chatOverrides"), dict)
                else {}
            )
            if proof.get("appLogTail"):
                surfaces.add("current_electron_dev_build")
            if _real_ui_proof_is_installed_app(proof):
                surfaces.add("installed_app_ui")
            if health.get("status") == "healthy" and health.get("model_loaded") is True:
                surfaces.add("real_loaded_model")
            if chat.get("turns"):
                surfaces.add("chat_completions")
            raw_parser_flag = chat.get("rawParserTagLeak")
            if not isinstance(raw_parser_flag, bool):
                raw_parser_flag = proof.get("rawParserLeak")
            reasoning_leaks = _real_ui_reasoning_leak_counts(proof, chat)
            reasoning_raw_flag = proof.get("reasoningRawParserLeak")
            if (
                raw_parser_flag is False
                and reasoning_raw_flag is not True
                and not _real_ui_chat_has_raw_parser_leak(chat)
                and not reasoning_leaks["raw"]
            ):
                surfaces.add("parser_leak_check")
            if (
                chat.get("cjkLeakCount") == 0
                and chat.get("koreanLeakCount") == 0
                and reasoning_leaks["cjk"] == 0
                and reasoning_leaks["korean"] == 0
                and reasoning_leaks["numeric"] == 0
            ):
                surfaces.add("language_leak_check")
            if (
                _json_number(proof, "cache", "cacheHitTokens")
                and _real_ui_cache_reconstruction_clean(proof)
            ):
                surfaces.add("cache_hit_telemetry")
            native_cache = (
                health.get("native_cache")
                if isinstance(health.get("native_cache"), dict)
                else {}
            )
            if (
                native_cache.get("family")
                and native_cache.get("schema")
                and native_cache.get("cache_type")
                and isinstance(native_cache.get("components"), list)
                and native_cache.get("prefix") is True
                and native_cache.get("paged") is True
                and native_cache.get("block_disk_l2") is True
            ):
                surfaces.add("native_cache_status")
            cache = proof.get("cache") if isinstance(proof.get("cache"), dict) else {}
            cache_before = cache.get("before") if isinstance(cache.get("before"), dict) else {}
            cache_after = cache.get("after") if isinstance(cache.get("after"), dict) else {}
            if (
                isinstance(cache_before.get("scheduler_cache"), dict)
                and isinstance(cache_after.get("scheduler_cache"), dict)
                and isinstance(cache_after.get("block_disk_cache"), dict)
                and isinstance(cache_after.get("cache_totals"), dict)
            ):
                surfaces.add("cache_endpoint_stats")
            if _real_ui_l2_disk_storage_seen(proof):
                surfaces.add("l2_disk_storage")
            if (
                proof.get("rendererWireApi") == "responses"
                and _json_number(proof, "eventCounts", "complete")
            ):
                surfaces.add("responses_api")
            if _real_ui_responses_delta_streaming_ok(proof):
                surfaces.add("responses_delta_streaming")
            if _real_ui_responses_cache_detail_usage_ok(proof):
                surfaces.add("responses_cache_detail_usage")
            if _real_ui_generation_defaults_applied_ok(proof):
                surfaces.add("generation_defaults_applied")
            if _real_ui_multi_turn_chat_ok(proof, chat):
                surfaces.add("multi_turn_chat")
            row = CURRENT_REAL_UI_LIVE_MODEL_PROOF_ROWS.get(row_id, {})
            family_id = str(row.get("family") or row_id)
            if _real_ui_architecture_cache_policy_ok(family_id, proof):
                surfaces.add("architecture_cache_policy")
            if _real_ui_live_speed_floor_ok(family_id, proof):
                surfaces.add("live_speed_floor")
            if (
                (event_counts.get("tool") or 0) >= 3
                and _real_ui_named_tool_result_count(proof) >= 2
                and _real_ui_named_tool_lifecycle_message_count(proof) >= 2
                and _real_ui_named_tool_error_count(proof) == 0
                and _real_ui_named_tool_probe_semantics_ok(proof)
            ):
                surfaces.add("long_tool_loop")
            if (
                (event_counts.get("reasoningDone") or 0) > 0
                or (proof.get("persistedReasoningCount") or 0) > 0
            ) and _real_ui_visible_assistant_turns_complete(proof, chat):
                surfaces.add("reasoning_display")
            if (
                "builtinToolsEnabled" in chat_overrides
                and chat_overrides.get("builtinToolsEnabled")
                == proof.get("requestedBuiltinTools")
            ):
                surfaces.add("settings_persistence")
            server_cache_controls = (
                proof.get("serverCacheControls")
                if isinstance(proof.get("serverCacheControls"), dict)
                else {}
            )
            if server_cache_controls.get("verified") is True:
                surfaces.add("server_cache_controls")
            if {
                "cache_endpoint_stats",
                "cache_hit_telemetry",
                "architecture_cache_policy",
                "generation_defaults_applied",
                "l2_disk_storage",
                "long_tool_loop",
                "responses_cache_detail_usage",
                "responses_delta_streaming",
                "server_cache_controls",
                "live_speed_floor",
            }.issubset(surfaces) and _real_ui_integrated_tool_l2_cache_ok(proof):
                surfaces.add(REAL_UI_INTEGRATED_TOOL_L2_CACHE_SURFACE)
            media = proof.get("media") if isinstance(proof.get("media"), dict) else {}
            if media.get("imageVerified") is True:
                surfaces.add("vl_image")
                if (
                    family_id == "step37"
                    and REAL_UI_INTEGRATED_TOOL_L2_CACHE_SURFACE in surfaces
                ):
                    surfaces.add(REAL_UI_INTEGRATED_VL_TOOL_L2_CACHE_SURFACE)
            if media.get("videoVerified") is True:
                surfaces.add("video_where_supported")

            current_model_name = str(proof.get("modelName") or row.get("model_name") or "")
            current_model_path = str(proof.get("modelPath") or row.get("model_path") or "")
            row_artifact = str(row.get("proof") or "")
            row_surface_artifacts = {
                surface: [row_artifact] for surface in surfaces if row_artifact
            }
            existing = covered_families.get(family_id)
            if existing:
                surfaces.update(existing.get("covered_surfaces", []))
                artifacts = list(existing.get("artifacts", []))
                if row.get("proof") and row.get("proof") not in artifacts:
                    artifacts.append(row.get("proof"))
                surface_artifacts = {
                    str(surface): [
                        str(artifact)
                        for artifact in artifact_list
                        if isinstance(artifact, str) and artifact
                    ]
                    for surface, artifact_list in (
                        existing.get("surface_artifacts") or {}
                    ).items()
                    if isinstance(artifact_list, list)
                }
                for surface, artifact_list in row_surface_artifacts.items():
                    current = surface_artifacts.setdefault(surface, [])
                    for artifact in artifact_list:
                        if artifact not in current:
                            current.append(artifact)
                model_names = {
                    str(name)
                    for name in existing.get("modelNames", [])
                    if isinstance(name, str) and name
                }
                if current_model_name:
                    model_names.add(current_model_name)
                model_paths = {
                    str(path)
                    for path in existing.get("modelPaths", [])
                    if isinstance(path, str) and path
                }
                if current_model_path:
                    model_paths.add(current_model_path)
                model_name = existing.get("modelName") or current_model_name
            else:
                artifacts = [row.get("proof")] if row.get("proof") else []
                surface_artifacts = row_surface_artifacts
                model_names = {current_model_name} if current_model_name else set()
                model_paths = {current_model_path} if current_model_path else set()
                model_name = current_model_name
            required_family_surfaces = REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY.get(
                family_id,
                REQUIRED_REAL_UI_LIVE_MODEL_SURFACES,
            )
            missing_family_surfaces = [
                surface
                for surface in required_family_surfaces
                if surface not in surfaces
            ]
            required_variants = REQUIRED_REAL_UI_LIVE_MODEL_VARIANTS_BY_FAMILY.get(
                family_id, {}
            )
            required_model_names = {
                str(name)
                for name in required_variants.get("model_names", set())
                if isinstance(name, str) and name
            }
            required_model_paths = {
                str(path)
                for path in required_variants.get("model_paths", set())
                if isinstance(path, str) and path
            }
            missing_required_model_variants = sorted(required_model_names - model_names)
            unexpected_model_names = sorted(
                model_names - required_model_names
            ) if required_model_names else []
            unexpected_model_paths = sorted(
                model_paths - required_model_paths
            ) if required_model_paths else []
            allowed_model_variant_union = bool(required_model_names) and (
                model_names == required_model_names
                and (not required_model_paths or model_paths == required_model_paths)
            )
            mixed_model_identity = (
                len(model_names) > 1 or len(model_paths) > 1
            ) and not allowed_model_variant_union
            covered_families[family_id] = {
                "status": (
                    "pass"
                    if (
                        not missing_family_surfaces
                        and not mixed_model_identity
                        and not missing_required_model_variants
                    )
                    else "partial"
                ),
                "modelName": model_name,
                "modelNames": sorted(model_names),
                "modelPaths": sorted(model_paths),
                "mixed_model_identity_union": mixed_model_identity,
                "allowed_model_variant_union": allowed_model_variant_union,
                "missing_required_model_variants": missing_required_model_variants,
                "unexpected_model_names": unexpected_model_names,
                "unexpected_model_paths": unexpected_model_paths,
                "artifact": artifacts[-1] if artifacts else row.get("proof"),
                "artifacts": artifacts,
                "required_surfaces": list(required_family_surfaces),
                "covered_surfaces": sorted(surfaces),
                "surface_artifacts": {
                    surface: artifact_list
                    for surface, artifact_list in sorted(surface_artifacts.items())
                    if surface in surfaces
                },
                "missing_surfaces": missing_family_surfaces,
            }

    covered_surface_union = {
        surface
        for family in covered_families.values()
        for surface in family.get("covered_surfaces", [])
    }
    missing_families = [
        family
        for family in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES
        if family not in covered_families
    ]
    missing_surfaces = [
        surface
        for surface in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES
        if surface not in covered_surface_union
    ]
    partial_families = [
        family
        for family in REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES
        if covered_families.get(family, {}).get("status") != "pass"
    ]
    mixed_model_identity_families = [
        family
        for family, row in covered_families.items()
        if row.get("mixed_model_identity_union") is True
    ]
    status = (
        "pass"
        if not missing_families and not missing_surfaces and not partial_families
        else "open"
    )
    return {
        "status": status,
        "release_blocker": (
            "Real Electron UI cross-family live model matrix is release-cleared"
        ),
        "required_families": list(REQUIRED_REAL_UI_LIVE_MODEL_FAMILIES),
        "required_surfaces": list(REQUIRED_REAL_UI_LIVE_MODEL_SURFACES),
        "required_surfaces_by_family": {
            family: list(surfaces)
            for family, surfaces in REQUIRED_REAL_UI_LIVE_MODEL_SURFACES_BY_FAMILY.items()
        },
        "covered_families": covered_families,
        "missing_families": missing_families,
        "missing_surfaces": missing_surfaces,
        "partial_families": partial_families,
        "mixed_model_identity_families": sorted(mixed_model_identity_families),
    }


def _real_ui_visible_assistant_turns_complete(
    proof: dict[str, Any],
    chat: dict[str, Any],
) -> bool:
    if proof.get("requestedEnableThinking") is not True:
        return True
    turns = chat.get("turns")
    if not isinstance(turns, list):
        return False
    saw_user = False
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        saw_user = True
        next_user_index = len(turns)
        for candidate_index in range(index + 1, len(turns)):
            candidate = turns[candidate_index]
            if isinstance(candidate, dict) and candidate.get("role") == "user":
                next_user_index = candidate_index
                break
        assistant_after_user = [
            candidate
            for candidate in turns[index + 1 : next_user_index]
            if isinstance(candidate, dict) and candidate.get("role") == "assistant"
        ]
        if not any(str(candidate.get("content") or "").strip() for candidate in assistant_after_user):
            return False
    return saw_user


def _validate_current_dev_ui_proof_artifacts(root: Path) -> dict[str, Any]:
    root = root.resolve()
    missing = [
        artifact
        for artifact in CURRENT_DEV_UI_PROOF_ARTIFACTS.values()
        if not (root / artifact).exists()
    ]
    failures: list[str] = []
    for key, artifact in CURRENT_DEV_UI_PROOF_ARTIFACTS.items():
        if not key.endswith("screenshot"):
            continue
        path = root / artifact
        if not path.exists():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            failures.append(f"screenshot_unreadable:{key}")
            continue
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 64:
            failures.append(f"invalid_png:{key}")

    proof: dict[str, Any] = {}
    proof_artifact = CURRENT_DEV_UI_PROOF_ARTIFACTS["proof"]
    proof_path = root / proof_artifact
    if proof_path.exists():
        try:
            loaded = json.loads(proof_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                proof = loaded
            else:
                failures.append("proof_json_not_object")
        except Exception as exc:  # noqa: BLE001 - report validation failure
            failures.append(f"proof_json_load_error:{type(exc).__name__}")

    if proof:
        if proof.get("repoDir") != str(root):
            failures.append("proof_repo_dir_mismatch")
        if proof.get("panelDir") != str(root / "panel"):
            failures.append("proof_panel_dir_mismatch")
        app_log_tail = proof.get("appLogTail")
        app_log_text = "\n".join(app_log_tail) if isinstance(app_log_tail, list) else ""
        if "http://localhost:5173/" not in app_log_text:
            failures.append("electron_dev_renderer_url_missing")
        if "start electron app" not in app_log_text:
            failures.append("electron_dev_launch_log_missing")
        screenshots = (
            proof.get("screenshots")
            if isinstance(proof.get("screenshots"), dict)
            else {}
        )
        expected_screenshots = {
            "chatSettings": str(
                (root / CURRENT_DEV_UI_PROOF_ARTIFACTS["chat_settings_screenshot"]).resolve()
            ),
            "serverCacheSettings": str(
                (root / CURRENT_DEV_UI_PROOF_ARTIFACTS["server_cache_screenshot"]).resolve()
            ),
        }
        for key, expected in expected_screenshots.items():
            if screenshots.get(key) != expected:
                failures.append(f"screenshot_path_mismatch:{key}")
        if proof.get("mockResponsesRequests") != 2:
            failures.append("mock_responses_request_count_not_two")
        if proof.get("finalAssistantContent") != "Done after tools.":
            failures.append("final_assistant_content_mismatch")
        if proof.get("rawThinkTagLeak") is not False:
            failures.append("raw_think_or_tool_tag_leak")
        if int(proof.get("persistedToolCount") or 0) < 14:
            failures.append("persisted_tool_count_too_low")
        if int(proof.get("runToolExecutingToResultMs") or 0) < 800:
            failures.append("tool_status_executing_result_gap_too_small")

        event_counts = proof.get("eventCounts") if isinstance(proof.get("eventCounts"), dict) else {}
        if int(event_counts.get("tool") or 0) < 14:
            failures.append("tool_event_count_too_low")
        if int(event_counts.get("reasoningDone") or 0) < 2:
            failures.append("reasoning_done_event_count_too_low")
        if int(event_counts.get("complete") or 0) < 1:
            failures.append("complete_event_missing")

        phases = proof.get("toolPhasesById") if isinstance(proof.get("toolPhasesById"), dict) else {}
        for tool_id in (
            "call_live_run",
            "call_live_list",
            "call_live_image",
            "call_live_video",
        ):
            seen = phases.get(tool_id) if isinstance(phases.get(tool_id), list) else []
            for phase in ("calling", "executing", "result"):
                if phase not in seen:
                    failures.append(f"{tool_id}_missing_{phase}")

        if proof.get("followupTailContentTypes") != ["text", "image_url", "video_url"]:
            failures.append("followup_media_content_types_mismatch")

        payload_keys = (
            proof.get("initialPayloadKeys")
            if isinstance(proof.get("initialPayloadKeys"), dict)
            else {}
        )
        for key in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if payload_keys.get(key) is not False:
                failures.append(f"unexpected_initial_payload_key:{key}")

        chat_ui = (
            proof.get("chatSettingsUi")
            if isinstance(proof.get("chatSettingsUi"), dict)
            else {}
        )
        server_ui = (
            proof.get("serverCacheUi")
            if isinstance(proof.get("serverCacheUi"), dict)
            else {}
        )
        if chat_ui.get("visible") is not True:
            failures.append("chat_settings_ui_not_visible")
        for label in (
            "Enable Built-in Coding Tools",
            "Working Directory",
            "Shell",
            "Search",
            "Utilities",
            "Hide Tool Status",
        ):
            if label not in (chat_ui.get("labels") or []):
                failures.append(f"chat_settings_label_missing:{label}")
        checked = chat_ui.get("checked") if isinstance(chat_ui.get("checked"), dict) else {}
        expected_checked = {
            "builtinToolsEnabled": True,
            "shellEnabled": True,
            "fileToolsEnabled": False,
            "gitEnabled": False,
            "hideToolStatus": True,
        }
        for key, expected in expected_checked.items():
            if checked.get(key) is not expected:
                failures.append(f"chat_settings_checked_mismatch:{key}")

        if server_ui.get("visible") is not True:
            failures.append("server_cache_ui_not_visible")
        for label in (
            "Enable Prefix Cache",
            "Use Paged KV Cache",
            "Block Disk Cache (L2)",
            "Enable Disk Cache",
            "Stored Cache Quantization",
        ):
            if label not in (server_ui.get("labels") or []):
                failures.append(f"server_cache_label_missing:{label}")
        block_toggle = (
            server_ui.get("afterBlockDiskToggle")
            if isinstance(server_ui.get("afterBlockDiskToggle"), dict)
            else {}
        )
        disk_toggle = (
            server_ui.get("afterDiskToggle")
            if isinstance(server_ui.get("afterDiskToggle"), dict)
            else {}
        )
        for key in ("enablePrefixCache", "usePagedCache", "enableBlockDiskCache"):
            if block_toggle.get(key) is not True:
                failures.append(f"block_disk_toggle_mismatch:{key}")
        if disk_toggle.get("enablePrefixCache") is not True:
            failures.append("legacy_disk_toggle_prefix_not_enabled")
        if disk_toggle.get("usePagedCache") is not False:
            failures.append("legacy_disk_toggle_paged_not_disabled")
        if disk_toggle.get("enableDiskCache") is not True:
            failures.append("legacy_disk_toggle_disk_not_enabled")

    return {
        "status": "pass" if not missing and not failures else "fail",
        "artifacts": dict(CURRENT_DEV_UI_PROOF_ARTIFACTS),
        "missing": missing,
        "failures": failures,
    }


def _validate_current_diagnostic_live_smoke_artifacts(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    not_expected: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}

    for row_id, expectation in CURRENT_DIAGNOSTIC_LIVE_SMOKE_ARTIFACTS.items():
        artifact = str(expectation["artifact"])
        expected_status = str(expectation["expected_status"])
        expected_label_reasons = dict(expectation["expected_label_reasons"])
        path = root / artifact
        if not path.exists():
            missing.append(artifact)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
            entry = {
                "artifact": artifact,
                "status": f"load_error:{type(exc).__name__}",
                "completed": 0,
                "failed": None,
                "matched_expected_failures": [],
                "unexpected_results": [],
            }
            artifacts[row_id] = entry
            not_expected.append(entry)
            continue

        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        matched_expected_failures: list[dict[str, str]] = []
        observed_label_reasons: dict[str, set[str]] = {}
        unexpected_results: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                unexpected_results.append(
                    {"model": None, "status": "invalid_result", "failures": []}
                )
                continue
            failures = result.get("failures")
            if not isinstance(failures, list):
                failures = []
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                label = failure.get("label")
                reason = failure.get("reason")
                if isinstance(label, str) and isinstance(reason, str):
                    observed_label_reasons.setdefault(label, set()).add(reason)

        for label, reason_or_reasons in expected_label_reasons.items():
            reasons = (
                [str(item) for item in reason_or_reasons]
                if isinstance(reason_or_reasons, list)
                else [str(reason_or_reasons)]
            )
            observed_reasons = observed_label_reasons.get(label, set())
            for reason in reasons:
                if reason not in observed_reasons:
                    unexpected_results.append(
                        {
                            "model": row_id,
                            "status": str(payload.get("status")),
                            "missing_expected_failure": {
                                "label": str(label),
                                "reason": str(reason),
                            },
                        }
                    )
                    continue
                matched_expected_failures.append(
                    {"label": str(label), "reason": str(reason)}
                )

        completed = payload.get("completed")
        try:
            completed_count = int(completed)
        except (TypeError, ValueError):
            completed_count = 0
        failed = payload.get("failed")
        try:
            failed_count = int(failed) if failed is not None else 0
        except (TypeError, ValueError):
            failed_count = 0
        status = str(payload.get("status")) if payload.get("status") is not None else ""
        if status != expected_status:
            unexpected_results.append(
                {
                    "model": row_id,
                    "status": status,
                    "expected_status": expected_status,
                }
            )
        if completed_count <= 0 or not results or failed_count <= 0:
            unexpected_results.append(
                {
                    "model": row_id,
                    "status": status,
                    "completed": completed_count,
                    "failed": failed_count,
                    "reason": "diagnostic_artifact_did_not_record_failure",
                }
            )

        entry = {
            "artifact": artifact,
            "status": status,
            "completed": completed_count,
            "failed": failed_count,
            "matched_expected_failures": matched_expected_failures,
            "unexpected_results": unexpected_results,
        }
        artifacts[row_id] = entry
        if unexpected_results:
            not_expected.append(entry)

    return {
        "status": "pass" if not missing and not not_expected else "fail",
        "artifacts": artifacts,
        "missing": missing,
        "not_expected": not_expected,
    }


def _validate_current_covered_live_smoke_artifacts(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    not_pass: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}

    for row_id, artifact in CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.items():
        path = root / artifact
        if not path.exists():
            missing.append(artifact)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
            status = f"load_error:{type(exc).__name__}"
            entry = {
                "artifact": artifact,
                "status": status,
                "completed": 0,
                "failed": None,
                "failing_results": [],
            }
            artifacts[row_id] = entry
            not_pass.append(entry)
            continue

        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        failing_results = []
        for result in results:
            if not isinstance(result, dict):
                failing_results.append(
                    {"model": None, "status": "invalid_result", "failures": []}
                )
                continue
            failures = result.get("failures")
            if not isinstance(failures, list):
                failures = []
            status = str(result.get("status"))
            requests = result.get("requests")
            if not isinstance(requests, list):
                requests = []
            seen_labels = {
                str(request.get("label"))
                for request in requests
                if isinstance(request, dict) and request.get("label") is not None
            }
            missing_request_labels = [
                label
                for label in _required_live_smoke_request_labels(row_id, result)
                if label not in seen_labels
            ]
            identity_mismatches = _live_smoke_identity_mismatches(row_id, result)
            capability_mismatches = _live_smoke_capability_mismatches(row_id, result)
            request_validation_failures: list[str] = []
            for request in requests:
                if not isinstance(request, dict):
                    continue
                request_validation_failures.extend(
                    _declared_request_validation_failures(request)
                )
            surface_issues = (
                _live_smoke_surface_issues(result)
                if (
                    status == "pass"
                    and not failures
                    and not missing_request_labels
                    and not identity_mismatches
                    and not capability_mismatches
                    and not request_validation_failures
                )
                else []
            )
            if (
                status == "pass"
                and not failures
                and not missing_request_labels
                and not identity_mismatches
                and not capability_mismatches
                and not surface_issues
                and not request_validation_failures
            ):
                for request in requests:
                    if not isinstance(request, dict):
                        continue
                    request_validation_failures.extend(
                        _live_smoke_cache_validation_failures(request)
                    )
            if (
                status != "pass"
                or failures
                or missing_request_labels
                or identity_mismatches
                or capability_mismatches
                or surface_issues
                or request_validation_failures
            ):
                row = result.get("row")
                row_name = row.get("name") if isinstance(row, dict) else None
                failing_result = {
                    "model": (
                        result.get("model")
                        or result.get("model_id")
                        or row_name
                    ),
                    "status": status,
                    "failures": [str(item) for item in failures],
                    "missing_request_labels": missing_request_labels,
                    "request_validation_failures": request_validation_failures,
                    "identity_mismatches": identity_mismatches,
                }
                if capability_mismatches:
                    failing_result["capability_mismatches"] = capability_mismatches
                if surface_issues:
                    failing_result["surface_issues"] = surface_issues
                failing_results.append(failing_result)

        completed = payload.get("completed")
        try:
            completed_count = int(completed)
        except (TypeError, ValueError):
            completed_count = 0
        failed = payload.get("failed")
        try:
            failed_count = int(failed) if failed is not None else len(failing_results)
        except (TypeError, ValueError):
            failed_count = len(failing_results)
        payload_status = payload.get("status")
        status = str(payload_status) if payload_status is not None else "missing_status"
        if (
            row_id in LEGACY_LIVE_SMOKE_MISSING_STATUS_ROWS
            and payload_status is None
            and completed_count > 0
            and results
            and not failed_count
            and not failing_results
        ):
            status = "pass"
        entry = {
            "artifact": artifact,
            "status": status,
            "completed": completed_count,
            "failed": failed_count,
            "failing_results": failing_results,
        }
        artifacts[row_id] = entry
        if (
            status != "pass"
            or completed_count <= 0
            or not results
            or failed_count
            or failing_results
        ):
            not_pass.append(entry)

    mimo_artifact = CURRENT_COVERED_LIVE_SMOKE_ARTIFACTS.get("mimo_v2_jang2l")
    non_mimo_missing = [artifact for artifact in missing if artifact != mimo_artifact]
    non_mimo_not_pass = [
        entry for entry in not_pass if entry.get("artifact") != mimo_artifact
    ]
    return {
        "status": "pass" if not missing and not not_pass else "fail",
        "non_mimo_status": (
            "pass" if not non_mimo_missing and not non_mimo_not_pass else "fail"
        ),
        "artifacts": artifacts,
        "missing": missing,
        "not_pass": not_pass,
        "non_mimo_missing": non_mimo_missing,
        "non_mimo_not_pass": non_mimo_not_pass,
    }


def _tool_smoke_request_validation_failures(request: dict[str, Any]) -> list[str]:
    failures = [
        str(item)
        for item in request.get("validation_failures") or []
        if item
    ]
    tool_calls = request.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    matching_call = None
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") == "record_fact":
            matching_call = function
            break
    if matching_call is None:
        failures.append("tool_required:expected_tool_missing")
        return failures
    try:
        args = json.loads(str(matching_call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        args = {}
    if args.get("value") != "blue-cat":
        failures.append("tool_required:expected_tool_argument_missing")
    content_head = request.get("content_head")
    if isinstance(content_head, str) and content_head.strip():
        failures.append("tool_required:visible_text_leaked")
    return failures


def _validate_current_covered_live_tool_smoke_artifacts(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    not_pass: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}

    for row_id, artifact in CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.items():
        path = root / artifact
        if not path.exists():
            missing.append(artifact)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
            status = f"load_error:{type(exc).__name__}"
            entry = {
                "artifact": artifact,
                "status": status,
                "completed": 0,
                "failed": None,
                "failing_results": [],
            }
            artifacts[row_id] = entry
            not_pass.append(entry)
            continue

        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        failing_results = []
        for result in results:
            if not isinstance(result, dict):
                failing_results.append(
                    {
                        "model": None,
                        "status": "invalid_result",
                        "failures": [],
                        "missing_request_labels": ["tool_required"],
                        "request_validation_failures": [],
                        "identity_mismatches": {},
                    }
                )
                continue
            failures = result.get("failures")
            if not isinstance(failures, list):
                failures = []
            status = str(result.get("status"))
            requests = result.get("requests")
            if not isinstance(requests, list):
                requests = []
            seen_labels = {
                str(request.get("label"))
                for request in requests
                if isinstance(request, dict) and request.get("label") is not None
            }
            missing_request_labels = [
                label for label in ("tool_required",) if label not in seen_labels
            ]
            identity_mismatches = _live_smoke_identity_mismatches(row_id, result)
            capability_mismatches = _live_smoke_capability_mismatches(row_id, result)
            request_validation_failures: list[str] = []
            for request in requests:
                if not isinstance(request, dict):
                    continue
                request_validation_failures.extend(
                    _declared_request_validation_failures(request)
                )
                if request.get("label") == "tool_required":
                    request_validation_failures.extend(
                        _tool_smoke_request_validation_failures(request)
                    )
            surface_issues = (
                _live_smoke_surface_issues(result)
                if (
                    status == "pass"
                    and not failures
                    and not missing_request_labels
                    and not identity_mismatches
                    and not capability_mismatches
                    and not request_validation_failures
                )
                else []
            )
            if (
                status == "pass"
                and not failures
                and not missing_request_labels
                and not identity_mismatches
                and not capability_mismatches
                and not surface_issues
                and not request_validation_failures
            ):
                for request in requests:
                    if not isinstance(request, dict):
                        continue
                    request_validation_failures.extend(
                        _live_smoke_cache_validation_failures(request)
                    )
            if (
                status != "pass"
                or failures
                or missing_request_labels
                or identity_mismatches
                or capability_mismatches
                or surface_issues
                or request_validation_failures
            ):
                row = result.get("row")
                row_name = row.get("name") if isinstance(row, dict) else None
                failing_result = {
                    "model": (
                        result.get("model")
                        or result.get("model_id")
                        or row_name
                    ),
                    "status": status,
                    "failures": [str(item) for item in failures],
                    "missing_request_labels": missing_request_labels,
                    "request_validation_failures": request_validation_failures,
                    "identity_mismatches": identity_mismatches,
                }
                if capability_mismatches:
                    failing_result["capability_mismatches"] = capability_mismatches
                if surface_issues:
                    failing_result["surface_issues"] = surface_issues
                failing_results.append(failing_result)

        completed = payload.get("completed")
        try:
            completed_count = int(completed)
        except (TypeError, ValueError):
            completed_count = 0
        failed = payload.get("failed")
        try:
            failed_count = int(failed) if failed is not None else len(failing_results)
        except (TypeError, ValueError):
            failed_count = len(failing_results)
        payload_status = payload.get("status")
        status = str(payload_status) if payload_status is not None else "missing_status"
        entry = {
            "artifact": artifact,
            "status": status,
            "completed": completed_count,
            "failed": failed_count,
            "failing_results": failing_results,
        }
        artifacts[row_id] = entry
        if (
            status != "pass"
            or completed_count <= 0
            or not results
            or failed_count
            or failing_results
        ):
            not_pass.append(entry)

    mimo_artifact = CURRENT_COVERED_LIVE_TOOL_SMOKE_ARTIFACTS.get("mimo_v2_jang2l")
    non_mimo_missing = [artifact for artifact in missing if artifact != mimo_artifact]
    non_mimo_not_pass = [
        entry for entry in not_pass if entry.get("artifact") != mimo_artifact
    ]
    return {
        "status": "pass" if not missing and not not_pass else "fail",
        "non_mimo_status": (
            "pass" if not non_mimo_missing and not non_mimo_not_pass else "fail"
        ),
        "artifacts": artifacts,
        "missing": missing,
        "not_pass": not_pass,
        "non_mimo_missing": non_mimo_missing,
        "non_mimo_not_pass": non_mimo_not_pass,
    }


def _live_smoke_gap_is_expected_mimo_open(
    summary: dict[str, Any],
    expected_artifact: str,
) -> bool:
    return False


def _validate_current_regression_suite_artifact(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact": CURRENT_REGRESSION_SUITE_ARTIFACT,
        "status": "missing",
        "failed_steps": [],
        "open_requirements": [],
        "open_requirement_details": {},
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS.copy(),
    }
    path = root / CURRENT_REGRESSION_SUITE_ARTIFACT
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    open_requirements = [str(item) for item in payload.get("open_requirements", [])]
    open_requirement_details = payload.get("open_requirement_details")
    if not isinstance(open_requirement_details, dict):
        open_requirement_details = {}
    failed_steps = [str(item) for item in payload.get("failed_steps", [])]
    open_requirement_detail_failures = _validate_open_requirement_details(
        open_requirement_details
    )
    progress_checkpoint_failures = _validate_current_suite_progress_checkpoints(
        payload
    )
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    stale_source_hashes: list[dict[str, str]] = []
    missing_source_hashes: list[str] = []
    for rel in CURRENT_SUITE_SOURCE_HASH_FILES:
        artifact_hash = source_hashes.get(rel)
        path = root / rel
        if not path.exists():
            continue
        if not isinstance(artifact_hash, str):
            missing_source_hashes.append(rel)
            continue
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_hash != artifact_hash:
            stale_source_hashes.append(
                {
                    "path": rel,
                    "artifact_sha256": artifact_hash,
                    "current_sha256": current_hash,
                }
            )
    unexpected_open_requirements = [
        item for item in open_requirements if item not in EXPECTED_CURRENT_OPEN_REQUIREMENTS
    ]
    missing_expected_open_requirements = [
        item for item in EXPECTED_CURRENT_OPEN_REQUIREMENTS if item not in open_requirements
    ]

    result.update(
        {
            "status": str(payload.get("status")),
            "failed_steps": failed_steps,
            "open_requirements": open_requirements,
            "open_requirement_details": open_requirement_details,
            "unexpected_open_requirements": unexpected_open_requirements,
            "missing_expected_open_requirements": missing_expected_open_requirements,
            "open_requirement_detail_failures": open_requirement_detail_failures,
        }
    )
    if source_hashes:
        result["source_hashes"] = source_hashes
    if missing_source_hashes:
        result["missing_source_hashes"] = missing_source_hashes
    if stale_source_hashes:
        result["stale_source_hashes"] = stale_source_hashes
    if progress_checkpoint_failures:
        result["progress_checkpoint_failures"] = progress_checkpoint_failures
    return result


def _validate_current_suite_progress_checkpoints(
    payload: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if "current_step" not in payload:
        failures.append("missing_current_step")
    if "completed_steps" not in payload:
        failures.append("missing_completed_steps")

    completed_steps = payload.get("completed_steps")
    if "completed_steps" in payload and not isinstance(completed_steps, list):
        failures.append("completed_steps_not_list")

    steps = payload.get("steps")
    if not isinstance(steps, dict):
        return failures

    step_names = [str(name) for name in steps]
    completed_step_names: list[str] = []
    if isinstance(completed_steps, list):
        completed_step_names = [str(name) for name in completed_steps]
        unknown_completed = [
            name for name in completed_step_names if name not in step_names
        ]
        if unknown_completed:
            failures.append(
                f"completed_steps_unknown:{','.join(unknown_completed[:5])}"
            )

    current_step = payload.get("current_step")
    if current_step is not None and str(current_step) not in step_names:
        failures.append(f"current_step_unknown:{current_step}")

    if (
        "current_step" in payload
        and current_step is None
        and payload.get("status") in {"open", "pass"}
        and isinstance(completed_steps, list)
    ):
        missing_completed = [
            name for name in step_names if name not in completed_step_names
        ]
        if missing_completed:
            failures.append(
                f"completed_steps_missing_final_steps:{','.join(missing_completed[:5])}"
            )
    return failures


def _current_regression_suite_state_is_acceptable(
    regression_suite: dict[str, Any],
) -> bool:
    if regression_suite["unexpected_open_requirements"]:
        return False
    if regression_suite["missing_expected_open_requirements"]:
        return False
    if regression_suite.get("missing_source_hashes"):
        return False
    if regression_suite.get("stale_source_hashes"):
        return False
    if regression_suite.get("open_requirement_detail_failures"):
        return False
    if regression_suite.get("progress_checkpoint_failures"):
        return False
    if regression_suite["status"] == "pass" and not regression_suite["failed_steps"]:
        return True
    if regression_suite["status"] == "open" and not regression_suite["failed_steps"]:
        return True
    return (
        regression_suite["status"] == "open"
        and regression_suite["failed_steps"] == ["release_regression_manifest"]
    )


def _validate_current_objective_digest_artifact(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "artifact": CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
        "status": "missing",
        "open_requirements": [],
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS.copy(),
        "invalid_requirements": [],
        "failed_requirements": [],
        "missing_evidence_requirements": [],
        "open_requirement_details": {},
        "open_requirement_detail_failures": [],
    }
    path = root / CURRENT_OBJECTIVE_DIGEST_ARTIFACT
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        result["status"] = "invalid"
        result["invalid_requirements"] = ["requirements_not_list"]
        return result

    open_requirements: list[str] = []
    open_requirement_details: dict[str, dict[str, Any]] = {}
    invalid_requirements: list[dict[str, Any]] = []
    failed_requirements: list[dict[str, str]] = []
    missing_evidence_requirements: list[dict[str, Any]] = []
    for index, item in enumerate(requirements):
        if not isinstance(item, dict):
            invalid_requirements.append({"index": index, "reason": "not_object"})
            continue
        requirement = str(item.get("requirement") or item.get("name") or "")
        status = str(item.get("status") or "")
        if not requirement:
            invalid_requirements.append(
                {"index": index, "reason": "missing_requirement"}
            )
            continue
        if status == "open":
            open_requirements.append(requirement)
            open_requirement_details[requirement] = {
                key: item[key]
                for key in ("status", "caveat", "evidence", "details")
                if key in item
            }
        elif status != "pass":
            failed_requirements.append(
                {"requirement": requirement, "status": status or "missing"}
            )

        details = item.get("details")
        if not isinstance(details, dict):
            continue
        missing_evidence = details.get("missing_evidence")
        if not isinstance(missing_evidence, list):
            missing_evidence = []
        evidence_files_present = details.get("evidence_files_present")
        evidence_map_failures: list[str] = []
        if isinstance(evidence_files_present, dict):
            evidence_map_failures = [
                str(key) for key, value in evidence_files_present.items() if not value
            ]
        elif evidence_files_present is False:
            evidence_map_failures = ["evidence_files_present_false"]
        if missing_evidence or evidence_map_failures:
            missing_evidence_requirements.append(
                {
                    "requirement": requirement,
                    "missing_evidence": [str(item) for item in missing_evidence],
                    "evidence_files_present_failures": evidence_map_failures,
                }
            )

    unexpected_open_requirements = [
        item for item in open_requirements if item not in EXPECTED_CURRENT_OPEN_REQUIREMENTS
    ]
    missing_expected_open_requirements = [
        item for item in EXPECTED_CURRENT_OPEN_REQUIREMENTS if item not in open_requirements
    ]
    open_requirement_detail_failures = (
        _validate_objective_digest_open_requirement_details(open_requirement_details)
    )
    result.update(
        {
            "status": "pass",
            "open_requirements": open_requirements,
            "unexpected_open_requirements": unexpected_open_requirements,
            "missing_expected_open_requirements": missing_expected_open_requirements,
            "invalid_requirements": invalid_requirements,
            "failed_requirements": failed_requirements,
            "missing_evidence_requirements": missing_evidence_requirements,
            "open_requirement_details": open_requirement_details,
            "open_requirement_detail_failures": open_requirement_detail_failures,
        }
    )
    return result


def _validate_objective_digest_open_requirement_details(
    open_requirement_details: dict[str, Any],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    requirement = "MiniMax-M2.7-JANGTQ_K reporter parity/root cause is release-cleared"
    if requirement not in EXPECTED_CURRENT_OPEN_REQUIREMENTS:
        return failures
    row = open_requirement_details.get(requirement)
    if not isinstance(row, dict):
        failures.append(
            {
                "requirement": requirement,
                "reason": "missing_or_stale_issue179_root_cause_boundary",
            }
        )
        return failures
    details = row.get("details")
    if not isinstance(details, dict):
        details = {}
    not_proven = details.get("not_proven")
    if not isinstance(not_proven, list):
        not_proven = []
    release_boundary = details.get("release_boundary")
    if (
        details.get("release_blocker_id") != "issue179_minimax_k_root_cause_audit"
        or not any(
            str(item)
            == "reporter installed app bundle hash matches public/local server.py route proof"
            for item in not_proven
        )
        or not isinstance(release_boundary, str)
        or "#179 remains open" not in release_boundary
    ):
        failures.append(
            {
                "requirement": requirement,
                "reason": "missing_or_stale_issue179_root_cause_boundary",
            }
        )
    return failures


def _validate_open_requirement_details(
    open_requirement_details: dict[str, Any],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    real_ui_requirement = (
        "Real Electron UI cross-family live model matrix is release-cleared"
    )
    real_ui_row = open_requirement_details.get(real_ui_requirement)
    if not isinstance(real_ui_row, dict):
        failures.append(
            {
                "requirement": real_ui_requirement,
                "reason": "missing_or_stale_real_ui_live_matrix_boundary",
            }
        )
    else:
        real_ui_details = real_ui_row.get("details")
        if not isinstance(real_ui_details, dict):
            real_ui_details = {}
        surfaces = real_ui_details.get("required_real_ui_surfaces")
        if not isinstance(surfaces, list):
            surfaces = []
        required_surfaces = {
            "current Electron dev build",
            "real loaded model per target family",
            "chat settings persistence and non-sticky defaults",
            "server cache setting controls",
            "Responses and Chat Completions paths",
            "long multi-turn tool calls",
            "reasoning/tool parser display",
            "prefix/cache hit telemetry",
            "image/video follow-up where supported",
            "no raw parser or reasoning tag leakage",
        }
        real_ui_matrix = real_ui_details.get("real_ui_live_model_matrix")
        if not isinstance(real_ui_matrix, dict):
            real_ui_matrix = {}
        valid_boundary = real_ui_details.get("release_boundary") in {
            "mock_ui_plus_server_smoke_is_not_real_ui_live_model_clearance",
            "current_real_ui_matrix_is_authoritative_for_unblocked_non_mimo",
        }
        valid_current_matrix = (
            real_ui_details.get("release_boundary")
            == "current_real_ui_matrix_is_authoritative_for_unblocked_non_mimo"
            and real_ui_matrix.get("unblocked_non_mimo_status") in {"pass", "open"}
            and isinstance(real_ui_matrix.get("covered_families"), dict)
            and isinstance(real_ui_matrix.get("mixed_model_identity_families"), list)
            and isinstance(
                real_ui_matrix.get("unblocked_non_mimo_missing_families"), list
            )
            and isinstance(
                real_ui_matrix.get("unblocked_non_mimo_partial_families"), list
            )
        )
        valid_legacy_boundary = (
            real_ui_details.get("release_boundary")
            == "mock_ui_plus_server_smoke_is_not_real_ui_live_model_clearance"
        )
        if (
            not valid_boundary
            or not required_surfaces.issubset({str(item) for item in surfaces})
            or not (valid_current_matrix or valid_legacy_boundary)
        ):
            failures.append(
                {
                    "requirement": real_ui_requirement,
                    "reason": "missing_or_stale_real_ui_live_matrix_boundary",
                }
            )

    requirement = "DSV4 long-output/code/file-generation quality is release-cleared"
    row = open_requirement_details.get(requirement)
    if not isinstance(row, dict):
        failures.append(
            {
                "requirement": requirement,
                "reason": "missing_or_stale_exact_code_root_boundary",
            }
        )
        return failures
    details = row.get("details")
    if not isinstance(details, dict):
        details = {}
    direct = details.get("direct_off_exactness_boundary")
    if not isinstance(direct, dict):
        direct = {}
    root = details.get("exact_code_root_boundary")
    if not isinstance(root, dict):
        root = {}
    source_preflight = details.get("current_source_full_output_preflight")
    if not isinstance(source_preflight, dict):
        source_preflight = {}

    required_direct = (
        direct.get("direct_off_failure_spans_chat_responses_and_completion") is True
        and direct.get("requested_thinking_success_is_diagnostic_only") is True
        and direct.get("requested_thinking_success_clears_direct_off") is False
        and direct.get("hidden_force_on_would_be_false_clearance") is True
    )
    current_memory_gated_direct = (
        direct.get("present") is True
        and direct.get("direct_off_release_blocker_active") is True
        and direct.get("requested_thinking_success_is_diagnostic_only") is True
        and direct.get("requested_thinking_success_clears_direct_off") is False
        and direct.get("hidden_force_on_would_be_false_clearance") is True
    )
    required_root = (
        root.get("direct_off_route_wide_failure") is True
        and root.get("requested_thinking_is_diagnostic_only") is True
        and root.get("prefix_cache_not_sufficient_root_cause") is True
        and root.get("forced_sampler_controls_not_sufficient_root_cause") is True
        and root.get("bundle_defaults_do_not_clear_direct_off") is True
        and root.get("direct_off_three_dot_identifier_boundary") is True
        and root.get("current_primary_failure")
        == "direct_off_exact_code_generation"
    )
    current_memory_gated_root = (
        root.get("present") is True
        and root.get("current_primary_failure") == "insufficient_evidence"
        and root.get("source_full_output_clearance_missing") is True
        and root.get("requested_thinking_is_diagnostic_only") is True
    )
    if not (
        (required_direct and required_root)
        or (current_memory_gated_direct and current_memory_gated_root)
    ):
        failures.append(
            {
                "requirement": requirement,
                "reason": "missing_or_stale_exact_code_root_boundary",
            }
        )
    required_available = source_preflight.get("required_available_gb")
    gate_available = source_preflight.get("available_for_gate_gb")
    if not isinstance(gate_available, int | float):
        gate_available = source_preflight.get("free_plus_speculative_purgeable_gb")
    if not isinstance(gate_available, int | float):
        gate_available = source_preflight.get("available_gb")
    source_preflight_created_at = source_preflight.get("created_at")
    active_heavy_processes = source_preflight.get("active_heavy_processes")
    selected_cases = source_preflight.get("selected_cases")
    if not isinstance(selected_cases, list):
        selected_cases = []
    selected_case_names = {str(name) for name in selected_cases}
    required_source_preflight_cases = set(EXPECTED_DSV4_SOURCE_PREFLIGHT_CASES)
    case_count = source_preflight.get("case_count")
    required_source_preflight = (
        source_preflight.get("artifact_present") is True
        and source_preflight.get("artifact")
        == CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT
        and isinstance(source_preflight_created_at, str)
        and bool(source_preflight_created_at)
        and source_preflight.get("status") == "skipped"
        and source_preflight.get("reason")
        in {"insufficient_free_memory", "insufficient_vm_stat_memory"}
        and source_preflight.get("did_not_launch") is True
        and source_preflight.get("launch_decision") == "do_not_launch"
        and source_preflight.get("launch_allowed") is False
        and str(source_preflight.get("model") or "").endswith(
            (
                "DeepSeek-V4-Flash-JANGTQ-K-HeadBF16-Probe-20260520",
                "DeepSeek-V4-Flash-JANGTQ-K",
            )
        )
        and source_preflight.get("preflight_memory_source")
        == "vm_stat_free_plus_speculative_purgeable"
        and source_preflight.get("active_heavy_process_count") == 0
        and isinstance(active_heavy_processes, list)
        and not active_heavy_processes
        and isinstance(required_available, int | float)
        and required_available >= 120.0
        and isinstance(gate_available, int | float)
        and gate_available < required_available
        and required_source_preflight_cases.issubset(selected_case_names)
        and isinstance(case_count, int)
        and case_count >= len(EXPECTED_DSV4_SOURCE_PREFLIGHT_CASES)
    )
    if not required_source_preflight:
        failures.append(
            {
                "requirement": requirement,
                "reason": "missing_or_stale_dsv4_source_full_output_preflight",
            }
        )
    return failures


def _validate_current_cache_architecture_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["cache-architecture-family-classification"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "missing_api_checks": [],
        "missing_api_command_markers": [],
        "missing_panel_markers": [],
        "missing_family_rows": list(EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS),
        "failed_family_rows": list(EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS),
        "cache_family_matrix": {},
        "failed_checks": list(EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_CACHE_ARCHITECTURE_CHECKS if checks.get(name) is not True
    ]
    cache_family_matrix = {
        str(row_id): dict(row)
        for row_id, row in dict(payload.get("cache_family_matrix", {})).items()
    }
    missing_family_rows = [
        row for row in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS if row not in cache_family_matrix
    ]
    failed_family_rows = [
        row
        for row in EXPECTED_CURRENT_CACHE_FAMILY_MATRIX_ROWS
        if cache_family_matrix.get(row, {}).get("status") != "pass"
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "missing_api_checks": [str(item) for item in payload.get("missing_api_checks", [])],
            "missing_api_command_markers": [
                str(item) for item in payload.get("missing_api_command_markers", [])
            ],
            "missing_panel_markers": [
                str(item) for item in payload.get("missing_panel_markers", [])
            ],
            "missing_family_rows": missing_family_rows,
            "failed_family_rows": failed_family_rows,
            "cache_family_matrix": cache_family_matrix,
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_model_artifact_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-artifact-format-detection"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_markers = [str(item) for item in payload.get("missing_markers", [])]
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_MODEL_ARTIFACT_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": missing_markers,
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_parser_registry_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["parser-registry-tool-reasoning-parity"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_PARSER_REGISTRY_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_generation_defaults_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["generation-defaults-no-hidden-forcing"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "missing_family_rows": list(EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS),
        "failed_family_rows": list(EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS),
        "generation_defaults_family_matrix": {},
        "failed_checks": list(EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_GENERATION_DEFAULTS_CHECKS if checks.get(name) is not True
    ]
    generation_defaults_family_matrix = {
        str(row_id): dict(row)
        for row_id, row in dict(
            payload.get("generation_defaults_family_matrix", {})
        ).items()
    }
    missing_family_rows = [
        row
        for row in EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS
        if row not in generation_defaults_family_matrix
    ]
    failed_family_rows = [
        row
        for row in EXPECTED_CURRENT_GENERATION_DEFAULTS_FAMILY_MATRIX_ROWS
        if generation_defaults_family_matrix.get(row, {}).get("status") != "pass"
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "missing_family_rows": missing_family_rows,
            "failed_family_rows": failed_family_rows,
            "generation_defaults_family_matrix": generation_defaults_family_matrix,
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_noheavy_api_cache_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["noheavy-api-cache-endpoint-runtime"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS if name not in checks
    ]
    failed_checks = [
        name
        for name in EXPECTED_CURRENT_NOHEAVY_API_CACHE_CHECKS
        if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_api_surface_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["api-chat-responses-anthropic-ollama-parity"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_nested_checks": [],
        "missing_nested_markers": [],
        "missing_panel_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_API_SURFACE_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_API_SURFACE_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_API_SURFACE_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_API_SURFACE_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_nested_checks": [
                str(item) for item in payload.get("missing_nested_checks", [])
            ],
            "missing_nested_markers": [
                str(item) for item in payload.get("missing_nested_markers", [])
            ],
            "missing_panel_markers": [
                str(item) for item in payload.get("missing_panel_markers", [])
            ],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_reasoning_template_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["reasoning-template-no-think-tag-leak"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_REASONING_TEMPLATE_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_tool_call_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["tool-call-loop-parser-cleanup"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_TOOL_CALL_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_TOOL_CALL_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_TOOL_CALL_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_panel_tool_security_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["panel-tool-security-loop-boundary"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "returncode": None,
        "missing": [artifact],
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    result.update(
        {
            "status": str(payload.get("status")),
            "returncode": payload.get("returncode"),
            "missing": [],
        }
    )
    return result


def _validate_current_native_mtp_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["native-mtp-d3-effect-policy"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_NATIVE_MTP_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_NATIVE_MTP_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_NATIVE_MTP_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_vl_media_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["vl-media-cache-tool-followup"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_engine_markers": [],
        "missing_panel_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_VL_MEDIA_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_VL_MEDIA_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_VL_MEDIA_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_engine_markers": [
                str(item) for item in payload.get("missing_engine_markers", [])
            ],
            "missing_panel_markers": [
                str(item) for item in payload.get("missing_panel_markers", [])
            ],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_mcp_policy_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["mcp-policy-ui-gateway"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed_checks": list(EXPECTED_CURRENT_MCP_POLICY_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_MCP_POLICY_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS if name not in checks
    ]
    failed_checks = [
        name for name in EXPECTED_CURRENT_MCP_POLICY_CHECKS if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_max_output_context_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["chat-settings-max-output-context-ui"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "missing_markers": [],
        "failed": [],
        "failed_checks": list(EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS if name not in checks
    ]
    failed_checks = [
        name
        for name in EXPECTED_CURRENT_MAX_OUTPUT_CONTEXT_CHECKS
        if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "missing_markers": [str(item) for item in payload.get("missing_markers", [])],
            "failed": [str(item) for item in payload.get("failed", [])],
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_packaged_integrity_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["packaged-release-integrity"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "failed": [],
        "known_expected_release_gate_open_requirements": [],
        "package_signing_preflight": {},
        "release_blockers": [],
        "unexpected_open_requirements": [],
        "missing_expected_open_requirements": EXPECTED_CURRENT_OPEN_REQUIREMENTS.copy(),
        "failed_checks": list(EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    open_requirements = [
        str(item)
        for item in payload.get("known_expected_release_gate_open_requirements", [])
    ]
    unexpected_open_requirements = [
        item for item in open_requirements if item not in EXPECTED_CURRENT_OPEN_REQUIREMENTS
    ]
    missing_expected_open_requirements = [
        item for item in EXPECTED_CURRENT_OPEN_REQUIREMENTS if item not in open_requirements
    ]
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS if name not in checks
    ]
    failed_checks = [
        name
        for name in EXPECTED_CURRENT_PACKAGED_INTEGRITY_CHECKS
        if checks.get(name) is not True
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "failed": [str(item) for item in payload.get("failed", [])],
            "known_expected_release_gate_open_requirements": open_requirements,
            "package_signing_preflight": dict(
                payload.get("package_signing_preflight", {})
            )
            if isinstance(payload.get("package_signing_preflight"), dict)
            else {},
            "release_blockers": [
                dict(item)
                for item in payload.get("release_blockers", [])
                if isinstance(item, dict)
            ],
            "unexpected_open_requirements": unexpected_open_requirements,
            "missing_expected_open_requirements": missing_expected_open_requirements,
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
        }
    )
    return result


def _validate_current_jang_model_compat_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["jang-model-compat-runtime-boundary"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "failed": [],
        "missing": [artifact],
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    result.update(
        {
            "status": str(payload.get("status")),
            "failed": [str(item) for item in payload.get("failed", [])],
            "missing": [],
        }
    )
    return result


def _validate_current_release_surface_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["public-release-surface-preflight"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "checks": {},
        "failed_checks": list(EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS),
        "missing_expected_checks": list(EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    checks = {
        str(name): bool(value)
        for name, value in dict(payload.get("checks", {})).items()
    }
    missing_expected_checks = [
        name for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS if name not in checks
    ]
    payload_status_failed_checks = payload.get("status_failed_checks")
    if isinstance(payload_status_failed_checks, list):
        failed_checks = [
            str(name)
            for name in payload_status_failed_checks
            if str(name) in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
        ]
    else:
        failed_checks = [
            name
            for name in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
            if checks.get(name) is not True
        ]
    result.update(
        {
            "status": str(payload.get("status")),
            "checks": checks,
            "failed_checks": failed_checks,
            "missing_expected_checks": missing_expected_checks,
            "informational_false_checks": [
                str(name)
                for name in payload.get("informational_false_checks", [])
                if str(name) in EXPECTED_CURRENT_RELEASE_SURFACE_CHECKS
            ]
            if isinstance(payload.get("informational_false_checks"), list)
            else [],
        }
    )
    return result


def _validate_current_model_family_matrix_artifact(root: Path) -> dict[str, Any]:
    artifact = CURRENT_POST_BUDGET_EDGE_ARTIFACTS["model-family-detection-noheavy"]
    result: dict[str, Any] = {
        "artifact": artifact,
        "status": "missing",
        "matched_rows": [],
        "missing_rows": [],
        "missing_expected_rows": list(EXPECTED_CURRENT_MODEL_FAMILY_ROWS),
    }
    path = root / artifact
    if not path.exists():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic helper reports load failures
        result["status"] = f"load_error:{type(exc).__name__}"
        return result

    matched_rows = [str(item) for item in payload.get("matched_rows", [])]
    missing_rows = [str(item) for item in payload.get("missing_rows", [])]
    missing_expected_rows = [
        row for row in EXPECTED_CURRENT_MODEL_FAMILY_ROWS if row not in matched_rows
    ]
    result.update(
        {
            "status": str(payload.get("status")),
            "matched_rows": matched_rows,
            "missing_rows": missing_rows,
            "missing_expected_rows": missing_expected_rows,
        }
    )
    return result


CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS = (
    CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT,
    CURRENT_REGRESSION_SUITE_ARTIFACT,
    CURRENT_OBJECTIVE_DIGEST_ARTIFACT,
)
CURRENT_DSV4_PROOF_ARTIFACT_STALE_MARKERS = ("second-local-check",)
CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS = (
    CURRENT_REAL_UI_DSV4_MEMORY_PREFLIGHT_ARTIFACT,
    CURRENT_DSV4_SOURCE_MEMORY_PREFLIGHT_ARTIFACT,
)


def _stale_marker_scan_text(path: Path, text: str) -> str:
    if path.suffix != ".json":
        return text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if key != "stale_markers"
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return json.dumps(scrub(payload), sort_keys=True)


def validate_current_dsv4_proof_artifact_freshness(
    root: Path,
    *,
    require_manifest_artifact: bool = True,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    checked_artifacts = (
        CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS
        if require_manifest_artifact
        else tuple(
            artifact
            for artifact in CURRENT_DSV4_PROOF_ARTIFACT_FRESHNESS_ARTIFACTS
            if artifact != CURRENT_RELEASE_REGRESSION_MANIFEST_ARTIFACT
        )
    )
    for artifact in checked_artifacts:
        path = root / artifact
        if not path.exists():
            artifacts.append(
                {
                    "path": artifact,
                    "missing": True,
                    "stale_references": [],
                    "missing_required_artifacts": list(
                        CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS
                    ),
                }
            )
            continue

        text = path.read_text(encoding="utf-8")
        scan_text = _stale_marker_scan_text(path, text)
        artifacts.append(
            {
                "path": artifact,
                "missing": False,
                "stale_references": [
                    marker
                    for marker in CURRENT_DSV4_PROOF_ARTIFACT_STALE_MARKERS
                    if marker in scan_text
                ],
                "missing_required_artifacts": [
                    required
                    for required in CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS
                    if required not in text
                ],
            }
        )

    failures = [
        artifact
        for artifact in artifacts
        if artifact["missing"]
        or artifact["stale_references"]
        or artifact["missing_required_artifacts"]
    ]
    return {
        "status": "fail" if failures else "pass",
        "artifacts": artifacts,
        "checked_artifacts": list(checked_artifacts),
        "require_manifest_artifact": require_manifest_artifact,
        "stale_markers": list(CURRENT_DSV4_PROOF_ARTIFACT_STALE_MARKERS),
        "required_artifacts": list(CURRENT_DSV4_PROOF_ARTIFACT_REQUIRED_ARTIFACTS),
        "failures": failures,
    }
