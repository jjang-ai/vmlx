#!/usr/bin/env python3
"""Run the current-source no-heavy API/cache contract proof.

This helper intentionally does not load models. It runs focused tests that pin:

- OpenAI Chat Completions and Responses request sampling/default propagation;
- OpenAI legacy Completions output/context cap propagation;
- Anthropic and Ollama adapter surfaces;
- DSV4 native SWA+CSA/HCA cache status separate from generic TurboQuant KV;
- ZAYA typed CCA status;
- plain attention/KV native cache status;
- hybrid SSM partial reuse and cache-detail telemetry;
- TurboQuant KV runtime and disk-cache serialization contracts.
- MLXStudio gateway stale-port and standby-wake routing contracts.
- Structured JSON/schema repair and post-generation repair diagnostics.
- Structured XML repair/validation diagnostics for benchmark/catalog callers.
- Structured repair benchmark raw-vs-repaired/validated summary rates.
- Guided JSON/schema token masking on compatible llguidance text paths.
- Live-smoke benchmark request adoption for JSON schema response_format.
- One-shot strict JSON-only retry after unrecoverable repair/schema failure.
- One-shot strict XML-only retry after unrecoverable XML repair failure.
- Strict stream-end XML validation for Chat and Responses SSE.
- API docs boundary for repair/validation vs hard constrained decoding.
- Responses SSE function-call argument, output-index, and required-arg fail-closed
  contracts.
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


DEFAULT_OUT = Path("build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json")
SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/api/models.py",
    "vmlx_engine/api/tool_calling.py",
    "vmlx_engine/tool_parsers/dsml_tool_parser.py",
    "vmlx_engine/api/anthropic_adapter.py",
    "vmlx_engine/api/ollama_adapter.py",
    "docs/guides/server.md",
    "docs/reference/configuration.md",
    "bench/structured_output_repair_report.py",
    "bench/all_local_model_smoke.py",
    "tests/test_all_local_model_smoke.py",
    "tests/test_structured_output.py",
    "tests/test_structured_output_repair_report.py",
    "tests/test_llm.py",
    "tests/test_server.py",
    "tests/test_engine_audit.py",
    "tests/test_batching.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_tq_disk_cache.py",
    "tests/test_dsml_tool_parser.py",
    "tests/test_responses_history.py",
    "tests/test_tool_format.py",
    "panel/src/main/api-gateway.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/src/renderer/src/components/chat/ToolCallStatus.tsx",
    "panel/tests/api-gateway-single-model.behavior.test.ts",
    "panel/tests/tool-status-responsiveness.test.ts",
    "tests/cross_matrix/run_noheavy_api_cache_contract.py",
)

REQUIRED_NOHEAVY_API_CACHE_TEST_MARKERS = (
    "test_chat_and_responses_log_and_forward_supported_sampling_kwargs",
    "test_request_output_caps_override_server_default_without_touching_context_cap",
    "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap",
    "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap",
    "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap",
    "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap",
    "test_prompt_context_aliases_clamp_without_rewriting_output_caps",
    "test_anthropic_messages_omitted_max_tokens_uses_bundle_default",
    "test_ollama_streaming_suppresses_duplicate_done_chunks",
    "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap",
    "test_chat_completions_nonstreaming",
    "test_responses_nonstreaming",
    "test_chat_completions_streaming",
    "test_responses_streaming",
    "test_responses_nonstreaming_forwards_tc_id",
    "test_chat_completion_streaming_validates_json",
    "test_responses_api_streaming_validates_json",
    "test_streaming_emits_error_on_strict_failure",
    "test_text_format_has_json_schema_field",
    "test_text_format_preserves_schema_data",
    "test_server_docs_state_response_format_is_not_constrained_decoding",
    "test_json_schema_decodes_nested_object_string",
    "test_reports_nested_object_json_string_schema_decode",
    "test_reports_markdown_extraction_as_repair_with_required_fields",
    "test_repair_records_supports_xml_root_and_required_fields",
    "test_repair_records_reports_raw_and_repaired_rates",
    "test_json_schema_processor_masks_non_json_start_token",
    "test_chat_response_format_forwards_guided_json_hint",
    "test_responses_text_format_forwards_guided_json_hint",
    "test_generate_installs_guided_json_logits_processor",
    "test_structured_json_probes_request_json_schema_response_format",
    "test_mimo_structured_json_sentinel_requests_literal_schema_response_format",
    "test_chat_response_format_strict_retries_failed_json_only",
    "test_responses_text_format_strict_retries_failed_json_only",
    "test_chat_response_format_strict_retries_failed_xml_only",
    "test_responses_text_format_strict_retries_failed_xml_only",
    "test_streaming_chat_strict_xml_validates_final_text",
    "test_streaming_responses_strict_xml_validates_final_text",
    "test_streaming_responses_tool_call_arguments_survive_buffering",
    "test_streaming_responses_reasoning_tool_call_keeps_arguments",
    "test_streaming_responses_tool_call_uses_next_output_index_without_text",
    "test_streaming_responses_required_empty_xml_tool_call_is_rejected",
    "test_streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments",
    "test_tool_parser_drops_empty_xml_call_and_strips_markup_for_nonstream_paths",
    "test_streaming_chat_preamble_empty_xml_tool_call_never_emits_empty_arguments",
    "test_chat_stream_tracks_cache_detail_alongside_cached_tokens",
    "test_chat_stream_finish_chunks_emit_cache_detail",
    "test_responses_stream_tracks_cache_detail_alongside_cached",
    "test_responses_stream_finish_emits_cache_detail",
    "test_usage_builders_preserve_cache_detail_without_cached_tokens",
    "test_chat_stream_usage_preserves_cache_detail_without_cached_tokens",
    "test_responses_stream_usage_preserves_cache_detail_without_cached_tokens",
    "test_responses_streaming_stores_history_for_previous_response_id",
    "test_responses_streaming_reasoning_only_stores_placeholder_and_marker",
    "test_chained_response_helper_emits_warning_for_reasoning_only_predecessor",
    "test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry",
    "test_cache_entries_endpoint_lists_paged_prefix_blocks",
    "test_cache_warm_endpoint_prefills_and_stores_block_cache",
    "test_clear_cache_prefix_clears_prefix_l2_without_multimodal",
    "test_native_cache_status_reports_dsv4_separately_from_tq_kv",
    "test_native_cache_status_reports_zaya_typed_cca",
    "test_native_cache_status_reports_plain_attention_kv",
    "test_acceleration_status_reports_internal_jangtq_acceleration_when_enabled",
    "test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw",
    "test_dsv4_encoder_keeps_function_arguments_as_dsml_params",
    "test_dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail",
    "test_responses_extracts_suppressed_reasoning_tool_calls_before_finalize",
    "test_visible_text_around_invoke_preserved_no_dsml_leak",
    "test_tools_called_implies_no_dsml_in_content",
    "test_server_repairs_dsv4_partial_tool_intent_from_request_args",
    "test_dsv4_fallback_tool_prompt_uses_canonical_tool_calls_wrapper",
    "test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint",
    "test_prompt_disk_l2_hit_backfills_paged_cache_for_partial_reuse",
    "test_hybrid_ssm_checkpoint_alignment_falls_back_to_exact_aligned_state",
    "test_serialize_tq_cache_mixed_hybrid",
    "test_tq_tensors_roundtrip_via_safetensors",
    "test_hybrid_ssm_capture_stores_block_aligned_checkpoints",
    "allows gateway startup on ports used only by stopped or remote saved sessions",
    "auto-switches to a standby model by waking it before direct OpenAI streaming",
    "passes Responses function-call argument SSE through unchanged",
    "passes Responses argument SSE with reasoning and empty final item arguments",
    "returns backend-unavailable for stale Responses session ports",
    "recovers Responses function-call arguments from argument delta and done events",
)

COMMANDS: dict[str, list[str]] = {
    "api_route_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_engine_audit.py",
        "-k",
        (
            "chat_and_responses_log_and_forward_supported_sampling_kwargs "
            "or anthropic_messages_omitted_max_tokens_uses_bundle_default "
            "or request_output_caps_override_server_default_without_touching_context_cap "
            "or chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap "
            "or legacy_completions_output_cap_overrides_server_default_without_touching_context_cap "
            "or legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap "
            "or anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap "
            "or prompt_context_aliases_clamp_without_rewriting_output_caps "
            "or ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap "
            "or responses_request_has_sampling_fields "
            "or media_diag_hooks_cover_anthropic_and_ollama_streaming_ingress "
            "or responses_nonstreaming_forwards_tc_id "
            "or chat_completion_streaming_validates_json "
            "or responses_api_streaming_validates_json "
            "or streaming_emits_error_on_strict_failure "
            "or server_docs_state_response_format_is_not_constrained_decoding "
            "or text_format_has_json_schema_field "
            "or text_format_preserves_schema_data "
            "or chat_completions_nonstreaming "
            "or responses_nonstreaming "
            "or chat_completions_streaming "
            "or responses_streaming "
            "or chat_stream_tracks_cache_detail_alongside_cached_tokens "
            "or chat_stream_finish_chunks_emit_cache_detail "
            "or responses_stream_tracks_cache_detail_alongside_cached "
            "or responses_stream_finish_emits_cache_detail "
            "or usage_builders_preserve_cache_detail_without_cached_tokens "
            "or chat_stream_usage_preserves_cache_detail_without_cached_tokens "
            "or responses_stream_usage_preserves_cache_detail_without_cached_tokens "
            "or generic_turboquant_patcher_skips_hybrid_ssm "
            "or ollama_streaming_suppresses_duplicate_done_chunks "
            "or cache_stats_endpoint_projects_cache_reuse_skip_telemetry "
            "or cache_entries_endpoint_lists_paged_prefix_blocks "
            "or cache_warm_endpoint_prefills_and_stores_block_cache "
            "or clear_cache_prefix_clears_prefix_l2_without_multimodal "
            "or native_cache_status_reports_dsv4_separately_from_tq_kv "
            "or native_cache_status_reports_zaya_typed_cca "
            "or native_cache_status_reports_plain_attention_kv "
            "or acceleration_status_reports_internal_jangtq_acceleration_when_enabled"
        ),
    ],
    "scheduler_cache_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_batching.py",
        "-k",
        "dsv4 or hybrid_ssm or cache_detail or prompt_disk_l2",
    ],
    "tq_and_mllm_cache_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
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
        "-vv",
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
    "responses_history_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_responses_history.py",
        "-k",
        (
            "responses_streaming_stores_history_for_previous_response_id "
            "or responses_streaming_reasoning_only_stores_placeholder_and_marker "
            "or chained_response_helper_emits_warning_for_reasoning_only_predecessor"
        ),
    ],
    "structured_output_repair_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_structured_output.py",
        "tests/test_structured_output_repair_report.py",
        "-k",
        (
            "json_schema_decodes_nested_object_string "
            "or reports_nested_object_json_string_schema_decode "
            "or json_schema_repairs_qwen_adjacent_string_array "
            "or json_schema_coerces_string_to_array "
            "or chat_completion_repairs_and_canonicalizes_schema_json "
            "or repair_records_preserves_raw_vs_repaired_counts "
            "or repair_records_reports_raw_and_repaired_rates "
            "or reports_markdown_extraction_as_repair_with_required_fields "
            "or repair_records_supports_xml_root_and_required_fields"
        ),
    ],
    "structured_output_retry_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_server.py",
        "-k",
        (
            "chat_response_format_strict_retries_failed_json_only "
            "or responses_text_format_strict_retries_failed_json_only "
            "or chat_response_format_strict_retries_failed_xml_only "
            "or responses_text_format_strict_retries_failed_xml_only "
            "or streaming_chat_strict_xml_validates_final_text "
            "or streaming_responses_strict_xml_validates_final_text"
        ),
    ],
    "responses_streaming_tool_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_server.py",
        "-k",
        (
            "streaming_responses_tool_call_arguments_survive_buffering "
            "or streaming_responses_reasoning_tool_call_keeps_arguments "
            "or streaming_responses_tool_call_uses_next_output_index_without_text "
            "or streaming_responses_required_empty_xml_tool_call_is_rejected "
            "or streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments "
            "or tool_parser_drops_empty_xml_call_and_strips_markup_for_nonstream_paths "
            "or streaming_chat_preamble_empty_xml_tool_call_never_emits_empty_arguments"
        ),
    ],
    "structured_guided_decoding_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_structured_output.py",
        "tests/test_server.py",
        "tests/test_llm.py",
        "-k",
        (
            "json_schema_processor_masks_non_json_start_token "
            "or chat_response_format_forwards_guided_json_hint "
            "or responses_text_format_forwards_guided_json_hint "
            "or generate_installs_guided_json_logits_processor"
        ),
    ],
    "structured_smoke_response_format_contracts": [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-vv",
        "tests/test_all_local_model_smoke.py",
        "-k",
        (
            "structured_json_probes_request_json_schema_response_format "
            "or mimo_structured_json_sentinel_requests_literal_schema_response_format"
        ),
    ],
    "panel_gateway_contracts": [
        "npm",
        "--prefix",
        "panel",
        "exec",
        "vitest",
        "run",
        "tests/api-gateway-single-model.behavior.test.ts",
        "--",
        "--reporter",
        "verbose",
        "--testNamePattern",
        (
            "allows gateway startup on ports used only by stopped or remote saved sessions|"
            "auto-switches to a standby model by waking it before direct OpenAI streaming|"
            "passes Responses function-call argument SSE through unchanged|"
            "passes Responses argument SSE with reasoning and empty final item arguments|"
            "returns backend-unavailable for stale Responses session ports"
        ),
    ],
    "panel_tool_status_contracts": [
        "npm",
        "--prefix",
        "panel",
        "exec",
        "vitest",
        "run",
        "tests/tool-status-responsiveness.test.ts",
        "--",
        "--reporter",
        "verbose",
        "--testNamePattern",
        "recovers Responses function-call arguments from argument delta and done events",
    ],
}


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    deselected = None
    matches = re.findall(r"(\d+) passed", output)
    if matches:
        passed = int(matches[-1])
    matches = re.findall(r"(\d+) deselected", output)
    if matches:
        deselected = int(matches[-1])
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
        "stdout": output,
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
    stdout = "\n".join(str(result.get("stdout", "")) for result in commands.values())
    missing_markers = [
        marker
        for marker in REQUIRED_NOHEAVY_API_CACHE_TEST_MARKERS
        if marker not in stdout
    ]
    api_ok = commands["api_route_contracts"]["returncode"] == 0
    scheduler_ok = commands["scheduler_cache_contracts"]["returncode"] == 0
    tq_ok = commands["tq_and_mllm_cache_contracts"]["returncode"] == 0
    dsml_ok = commands["dsv4_dsml_tool_contracts"]["returncode"] == 0
    responses_history_ok = commands["responses_history_contracts"]["returncode"] == 0
    structured_output_ok = commands["structured_output_repair_contracts"]["returncode"] == 0
    structured_retry_ok = commands["structured_output_retry_contracts"]["returncode"] == 0
    responses_stream_tools_ok = commands["responses_streaming_tool_contracts"]["returncode"] == 0
    structured_guided_ok = commands["structured_guided_decoding_contracts"]["returncode"] == 0
    structured_smoke_ok = commands["structured_smoke_response_format_contracts"]["returncode"] == 0
    panel_gateway_ok = commands["panel_gateway_contracts"]["returncode"] == 0
    panel_tool_status_ok = commands["panel_tool_status_contracts"]["returncode"] == 0
    checks = {
        "openai_chat_sampling_kwargs": (
            api_ok and "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" not in missing_markers
        ),
        "responses_sampling_kwargs": (
            api_ok and "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" not in missing_markers
        ),
        "request_output_caps_override_server_default": (
            api_ok
            and "test_request_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
        ),
        "prompt_context_caps_stay_separate_from_output_caps": (
            api_ok
            and "test_request_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "test_prompt_context_aliases_clamp_without_rewriting_output_caps" not in missing_markers
        ),
        "legacy_completions_output_caps_override_server_default": (
            api_ok
            and "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap" not in missing_markers
            and "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap" not in missing_markers
        ),
        "anthropic_bundle_defaults": (
            api_ok
            and "test_anthropic_messages_omitted_max_tokens_uses_bundle_default" not in missing_markers
            and "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap" not in missing_markers
        ),
        "ollama_adapter_surface": (
            api_ok
            and "test_ollama_streaming_suppresses_duplicate_done_chunks" not in missing_markers
            and "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap" not in missing_markers
        ),
        "streaming_cache_detail_usage": (
            api_ok
            and "test_chat_stream_tracks_cache_detail_alongside_cached_tokens" not in missing_markers
            and "test_chat_stream_finish_chunks_emit_cache_detail" not in missing_markers
            and "test_responses_stream_tracks_cache_detail_alongside_cached" not in missing_markers
            and "test_responses_stream_finish_emits_cache_detail" not in missing_markers
            and "test_usage_builders_preserve_cache_detail_without_cached_tokens" not in missing_markers
            and "test_chat_stream_usage_preserves_cache_detail_without_cached_tokens" not in missing_markers
            and "test_responses_stream_usage_preserves_cache_detail_without_cached_tokens" not in missing_markers
        ),
        "json_response_format_stream_validation": (
            api_ok
            and "test_chat_completion_streaming_validates_json" not in missing_markers
            and "test_responses_api_streaming_validates_json" not in missing_markers
            and "test_streaming_emits_error_on_strict_failure" not in missing_markers
        ),
        "responses_text_format_json_schema_preserved": (
            api_ok
            and "test_text_format_has_json_schema_field" not in missing_markers
            and "test_text_format_preserves_schema_data" not in missing_markers
        ),
        "response_format_docs_repair_validation_boundary": (
            api_ok
            and "test_server_docs_state_response_format_is_not_constrained_decoding"
            not in missing_markers
        ),
        "structured_schema_decode_repair": (
            structured_output_ok
            and "test_json_schema_decodes_nested_object_string"
            not in missing_markers
            and "test_reports_nested_object_json_string_schema_decode"
            not in missing_markers
        ),
        "structured_xml_repair_validation_boundary": (
            structured_output_ok
            and "test_reports_markdown_extraction_as_repair_with_required_fields"
            not in missing_markers
            and "test_repair_records_supports_xml_root_and_required_fields"
            not in missing_markers
        ),
        "structured_repair_report_rates": (
            structured_output_ok
            and "test_repair_records_reports_raw_and_repaired_rates"
            not in missing_markers
        ),
        "structured_json_retry_after_repair_failure": (
            structured_retry_ok
            and "test_chat_response_format_strict_retries_failed_json_only"
            not in missing_markers
            and "test_responses_text_format_strict_retries_failed_json_only"
            not in missing_markers
        ),
        "structured_xml_retry_after_repair_failure": (
            structured_retry_ok
            and "test_chat_response_format_strict_retries_failed_xml_only"
            not in missing_markers
            and "test_responses_text_format_strict_retries_failed_xml_only"
            not in missing_markers
        ),
        "structured_xml_stream_validation": (
            structured_retry_ok
            and "test_streaming_chat_strict_xml_validates_final_text"
            not in missing_markers
            and "test_streaming_responses_strict_xml_validates_final_text"
            not in missing_markers
        ),
        "structured_guided_json_schema_token_masking": (
            structured_guided_ok
            and "test_json_schema_processor_masks_non_json_start_token"
            not in missing_markers
            and "test_chat_response_format_forwards_guided_json_hint"
            not in missing_markers
            and "test_responses_text_format_forwards_guided_json_hint"
            not in missing_markers
            and "test_generate_installs_guided_json_logits_processor"
            not in missing_markers
        ),
        "structured_live_smoke_response_format_adoption": (
            structured_smoke_ok
            and "test_structured_json_probes_request_json_schema_response_format"
            not in missing_markers
            and "test_mimo_structured_json_sentinel_requests_literal_schema_response_format"
            not in missing_markers
        ),
        "responses_previous_response_history": (
            responses_history_ok
            and "test_responses_streaming_stores_history_for_previous_response_id"
            not in missing_markers
            and "test_responses_streaming_reasoning_only_stores_placeholder_and_marker"
            not in missing_markers
            and "test_chained_response_helper_emits_warning_for_reasoning_only_predecessor"
            not in missing_markers
        ),
        "responses_streaming_tool_call_arguments_and_indexes": (
            responses_stream_tools_ok
            and "test_streaming_responses_tool_call_arguments_survive_buffering"
            not in missing_markers
            and "test_streaming_responses_reasoning_tool_call_keeps_arguments"
            not in missing_markers
            and "test_streaming_responses_tool_call_uses_next_output_index_without_text"
            not in missing_markers
            and "test_streaming_responses_required_empty_xml_tool_call_is_rejected"
            not in missing_markers
            and "test_streaming_responses_preamble_empty_xml_tool_call_never_emits_empty_arguments"
            not in missing_markers
        ),
        "cache_stats_reuse_skip_telemetry": (
            api_ok
            and "test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry"
            not in missing_markers
        ),
        "cache_reuse_endpoints": (
            api_ok
            and "test_cache_stats_endpoint_projects_cache_reuse_skip_telemetry" not in missing_markers
            and "test_cache_entries_endpoint_lists_paged_prefix_blocks" not in missing_markers
            and "test_cache_warm_endpoint_prefills_and_stores_block_cache" not in missing_markers
            and "test_clear_cache_prefix_clears_prefix_l2_without_multimodal" not in missing_markers
        ),
        "gateway_stale_port_startup": (
            panel_gateway_ok
            and "allows gateway startup on ports used only by stopped or remote saved sessions"
            not in missing_markers
        ),
        "gateway_standby_wake_routing": (
            panel_gateway_ok
            and "auto-switches to a standby model by waking it before direct OpenAI streaming"
            not in missing_markers
        ),
        "gateway_responses_function_call_arguments_streaming": (
            panel_gateway_ok
            and "passes Responses function-call argument SSE through unchanged"
            not in missing_markers
        ),
        "gateway_responses_reasoning_empty_final_arguments_streaming": (
            panel_gateway_ok
            and "passes Responses argument SSE with reasoning and empty final item arguments"
            not in missing_markers
        ),
        "gateway_stale_responses_port_rejection": (
            panel_gateway_ok
            and "returns backend-unavailable for stale Responses session ports"
            not in missing_markers
        ),
        "panel_tool_status_responses_argument_recovery": (
            panel_tool_status_ok
            and "recovers Responses function-call arguments from argument delta and done events"
            not in missing_markers
        ),
        "dsv4_native_cache_status": (
            api_ok and scheduler_ok and "test_native_cache_status_reports_dsv4_separately_from_tq_kv" not in missing_markers
        ),
        "dsv4_dsml_parser_residue_rejection": (
            dsml_ok and "test_tools_called_implies_no_dsml_in_content" not in missing_markers
        ),
        "dsv4_dsml_valid_tool_call_preserved": (
            dsml_ok and "test_dsv4_encoder_keeps_function_arguments_as_dsml_params" not in missing_markers
        ),
        "dsv4_suppressed_tool_markup_not_stored": (
            dsml_ok and "test_responses_extracts_suppressed_reasoning_tool_calls_before_finalize" not in missing_markers
        ),
        "zaya_typed_cca_status": (
            api_ok and "test_native_cache_status_reports_zaya_typed_cca" not in missing_markers
        ),
        "plain_attention_kv_status": (
            api_ok and "test_native_cache_status_reports_plain_attention_kv" not in missing_markers
        ),
        "jangtq_mpp_nax_health_kernel_name": (
            api_ok
            and "test_acceleration_status_reports_internal_jangtq_acceleration_when_enabled"
            not in missing_markers
        ),
        "hybrid_ssm_partial_reuse": (
            scheduler_ok
            and tq_ok
            and "test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint" not in missing_markers
            and "test_hybrid_ssm_capture_stores_block_aligned_checkpoints" not in missing_markers
        ),
        "turboquant_kv_runtime_contract": (
            tq_ok and "test_serialize_tq_cache_mixed_hybrid" not in missing_markers
        ),
        "turboquant_disk_roundtrip": (
            tq_ok and "test_tq_tensors_roundtrip_via_safetensors" not in missing_markers
        ),
        "no_generic_tq_on_hybrid_ssm": api_ok,
        "all_required_named_rows_ran": not missing_markers,
    }
    public_commands = {
        name: {key: value for key, value in result.items() if key != "stdout"}
        for name, result in commands.items()
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
        "missing_markers": missing_markers,
        "source_hashes": source_hashes(root),
        "commands": public_commands,
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
    print("missing_markers=" + json.dumps(artifact["missing_markers"]))
    for name, result in artifact["commands"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
