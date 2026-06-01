from pathlib import Path


def test_api_surface_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_api_surface_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-api-surface-contract-20260601-child-image-wrapped-epipe-refresh.json"
    )
    assert gate.NESTED_OUT == Path(
        "build/current-api-cache-contract-api-surface-check-20260528-epipe-aggregate-guard.json"
    )


def test_api_surface_contract_pins_named_public_surface_edges():
    from tests.cross_matrix import run_api_surface_contract as gate

    nested = gate.REQUIRED_NESTED_API_CHECKS
    panel = gate.REQUIRED_PANEL_API_TEST_MARKERS

    assert "openai_chat_sampling_kwargs" in nested
    assert "responses_sampling_kwargs" in nested
    assert "legacy_completions_output_caps_override_server_default" in nested
    assert "request_output_caps_override_server_default" in nested
    assert "prompt_context_caps_stay_separate_from_output_caps" in nested
    assert "anthropic_bundle_defaults" in nested
    assert "ollama_adapter_surface" in nested
    assert "streaming_cache_detail_usage" in nested
    assert "cache_reuse_endpoints" in nested
    assert "dsv4_native_cache_status" in nested
    assert "zaya_typed_cca_status" in nested
    assert "jangtq_mpp_nax_health_kernel_name" in nested
    assert "dsv4_dsml_parser_residue_rejection" in nested
    assert "turboquant_disk_roundtrip" in nested

    assert "omits sampling and token defaults when unset so the engine resolves bundle metadata" in panel
    assert "keeps per-chat maxTokens as output budget only, never prompt context" in panel
    assert "keeps Responses maxTokens as output budget only, never prompt context" in panel
    assert "preserves DSV4 Responses max_output_tokens for Max thinking" in panel
    assert "omits malformed Ollama context values instead of poisoning max_prompt_tokens" in panel
    assert "omits unset and disabled sampling sentinels without dropping explicit overrides" in panel
    assert "omits malformed Ollama num_predict values instead of poisoning max_tokens" in panel
    assert "applies gateway timeout handling to Ollama embeddings proxy requests" in panel
    assert "auto-switches by model id in single-model mode before preserving streaming deltas" in panel
    assert "refuses auto-switch when previous local model cannot unload before starting target" in panel
    assert "auto-switches single-model Ollama chat while emitting incremental content chunks" in panel
    assert "auto-switches single-model Ollama generate while emitting incremental response chunks" in panel
    assert "auto-switches single-model Ollama embeddings before proxying embedding data" in panel
    assert "refuses single-model Ollama routes when previous local model cannot unload" in panel
    assert "auto-switches direct OpenAI streaming by model id without mutating payload or deltas" in panel
    assert "serializes concurrent single-model switches before starting a second target" in panel
    assert "auto-switches to a standby model by waking it before direct OpenAI streaming" in panel
    assert "auto-switches Responses API streaming by model id while preserving output text deltas" in panel
    assert "auto-switches model capability requests by path model before proxying" in panel
    assert "auto-switches cache endpoints by query model before proxying cache stats" in panel
    assert "auto-switches cache entries and clear endpoints by query model before proxying cache endpoints" in panel
    assert "auto-switches cache warm by body model before proxying warm prompts" in panel
    assert "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap" in panel
    assert "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults" in panel
    assert "Auto chat maxTokens omits per-request output caps so server default can apply" in panel
    assert "guards gateway streaming writes against client disconnect EPIPE errors" in panel
    assert "guards every Ollama backend response stream error as a disconnect boundary" in panel
    assert "aborts Ollama backend response streams when the client response closes" in panel
    assert "writes each streamed gateway response chunk once and treats EPIPE as disconnect" in panel
    assert "does not write gateway response chunks after the client socket is destroyed" in panel
    assert "does not write gateway response chunks after Node marks the response closed" in panel
    assert "does not write proxied request bodies after the backend socket is destroyed" in panel
    assert "does not write proxied request bodies after Node marks the request closed" in panel
    assert "treats top-level request handler EPIPE failures as client disconnects" in panel
    assert "treats nested broken-pipe stream errors as client disconnects" in panel
    assert "guards child process stdio stream EPIPE across app-managed process lanes" in panel
    assert "does not end proxied requests after the backend socket is destroyed" in panel
    assert "does not end proxied requests after Node marks the request closed" in panel
    assert "does not leave raw backend request end calls unguarded after disconnect" in panel
    assert "does not leave raw chat IPC backend request finalization unguarded" in panel
    assert "routes local image server request writes through EPIPE-aware helpers" in panel
    assert "image requests disable connection reuse and normalize reset-like socket errors" in panel
    assert "panel_gateway_streaming_disconnect_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_gateway_ollama_proxy_response_error_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_ipc_backend_request_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_child_process_stdio_epipe_guard" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()
    assert "panel_gateway_single_model_auto_switch_cache_endpoints" in Path(
        "tests/cross_matrix/run_api_surface_contract.py"
    ).read_text()

    panel_command = gate.COMMANDS["panel_api_request_builders"][1]
    assert "tests/api-gateway-ollama.test.ts" in panel_command
    assert "tests/image-generation-state.test.ts" in panel_command
    assert "tests/image-system.test.ts" in panel_command
    assert "--reporter=verbose" in panel_command


def test_api_surface_contract_status_fails_when_required_panel_markers_are_missing():
    from tests.cross_matrix import run_api_surface_contract as gate

    source = Path("tests/cross_matrix/run_api_surface_contract.py").read_text()

    assert "all_required_panel_api_markers_present" in source
    assert "not missing_panel_markers" in source


def test_api_surface_contract_source_hash_files_exist():
    from tests.cross_matrix import run_api_surface_contract as gate

    assert "panel/src/main/ipc/models.ts" in gate.SOURCE_HASH_FILES
    assert "panel/src/main/tools/executor.ts" in gate.SOURCE_HASH_FILES

    missing = [rel for rel in gate.SOURCE_HASH_FILES if not Path(rel).exists()]

    assert missing == []


def test_noheavy_api_cache_contract_pins_named_server_rows():
    from tests.cross_matrix import run_noheavy_api_cache_contract as gate

    required = gate.REQUIRED_NOHEAVY_API_CACHE_TEST_MARKERS

    assert "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" in required
    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_prompt_context_aliases_clamp_without_rewriting_output_caps" in required
    assert "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap" in required
    assert "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap" in required
    assert "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap" in required
    assert "test_anthropic_messages_omitted_max_tokens_uses_bundle_default" in required
    assert "test_ollama_streaming_suppresses_duplicate_done_chunks" in required
    assert "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap" in required
    assert "test_chat_stream_tracks_cache_detail_alongside_cached_tokens" in required
    assert "test_chat_stream_finish_chunks_emit_cache_detail" in required
    assert "test_responses_stream_tracks_cache_detail_alongside_cached" in required
    assert "test_responses_stream_finish_emits_cache_detail" in required
    assert "test_cache_entries_endpoint_lists_paged_prefix_blocks" in required
    assert "test_cache_warm_endpoint_prefills_and_stores_block_cache" in required
    assert "test_clear_cache_prefix_clears_prefix_l2_without_multimodal" in required
    assert "test_native_cache_status_reports_dsv4_separately_from_tq_kv" in required
    assert "test_native_cache_status_reports_zaya_typed_cca" in required
    assert "test_acceleration_status_reports_internal_jangtq_acceleration_when_enabled" in required
    assert "test_responses_extracts_suppressed_reasoning_tool_calls_before_finalize" in required
    assert "test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw" in required
    assert "test_dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail" in required

    for command in gate.COMMANDS.values():
        assert "-vv" in command
