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
    assert "dsv4_native_cache_status" in nested
    assert "zaya_typed_cca_status" in nested
    assert "dsv4_dsml_parser_residue_rejection" in nested
    assert "turboquant_disk_roundtrip" in nested

    assert "omits sampling and token defaults when unset so the engine resolves bundle metadata" in panel
    assert "keeps per-chat maxTokens as output budget only, never prompt context" in panel
    assert "keeps Responses maxTokens as output budget only, never prompt context" in panel
    assert "preserves DSV4 Responses max_output_tokens for Max thinking" in panel
    assert "omits unset and disabled sampling sentinels without dropping explicit overrides" in panel
    assert "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap" in panel
    assert "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults" in panel

    panel_command = gate.COMMANDS["panel_api_request_builders"][1]
    assert "--reporter=verbose" in panel_command


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
    assert "test_native_cache_status_reports_dsv4_separately_from_tq_kv" in required
    assert "test_native_cache_status_reports_zaya_typed_cca" in required
    assert "test_responses_extracts_suppressed_reasoning_tool_calls_before_finalize" in required
    assert "test_dsml_issue_165_server_tool_call_arguments_are_not_empty_or_raw" in required
    assert "test_dsv4_encoder_preserves_code_identifiers_on_direct_chat_rail" in required

    for command in gate.COMMANDS.values():
        assert "-vv" in command
