def test_generation_defaults_contract_pins_named_default_edges():
    from tests.cross_matrix import run_generation_defaults_contract as gate

    required = gate.REQUIRED_GENERATION_DEFAULT_TEST_MARKERS

    assert "uses standard MLX generation_config.json for non-JANG and VLM bundles" in required
    assert "lets JANG chat sampling metadata override generation_config.json" in required
    assert "normalizes disabled top_k sentinels to Off/0 for UI and requests" in required
    assert "uses chat repetition penalty when bundle default reasoning mode is not thinking" in required
    assert "does not synthesize server --default sampling flags from UI/session config" in required
    assert "does not copy model max_new_tokens into hidden startup maxTokens config" in required
    assert "database clears legacy session maxTokens before settings UI or launch can reuse them" in required
    assert "server startup generation defaults are model-owned and not editable sliders" in required
    assert "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" in required
    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_request_output_caps_can_go_below_or_above_startup_default" in required
    assert "test_reasoning_effort_preserves_bundle_max_new_tokens" in required
    assert "test_omitted_server_max_tokens_uses_bundle_max_new_tokens" in required
    assert "test_omitted_server_max_tokens_without_bundle_default_is_bounded" in required
    assert "test_max_tokens_resolution_contract_applies_to_every_registered_family" in required
    assert "test_effort_does_not_synthesize_max_tokens_when_unset" in required
    assert "test_session_command_preview_mirrors_runtime_default_flags" in required
    assert "test_local_generation_metadata_audit_flags_thinking_template_without_budget" in required
    assert "test_local_generation_metadata_audit_accepts_template_budget_support" in required

    panel_command = gate.COMMANDS["panel_generation_defaults"][1]
    engine_command = gate.COMMANDS["engine_generation_defaults"][1]
    assert "--reporter=verbose" in panel_command
    assert "-vv" in engine_command
