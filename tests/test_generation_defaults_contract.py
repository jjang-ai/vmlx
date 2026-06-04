from pathlib import Path


def test_generation_defaults_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_generation_defaults_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-generation-defaults-contract-gemma4-release-boundary-20260604.json"
    )


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
    assert "text additional args cannot override app-owned generation, parser, cache, or MTP flags" in required
    assert "text additional args cannot override app-owned server, template, model-name, or MCP flags" in required
    assert "DSV4 additional args cannot reenable native MTP or deterministic sampling policy" in required
    assert "DSV4 additional args strips blocked equals-form serve overrides" in required
    assert "test_chat_and_responses_log_and_forward_supported_sampling_kwargs" in required
    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_request_output_caps_can_go_below_or_above_startup_default" in required
    assert "test_reasoning_effort_preserves_bundle_max_new_tokens" in required
    assert "test_omitted_server_max_tokens_uses_bundle_max_new_tokens" in required
    assert "test_omitted_server_max_tokens_without_bundle_default_is_bounded" in required
    assert "test_max_tokens_resolution_contract_applies_to_every_registered_family" in required
    assert "test_effort_does_not_synthesize_max_tokens_when_unset" in required
    assert "surfaces Max Output Tokens separately from Max Context Tokens" in required
    assert "test_session_command_preview_mirrors_runtime_default_flags" in required
    assert "test_local_generation_metadata_audit_flags_thinking_template_without_budget" in required
    assert "test_local_generation_metadata_audit_accepts_template_budget_support" in required
    assert "marks thinking-budget unsupported when the template has thinking but no budget variable" in required
    assert "surfaces maxThinkingTokens only when the template consumes thinking_budget" in required
    assert "chat settings hides max-thinking tokens when template metadata says budget is unsupported" in required

    panel_command = gate.COMMANDS["panel_generation_defaults"][1]
    engine_command = gate.COMMANDS["engine_generation_defaults"][1]
    assert "--reporter=verbose" in panel_command
    assert "-vv" in engine_command


def test_generation_defaults_contract_publishes_structured_family_matrix():
    from tests.cross_matrix import run_generation_defaults_contract as gate

    expected_rows = {
        "standard_mlx_generation_config",
        "jang_chat_sampling_overrides",
        "disabled_top_k_sentinel",
        "dsv4_direct_chat_repetition_policy",
        "max_output_context_separation",
        "thinking_budget_template_support",
        "additional_args_no_override",
        "registered_family_max_token_contract",
    }

    assert set(gate.REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX) == expected_rows
    for row_id, requirement in gate.REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX.items():
        assert requirement["checks"], row_id
        assert requirement["markers"], row_id

    dsv4 = gate.REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX[
        "dsv4_direct_chat_repetition_policy"
    ]
    assert "mode_specific_jang_repetition_penalty_is_metadata_owned" in dsv4["checks"]
    assert "uses neutral generic repetition penalty for DSV4 direct chat defaults" in dsv4["markers"]

    max_tokens = gate.REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX[
        "registered_family_max_token_contract"
    ]
    assert "omitted_max_tokens_without_bundle_default_is_bounded" in max_tokens["checks"]
    assert "test_max_tokens_resolution_contract_applies_to_every_registered_family" in max_tokens["markers"]

    output_context = gate.REQUIRED_GENERATION_DEFAULT_FAMILY_MATRIX[
        "max_output_context_separation"
    ]
    assert "panel_max_output_context_labels_are_separated" in output_context["checks"]
    assert "surfaces Max Output Tokens separately from Max Context Tokens" in output_context["markers"]
