def test_max_output_context_contract_covers_all_public_api_surfaces():
    from tests.cross_matrix import run_max_output_context_contract as gate

    required = gate.REQUIRED_MAX_OUTPUT_CONTEXT_TEST_MARKERS
    sources = set(gate.SOURCE_HASH_FILES)
    assert "vmlx_engine/api/models.py" in sources
    assert "vmlx_engine/api/anthropic_adapter.py" in sources
    assert "vmlx_engine/api/ollama_adapter.py" in sources
    assert "panel/src/main/api-gateway.ts" in sources
    assert "panel/src/shared/dsv4RequestBudget.ts" in sources
    assert "tests/test_ollama_adapter.py" in sources
    assert "panel/tests/chat-settings-compatibility.test.ts" in sources

    joined_commands = "\n".join(
        " ".join(command)
        for _root, command in gate.COMMANDS.values()
    )
    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in joined_commands
    assert "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" in joined_commands
    assert "test_streaming_output_caps_do_not_mutate_server_default_across_later_omitted_requests" in joined_commands
    assert "test_explicit_startup_max_tokens_is_default_not_request_ceiling" in joined_commands
    assert "test_request_output_caps_can_go_below_or_above_startup_default" in joined_commands
    assert "test_request_output_caps_do_not_mutate_server_default_across_later_omitted_requests" in joined_commands
    assert "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap" in joined_commands
    assert "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap" in joined_commands
    assert "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap" in joined_commands
    assert "test_prompt_context_aliases_clamp_without_rewriting_output_caps" in joined_commands
    assert "test_anthropic_messages_omitted_max_tokens_uses_bundle_default" in joined_commands
    assert "test_ollama_generate_default_uses_chat_template_request_shape" in joined_commands
    assert "test_ollama_chat_omits_non_positive_num_predict_sentinels" in joined_commands
    assert "test_ollama_generate_omits_non_positive_num_predict_sentinels" in joined_commands
    assert "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap" in joined_commands
    assert "test_public_api_models_reject_non_positive_output_caps" in joined_commands

    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" in required
    assert "test_streaming_output_caps_do_not_mutate_server_default_across_later_omitted_requests" in required
    assert "test_explicit_startup_max_tokens_is_default_not_request_ceiling" in required
    assert "test_request_output_caps_can_go_below_or_above_startup_default" in required
    assert "test_request_output_caps_do_not_mutate_server_default_across_later_omitted_requests" in required
    assert "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap" in required
    assert "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap" in required
    assert "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap" in required
    assert "test_explicit_server_max_tokens_overrides_bundle_max_new_tokens" in required
    assert "test_omitted_server_max_tokens_uses_bundle_max_new_tokens" in required
    assert "test_omitted_server_max_tokens_without_bundle_default_is_bounded" in required
    assert "test_prompt_context_aliases_clamp_without_rewriting_output_caps" in required
    assert "test_max_tokens_resolution_contract_applies_to_every_registered_family" in required
    assert "test_wake_reload_preserves_max_tokens_explicitness" in required
    assert "test_cli_serve_implicit_max_tokens_uses_bounded_fallback" in required
    assert "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap" in required
    assert "test_public_api_models_reject_non_positive_output_caps" in required
    assert "surfaces Max Output Tokens separately from Max Context Tokens" in required
    assert "does not copy model max_new_tokens into hidden startup maxTokens config" in required
    assert "chat settings expose per-chat max tokens without hidden DSV4 floors" in required
    assert "keeps per-chat maxTokens as output budget only, never prompt context" in required
    assert "keeps Responses maxTokens as output budget only, never prompt context" in required
    assert "per-chat maxTokens below or above the server startup default remain request scoped for Responses" in required
    assert "does not invent Responses sampler or output-budget values when chat overrides are absent" in required
    assert "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap" in required
    assert "clears legacy session maxTokens=32768 before launch can reuse it" in required
    assert "switching chats never carries a previous chat maxTokens into Auto Chat Completions" in required
    assert "switching chats never carries a previous chat maxTokens into Auto Responses" in required
    assert "new chats preserve model-owned maxTokens while refusing inherited output caps" in required
    assert "persisted chat maxTokens cannot relaunch server with a new startup maxTokens" in required

    assert "new_chat_output_caps_are_not_inherited_or_made_sticky" in gate.build_artifact.__code__.co_consts
    assert "chat_output_cap_remains_per_chat_override" in gate.build_artifact.__code__.co_consts

    engine_command = gate.COMMANDS["engine_output_context_resolution"][1]
    panel_command = gate.COMMANDS["panel_output_context_wiring"][1]
    assert "-vv" in engine_command
    assert "--reporter=verbose" in panel_command
