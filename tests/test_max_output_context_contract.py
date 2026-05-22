def test_max_output_context_contract_covers_all_public_api_surfaces():
    from tests.cross_matrix import run_max_output_context_contract as gate

    sources = set(gate.SOURCE_HASH_FILES)
    assert "vmlx_engine/api/anthropic_adapter.py" in sources
    assert "vmlx_engine/api/ollama_adapter.py" in sources
    assert "tests/test_ollama_adapter.py" in sources

    joined_commands = "\n".join(
        " ".join(command)
        for _root, command in gate.COMMANDS.values()
    )
    assert "test_request_output_caps_override_server_default_without_touching_context_cap" in joined_commands
    assert "test_explicit_startup_max_tokens_is_default_not_request_ceiling" in joined_commands
    assert "test_anthropic_messages_omitted_max_tokens_uses_bundle_default" in joined_commands
    assert "test_ollama_generate_default_uses_chat_template_request_shape" in joined_commands
    assert "test_ollama_chat_omits_non_positive_num_predict_sentinels" in joined_commands
    assert "test_ollama_generate_omits_non_positive_num_predict_sentinels" in joined_commands
