from pathlib import Path


def test_reasoning_template_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_reasoning_template_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-reasoning-template-contract-20260526-settings-audit.json"
    )


def test_reasoning_template_contract_pins_named_reasoning_edges():
    from tests.cross_matrix import run_reasoning_template_contract as gate

    required = gate.REQUIRED_REASONING_TEMPLATE_TEST_MARKERS

    assert "test_dsv4_reasoning_effort_preserves_requested_rails" in required
    assert "test_dsv4_thinking_policy_does_not_force_tool_calls_to_direct_rail" in required
    assert "test_dsv4_bundle_defaults_apply_only_when_request_omits_values" in required
    assert "test_minimax_m2_preserves_sampling_values_without_family_floor" in required
    assert "test_ling_suppresses_reasoning_parser_and_stale_think_in_template" in required
    assert "test_hy_v3_qwen3_reasoning_parser_no_think_does_not_leak_tags" in required
    assert "test_deepseek_r1_reasoning_parser_orphan_close_is_not_visible" in required
    assert "test_gemma4_reasoning_parser_orphan_channel_close_is_not_visible" in required
    assert "test_implicit_reasoning_on_tool_followup" in required
    assert "test_qwen3_reasoning_then_multiple_tool_calls" in required
    assert "test_think_tags_stripped_before_parsing" in required
    assert "test_tools_called_implies_no_channel_marker_in_content" in required
    assert "test_used_by_documented_families" in required
    assert "content with <think> tags but server provides reasoning_content — no double-extraction" in required
    assert "response.output_text.delta also triggers reasoningDone if was reasoning" in required
    assert "Responses: local Auto omits enable_thinking so engine auto-detects" in required
    assert "Tool iteration: reasoning in iteration 1, tool call, clean iteration 2" in required
    assert "server-side DeepSeek reasoning_content handles implicit correctly" in required
    assert "replaces old reasoning segments during live interleaved streaming, then can show all after completion" in required
    assert "live-replaces previous reasoning segments while streaming and shows all after completion" in required

    engine_command = gate.COMMANDS["engine_reasoning_template"][1]
    panel_command = gate.COMMANDS["panel_reasoning_rendering"][1]
    assert "-vv" in engine_command
    assert "--reporter=verbose" in panel_command
