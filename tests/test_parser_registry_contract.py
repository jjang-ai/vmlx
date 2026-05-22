from pathlib import Path


def test_parser_registry_contract_pins_named_parser_edges():
    from tests.cross_matrix import run_parser_registry_contract as gate

    required = gate.REQUIRED_PARSER_TEST_MARKERS

    assert "test_all_reasoning_parsers_registered" in required
    assert "test_cli_tool_parser_choices_cover_family_registry_parsers" in required
    assert "test_cli_reasoning_parser_choices_cover_family_registry_parsers" in required
    assert "test_registry_overrides_stale_minimax_qwen3_sidecar" in required
    assert "test_zaya1_vl_registered_with_full_contract" in required
    assert "test_ling_is_not_a_reasoning_model_per_eric_2026_05_11" in required
    assert "test_qwen2_must_not_have_reasoning" in required
    assert "test_qwen2_vl_must_not_have_reasoning" in required
    assert "test_gemma3_reasoning_parser" in required
    assert "test_glm_flash_vs_base_reasoning_parser_differs" in required
    assert "test_deepseek_v4_eos_includes_latest_reminder" in required
    assert "test_minimax_eos_includes_role_boundary_marker" in required
    assert "canonicalizes legacy DSV4 and Hy3 parser aliases before launch" in required
    assert "tool parser dropdown exposes DSV4 DSML, Hy3, and ZAYA parsers" in required
    assert "reasoning parser dropdown covers every parser the panel registry can emit" in required
    assert "passes MiniMax through the registered minimax_m2 reasoning parser" in required
    assert "uses the registered MiniMax reasoning parser even when bundle sidecars say qwen3" in required
    assert "detects Hy3 as text-only KV with Hunyuan tools and qwen3 reasoning" in required

    engine_command = gate.COMMANDS["engine_parser_registry"][1]
    panel_command = gate.COMMANDS["panel_parser_registry"][1]
    assert "-vv" in engine_command
    assert "--reporter=verbose" in panel_command


def test_parser_registry_contract_status_fails_when_required_markers_are_missing():
    source = Path("tests/cross_matrix/run_parser_registry_contract.py").read_text()

    assert "all_required_parser_markers_present" in source
    assert "not missing_markers" in source
