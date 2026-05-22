from pathlib import Path


def test_family_detection_contract_pins_named_release_rows():
    from tests.cross_matrix import run_model_family_detection_contract as gate

    names = set(gate.REQUIRED_ROWS)
    assert {
        "dsv4_deepseek_v4_native_cache_and_parser",
        "zaya_text_cca_tools_reasoning",
        "zaya1_vl_cca_multimodal",
        "ling_bailing_hybrid_plain_content",
        "nemotron_h_hybrid_text_not_stale_omni",
        "qwen36_vl_video_hybrid",
        "qwen36_moe_vl_video_hybrid",
        "qwen36_mxfp4_mxfp8_vl",
        "qwen36_native_mtp_vl",
        "minimax_m2_parser_alias",
        "hy3_jangtq_hunyuan_qwen3",
    }.issubset(names)


def test_family_detection_contract_hashes_app_and_engine_sources():
    from tests.cross_matrix import run_model_family_detection_contract as gate

    required = {
        "vmlx_engine/model_config_registry.py",
        "vmlx_engine/model_configs.py",
        "vmlx_engine/native_mtp.py",
        "panel/src/main/model-config-registry.ts",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/tests/model-config-registry.test.ts",
        "tests/cross_matrix/run_decode_speed_gate.py",
        "tests/test_model_config_registry.py",
    }

    assert required.issubset(set(gate.SOURCE_HASH_FILES))


def test_family_detection_contract_runs_verbose_engine_and_panel_rows():
    from tests.cross_matrix import run_model_family_detection_contract as gate

    engine_cmd = " ".join(gate.COMMANDS["engine_family_detection"][1])
    panel_cmd = " ".join(gate.COMMANDS["panel_family_detection"][1])

    assert "tests/test_model_config_registry.py" in engine_cmd
    assert "-vv" in engine_cmd
    assert "tests/model-config-registry.test.ts" in panel_cmd
    assert "--reporter" in panel_cmd
    assert "verbose" in panel_cmd


def test_family_detection_contract_marker_validation_fails_if_required_row_missing(tmp_path):
    from tests.cross_matrix import run_model_family_detection_contract as gate

    result = {
        "engine_family_detection": {
            "returncode": 0,
            "stdout_tail": ["test_qwen3_5_config PASSED"],
        },
        "panel_family_detection": {
            "returncode": 0,
            "stdout_tail": [
                "detects backend-covered model_type=nemotron_h_v2",
                "does not route Nemotron-H text extracts through MLLM",
            ],
        },
    }

    checks = gate._build_checks(result)

    assert checks["qwen36_vl_video_hybrid"] is False
    assert checks["nemotron_h_hybrid_text_not_stale_omni"] is True


def test_decode_speed_gate_uses_canonical_release_parsers_for_dsv4_and_minimax():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    for row_name in ("minimax", "minimax_k", "minimax_jang2l_crack"):
        row = ROWS[row_name]
        assert row.tool_parser == "minimax"
        assert row.reasoning_parser == "minimax_m2"

    for row_name in ("dsv4_k", "dsv4_jang_dq2_gate3math6"):
        row = ROWS[row_name]
        assert row.tool_parser == "dsml"
        assert row.reasoning_parser == "deepseek_r1"


def test_decode_speed_gate_does_not_force_legacy_32k_startup_output_cap():
    source = Path("tests/cross_matrix/run_decode_speed_gate.py").read_text()

    assert '"--max-tokens",' not in source
    assert '"32768",' not in source
