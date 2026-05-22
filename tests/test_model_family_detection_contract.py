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


def test_decode_speed_gate_has_explicit_qwen36_mxfp8_and_native_mtp_rows():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    required = {
        "qwen27_mxfp8_mtp": {
            "path": "/Users/example/models/JANGQ/Qwen3.6-27B-MXFP8-MTP",
            "is_mllm": True,
            "tool_parser": "qwen",
            "reasoning_parser": "qwen3",
        },
        "qwen35_mxfp8_mtp": {
            "path": "/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP",
            "is_mllm": True,
            "tool_parser": "qwen",
            "reasoning_parser": "qwen3",
        },
        "qwen27_jang4m_mtp": {
            "path": "/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP",
            "is_mllm": False,
            "tool_parser": "qwen",
            "reasoning_parser": "qwen3",
        },
    }

    for row_name, expected in required.items():
        row = ROWS[row_name]
        assert row.path == expected["path"]
        assert row.is_mllm is expected["is_mllm"]
        assert row.tool_parser == expected["tool_parser"]
        assert row.reasoning_parser == expected["reasoning_parser"]
        assert row.max_tokens <= 320


def test_decode_speed_gate_has_explicit_nemotron_omni_nano_jangtq4_row():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    row = ROWS["nemotron_omni_nano_jangtq4"]
    assert row.path == "/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK"
    assert row.tool_parser == "nemotron"
    assert row.reasoning_parser == "deepseek_r1"
    assert row.max_tokens <= 320


def test_decode_speed_gate_matches_registry_parser_policy_for_ling_and_nemotron():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    for row_name in ("ling", "ling_crack", "ling_mxfp4"):
        row = ROWS[row_name]
        assert row.tool_parser == "deepseek"
        assert row.reasoning_parser is None

    for row_name in ("nemotron_jangtq", "nemotron_omni_nano_jangtq4", "nemotron_mxfp4"):
        row = ROWS[row_name]
        assert row.tool_parser == "nemotron"
        assert row.reasoning_parser == "deepseek_r1"


def test_existing_decode_speed_rows_match_engine_registry_parser_policy():
    from tests.cross_matrix.run_decode_speed_gate import ROWS
    from vmlx_engine.model_config_registry import get_model_config_registry

    registry = get_model_config_registry()
    registry.clear_cache()
    mismatches = []

    for row_name, row in ROWS.items():
        if not Path(row.path).exists():
            continue
        cfg = registry.lookup(row.path)
        if row.tool_parser != cfg.tool_parser:
            mismatches.append(
                f"{row_name}: tool row={row.tool_parser!r} registry={cfg.tool_parser!r}"
            )
        if row.reasoning_parser != cfg.reasoning_parser:
            mismatches.append(
                f"{row_name}: reasoning row={row.reasoning_parser!r} registry={cfg.reasoning_parser!r}"
            )

    assert mismatches == []


def test_existing_decode_speed_rows_match_engine_registry_modality_policy():
    from tests.cross_matrix.run_decode_speed_gate import ROWS
    from vmlx_engine.model_config_registry import get_model_config_registry

    registry = get_model_config_registry()
    registry.clear_cache()
    mismatches = []

    for row_name, row in ROWS.items():
        if not Path(row.path).exists():
            continue
        cfg = registry.lookup(row.path)
        if row.is_mllm != cfg.is_mllm:
            mismatches.append(
                f"{row_name}: is_mllm row={row.is_mllm!r} registry={cfg.is_mllm!r}"
            )

    assert mismatches == []


def test_dsv4_live_cache_gates_use_canonical_parser_and_no_legacy_32k_startup_cap():
    gate_paths = [
        Path("tests/cross_matrix/run_dsv4_long_context_gate.py"),
        Path("tests/cross_matrix/run_dsv4_responses_cache_gate.py"),
    ]

    for path in gate_paths:
        source = path.read_text()
        assert '"--tool-call-parser",\n        "deepseek",' not in source
        assert '"--tool-call-parser",\n        "dsml",' in source
        assert '"--max-tokens",\n        "32768",' not in source
