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
        "decode_speed_dsv4_minimax_canonical_parsers",
        "decode_speed_no_legacy_32k_startup_cap",
        "decode_speed_qwen36_mxfp8_native_mtp_rows",
        "decode_speed_nemotron_omni_nano_jangtq4_row",
        "decode_speed_distinct_jang_jangtq_mxfp_speed_rows",
        "decode_speed_all_declared_parsers_are_engine_registered",
        "decode_speed_existing_rows_match_engine_parser_policy",
        "decode_speed_existing_rows_match_engine_modality_policy",
        "decode_speed_registry_cache_metadata_health",
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
    assert "tests/test_model_family_detection_contract.py" in engine_cmd
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


def test_decode_speed_gate_has_distinct_jang_jangtq_mxfp_speed_rows():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    groups = {
        "jang_only_mx_matmul": ("qwen27_jang4m", "minimax_jang2l_crack"),
        "jangtq_mxtq_turboquant": ("qwen35_jangtq", "qwen36_35_jangtq4_ext"),
        "mxfp4_mlx": ("qwen27_mxfp4", "zaya_text_mxfp4", "nemotron_mxfp4"),
        "mxfp8_mlx_mtp": ("qwen27_mxfp8_mtp", "qwen35_mxfp8_mtp"),
    }

    for row_names in groups.values():
        for row_name in row_names:
            row = ROWS[row_name]
            assert row.expected_min_tps is not None, row_name
            assert row.expected_min_tps > 0, row_name
            assert row.expected_min_pp is not None, row_name
            assert row.expected_min_pp > 0, row_name

    assert all("JANGTQ" not in ROWS[name].path for name in groups["jang_only_mx_matmul"])
    assert all("JANG_" in ROWS[name].path for name in groups["jang_only_mx_matmul"])
    assert all("JANGTQ" in ROWS[name].path for name in groups["jangtq_mxtq_turboquant"])
    assert all("MXFP4" in ROWS[name].path or "mxfp4" in ROWS[name].path for name in groups["mxfp4_mlx"])
    assert all("MXFP8" in ROWS[name].path for name in groups["mxfp8_mlx_mtp"])


def test_decode_speed_gate_declared_parsers_are_engine_registered():
    from tests.cross_matrix.run_decode_speed_gate import ROWS
    from vmlx_engine.reasoning import list_parsers
    from vmlx_engine.tool_parsers import ToolParserManager

    registered_tool_parsers = set(ToolParserManager.list_registered())
    registered_reasoning_parsers = set(list_parsers())
    bad_tool_rows = []
    bad_reasoning_rows = []

    for row_name, row in ROWS.items():
        if row.tool_parser and row.tool_parser not in registered_tool_parsers:
            bad_tool_rows.append(f"{row_name}:{row.tool_parser}")
        if row.reasoning_parser and row.reasoning_parser not in registered_reasoning_parsers:
            bad_reasoning_rows.append(f"{row_name}:{row.reasoning_parser}")

    assert bad_tool_rows == []
    assert bad_reasoning_rows == []


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


def test_decode_speed_gate_records_registry_cache_metadata_for_existing_rows():
    from tests.cross_matrix.run_decode_speed_gate import (
        ROWS,
        resolve_row_registry_metadata,
    )
    from vmlx_engine.model_config_registry import get_model_config_registry

    registry = get_model_config_registry()
    registry.clear_cache()
    rows_seen = []
    cache_families = set()
    cache_subtypes = set()

    for row_name, row in ROWS.items():
        if not Path(row.path).exists():
            continue
        cfg = registry.lookup(row.path)
        metadata = resolve_row_registry_metadata(row)
        rows_seen.append(row_name)
        cache_families.add(metadata["cache_type"])
        cache_subtypes.add(metadata["cache_subtype"])

        assert metadata["family_name"] == cfg.family_name
        assert metadata["cache_type"] == cfg.cache_type
        assert metadata["cache_subtype"] == cfg.cache_subtype
        assert metadata["tool_parser"] == cfg.tool_parser
        assert metadata["reasoning_parser"] == cfg.reasoning_parser
        assert metadata["is_mllm"] == cfg.is_mllm

    assert rows_seen
    assert {"kv", "hybrid"}.issubset(cache_families)
    assert "zaya_cca" in cache_subtypes
    assert "deepseek_v4_composite" in cache_subtypes


def test_decode_speed_gate_detects_cache_health_mismatches():
    from tests.cross_matrix.run_decode_speed_gate import cache_health_mismatches

    assert cache_health_mismatches(
        {
            "family_name": "deepseek_v4",
            "cache_type": "kv",
            "cache_subtype": "deepseek_v4_composite",
        },
        {
            "cache_type": "paged_kv",
            "generic_turboquant_kv": {"enabled": True},
        },
    ) == [
        "registry deepseek_v4_composite expected health cache_type=native_composite",
        "registry deepseek_v4_composite expected generic_turboquant_kv.enabled=false",
    ]

    assert cache_health_mismatches(
        {
            "family_name": "zaya1_vl",
            "cache_type": "hybrid",
            "cache_subtype": "zaya_cca",
        },
        {
            "cache_type": "paged_kv",
            "generic_turboquant_kv": {"enabled": True},
        },
    ) == [
        "registry zaya_cca expected health cache_type=typed_cca",
        "registry zaya_cca expected generic_turboquant_kv.enabled=false",
    ]

    assert cache_health_mismatches(
        {
            "family_name": "qwen3_5",
            "cache_type": "hybrid",
            "cache_subtype": None,
        },
        {
            "cache_type": "hybrid_ssm_typed",
            "generic_turboquant_kv": {"enabled": True},
        },
    ) == [
        "registry hybrid cache expected generic_turboquant_kv.enabled=false",
    ]


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
