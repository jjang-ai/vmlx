import ast
from pathlib import Path


def test_family_detection_contract_default_out_tracks_current_release_proof_artifact():
    from tests.cross_matrix import run_model_family_detection_contract as gate

    assert gate.DEFAULT_OUT == Path(
        "build/current-model-family-detection-contract-20260531-post-step-lfm-refresh.json"
    )


def test_family_detection_contract_pins_named_release_rows():
    from tests.cross_matrix import run_model_family_detection_contract as gate

    names = set(gate.REQUIRED_ROWS)
    assert {
        "dsv4_deepseek_v4_native_cache_and_parser",
        "zaya_text_cca_tools_reasoning",
        "zaya_stale_stamp_reasoning_policy",
        "zaya1_vl_cca_multimodal",
        "zaya1_vl_jangtq_profiles_reasoning_policy",
        "ling_bailing_hybrid_plain_content",
        "nemotron_h_hybrid_text_not_stale_omni",
        "nemotron_h_registry_hybrid_cache",
        "qwen36_dense_linear_attention_hybrid_cache",
        "qwen36_moe_text_linear_attention_hybrid_cache",
        "qwen36_vl_video_hybrid",
        "qwen36_moe_vl_video_hybrid",
        "qwen36_release_rows_use_qwen35_family_alias",
        "qwen36_affine_jang_native_mtp_vl_video",
        "qwen36_mxfp4_mxfp8_vl",
        "qwen36_native_mtp_vl",
        "minimax_m2_parser_alias",
        "hy3_jangtq_hunyuan_qwen3",
        "hy3_jangtq_k_reasoning_policy",
        "step37_flash_vlm_full_sliding_kv",
        "lfm25_moe_hybrid_registry",
        "decode_speed_dsv4_minimax_canonical_parsers",
        "decode_speed_no_legacy_32k_startup_cap",
        "decode_speed_qwen36_mxfp8_native_mtp_rows",
        "decode_speed_nemotron_omni_nano_jangtq4_row",
        "decode_speed_jang_only_mx_matmul_policy",
        "decode_speed_distinct_jang_jangtq_mxfp_speed_rows",
        "decode_speed_plain_mlx_4bit_qwen36_row",
        "decode_speed_artifact_format_coverage_matrix",
        "decode_speed_all_declared_parsers_are_engine_registered",
        "decode_speed_all_declared_parsers_are_cli_choices",
        "decode_speed_existing_rows_match_engine_parser_policy",
        "decode_speed_existing_rows_match_engine_modality_policy",
        "decode_speed_build_command_parser_modality_policy",
        "panel_session_launch_parser_modality_policy",
        "decode_speed_large_external_jangtq_mxfp_gptoss_rows",
        "decode_speed_external_nemotron3_jangtq_mxfp_rows",
        "decode_speed_registry_cache_metadata_health",
        "decode_speed_plain_kv_cache_health_not_native",
        "decode_speed_local_high_risk_rows_match_engine_registry",
        "panel_local_high_risk_rows_match_detector_policy",
    }.issubset(names)


def test_family_detection_contract_hashes_app_and_engine_sources():
    from tests.cross_matrix import run_model_family_detection_contract as gate

    required = {
        "vmlx_engine/model_config_registry.py",
        "vmlx_engine/model_configs.py",
        "vmlx_engine/native_mtp.py",
        "panel/src/main/model-config-registry.ts",
        "panel/src/main/sessions.ts",
        "panel/src/shared/reasoningParserAliases.ts",
        "panel/tests/model-config-registry.test.ts",
        "panel/tests/settings-flow.test.ts",
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

    launch_cmd = " ".join(gate.COMMANDS["panel_session_launch_wiring"][1])
    assert "tests/settings-flow.test.ts" in launch_cmd
    assert "--reporter" in launch_cmd
    assert "verbose" in launch_cmd


def test_family_detection_contract_marker_validation_fails_if_required_row_missing(tmp_path):
    from tests.cross_matrix import run_model_family_detection_contract as gate

    result = {
        "engine_family_detection": {
            "returncode": 0,
            "stdout_tail": [
                "test_qwen3_5_config PASSED",
                "test_nemotron_h_stale_omni_stamp_without_media_stays_text_hybrid PASSED",
            ],
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


def test_decode_speed_gate_server_import_isolated_from_repo_cwd():
    import inspect

    from tests.cross_matrix.run_decode_speed_gate import (
        ROWS,
        SAFE_SERVER_CWD,
        build_serve_command,
        run_row,
    )

    cmd = build_serve_command(
        ROWS["qwen27_jang4m"],
        python=Path("/bundle/python3"),
        port=8799,
        prefill_step_size=2048,
    )

    assert cmd[:4] == ["/bundle/python3", "-B", "-s", "-P"]
    assert SAFE_SERVER_CWD == Path("/tmp")
    assert "cwd=str(SAFE_SERVER_CWD)" in inspect.getsource(run_row)


def test_decode_speed_gate_keeps_venv_python_symlink_identity():
    from tests.cross_matrix.run_decode_speed_gate import ROWS, build_serve_command

    python = Path(".venv/bin/python")

    cmd = build_serve_command(
        ROWS["qwen27_jang4m"],
        python=python,
        port=8799,
        prefill_step_size=2048,
    )

    assert cmd[0] == str(python.absolute())
    assert "uv/python" not in cmd[0]


def test_decode_speed_gate_wheel_probe_keeps_venv_python_symlink_identity(monkeypatch):
    import json
    import subprocess

    from tests.cross_matrix.run_decode_speed_gate import resolve_runtime_wheel_tags

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "mlx": ["cp313-cp313-macosx_26_0_arm64"],
                    "mlx-metal": ["py3-none-macosx_26_0_arm64"],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(subprocess, "run", fake_run)

    tags = resolve_runtime_wheel_tags(Path(".venv/bin/python"))

    assert captured["cmd"][:4] == [
        str(Path(".venv/bin/python").absolute()),
        "-B",
        "-s",
        "-P",
    ]
    assert "uv/python" not in captured["cmd"][0]
    assert captured["cwd"] == "/tmp"
    assert tags["mlx"] == ["cp313-cp313-macosx_26_0_arm64"]


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
            "is_mllm": True,
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


def test_decode_speed_gate_jang_only_rows_keep_text_mx_matmul_launch_policy():
    from tests.cross_matrix.run_decode_speed_gate import ROWS, build_serve_command

    jang_only_rows = {
        "qwen27_jang4m": ("qwen", "qwen3"),
        "qwen35_jang4k_ext": ("qwen", "qwen3"),
        "minimax_jang2l_crack": ("minimax", "minimax_m2"),
    }

    for row_name, (tool_parser, reasoning_parser) in jang_only_rows.items():
        row = ROWS[row_name]
        cmd = build_serve_command(
            row,
            python=Path("/bundle/python3"),
            port=8793,
            prefill_step_size=2048,
        )
        joined = " ".join(cmd)

        assert "JANG_" in row.path, row_name
        assert "JANGTQ" not in row.path, row_name
        assert "MXFP" not in row.path.upper(), row_name
        assert row.is_mllm is False, row_name
        assert "--is-mllm" not in cmd, row_name
        assert "--max-tokens" not in cmd, row_name
        assert cmd[cmd.index("--tool-call-parser") + 1] == tool_parser
        assert cmd[cmd.index("--reasoning-parser") + 1] == reasoning_parser
        assert "JANGTQ_MPP" not in joined
        assert "JANGTQ_DISABLE_DSV4" not in joined

    mtp_vlm = ROWS["qwen27_jang4m_mtp"]
    mtp_vlm_cmd = build_serve_command(
        mtp_vlm,
        python=Path("/bundle/python3"),
        port=8794,
        prefill_step_size=2048,
    )
    assert "JANG_" in mtp_vlm.path
    assert "JANGTQ" not in mtp_vlm.path
    assert mtp_vlm.is_mllm is True
    assert "--is-mllm" in mtp_vlm_cmd
    assert "--max-tokens" not in mtp_vlm_cmd
    assert mtp_vlm_cmd[mtp_vlm_cmd.index("--tool-call-parser") + 1] == "qwen"
    assert mtp_vlm_cmd[mtp_vlm_cmd.index("--reasoning-parser") + 1] == "qwen3"


def test_decode_speed_gate_extra_serve_arg_is_opt_in_without_mutating_row_defaults():
    from tests.cross_matrix.run_decode_speed_gate import (
        ROWS,
        build_serve_command,
        row_with_extra_serve_args,
    )

    base = ROWS["qwen27_jang4m"]
    assert "--prefill-keep-alloc" not in base.extra_args

    row = row_with_extra_serve_args(base, ["--prefill-keep-alloc"])
    cmd = build_serve_command(
        row,
        python=Path("/bundle/python3"),
        port=8793,
        prefill_step_size=2048,
    )

    assert "--prefill-keep-alloc" in cmd
    assert "--prefill-keep-alloc" not in base.extra_args


def test_decode_speed_gate_extra_serve_env_is_opt_in_without_mutating_row_defaults(
    monkeypatch, tmp_path
):
    from tests.cross_matrix import run_decode_speed_gate as gate

    base = gate.ROWS["qwen27_jang4m_mtp"]
    assert not getattr(base, "extra_serve_env", {})

    row = gate.row_with_extra_serve_env(
        base,
        ["VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL=0"],
    )
    assert row.extra_serve_env == {"VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL": "0"}
    assert not getattr(base, "extra_serve_env", {})

    captured: dict[str, object] = {}

    class FakeProc:
        def poll(self):
            return 0

        def send_signal(self, _signal):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return FakeProc()

    monkeypatch.setattr(gate.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gate.Path, "exists", lambda self: True)
    monkeypatch.setattr(gate, "resolve_runtime_wheel_tags", lambda _python: {})
    monkeypatch.setattr(gate, "resolve_row_registry_metadata", lambda _row: {})
    monkeypatch.setattr(gate, "wait_health", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        gate,
        "post_json",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "1 2 3"}}],
            "usage": {"completion_tokens": 3, "prompt_tokens": 3},
        },
    )
    monkeypatch.setattr(gate, "get_json", lambda *_args, **_kwargs: {})

    result = gate.run_row(
        row,
        port=8801,
        python=tmp_path / "python3",
        keep_logs=tmp_path,
        timeout_s=1,
        prefill_step_size=2048,
    )

    assert result["status"] == "pass"
    assert captured["env"]["VMLINUX_QWEN_VLM_GATED_DELTA_KERNEL"] == "0"


def test_decode_speed_gate_records_runtime_mlx_wheel_tags():
    from tests.cross_matrix.run_decode_speed_gate import resolve_runtime_wheel_tags

    tags = resolve_runtime_wheel_tags(Path("/bundle/python3"))

    assert "mlx" in tags
    assert "mlx-metal" in tags
    assert any(tag.startswith("unavailable:") for tag in tags["mlx"])
    assert any(tag.startswith("unavailable:") for tag in tags["mlx-metal"])


def test_decode_speed_gate_has_plain_mlx_qwen36_4bit_row():
    from tests.cross_matrix.run_decode_speed_gate import ROWS, build_serve_command

    row = ROWS["qwen35_4bit"]
    cmd = build_serve_command(
        row,
        python=Path("/bundle/python3"),
        port=8793,
        prefill_step_size=2048,
    )

    assert row.path == "/Users/example/models/Qwen3.6-35B-A3B-4bit"
    assert "JANG" not in row.path
    assert "JANGTQ" not in row.path
    assert "MXFP" not in row.path.upper()
    assert row.is_mllm is True
    assert row.tool_parser == "qwen"
    assert row.reasoning_parser == "qwen3"
    assert row.max_tokens <= 320
    assert row.expected_min_tps is not None and row.expected_min_tps > 0
    assert row.expected_min_pp is not None and row.expected_min_pp > 0
    assert "--max-tokens" not in cmd
    assert "--is-mllm" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"


def test_decode_speed_gate_artifact_format_coverage_matrix():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    matrix = {
        "jang_only_mx_matmul": ("qwen27_jang4m", "qwen35_jang4k_ext", "minimax_jang2l_crack"),
        "jang_mtp_vlm": ("qwen27_jang4m_mtp",),
        "jangtq_mxtq": ("qwen35_jangtq", "qwen36_35_jangtq4_ext", "hy3", "laguna"),
        "plain_mlx_4bit": ("qwen35_4bit",),
        "mxfp4_mlx": ("qwen27_mxfp4", "zaya_text_mxfp4", "ling_mxfp4", "nemotron_mxfp4"),
        "mxfp8_mtp": ("qwen27_mxfp8_mtp", "qwen35_mxfp8_mtp"),
        "dsv4_native_composite": ("dsv4_k", "dsv4_jang_dq2_gate3math6"),
    }

    for row_names in matrix.values():
        for row_name in row_names:
            assert row_name in ROWS
            assert ROWS[row_name].expected_min_tps is not None
            assert ROWS[row_name].expected_min_pp is not None

    assert all("JANG_" in ROWS[name].path and "JANGTQ" not in ROWS[name].path for name in matrix["jang_only_mx_matmul"])
    assert all("JANGTQ" in ROWS[name].path for name in matrix["jangtq_mxtq"])
    assert all("JANG_" in ROWS[name].path and "MTP" in ROWS[name].path and ROWS[name].is_mllm is True for name in matrix["jang_mtp_vlm"])
    assert all("JANG" not in ROWS[name].path and "MXFP" not in ROWS[name].path.upper() for name in matrix["plain_mlx_4bit"])
    assert all("MXFP4" in ROWS[name].path or "mxfp4" in ROWS[name].path for name in matrix["mxfp4_mlx"])
    assert all("MXFP8" in ROWS[name].path and "MTP" in ROWS[name].path for name in matrix["mxfp8_mtp"])
    assert all(ROWS[name].tool_parser == "dsml" for name in matrix["dsv4_native_composite"])
    assert all(ROWS[name].reasoning_parser == "deepseek_r1" for name in matrix["dsv4_native_composite"])


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


def test_decode_speed_gate_declared_parsers_are_cli_choices():
    from tests.cross_matrix.run_decode_speed_gate import ROWS
    from vmlx_engine.reasoning import list_parsers

    cli_source = Path("vmlx_engine/cli.py").read_text()
    cli_ast = ast.parse(cli_source)
    tool_choices = None
    for node in ast.walk(cli_ast):
        if not isinstance(node, ast.Call):
            continue
        if not any(
            isinstance(arg, ast.Constant) and arg.value == "--tool-call-parser"
            for arg in node.args
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                tool_choices = set(ast.literal_eval(keyword.value))
                break

    assert tool_choices is not None
    reasoning_choices = {"auto", "none", *list_parsers()}
    bad_tool_rows = []
    bad_reasoning_rows = []

    for row_name, row in ROWS.items():
        if row.tool_parser and row.tool_parser not in tool_choices:
            bad_tool_rows.append(f"{row_name}:{row.tool_parser}")
        if row.reasoning_parser and row.reasoning_parser not in reasoning_choices:
            bad_reasoning_rows.append(f"{row_name}:{row.reasoning_parser}")

    assert bad_tool_rows == []
    assert bad_reasoning_rows == []


def test_decode_speed_gate_build_command_preserves_row_parser_modality_policy():
    from tests.cross_matrix.run_decode_speed_gate import (
        ROWS,
        build_clean_env,
        build_serve_command,
    )

    representative_rows = {
        "dsv4_k": {"mllm": False, "tool": "dsml", "reasoning": "deepseek_r1"},
        "qwen27_jang4m": {"mllm": False, "tool": "qwen", "reasoning": "qwen3"},
        "qwen27_mxfp8_mtp": {"mllm": True, "tool": "qwen", "reasoning": "qwen3"},
        "zaya_vl_jangtq4": {"mllm": True, "tool": "zaya_xml", "reasoning": "qwen3"},
        "ling_mxfp4": {"mllm": False, "tool": "deepseek", "reasoning": None},
        "nemotron_omni_nano_jangtq4": {
            "mllm": False,
            "tool": "nemotron",
            "reasoning": "deepseek_r1",
        },
        "minimax_jang2l_crack": {
            "mllm": False,
            "tool": "minimax",
            "reasoning": "minimax_m2",
        },
    }

    for row_name, expected in representative_rows.items():
        row = ROWS[row_name]
        cmd = build_serve_command(
            row,
            python=Path("/bundle/python3"),
            port=8790,
            prefill_step_size=2048,
        )
        joined = " ".join(cmd)

        assert "--max-tokens" not in cmd, row_name
        assert "--is-mllm" in cmd if expected["mllm"] else "--is-mllm" not in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == expected["tool"]
        assert "--enable-auto-tool-choice" in cmd
        if expected["reasoning"] is None:
            assert "--reasoning-parser" not in cmd
        else:
            assert cmd[cmd.index("--reasoning-parser") + 1] == expected["reasoning"]
        assert "JANGTQ_MPP" not in joined
        assert "JANGTQ_DISABLE_DSV4" not in joined

    clean_env = build_clean_env(
        {
            "PYTHONPATH": "/source/tree",
            "JANGTQ_MPP_NAX": "1",
            "JANGTQ_MPP_NAX_DISABLE": "1",
            "JANGTQ_MPP_NAX_STRICT": "1",
            "JANGTQ_MPP_DENSE": "1",
            "JANGTQ_MPP_DENSE_STRICT": "1",
            "JANGTQ_DISABLE_DSV4_STREAM_LOAD": "1",
            "JANGTQ_DISABLE_DSV4_FAST_LOAD": "1",
            "KEEP_ME": "1",
        }
    )
    assert clean_env["KEEP_ME"] == "1"
    assert clean_env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert clean_env["PYTHONNOUSERSITE"] == "1"
    for forbidden in (
        "PYTHONPATH",
        "JANGTQ_MPP_NAX",
        "JANGTQ_MPP_NAX_DISABLE",
        "JANGTQ_MPP_NAX_STRICT",
        "JANGTQ_MPP_DENSE",
        "JANGTQ_MPP_DENSE_STRICT",
        "JANGTQ_DISABLE_DSV4_STREAM_LOAD",
        "JANGTQ_DISABLE_DSV4_FAST_LOAD",
    ):
        assert forbidden not in clean_env


def test_decode_speed_gate_has_large_external_mistral_gptoss_rows():
    from tests.cross_matrix.run_decode_speed_gate import ROWS, build_serve_command

    expected_rows = {
        "mistral_medium_jangtq_ext": {
            "path": "/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-JANGTQ",
            "mllm": True,
            "tool": "mistral",
            "reasoning": "mistral",
            "format_marker": "JANGTQ",
        },
        "mistral_medium_mxfp4_ext": {
            "path": "/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-mxfp4",
            "mllm": True,
            "tool": "mistral",
            "reasoning": "mistral",
            "format_marker": "mxfp4",
        },
        "gpt_oss_ext": {
            "path": "/Volumes/ModelDrive/dealignai/GPT-OSS-120B-MLX-CRACK",
            "mllm": False,
            "tool": "glm47",
            "reasoning": "openai_gptoss",
            "format_marker": "GPT-OSS",
        },
    }

    for row_name, expected in expected_rows.items():
        row = ROWS[row_name]
        cmd = build_serve_command(
            row,
            python=Path("/bundle/python3"),
            port=8791,
            prefill_step_size=2048,
        )

        assert row.path == expected["path"]
        assert expected["format_marker"] in row.path
        assert row.is_mllm is expected["mllm"]
        assert row.tool_parser == expected["tool"]
        assert row.reasoning_parser == expected["reasoning"]
        assert row.max_tokens <= 192
        assert row.expected_min_tps is not None and row.expected_min_tps > 0
        assert row.expected_min_pp is not None and row.expected_min_pp > 0
        assert cmd[cmd.index("--served-model-name") + 1] == row.name
        assert "--max-tokens" not in cmd
        assert "--is-mllm" in cmd if expected["mllm"] else "--is-mllm" not in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == expected["tool"]
        assert cmd[cmd.index("--reasoning-parser") + 1] == expected["reasoning"]


def test_decode_speed_gate_has_external_nemotron3_jangtq_mxfp_rows():
    from tests.cross_matrix.run_decode_speed_gate import ROWS, build_serve_command

    expected_rows = {
        "nemotron3_jangtq2_ext": {
            "path": "/Volumes/ModelDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
            "format_marker": "JANGTQ2",
            "min_tps": 20.0,
            "min_pp": 400.0,
        },
        "nemotron3_mxfp4_ext": {
            "path": "/Volumes/ModelDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-MXFP4",
            "format_marker": "MXFP4",
            "min_tps": 18.0,
            "min_pp": 400.0,
        },
    }

    for row_name, expected in expected_rows.items():
        row = ROWS[row_name]
        cmd = build_serve_command(
            row,
            python=Path("/bundle/python3"),
            port=8792,
            prefill_step_size=2048,
        )

        assert row.path == expected["path"]
        assert expected["format_marker"] in row.path
        assert row.is_mllm is False
        assert row.tool_parser == "nemotron"
        assert row.reasoning_parser == "deepseek_r1"
        assert row.max_tokens <= 320
        assert row.expected_min_tps == expected["min_tps"]
        assert row.expected_min_pp == expected["min_pp"]
        assert cmd[cmd.index("--served-model-name") + 1] == row.name
        assert "--max-tokens" not in cmd
        assert "--is-mllm" not in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == "nemotron"
        assert cmd[cmd.index("--reasoning-parser") + 1] == "deepseek_r1"


def test_decode_speed_local_high_risk_rows_match_current_engine_registry():
    from pathlib import Path

    from tests.cross_matrix.run_decode_speed_gate import ROWS
    from vmlx_engine.model_config_registry import get_model_config_registry

    row_names = (
        "dsv4_k",
        "qwen27_jang4m",
        "qwen27_jang4m_mtp",
        "qwen27_mxfp4",
        "qwen27_mxfp8_mtp",
        "qwen35_jangtq",
        "qwen35_4bit",
        "qwen35_mxfp8_mtp",
        "hy3",
        "nemotron_jangtq",
        "nemotron_omni_nano_jangtq4",
        "nemotron_mxfp4",
    )
    expected_families = {
        "dsv4_k": ("deepseek_v4", "kv", "deepseek_v4_composite"),
        "qwen27_jang4m": ("qwen3_5", "hybrid", None),
        "qwen27_jang4m_mtp": ("qwen3_5", "hybrid", None),
        "qwen27_mxfp4": ("qwen3_5", "hybrid", None),
        "qwen27_mxfp8_mtp": ("qwen3_5", "hybrid", None),
        "qwen35_jangtq": ("qwen3_5_moe", "hybrid", None),
        "qwen35_4bit": ("qwen3_5_moe", "hybrid", None),
        "qwen35_mxfp8_mtp": ("qwen3_5_moe", "hybrid", None),
        "hy3": ("hy_v3", "kv", "kv"),
        "nemotron_jangtq": ("nemotron_h", "hybrid", "nemotron_h_ssm_attention"),
        "nemotron_omni_nano_jangtq4": ("nemotron_h", "hybrid", "nemotron_h_ssm_attention"),
        "nemotron_mxfp4": ("nemotron_h", "hybrid", "nemotron_h_ssm_attention"),
    }

    registry = get_model_config_registry()

    for row_name in row_names:
        row = ROWS[row_name]
        assert Path(row.path).is_dir(), f"{row_name} path missing: {row.path}"
        detected = registry.lookup(row.path)
        family_name, cache_type, cache_subtype = expected_families[row_name]

        assert detected.family_name == family_name, row_name
        assert detected.cache_type == cache_type, row_name
        assert detected.cache_subtype == cache_subtype, row_name
        assert detected.tool_parser == row.tool_parser, row_name
        assert detected.reasoning_parser == row.reasoning_parser, row_name
        assert detected.is_mllm is row.is_mllm, row_name


def test_decode_speed_gate_matches_registry_parser_policy_for_ling_and_nemotron():
    from tests.cross_matrix.run_decode_speed_gate import ROWS

    for row_name in ("ling", "ling_crack", "ling_mxfp4"):
        row = ROWS[row_name]
        assert row.tool_parser == "deepseek"
        assert row.reasoning_parser is None

    for row_name in (
        "nemotron_jangtq",
        "nemotron_omni_nano_jangtq4",
        "nemotron_mxfp4",
        "nemotron3_jangtq2_ext",
        "nemotron3_mxfp4_ext",
    ):
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
            "generic_turboquant_kv": {
                "enabled": True,
                "reason": "hybrid_attention_kv_only",
            },
            "live_attention_tq_kv": {
                "enabled": True,
                "applies_to": "attention_kv_layers_only",
                "ssm_policy": "native_full_precision_companion_state",
            },
        },
    ) == []

    assert cache_health_mismatches(
        {
            "family_name": "bailing_hybrid",
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


def test_decode_speed_gate_detects_plain_kv_cache_health_mismatches():
    from tests.cross_matrix.run_decode_speed_gate import cache_health_mismatches

    plain_kv_metadata = {
        "family_name": "qwen3_5",
        "cache_type": "kv",
        "cache_subtype": None,
    }

    assert cache_health_mismatches(
        plain_kv_metadata,
        {
            "cache_type": "native_composite",
            "generic_turboquant_kv": {"enabled": False},
        },
    ) == ["registry kv cache expected health cache_type=kv-or-paged_kv"]

    assert cache_health_mismatches(
        plain_kv_metadata,
        {
            "cache_type": "typed_cca",
            "generic_turboquant_kv": {"enabled": False},
        },
    ) == ["registry kv cache expected health cache_type=kv-or-paged_kv"]

    assert (
        cache_health_mismatches(
            plain_kv_metadata,
            {
                "cache_type": "paged_kv",
                "generic_turboquant_kv": {"enabled": True},
            },
        )
        == []
    )


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
