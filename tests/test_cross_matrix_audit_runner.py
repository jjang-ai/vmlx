"""Regression tests for the live production-family audit harness.

These tests keep the harness from producing false-positive PASS rows when a
model mentions the expected token inside reasoning but never emits visible
content.
"""

from tests.cross_matrix.run_production_family_audit import (
    ModelRow,
    ROWS,
    audit_child_env_for_row,
    cache_probe_content_ok,
    cache_exact_hit_probe,
    cache_exact_hit_required,
    capability_endpoint_contract_ok,
    chat_basic_turn_ok,
    dsv4_thinking_mode_max_ok,
    dsv4_long_context_full_output_ok,
    extract_anthropic_text_and_stop,
    extract_ollama_visible_text_and_stop,
    is_non_length_stop,
    normalize_python_executable,
    normalize_short_answer,
    simple_loop_score,
    static_audit,
)
from tests.cross_matrix import run_production_family_audit as audit_harness


def test_live_audit_root_is_this_checkout_not_stale_absolute_path():
    expected_root = audit_harness.Path(audit_harness.__file__).resolve().parents[2]

    assert audit_harness.ROOT == expected_root
    assert audit_harness.ROOT != audit_harness.Path("/Users/example/mlx/vllm-mlx")


def test_anthropic_exact_probe_ignores_reasoning_blocks():
    resp = {
        "content": [
            {"type": "thinking", "thinking": "I should answer anthropic-ok"},
            {"type": "text", "text": ""},
        ],
        "stop_reason": "max_tokens",
    }

    text, stop = extract_anthropic_text_and_stop(resp)

    assert normalize_short_answer(text).lower() != "anthropic-ok"
    assert not is_non_length_stop(stop)


def test_anthropic_exact_probe_reads_visible_text_only():
    resp = {
        "content": [
            {"type": "thinking", "thinking": "ignore me"},
            {"type": "text", "text": "\nanthropic-ok\n"},
        ],
        "stop_reason": "end_turn",
    }

    text, stop = extract_anthropic_text_and_stop(resp)

    assert normalize_short_answer(text).lower() == "anthropic-ok"
    assert is_non_length_stop(stop)


def test_ollama_exact_probe_ignores_thinking_field():
    resp = {
        "message": {
            "content": "",
            "thinking": "The answer should be ollama-ok",
        },
        "done_reason": "length",
    }

    text, stop = extract_ollama_visible_text_and_stop(resp)

    assert normalize_short_answer(text).lower() != "ollama-ok"
    assert not is_non_length_stop(stop)


def test_ollama_exact_probe_reads_visible_content():
    resp = {
        "message": {
            "content": "ollama-ok",
            "thinking": "reasoning",
        },
        "done_reason": "stop",
    }

    text, stop = extract_ollama_visible_text_and_stop(resp)

    assert normalize_short_answer(text).lower() == "ollama-ok"
    assert is_non_length_stop(stop)


def test_short_answer_normalizer_accepts_terminal_punctuation_only():
    assert normalize_short_answer("Paris.").lower() == "paris"
    assert normalize_short_answer('"Paris!"').lower() == "paris"
    assert normalize_short_answer("Paris. Explanation").lower() != "paris"


def test_zaya_rows_are_present_and_marked_cca():
    rows = {row.id: row for row in ROWS}

    assert rows["zaya_jangtq2"].cache_profile == "zaya_cca"
    assert rows["zaya_jangtq4"].cache_profile == "zaya_cca"
    assert rows["zaya_mxfp4"].cache_profile == "zaya_cca"


def test_zaya_vl_rows_are_present_and_marked_cca_mllm():
    rows = {row.id: row for row in ROWS}

    assert rows["zaya_vl_mxfp4"].family == "zaya1_vl"
    assert rows["zaya_vl_mxfp4"].kind == "vl"
    assert rows["zaya_vl_mxfp4"].cache_profile == "zaya_cca"
    assert rows["zaya_vl_jangtq4"].family == "zaya1_vl"
    assert rows["zaya_vl_jangtq4"].kind == "vl"
    assert rows["zaya_vl_jangtq4"].cache_profile == "zaya_cca"
    assert rows["zaya_vl_jangtq_k"].family == "zaya1_vl"
    assert rows["zaya_vl_jangtq_k"].kind == "vl"
    assert rows["zaya_vl_jangtq_k"].cache_profile == "zaya_cca"


def test_hy3_release_row_is_present_with_hunyuan_contract():
    rows = {row.id: row for row in ROWS}

    assert rows["hy3_preview_jangtq2"].family == "hy_v3"
    assert rows["hy3_preview_jangtq2"].expect_reasoning is True
    assert rows["hy3_preview_jangtq2"].expect_tool_parser == "hunyuan"


def test_ling_rows_do_not_expect_reasoning():
    rows = {row.id: row for row in ROWS}

    assert rows["ling_flash_tq"].expect_reasoning is False
    assert rows["ling_flash_tq2_crack"].expect_reasoning is False
    assert rows["ling_flash_mxfp4_crack"].expect_reasoning is False


def test_zaya_static_audit_exposes_cache_subtype_when_local_bundle_exists():
    row = next(row for row in ROWS if row.id == "zaya_jangtq2")
    static = static_audit(row)

    if not static["exists"]:
        return
    assert static["family_expected"] == "zaya"
    assert static["jang"]["cache_subtype"] == "zaya_cca"
    assert static["registry"]["cache_subtype"] == "zaya_cca"


def test_zaya_cca_rows_require_typed_exact_hit_cache_probe():
    rows = {row.id: row for row in ROWS}

    assert cache_exact_hit_required(rows["zaya_jangtq2"])
    assert cache_exact_hit_required(rows["zaya_jangtq4"])
    assert cache_exact_hit_required(rows["zaya_mxfp4"])
    assert cache_exact_hit_required(rows["dsv4_tq"])


def test_live_basic_turn_accepts_coherent_non_exact_acknowledgement():
    row = next(row for row in ROWS if row.id == "zaya_mxfp4")

    assert chat_basic_turn_ok(
        row,
        code=200,
        finish="stop",
        content="blue; cat",
        reasoning="",
    )


def test_live_basic_turn_rejects_reasoning_leak_for_non_reasoning_row():
    row = next(row for row in ROWS if row.id == "zaya_mxfp4")

    assert not chat_basic_turn_ok(
        row,
        code=200,
        finish="stop",
        content="blue; cat",
        reasoning="I should answer with the stored facts.",
    )


def test_cache_probe_accepts_coherent_non_exact_cached_answer():
    row = next(row for row in ROWS if row.id == "zaya_jangtq2")

    assert cache_probe_content_ok(
        row=row,
        expected="blue",
        first_content="The phrase repeats the color word blue.",
        repeat_content="The phrase repeats the color word blue.",
        strict_short_answer=True,
        min_count=1,
    )


def test_responses_tool_choice_rejects_visible_tool_markup_leak():
    assert not audit_harness.responses_tool_choice_output_ok(
        "<zyphra_tool_call>\n<function=list_directory>"
    )
    assert not audit_harness.responses_tool_choice_output_ok(
        "For a request to list files:\n<py>\ndef list_directory(path): pass"
    )
    assert audit_harness.responses_tool_choice_output_ok("")


def test_live_audit_can_opt_into_source_vmlx_imports():
    row = next(row for row in ROWS if row.id == "zaya_vl_jangtq4")

    env = audit_child_env_for_row(
        row,
        base_env={"VMLINUX_AUDIT_USE_SOURCE_VMLX": "1", "PYTHONPATH": "stale"},
    )

    assert env["PYTHONPATH"] == str(audit_harness.ROOT)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_live_audit_preserves_venv_python_symlink_for_subprocess_launch(tmp_path):
    py = normalize_python_executable(".venv/bin/python", cwd=tmp_path)

    assert py == tmp_path / ".venv/bin/python"


def test_live_audit_deferred_rows_keep_failure_details():
    row = next(row for row in ROWS if row.id == "zaya_vl_jangtq_k")
    result = {
        "checks": [
            {
                "name": "responses_tool_history_continuation",
                "ok": False,
                "detail": {"text": "<|box_start|>[100,100,188,188]<|box_end|>"},
            }
        ]
    }

    audit_harness.finalize_live_status(row, result)

    assert result["status"] == "DEFERRED"
    assert "Responses tool-history" in result["reason"]
    assert result["failures"][0]["name"] == "responses_tool_history_continuation"

def test_live_audit_root_points_at_current_checkout():
    """Worktree gates must not import modules from the stale main checkout."""
    expected = audit_harness.Path(__file__).resolve().parents[1]

    assert audit_harness.ROOT == expected


def test_dsv4_row_points_at_current_sub80_upload_candidate_bundle():
    rows = {row.id: row for row in ROWS}

    assert rows["dsv4_tq"].path.endswith("DeepSeek-V4-Flash-JANGTQ-K")


def test_dsv4_pure_affine_keeper_is_in_production_audit_matrix():
    rows = {row.id: row for row in ROWS}

    row = rows["dsv4_jang_dq2_gate3math6"]
    assert row.path == (
        "/Users/example/models/JANGQ/"
        "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
    )
    assert row.family == "deepseek_v4"
    assert row.cache_profile == "dsv4_composite"
    assert row.expect_tool_parser == "dsml"


def test_dsv4_decode_speed_gate_tracks_jangtq_and_pure_affine_lanes():
    from tests.cross_matrix import run_decode_speed_gate

    assert run_decode_speed_gate.ROWS["dsv4_k"].path.endswith(
        "DeepSeek-V4-Flash-JANGTQ-K"
    )
    assert run_decode_speed_gate.ROWS["dsv4_jang_dq2_gate3math6"].path.endswith(
        "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
    )


def test_dsv4_long_context_gate_defaults_to_pure_affine_keeper():
    from tests.cross_matrix import run_dsv4_long_context_gate

    assert run_dsv4_long_context_gate.DEFAULT_MODEL.endswith(
        "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
    )


def test_dsv4_long_context_gate_compares_cached_followup_to_no_cache():
    """Live DSV4 gate must separate cache corruption from artifact quality."""
    import inspect
    from tests.cross_matrix import run_dsv4_long_context_gate

    src = inspect.getsource(run_dsv4_long_context_gate.run)

    assert "follow_no_cache" in src
    assert '"skip_prefix_cache": True' in src
    assert '"repetition_penalty": 1.0' in src
    assert "GATE RUN ID" in src
    assert "follow_cached: missing cached_tokens evidence" in src
    assert "store_turn: unexpected prefix cache hit" in src
    assert "cached_vs_no_cache" in src


def test_dsv4_responses_cache_gate_defaults_to_pure_affine_keeper():
    from tests.cross_matrix import run_dsv4_responses_cache_gate

    assert run_dsv4_responses_cache_gate.DEFAULT_MODEL.endswith(
        "DeepSeek-V4-Flash-JANG_DQ2-Token8-DownG32-Gate3Math6-NoMTP"
    )


def test_dsv4_responses_cache_gate_uses_previous_response_and_no_cache_control():
    """DSV4 Responses cache proof must separate previous_response_id reuse from cold quality."""
    import inspect
    from tests.cross_matrix import run_dsv4_responses_cache_gate

    src = inspect.getsource(run_dsv4_responses_cache_gate.run)
    module_src = inspect.getsource(run_dsv4_responses_cache_gate)

    assert "/v1/responses" in src
    assert "previous_response_id" in src
    assert "explicit_no_cache_full_prompt" in src
    assert '"skip_prefix_cache": True' in src
    assert '"repetition_penalty": 1.0' in src
    assert "GATE RUN ID" in src
    assert "previous_response_follow: missing cached_tokens evidence" in src
    assert "previous_response_follow: missing dsv4 cache_detail" in src
    assert "store_turn: unexpected prefix cache hit" in src
    assert "explicit_no_cache_full_prompt: unexpected cache usage" in src
    assert "input_tokens_details" in module_src
    assert "native_cache" in src


def test_dsv4_responses_cache_gate_records_stream_ttft_and_usage():
    """Streaming TTFT must be measured from first real model delta, not SSE setup events."""
    import inspect
    from tests.cross_matrix import run_dsv4_responses_cache_gate

    events = [
        b'event: response.created\n',
        b'data: {"type":"response.created"}\n',
        b"\n",
        b'event: response.output_item.added\n',
        b'data: {"type":"response.output_item.added"}\n',
        b"\n",
        b'event: response.output_text.delta\n',
        b'data: {"type":"response.output_text.delta","delta":"CERULEAN / "}\n',
        b"\n",
        b'event: response.usage\n',
        b'data: {"type":"response.usage","usage":{"input_tokens":10,"output_tokens":1,"input_tokens_details":{"cached_tokens":9,"cache_detail":"paged+dsv4"}}}\n',
        b"\n",
        b'event: response.completed\n',
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":3,"input_tokens_details":{"cached_tokens":9,"cache_detail":"paged+dsv4"}}}}\n',
        b"\n",
    ]

    parsed = list(run_dsv4_responses_cache_gate.iter_sse_json_lines(events))
    assert [event["type"] for event in parsed] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.usage",
        "response.completed",
    ]

    module_src = inspect.getsource(run_dsv4_responses_cache_gate)
    run_src = inspect.getsource(run_dsv4_responses_cache_gate.run)
    assert "stream_previous_response_follow" in run_src
    assert '"stream": True' in run_src
    assert '"stream_options": {"include_usage": True}' in run_src
    assert "ttft_seconds" in module_src
    assert "response.output_text.delta" in module_src
    assert "stream_previous_response_follow: missing TTFT" in run_src
    assert "stream_previous_response_follow: missing dsv4 cache_detail" in run_src


def test_mistral_medium_jangtq_path_matches_current_drive_layout():
    rows = {row.id: row for row in ROWS}

    assert rows["mistral_medium35_tq"].path == (
        "/Volumes/ModelDrive/jangq-ai/Mistral-Medium-3.5-128B-JANGTQ"
    )


def test_dsv4_static_audit_exposes_actual_bit_plan_when_local_bundle_exists():
    rows = {row.id: row for row in ROWS}
    static = static_audit(rows["dsv4_tq"])

    if not static["exists"]:
        return
    bit_plan = static["dsv4_artifact_bit_plan"]
    assert bit_plan["checked"] is True
    assert bit_plan["routed_bit_counts"] == {"2": 38, "4": 5}
    assert bit_plan["metadata_matches_actual"] is True
    assert bit_plan["sidecar"]["missing_keys"] == []
    assert bit_plan["issues"] == []


def test_dsv4_static_audit_accepts_canonical_encoder_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"deepseek_v4","quantization":{"mxtq_bits":{"routed_expert":2}}}'
    )
    (tmp_path / "jang_config.json").write_text(
        """
        {
          "model_family": "deepseek_v4",
          "weight_format": "mxtq",
          "chat": {
            "encoder": "encoding_dsv4",
            "encoder_fn": "encode_messages",
            "reasoning": {"supported": true, "parser": "deepseek_r1"},
            "tool_calling": {"supported": true, "parser": "dsml"},
            "sampling_defaults": {"repetition_penalty_thinking": 1.0}
          }
        }
        """
    )
    row = ModelRow(
        id="dsv4_tmp",
        label="tmp dsv4",
        path=str(tmp_path),
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
    )

    static = static_audit(row)

    assert (
        "DSV4 missing chat template file/config; canonical encoder shim still required"
        not in static["issues"]
    )


def test_dsv4_static_audit_reports_mtp_drop_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_nextn_predict_layers":0}'
    )
    (tmp_path / "jang_config.json").write_text(
        '{"model_family":"deepseek_v4","weight_format":"mxtq","drop_mtp":true}'
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
    )
    row = ModelRow(
        id="dsv4_tmp",
        label="tmp dsv4",
        path=str(tmp_path),
        family="deepseek_v4",
        cache_profile="dsv4_composite",
    )

    static = static_audit(row)

    assert static["mtp"] == {
        "config_num_nextn_predict_layers": 0,
        "jang_drop_mtp": True,
        "index_has_mtp_tensors": False,
        "artifact_available": False,
        "runtime_available": False,
        "runtime_reason": "jang_config.drop_mtp=true",
        "status": "dropped",
        "issues": [],
    }
    assert "DSV4 MTP metadata inconsistent" not in "\n".join(static["issues"])


def test_dsv4_static_audit_rejects_missing_mtp_weights_when_config_expects_them(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
    )
    (tmp_path / "jang_config.json").write_text(
        '{"model_family":"deepseek_v4","weight_format":"mxtq","drop_mtp":false}'
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.embed.weight":"model-00001-of-00001.safetensors"}}'
    )
    row = ModelRow(
        id="dsv4_tmp",
        label="tmp dsv4",
        path=str(tmp_path),
        family="deepseek_v4",
        cache_profile="dsv4_composite",
    )

    static = static_audit(row)

    assert static["mtp"]["runtime_available"] is False
    assert any("DSV4 MTP metadata inconsistent" in issue for issue in static["issues"])


def test_dsv4_static_audit_does_not_claim_mtp_runtime_from_weights_only(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_nextn_predict_layers":1}'
    )
    (tmp_path / "jang_config.json").write_text(
        '{"model_family":"deepseek_v4","weight_format":"mxtq","drop_mtp":false}'
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"mtp.0.layers.0.self_attn.q_proj.weight":"model.safetensors"}}'
    )
    row = ModelRow(
        id="dsv4_tmp",
        label="tmp dsv4",
        path=str(tmp_path),
        family="deepseek_v4",
        cache_profile="dsv4_composite",
    )

    static = static_audit(row)

    assert static["mtp"]["artifact_available"] is True
    assert static["mtp"]["runtime_available"] is False
    assert static["mtp"]["status"] == "weights_present_runtime_unwired"
    assert "not wired" in static["mtp"]["runtime_reason"]


def test_dsv4_static_audit_reports_malformed_mtp_index(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"deepseek_v4","num_nextn_predict_layers":0}'
    )
    (tmp_path / "jang_config.json").write_text(
        '{"model_family":"deepseek_v4","weight_format":"mxtq","drop_mtp":false}'
    )
    (tmp_path / "model.safetensors.index.json").write_text("{")
    row = ModelRow(
        id="dsv4_tmp",
        label="tmp dsv4",
        path=str(tmp_path),
        family="deepseek_v4",
        cache_profile="dsv4_composite",
    )

    static = static_audit(row)

    assert static["mtp"]["runtime_available"] is False
    assert static["mtp"]["status"] == "metadata_inconsistent"
    assert any("model.safetensors.index.json" in issue for issue in static["mtp"]["issues"])
    assert any("DSV4 MTP metadata inconsistent" in issue for issue in static["issues"])


def test_dsv4_capability_contract_uses_native_cache_not_experimental_mode():
    rows = {row.id: row for row in ROWS}
    dsv4 = rows["dsv4_tq"]

    caps = {
        "supports_thinking": True,
        "supported_modes": ["instruct", "reasoning"],
        "experimental_modes": [],
        "reasoning_efforts": ["high", "max"],
        "cache": {
            "native": {
                "family": "deepseek_v4",
                "schema": "deepseek_v4_v7",
                "cache_type": "native_composite",
                "generic_turboquant_kv": {"enabled": False},
            }
        },
    }
    assert capability_endpoint_contract_ok(dsv4, caps)

    stale_caps = {
        "supports_thinking": False,
        "supported_modes": ["instruct"],
        "experimental_modes": ["raw-thinking"],
        "reasoning_efforts": [],
        "cache": {"dsv4_composite_state": True},
    }
    assert not capability_endpoint_contract_ok(dsv4, stale_caps)


def test_zaya_cache_probe_checks_cache_not_repeated_style_compliance():
    rows = {row.id: row for row in ROWS}
    probe = cache_exact_hit_probe(rows["zaya_mxfp4"])

    assert probe["expected"] == "blue"
    assert probe["min_count"] == 1
    assert probe["strict_short_answer"] is True
    assert probe["thinking"] is False
    assert "which color word repeats" in probe["messages"][0]["content"]


def test_loop_score_catches_no_space_cjk_and_emoji_repetition():
    assert simple_loop_score("音苷苷和音诺族的对策" * 80) >= 0.25
    assert simple_loop_score("👀" * 200) >= 0.25
    assert simple_loop_score("state " * 80) >= 0.25
    assert simple_loop_score("Paris is the capital of France.") < 0.25


def test_loop_score_does_not_false_fail_coherent_cyrillic_list():
    text = (
        "1. Сцена создаётся с травой и деревьями.\n"
        "2. Игрок управляется клавиатурой перемещаясь.\n"
        "3. Дробовик стреляет при нажатии специальной кнопки.\n"
        "4. Враги появляются случайным образом в пределах поля зрения.\n"
        "5. Игра завершается после определённого количества целей."
    )

    assert simple_loop_score(text) < 0.25


def test_dsv4_long_output_requires_stop_not_length_cap():
    complete_answer = (
        "Visible answer with a complete useful paragraph. "
        "It has enough substance to meet the long-row threshold, reaches an "
        "actual stop condition, and does not rely on a length cap."
    )
    assert dsv4_long_context_full_output_ok(
        code=200,
        finish="stop",
        full_text=f"Reasoning summary.\n{complete_answer}",
        content=complete_answer,
        split_ok=True,
        loop_score=0.0,
    )

    assert not dsv4_long_context_full_output_ok(
        code=200,
        finish="length",
        full_text=f"Reasoning summary.\n{complete_answer}",
        content=complete_answer,
        split_ok=True,
        loop_score=0.0,
    )


def test_dsv4_max_mode_requires_visible_exact_answer_not_reasoning_mention():
    assert dsv4_thinking_mode_max_ok(
        code=200,
        finish="stop",
        content="7",
        reasoning="3 plus 4 equals 7.",
    )

    assert dsv4_thinking_mode_max_ok(
        code=200,
        finish="stop",
        content="3 plus 4 equals 7. The final digit is 7.",
        reasoning="3 plus 4 equals 7.",
    )

    assert not dsv4_thinking_mode_max_ok(
        code=200,
        finish="stop",
        content="I should calculate it.",
        reasoning="3 plus 4 equals 7, so final answer should be 7.",
    )


def test_dsv4_live_gate_does_not_use_hard_repetition_block_env():
    rows = {row.id: row for row in ROWS}
    env = audit_child_env_for_row(
        rows["dsv4_tq"],
        {"VMLX_DSV4_HARD_REP_BLOCK": "1"},
    )

    assert "VMLX_DSV4_HARD_REP_BLOCK" not in env


def test_dsv4_live_gate_prefers_local_jang_source_when_available(tmp_path):
    rows = {row.id: row for row in ROWS}
    local_jang = tmp_path / "jang-tools"
    (local_jang / "jang_tools" / "dsv4").mkdir(parents=True)
    (local_jang / "jang_tools" / "dsv4" / "mlx_model.py").write_text("# local dsv4 runtime\n")

    env = audit_child_env_for_row(
        rows["dsv4_tq"],
        {
            "VMLINUX_AUDIT_USE_SOURCE_JANG": "1",
            "VMLINUX_JANG_TOOLS_SOURCE": str(local_jang),
            "PYTHONPATH": "/existing/path",
        },
    )

    assert env["PYTHONPATH"] == str(local_jang)


def test_live_gate_child_env_prevents_packaged_app_pycache_writes():
    rows = {row.id: row for row in ROWS}
    env = audit_child_env_for_row(
        rows["zaya_vl_jangtq4"],
        {"PYTHONDONTWRITEBYTECODE": "0", "PYTHONPATH": "/repo/shadow"},
    )

    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"] == ""


def test_live_gate_server_command_does_not_import_from_repo_cwd(tmp_path):
    command = getattr(audit_harness, "live_server_command", None)

    assert command is not None
    cmd = command(
        tmp_path / "python3",
        tmp_path / "model",
        8123,
        tmp_path / "block-cache",
    )
    assert cmd[:4] == [str(tmp_path / "python3"), "-B", "-s", "-P"]
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "vmlx_engine.cli"


def test_live_gate_server_command_does_not_force_sampling_defaults(tmp_path):
    cmd = audit_harness.live_server_command(
        tmp_path / "python3",
        tmp_path / "model",
        8123,
        tmp_path / "block-cache",
    )

    for flag in (
        "--default-temperature",
        "--default-top-p",
        "--default-top-k",
        "--default-min-p",
        "--default-repetition-penalty",
    ):
        assert flag not in cmd
