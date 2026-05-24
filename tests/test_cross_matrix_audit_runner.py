"""Regression tests for the live production-family audit harness.

These tests keep the harness from producing false-positive PASS rows when a
model mentions the expected token inside reasoning but never emits visible
content.
"""

import json

from tests.cross_matrix.run_production_family_audit import (
    ModelRow,
    ROWS,
    audit_child_env_for_row,
    cache_probe_content_ok,
    cache_exact_hit_probe,
    cache_exact_hit_required,
    capability_endpoint_contract_ok,
    chat_basic_turn_ok,
    dsv4_release_fault_matrix,
    dsv4_output_precision_boundary,
    dsv4_rope_scaling_contract,
    dsv4_thinking_mode_max_ok,
    dsv4_threejs_identifier_integrity_ok,
    dsv4_threejs_single_file_ok,
    dsv4_long_context_full_output_ok,
    dsv4_long_prompt_specs,
    extract_anthropic_text_and_stop,
    extract_ollama_visible_text_and_stop,
    is_non_length_stop,
    live_request_perf_summary,
    normalize_python_executable,
    normalize_short_answer,
    family_matches_expected,
    live_loop_probe_name,
    multilingual_loop_quality_ok,
    production_family_audit_summary,
    sampling_loop_risk_summary,
    simple_loop_score,
    static_audit,
    text_quality_summary,
)
from tests.cross_matrix import run_production_family_audit as audit_harness


def test_live_audit_root_is_this_checkout_not_stale_absolute_path():
    expected_root = audit_harness.Path(audit_harness.__file__).resolve().parents[2]
    source = audit_harness.Path(audit_harness.__file__).read_text(encoding="utf-8")

    assert audit_harness.ROOT == expected_root
    assert 'Path("/Users/example/mlx/vllm-mlx")' not in source


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

    for row_id in (
        "hy3_preview_jangtq2",
        "hy3_preview_jang_2k",
        "hy3_preview_jang_2l",
    ):
        assert rows[row_id].family == "hy_v3"
        assert rows[row_id].expect_reasoning is True
        assert rows[row_id].expect_tool_parser == "hunyuan"
        assert rows[row_id].slow is True


def test_hy3_affine_rows_track_generation_and_quantization_contract():
    rows = {row.id: row for row in ROWS}

    for row_id in ("hy3_preview_jang_2k", "hy3_preview_jang_2l"):
        static = static_audit(rows[row_id])

        assert static["registry"]["family_name"] == "hy_v3"
        assert static["registry"]["reasoning_parser"] == "qwen3"
        assert static["registry"]["tool_parser"] == "hunyuan"
        assert static["registry"]["cache_type"] == "kv"
        assert static["registry"]["think_in_template"] is False
        assert static["generation"]["max_new_tokens"] == 2048
        assert static["generation"]["temperature"] == 0.0
        assert static["generation"]["top_p"] == 1
        assert static["generation"]["top_k"] == 0
        assert static["sampling_loop_risk"]["max_new_tokens"] == 2048
        assert static["sampling_loop_risk"]["output_cap_source"] == "jang_config.chat.sampling_defaults.max_new_tokens"
        assert static["sampling_loop_risk"]["reasoning_default_mode"] == "no_think"
        assert "bundle omits repetition_penalty; engine must omit it rather than invent a hidden floor" in static["sampling_loop_risk"]["review_notes"]
        assert static["jang"]["quantization"]["group_size"] == 128
        assert static["jang"]["runtime"]["bundle_has_mtp"] is False
        assert not static["issues"]


def test_static_audit_surfaces_sampling_loop_risk_fields_without_forcing_defaults(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "hy_v3"}))
    (tmp_path / "generation_config.json").write_text(
        json.dumps(
            {
                "temperature": 0.9,
                "top_p": 1.0,
                "top_k": -1,
            }
        )
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "model_family": "hy_v3",
                "chat": {
                    "reasoning": {"supported": True},
                    "sampling_defaults": {
                        "temperature": 0.9,
                        "top_p": 1.0,
                        "top_k": -1,
                    },
                },
            }
        )
    )
    row = ModelRow(
        id="hy3_tmp",
        label="tmp hy3",
        path=str(tmp_path),
        family="hy_v3",
        expect_reasoning=True,
        expect_tool_parser="hunyuan",
    )

    static = static_audit(row)
    risk = static["sampling_loop_risk"]

    assert static["generation"]["top_k"] == -1
    assert static["generation"]["repetition_penalty"] is None
    assert risk["status"] == "review"
    assert risk["output_cap_source"] == "engine_fallback_4096"
    assert risk["top_k_disabled"] is True
    assert risk["repetition_sources"] == {}
    assert any("max_new_tokens" in note for note in risk["review_notes"])
    assert any("repetition_penalty" in note for note in risk["review_notes"])
    assert any("Auto must stay model/template-owned" in note for note in risk["review_notes"])


def test_sampling_loop_risk_prefers_jang_chat_defaults_over_generation_config():
    risk = sampling_loop_risk_summary(
        gen={
            "max_new_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1,
        },
        sampling={
            "max_new_tokens": 4096,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 0,
            "repetition_penalty_chat": 1.05,
            "repetition_penalty_thinking": 1.0,
        },
        reasoning={"supported": True, "default_mode": "chat"},
        registry={"reasoning_parser": "qwen3"},
    )

    assert risk["status"] == "review"
    assert risk["output_cap_source"] == "jang_config.chat.sampling_defaults.max_new_tokens"
    assert risk["max_new_tokens"] == 4096
    assert risk["temperature"] == 0.6
    assert risk["top_p"] == 0.9
    assert risk["top_k"] == 0
    assert risk["top_k_disabled"] is True
    assert risk["repetition_sources"] == {
        "jang_config.chat.sampling_defaults.repetition_penalty_chat": 1.05,
        "jang_config.chat.sampling_defaults.repetition_penalty_thinking": 1.0,
        "generation_config.repetition_penalty": 1.1,
    }
    assert risk["reasoning_default_mode"] == "chat"
    assert risk["review_notes"] == ["bundle disables top_k with a non-positive sentinel"]


def test_production_family_audit_summary_surfaces_missing_and_review_rows():
    results = {
        "rows": [
            {
                "static": {
                    "id": "present_ok",
                    "exists": True,
                    "issues": [],
                    "sampling_loop_risk": {"status": "covered"},
                }
            },
            {
                "static": {
                    "id": "missing_model",
                    "exists": False,
                    "issues": ["registry family mismatch"],
                    "sampling_loop_risk": {"status": "review"},
                }
            },
            {
                "static": {
                    "id": "present_review",
                    "exists": True,
                    "issues": ["needs live proof"],
                    "sampling_loop_risk": {"status": "review"},
                },
                "live": {"status": "DEFERRED"},
            },
        ]
    }

    summary = production_family_audit_summary(results)

    assert summary == {
        "row_count": 3,
        "missing_rows": ["missing_model"],
        "issue_rows": ["missing_model", "present_review"],
        "sampling_review_rows": ["missing_model", "present_review"],
        "live_status_counts": {"DEFERRED": 1},
    }


def test_ling_rows_do_not_expect_reasoning():
    rows = {row.id: row for row in ROWS}

    assert rows["ling_flash_tq"].expect_reasoning is False
    assert rows["ling_flash_tq2_crack"].expect_reasoning is False
    assert rows["ling_flash_mxfp4_crack"].expect_reasoning is False


def test_static_audit_accepts_registry_family_aliases_for_ling_and_nemotron():
    assert family_matches_expected("bailing_hybrid", "ling")
    assert family_matches_expected("nemotron_omni", "nemotron_h")
    assert not family_matches_expected("minimax", "ling")


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


def test_responses_tool_choice_requires_exact_function_arguments():
    good = [
        {
            "type": "function_call",
            "name": "list_directory",
            "arguments": '{"path": "."}',
        }
    ]
    wrapped = [
        {
            "type": "function_call",
            "name": "list_directory",
            "arguments": '{"path": "<value>.</value>"}',
        }
    ]
    placeholder = [
        {
            "type": "function_call",
            "name": "list_directory",
            "arguments": '{"path": "VALUE HERE"}',
        }
    ]

    assert audit_harness.responses_tool_call_arguments_ok(
        good,
        expected_name="list_directory",
        expected_arguments={"path": "."},
    )
    assert not audit_harness.responses_tool_call_arguments_ok(
        wrapped,
        expected_name="list_directory",
        expected_arguments={"path": "."},
    )
    assert not audit_harness.responses_tool_call_arguments_ok(
        placeholder,
        expected_name="list_directory",
        expected_arguments={"path": "."},
    )


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


def test_live_request_perf_summary_derives_wall_token_rates_from_chat_usage():
    perf = live_request_perf_summary(
        {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        },
        elapsed_sec=10.0,
    )

    assert perf == {
        "elapsed_sec": 10.0,
        "basis": "api_usage_wall_clock",
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_tokens_per_sec_wall": 12.0,
        "completion_tokens_per_sec_wall": 3.0,
        "total_tokens_per_sec_wall": 15.0,
    }


def test_live_request_perf_summary_derives_wall_token_rates_from_responses_usage():
    perf = live_request_perf_summary(
        {
            "input_tokens": 64,
            "output_tokens": 16,
            "total_tokens": 80,
        },
        elapsed_sec=4.0,
    )

    assert perf["prompt_tokens"] == 64
    assert perf["completion_tokens"] == 16
    assert perf["prompt_tokens_per_sec_wall"] == 16.0
    assert perf["completion_tokens_per_sec_wall"] == 4.0


def test_live_request_perf_summary_marks_missing_usage_without_fake_rates():
    perf = live_request_perf_summary(None, elapsed_sec=3.0)

    assert perf["elapsed_sec"] == 3.0
    assert perf["basis"] == "missing_api_usage"
    assert "completion_tokens_per_sec_wall" not in perf

def test_live_audit_root_points_at_current_checkout():
    """Worktree gates must not import modules from the stale main checkout."""
    expected = audit_harness.Path(__file__).resolve().parents[1]

    assert audit_harness.ROOT == expected


def test_dsv4_row_points_at_current_sub80_upload_candidate_bundle():
    rows = {row.id: row for row in ROWS}

    assert rows["dsv4_tq"].path.endswith("DeepSeek-V4-Flash-JANGTQ-K")


def test_dsv4_local_affine_artifact_is_in_production_audit_matrix():
    rows = {row.id: row for row in ROWS}

    row = rows["dsv4_jang_local"]
    assert row.path == "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG"
    assert row.family == "deepseek_v4"
    assert row.cache_profile == "dsv4_composite"
    assert row.expect_tool_parser == "dsml"


def test_dsv4_decode_speed_gate_tracks_jangtq_and_pure_affine_lanes():
    from tests.cross_matrix import run_decode_speed_gate

    assert run_decode_speed_gate.ROWS["dsv4_k"].path.endswith(
        "DeepSeek-V4-Flash-JANGTQ-K"
    )
    assert "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG" in (
        run_decode_speed_gate.DSV4_AFFINE_MODEL_CANDIDATES
    )
    assert run_decode_speed_gate.ROWS["dsv4_jang_dq2_gate3math6"].path in (
        run_decode_speed_gate.DSV4_AFFINE_MODEL_CANDIDATES
    )


def test_dsv4_production_audit_pure_affine_lane_points_at_present_bundle():
    rows = {row.id: row for row in ROWS}

    row = rows["dsv4_jang_dq2_gate3math6"]
    assert row.path == "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG"
    assert row.family == "deepseek_v4"
    assert row.cache_profile == "dsv4_composite"
    assert row.expect_tool_parser == "dsml"


def test_dsv4_decode_speed_gate_reads_nested_native_cache_health():
    from tests.cross_matrix.run_decode_speed_gate import cache_health_mismatches

    registry = {
        "cache_type": "kv",
        "cache_subtype": "deepseek_v4_composite",
    }
    health = {
        "native_cache": {
            "cache_type": "native_composite",
            "generic_turboquant_kv": {"enabled": False},
        }
    }

    assert cache_health_mismatches(registry, health) == []


def test_dsv4_long_context_gate_defaults_to_existing_affine_candidate():
    from tests.cross_matrix import run_dsv4_long_context_gate

    assert "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG" in (
        run_dsv4_long_context_gate.DSV4_AFFINE_MODEL_CANDIDATES
    )
    assert run_dsv4_long_context_gate.DEFAULT_MODEL in (
        run_dsv4_long_context_gate.DSV4_AFFINE_MODEL_CANDIDATES
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


def test_dsv4_responses_cache_gate_defaults_to_existing_affine_candidate():
    from tests.cross_matrix import run_dsv4_responses_cache_gate

    assert "/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANG" in (
        run_dsv4_responses_cache_gate.DSV4_AFFINE_MODEL_CANDIDATES
    )
    assert run_dsv4_responses_cache_gate.DEFAULT_MODEL in (
        run_dsv4_responses_cache_gate.DSV4_AFFINE_MODEL_CANDIDATES
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
    precision = static["dsv4_precision_boundary"]
    assert precision["output_head_quantized"] is True
    assert precision["needs_source_or_rebuild_clearance"] is True
    assert any("output-head/final-norm precision boundary" in issue for issue in static["issues"])


def test_dsv4_precision_boundary_marks_quantized_output_head(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    save_file(
        {
            "head.weight": np.zeros((4, 2), dtype=np.uint32),
            "head.scales": np.ones((4, 1), dtype=np.float16),
            "head.biases": np.zeros((4, 1), dtype=np.float16),
            "norm.weight": np.ones((2,), dtype=np.float16),
        },
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "head.weight": "model.safetensors",
                    "head.scales": "model.safetensors",
                    "head.biases": "model.safetensors",
                    "norm.weight": "model.safetensors",
                }
            }
        )
    )

    precision = dsv4_output_precision_boundary(tmp_path)

    assert precision["checked"] is True
    assert precision["output_head_dtype"] == "U32"
    assert precision["output_head_quantized"] is True
    assert precision["final_norm_dtype"] == "F16"
    assert precision["final_norm_lower_precision"] is True
    assert precision["needs_source_or_rebuild_clearance"] is True


def test_dsv4_precision_boundary_accepts_source_like_output_head(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    save_file(
        {
            "head.weight": np.zeros((4, 2), dtype=np.float32),
            "norm.weight": np.ones((2,), dtype=np.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "head.weight": "model.safetensors",
                    "norm.weight": "model.safetensors",
                }
            }
        )
    )

    precision = dsv4_output_precision_boundary(tmp_path)

    assert precision["checked"] is True
    assert precision["output_head_dtype"] == "F32"
    assert precision["output_head_quantized"] is False
    assert precision["final_norm_dtype"] == "F32"
    assert precision["final_norm_lower_precision"] is False
    assert precision["needs_source_or_rebuild_clearance"] is False


def test_dsv4_release_fault_matrix_separates_ruled_out_layers_from_open_blockers(tmp_path):
    matrix = dsv4_release_fault_matrix(
        model_dir=tmp_path,
        precision_boundary={
            "output_head_quantized": True,
            "final_norm_lower_precision": True,
            "needs_source_or_rebuild_clearance": True,
        },
        rope_scaling_contract={"runtime_repair_required": True},
    )

    assert matrix["checked"] is True
    assert matrix["release_blocked"] is True
    ruled_out = {item["layer"] for item in matrix["ruled_out"]}
    assert {
        "prompt_renderer",
        "api_endpoint_adapter",
        "stream_sse_or_response_assembly",
        "tokenizer_roundtrip",
        "mxtq_only_kernel_path",
        "output_head_or_final_norm_only",
    }.issubset(ruled_out)
    open_layers = {item["layer"] for item in matrix["open_blockers"]}
    assert {
        "source_or_broader_body_comparison",
        "shared_dsv4_runtime_or_body_precision",
        "full_output_identifier_gate",
        "dsv4_composite_prefix_cache_equivalence",
        "compressed_rope_scaling_metadata",
    }.issubset(open_layers)
    assert any("not a release pass" in step for step in matrix["required_next_proofs"])
    assert any("cache equivalence" in step for step in matrix["required_next_proofs"])


def test_dsv4_rope_scaling_contract_flags_null_flash_yarn_metadata(tmp_path):
    cfg = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 0, 4, 128, 0],
        "rope_scaling": None,
    }

    contract = dsv4_rope_scaling_contract(cfg, model_dir=tmp_path / "DeepSeek-V4-Flash-JANGTQ-K")

    assert contract["checked"] is True
    assert contract["runtime_repair_required"] is True
    assert contract["bundle_rope_scaling"] is None
    assert contract["runtime_rope_scaling"]["factor"] == 16


def test_dsv4_static_audit_exposes_missing_source_rope_scaling(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 160000,
        "compress_ratios": [0, 0, 4, 128, 0],
        "rope_scaling": None,
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "model_family": "deepseek_v4",
        "weight_format": "mxtq",
        "chat": {
            "encoder": "encoding_dsv4",
            "encoder_fn": "encode_messages",
            "sampling_defaults": {"repetition_penalty_thinking": 1.0},
        },
    }))
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

    assert static["dsv4_rope_scaling_contract"]["runtime_repair_required"] is True
    assert any("YaRN rope_scaling" in issue for issue in static["issues"])
    open_layers = {
        item["layer"]
        for item in static["dsv4_release_fault_matrix"]["open_blockers"]
    }
    assert "compressed_rope_scaling_metadata" in open_layers


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


def test_multilingual_loop_gate_rejects_stray_cjk_in_cyrillic_answer():
    text = (
        "1. Сцена создаётся с травой и деревьями.\n"
        "2. Игрок управляется клавиатурой и стреляет.\n"
        "3. Враги генерируются случайным образом в р的林.\n"
        "4. Урон отнимает здоровье при попадании.\n"
        "5. Победа зависит от выживания и счёта."
    )

    assert text_quality_summary(text)["cjk_chars"] > 0
    assert not multilingual_loop_quality_ok(text)


def test_multilingual_loop_gate_rejects_unrelated_latin_words_in_cyrillic_answer():
    text = (
        "1. Сцена создаётся с травой, деревьями и Himmel.\n"
        "2. Герой двигается по траектории, используя клавиатуру.\n"
        "3. Оружие визуализируется и стреляет боеприпасами.\n"
        "4. Враги меняют траекторию и избегают попаданий.\n"
        "5. Система оценивает очки, здоровье и завершает уровень."
    )

    assert text_quality_summary(text)["latin_chars"] > 0
    assert not multilingual_loop_quality_ok(text)


def test_multilingual_loop_gate_allows_expected_web_technical_tokens():
    text = (
        "1. HTML сцена создаётся с лесом и небом.\n"
        "2. Three.js управляет камерой и объектами.\n"
        "3. JavaScript обновляет движение игрока.\n"
        "4. CSS настраивает размер полотна.\n"
        "5. WebGL выводит врагов и эффекты."
    )

    assert text_quality_summary(text)["latin_chars"] > 0
    assert multilingual_loop_quality_ok(text)


def test_multilingual_loop_gate_allows_prompted_wasd_control_acronym():
    text = (
        "1. Игрок управляет охотником, который бежит по лесной тропе, используя WASD или стрелки.\n"
        "2. Дробовик позволяет стрелять по врагам и создавать короткий луч выстрела вперёд.\n"
        "3. Враги появляются случайно из-за деревьев и движутся к охотнику с разной скоростью.\n"
        "4. При попадании выстрела враг получает урон, теряет очки здоровья и исчезает.\n"
        "5. За каждого врага начисляются очки, а игра заканчивается при истощении здоровья."
    )

    assert text_quality_summary(text)["latin_chars"] == 4
    assert multilingual_loop_quality_ok(text)


def test_loop_sensitive_rows_require_live_loop_probe():
    rows = {row.id: row for row in ROWS}

    assert live_loop_probe_name(rows["ling_flash_tq"]) == "ling_multilingual_loop_trigger"
    assert live_loop_probe_name(rows["minimax_m27_small_tq"]) == "minimax_multilingual_loop_trigger"
    assert live_loop_probe_name(rows["hy3_preview_jangtq2"]) == "hy3_multilingual_loop_trigger"
    assert live_loop_probe_name(rows["qwen36_dense_jang"]) is None


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

    assert not dsv4_long_context_full_output_ok(
        code=200,
        finish="stop",
        full_text=f"Reasoning summary.\n{complete_answer}",
        content=complete_answer,
        split_ok=True,
        loop_score=0.0,
        content_contract_ok=False,
    )


def test_dsv4_game_gate_requires_complete_threejs_html_not_canvas_sketch():
    good = """
<!DOCTYPE html>
<html><body><script src="https://cdn.jsdelivr.net/npm/three/build/three.min.js"></script>
<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
const renderer = new THREE.WebGLRenderer();
const player = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
function spawnEnemy(){}
function updateProjectiles(){ /* collision and health score updates */ }
function loop(){ requestAnimationFrame(loop); }
loop();
</script><div>health score enemy projectile collision</div></body></html>
"""
    bad_canvas = """
<!DOCTYPE html>
<html><body><canvas id="c"></canvas><script>
function loop(){ requestAnimationFrame(loop); }
// health score enemy projectile collision
</script></body></html>
"""
    fenced = f"```html\n{good}\n```"
    placeholder = good.replace("</script>", "// TODO omitted for brevity\n</script>")

    assert dsv4_threejs_single_file_ok(good)
    assert not dsv4_threejs_single_file_ok(bad_canvas)
    assert not dsv4_threejs_single_file_ok(fenced)
    assert not dsv4_threejs_single_file_ok(placeholder)

    corrupt_api = good.replace("THREE.Scene", "THREE.ScScene").replace(
        "THREE.PerspectiveCamera", "THREE.PPerspectiveCamera"
    )
    assert not dsv4_threejs_single_file_ok(corrupt_api)


def test_dsv4_identifier_integrity_gate_rejects_corrupted_api_names():
    good = "\n".join(
        [
            "const scene = new THREE.Scene();",
            "const renderer = new THREE.WebGLRenderer();",
            "const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);",
            "const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());",
        ]
    )
    corrupt = good.replace("THREE.Scene", "THREE.ScScene").replace(
        "THREE.WebGLRenderer", "THREE.WebWebGLRenderer"
    )

    assert dsv4_threejs_identifier_integrity_ok(good)
    assert not dsv4_threejs_identifier_integrity_ok(f"```javascript\n{good}\n```")
    assert not dsv4_threejs_identifier_integrity_ok(corrupt)


def test_dsv4_live_gate_skips_full_game_when_identifier_gate_failed():
    source = audit_harness.Path(audit_harness.__file__).read_text(encoding="utf-8")

    assert "skipped_due_to_identifier_integrity_failure" in source
    assert "identifier_artifact" in source


def test_dsv4_identifier_gate_requests_token_logprobs_for_root_cause_artifacts():
    source = audit_harness.Path(audit_harness.__file__).read_text(encoding="utf-8")
    start = source.index('"dsv4_threejs_identifier_integrity"')
    block = source[start : source.index("ident_content", start)]

    assert '"logprobs": True' in block
    assert '"top_logprobs": 5' in block


def test_dsv4_long_prompt_specs_use_right_rail_and_budget():
    specs = {spec["name"]: spec for spec in dsv4_long_prompt_specs()}

    vc = specs["vc_project_plan"]
    assert vc["enable_thinking"] is True
    assert vc["max_tokens"] >= 2500

    game = specs["game_design_long_context"]
    assert game["enable_thinking"] is False
    assert game["max_tokens"] == 2600
    assert game["temperature"] == 0.0
    assert game["top_p"] == 1.0
    assert game["repetition_penalty"] == 1.0
    assert "THREE.Scene" in game["prompt"]
    assert "THREE.WebGLRenderer" in game["prompt"]
    assert "ending with </html>" in game["prompt"]


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
        "--max-tokens",
    ):
        assert flag not in cmd


def test_live_gate_server_command_uses_dsv4_native_prefix_flag(tmp_path):
    rows = {row.id: row for row in ROWS}

    dsv4_cmd = audit_harness.live_server_command(
        tmp_path / "python3",
        tmp_path / "dsv4",
        8123,
        tmp_path / "block-cache",
        row=rows["dsv4_jang_local"],
    )
    assert "--dsv4-enable-prefix-cache" in dsv4_cmd
    assert "--enable-prefix-cache" not in dsv4_cmd

    zaya_cmd = audit_harness.live_server_command(
        tmp_path / "python3",
        tmp_path / "zaya",
        8123,
        tmp_path / "block-cache",
        row=rows["zaya_mxfp4"],
    )
    assert "--enable-prefix-cache" in zaya_cmd
    assert "--dsv4-enable-prefix-cache" not in zaya_cmd
