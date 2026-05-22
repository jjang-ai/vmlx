def test_vl_media_cache_contract_pins_named_engine_rows():
    from tests.cross_matrix import run_vl_media_cache_contract as gate

    required = set(gate.REQUIRED_VL_MEDIA_CACHE_TEST_MARKERS)

    assert "test_video_url_parsed" in required
    assert "test_image_and_video_url_content_part_schemas" in required
    assert "test_jangtq_video_fallback_no_crash" in required
    assert "test_qwen36_jangtq_is_detected_as_vl" in required
    assert "test_jang_config_qwen36_can_be_video" in required
    assert "test_hybrid_ssm_auto_mode_disables_live_tq_kv" in required
    assert "test_deferred_rederive_covers_post_output_hybrid_ssm_paths" in required
    assert "test_mllm_side_captures_at_prefill_not_finalize" in required
    assert "test_multi_turn_alternating_image_text_distinct_keys" in required
    assert "test_mllm_chat_template_receives_effective_tool_schemas" in required
    assert "test_simple_engine_forwards_effective_tools_to_mllm_chat" in required

    engine_command = gate.COMMANDS["engine_vl_media_cache_contracts"][1]
    assert "-vv" in engine_command


def test_vl_media_cache_contract_pins_named_panel_rows():
    from tests.cross_matrix import run_vl_media_cache_contract as gate

    required = set(gate.REQUIRED_PANEL_VL_MEDIA_TEST_MARKERS)

    assert "exposes read_video as a file tool next to read_image" in required
    assert "injects image and video tool results as multimodal follow-up content parts" in required
    assert "builds real multimodal follow-up content in text-image-video order" in required
    assert "keeps media bytes out of plain tool text and returns data URLs separately" in required
    assert "VLM gets --continuous-batching for BatchedEngine with MLLMScheduler" in required
    assert "VLM continuous batching off emits explicit opt-out and suppresses cache stack" in required
    assert "keeps ZAYA1-VL multimodal when a stale stamp says text" in required
    assert "marks Qwen3.6 VL JANG bundles with indexed MTP tensors as native MTP capable" in required
    assert "keeps MXTQ/JANGTQ Qwen hybrid VLM multimodal" in required
    assert "keeps mxfp4 Qwen hybrid VLM multimodal" in required
    assert "keeps mxfp8 Qwen hybrid VLM multimodal" in required
    assert "marks non-JANG Qwen 3.6 MoE bundles with vision/video metadata as multimodal" in required
    assert "does not route Nemotron-H text extracts through MLLM from stale sidecars" in required
    assert "multimodal/VLM detection suppresses --enable-jit because mlx-vlm streaming is not compile-safe" in required
    assert "VLM with all caching features works together" in required

    for name in (
        "panel_vl_media_followup_contracts",
        "panel_vlm_settings_contracts",
        "panel_vlm_family_detection",
    ):
        command = gate.COMMANDS[name][1]
        assert "--reporter=verbose" in command

    family_command = " ".join(gate.COMMANDS["panel_vlm_family_detection"][1])
    assert "indexed MTP" in family_command


def test_vl_media_cache_contract_marker_validation_fails_if_required_row_missing():
    from tests.cross_matrix import run_vl_media_cache_contract as gate

    results = {
        "engine_vl_media_cache_contracts": {
            "returncode": 0,
            "stdout": "test_video_url_parsed PASSED",
            "stdout_tail": [],
        },
        "panel_vl_media_followup_contracts": {
            "returncode": 0,
            "stdout": "exposes read_video as a file tool next to read_image",
            "stdout_tail": [],
        },
        "panel_vlm_settings_contracts": {
            "returncode": 0,
            "stdout": "VLM gets --continuous-batching for BatchedEngine with MLLMScheduler",
            "stdout_tail": [],
        },
        "panel_vlm_family_detection": {
            "returncode": 0,
            "stdout": "",
            "stdout_tail": [],
        },
    }

    assert "test_image_and_video_url_content_part_schemas" in gate._missing_engine_markers(results)
    assert (
        "injects image and video tool results as multimodal follow-up content parts"
        in gate._missing_panel_markers(results)
    )
    assert (
        "marks non-JANG Qwen 3.6 MoE bundles with vision/video metadata as multimodal"
        in gate._missing_panel_markers(results)
    )
