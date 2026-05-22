def test_cache_architecture_contract_pins_named_cache_edges():
    from tests.cross_matrix import run_cache_architecture_contract as gate

    required_pytest = gate.REQUIRED_CACHE_TEST_MARKERS
    required_api = gate.REQUIRED_API_CACHE_CHECKS
    required_api_command = gate.REQUIRED_API_CACHE_COMMAND_MARKERS

    assert "test_memory_pressure_reuses_shorter_dsv4_terminal_composite_prefix" in required_pytest
    assert "test_memory_pressure_refuses_dsv4_partial_without_terminal_composite" in required_pytest
    assert "test_memory_pressure_partially_reuses_hybrid_ssm_with_aligned_checkpoint" in required_pytest
    assert "test_prompt_disk_l2_hit_backfills_paged_cache_for_partial_reuse" in required_pytest
    assert "test_hybrid_ssm_checkpoint_alignment_falls_back_to_exact_aligned_state" in required_pytest
    assert "test_hybrid_ssm_auto_mode_disables_live_tq_but_keeps_stored_kv_q4" in required_pytest
    assert "test_qwen3_5_moe_linear_attention_keeps_selective_live_tq_and_stored_kv_q4" in required_pytest
    assert "test_serialize_tq_cache_mixed_hybrid" in required_pytest
    assert "test_tq_tensors_roundtrip_via_safetensors" in required_pytest
    assert "test_mllm_stats_include_cache_fields" in required_pytest
    assert "native_cache_status_reports_dsv4_separately_from_tq_kv" in required_api_command
    assert "native_cache_status_reports_zaya_typed_cca" in required_api_command
    assert "dsv4_native_cache_status" in required_api
    assert "zaya_typed_cca_status" in required_api
    assert "hybrid_ssm_partial_reuse" in required_api
    assert "turboquant_disk_roundtrip" in required_api

    command = gate.COMMANDS["cache_family_pytest"]
    assert "-vv" in command
